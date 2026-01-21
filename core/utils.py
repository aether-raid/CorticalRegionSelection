import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import random
import warnings
from typing import Optional
from pathlib import Path
import re
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from .encoder import spike_encoder
from models.snn.decoder import Decoder
from contextlib import nullcontext
from sklearn.model_selection import GroupShuffleSplit


def set_deterministic(
    seed: int,
    *,
    enforce_cublas: bool = True,
    enable_tf32: bool = False,
    warn_only: bool = True,
) -> int:
    """Set global seeds and enable deterministic behavior for reproducible runs.

    Parameters
    ----------
    seed: int
        Base seed to use for python, numpy, and torch RNGs.
    enforce_cublas: bool
        If True and CUDA is available, sets `CUBLAS_WORKSPACE_CONFIG` (recommended
        for deterministic cuBLAS). If left unset, you may still get nondeterministic
        cuBLAS behavior.
    enable_tf32: bool
        Whether to enable TF32 math on supported hardware. Disabling TF32 reduces
        nondeterministic variations from mixed-precision algorithms.
    warn_only: bool
        If True (default), PyTorch's `torch.use_deterministic_algorithms` will be
        called with `warn_only=True` on supported versions, so unsupported ops
        will not raise at runtime but will emit a warning. If False, strict
        determinism is enforced and PyTorch will raise on unsupported ops.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    if enforce_cublas and torch.cuda.is_available():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if not enable_tf32:
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prefer 'warn_only=True' to avoid runtime exceptions when a specific op lacks
    # a deterministic implementation (e.g. adaptive_max_pool2d backward on CUDA).
    # Users may set warn_only=False to strictly enforce determinism (will raise on
    # unsupported ops). We attempt to call PyTorch API accordingly and fall back
    # gracefully for older PyTorch versions.
    try:
        # torch.use_deterministic_algorithms accepts warn_only on newer PyTorch
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:
        # Older torch versions may not support the warn_only kwarg
        try:
            torch.use_deterministic_algorithms(True)
            if warn_only:
                warnings.warn("PyTorch does not support warn_only on this version; strict deterministic algorithms were enabled and may raise on unsupported ops.", RuntimeWarning)
        except Exception as exc:  # pragma: no cover - older torch versions
            warnings.warn(f"Deterministic algorithms unavailable: {exc}", RuntimeWarning)
    except Exception as exc:
        # If strict and an error occurs, re-raise. For warn_only True, show a warning.
        if warn_only:
            warnings.warn(f"Deterministic algorithms could not be fully enabled: {exc}; falling back to warn_only behavior.", RuntimeWarning)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                # As a last resort try enabling deterministic algorithms without warn_only
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception as exc2:  # pragma: no cover - older torch versions
                    warnings.warn(f"Failed to enable deterministic algorithms: {exc2}", RuntimeWarning)
        else:
            raise

    return seed


def make_worker_init_fn(base_seed: int):
    """Return a DataLoader worker init function that preserves determinism."""

    def _worker_init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn


def make_generator(seed: Optional[int]):
    """Create a torch.Generator seeded for deterministic DataLoader shuffles."""
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# -------------------------
# Subject/group utilities
# -------------------------

def extract_subject_id(file_id: str | dict | object) -> str:
    """Extract a robust subject/person group ID from dataset sample metadata.

    Accepts a filename stem or a full sample dict/object. Handles common
    conventions (case-insensitive):
    - BIDS-like: sub-01, subject_03, subject12
    - Shorthand: S01, S_1, S-12, s12
    - Participants: participant07, user05, person7, pp03
    - Generic IDs: P01, ID12
    - Fallback: first token before '_' or '-' of the stem

    Returns an uppercase canonical group string, e.g. 'S01', 'PARTICIPANT07'.
    """

    # If a full sample is passed, try common keys first
    if isinstance(file_id, dict):
        for key in ("subject", "participant", "user", "person", "subj", "pid", "id"):
            if key in file_id and file_id[key] is not None:
                val = str(file_id[key])
                m = re.search(r"(\d{1,4})", val)
                if m:
                    num = int(m.group(1))
                    prefix = key.upper()
                    return f"{prefix}{num:02d}"
                return val.upper()
        # Fallback to explicit file_id if present
        if "file_id" in file_id:
            s = str(file_id["file_id"]) 
        else:
            s = str(file_id)
    else:
        s = str(file_id)

    # Normalize separators for easier token handling
    s_norm = s.replace(" ", "_")

    # 0) UN_ pattern (Universe dataset: UN_XXX)
    m = re.search(r"(?i)(?<![A-Za-z0-9])un[-_]?(\d{1,4})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"UN{int(m.group(1)):03d}"

    # 1) SUBJECT/SUB patterns
    m = re.search(r"(?i)(?<![A-Za-z0-9])sub(?:ject)?[-_]?(\d{1,3})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"S{int(m.group(1)):02d}"

    # 2) S patterns (avoid matching 'session') using boundaries
    m = re.search(r"(?i)(?<![A-Za-z0-9])s[-_]?(\d{1,3})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"S{int(m.group(1)):02d}"

    # 3) Participant/user/person/pp
    m = re.search(r"(?i)(?<![A-Za-z0-9])participant[-_]?(\d{1,3})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"PARTICIPANT{int(m.group(1)):02d}"
    m = re.search(r"(?i)(?<![A-Za-z0-9])user[-_]?(\d{1,3})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"USER{int(m.group(1)):02d}"
    m = re.search(r"(?i)(?<![A-Za-z0-9])person[-_]?(\d{1,3})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"PERSON{int(m.group(1)):02d}"
    m = re.search(r"(?i)(?<![A-Za-z0-9])pp[-_]?(\d{1,3})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"PP{int(m.group(1)):02d}"

    # 4) Pxx, IDxx
    m = re.search(r"(?i)(?<![A-Za-z0-9])p[-_]?(\d{1,3})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"P{int(m.group(1)):02d}"
    m = re.search(r"(?i)(?<![A-Za-z0-9])id[-_]?(\d{1,4})(?![A-Za-z0-9])", s_norm)
    if m:
        return f"ID{int(m.group(1)):02d}"

    # 5) Fallbacks: take leading token before '_' or '-', drop trailing session markers
    lead = re.split(r"[-_]", s_norm)[0]
    return lead.upper()


def build_groups_from_dataset(raw_dataset, indices: Optional[list] = None) -> list:
    """Return a list of group IDs (subject IDs) aligned with the given indices."""
    if indices is None:
        indices = list(range(len(raw_dataset)))
    groups = []
    for i in indices:
        sample = raw_dataset[i]
        if isinstance(sample, dict):
            # Prefer explicit subject-like keys when available
            for key in ("subject", "participant", "user", "person", "subj", "pid", "id"):
                if key in sample and sample[key] is not None:
                    groups.append(extract_subject_id(sample))
                    break
            else:
                file_id = sample.get('file_id', str(i))
                groups.append(extract_subject_id(file_id))
        else:
            file_id = getattr(sample, 'file_id', str(i))
            groups.append(extract_subject_id(file_id))
    print(f"Found {len(set(groups))} unique subject/group IDs.")
    return groups


def split_by_group(all_indices: list, groups: list, *, val_size=0.15, test_size=0.15, random_state=42):
    """Leakage-free train/val/test split by subject groups.

    Performs two GroupShuffleSplit passes: train vs temp, then temp -> val/test.
    Returns (train_idx, val_idx, test_idx) as index lists relative to the original dataset.
    
    Fallback: If only 1 unique group exists, falls back to random split (no group guarantee).
    """
    assert len(all_indices) == len(groups)
    
    # Check if we have enough unique groups for splitting
    unique_groups = len(set(groups))
    if unique_groups < 3:
        print(f"⚠️  Warning: Only {unique_groups} unique group(s) found. Falling back to random split (no subject-wise guarantee).")
        from sklearn.model_selection import train_test_split
        
        # First split: train vs temp
        total_temp = val_size + test_size
        train_idx, temp_idx = train_test_split(
            all_indices, test_size=total_temp, random_state=random_state
        )
        
        # Second split: val vs test
        val_ratio = val_size / max(total_temp, 1e-8)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1.0 - val_ratio), random_state=random_state
        )
        
        return train_idx, val_idx, test_idx
    
    # Standard group-based splitting
    total_temp = val_size + test_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=total_temp, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(all_indices, groups=groups))

    # Split temp into val/test
    temp_indices = [all_indices[i] for i in temp_idx]
    temp_groups = [groups[i] for i in temp_idx]
    # ratio for val within temp
    val_ratio = val_size / max(total_temp, 1e-8)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_ratio), random_state=random_state)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_indices, groups=temp_groups))
    val_idx = [temp_indices[i] for i in val_rel_idx]
    test_idx = [temp_indices[i] for i in test_rel_idx]

    return train_idx, val_idx, test_idx

def create_cv_folds(raw_dataset, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = [raw_dataset[i]['label'] for i in range(len(raw_dataset))]
    return list(skf.split(range(len(raw_dataset)), labels))

def get_sklearn_data_idtthisused(dataset, indices):
    features = [dataset[idx]['features'].numpy() for idx in indices]
    labels = [dataset[idx]['label'] for idx in indices]
    return np.array(features), np.array(labels)

def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    *,
    num_workers: int = 0,
    generator: Optional[torch.Generator] = None,
    worker_init_fn=None,
    pin_memory: bool = False,
):
    """Create train/val/test DataLoaders with optional deterministic helpers."""

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": worker_init_fn,
        "pin_memory": pin_memory,
    }
    if generator is not None:
        loader_kwargs["generator"] = generator

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

def evaluate_model(model, data_loader, criterion, optimizer, device, is_eval=False, task_type="classification"):
    # For val/test, is_eval = True
    if is_eval:
        model.eval()
    else:
        model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []  
    
    context = torch.no_grad() if is_eval else nullcontext()
    
    with context:
        for batch in data_loader:
            inputs = batch['features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if not is_eval:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)

            if task_type == "classification":
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = outputs.squeeze()

            all_preds.extend(np.atleast_1d(predicted.detach().cpu().numpy()))
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()))

    loss = running_loss / len(data_loader.dataset)

    if task_type == "classification":
        metric = accuracy_score(all_labels, all_preds)
    else:
        # Calculate RMSE - compatible with all scikit-learn versions
        mse = mean_squared_error(all_labels, all_preds)
        metric = np.sqrt(mse)  # RMSE

    return loss, metric, all_preds, all_labels


def evaluate_snn(model, model_name, epoch, encoder, cfg, data_loader, criterion, optimizer, device, snn_type, is_eval, task_type, num_classes, gradient_clip, decoder=None):
    # For val/test, is_eval = True
    if is_eval:
        model.eval()
        if encoder:
            encoder.eval()
        if decoder:
            decoder.eval()
    else:
        model.train()
        if encoder:
            encoder.train()
        if decoder:
            decoder.train()

    running_loss = 0.0
    all_preds, all_labels = [], []
    
    context = torch.no_grad() if is_eval else nullcontext()
    with context:
        for batch in data_loader:
            raw_inputs = batch['tensor'].to(device)
            labels = batch['label'].to(device)
            inputs = spike_encoder(cfg, raw_inputs, encoder)
            
            # Inputs shape: T, B, N, C            
            if model_name.upper() == "STDP_LSM" or model_name.lower() == "stdp_lsm":
                spk_rec = model(inputs, epoch, is_eval)
            else:
                spk_rec = model(inputs)
            
            # Initialize decoder on first batch if not provided
            if decoder is None:
                if task_type=="classification":
                    decoder = Decoder(spk_rec.shape[-1],num_classes).to(device)  
                else:   
                    decoder = Decoder(spk_rec.shape[-1], 1).to(device)
                # Set decoder to appropriate mode
                if is_eval:
                    decoder.eval()
                else:
                    decoder.train()
                
            logits = decoder(spk_rec)
            loss = criterion(logits, labels)
            
            #for train 
            if not is_eval:
                optimizer.zero_grad()
                loss.backward()
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()                
                
            running_loss += loss.item() * inputs.size(0)
            if task_type=="classification":
                _, predicted = logits.max(dim=1)
            else:
                predicted = logits.squeeze()
            
            # Detach predictions before converting to numpy (handles both train and eval modes)
            # Convert to list to handle both scalars and arrays
            pred_numpy = predicted.detach().cpu().numpy()
            label_numpy = labels.detach().cpu().numpy()
            
            # Use tolist() to handle both 0-d and multi-d arrays
            if pred_numpy.ndim == 0:
                all_preds.append(pred_numpy.item())
                all_labels.append(label_numpy.item())
            else:
                all_preds.extend(pred_numpy.tolist())
                all_labels.extend(label_numpy.tolist())

    
    loss = running_loss / len(data_loader.dataset)
    
    if task_type=="classification":
        metric = accuracy_score(all_labels, all_preds)
        metric_name="Accuracy"
    else:
        # Calculate RMSE - compatible with all scikit-learn versions
        mse = mean_squared_error(all_labels, all_preds)
        metric = np.sqrt(mse)  # RMSE
        metric_name="RMSE"
    return loss, metric, all_preds, all_labels, metric_name, decoder






# add these imports at the top of the file if not present
# from models.snn.decoder import Decoder
# from span import SPANOptimizer, build_latency_targets

from contextlib import nullcontext

def build_latency_targets(labels, T, O, t_min=1, t_max=None, device=None):
    """
    Build latency-coded targets for SPAN training in classification.
    Each class gets a spike at a specific time, with earlier times for lower class indices.
    
    Args:
        labels: [B] tensor of class indices
        T: number of time steps
        O: number of output classes
        t_min: minimum spike time
        t_max: maximum spike time (if None, use T-1)
        device: torch device
    
    Returns:
        targets: [T, B, O] tensor with one spike per class per sample
    """
    if t_max is None:
        t_max = T - 1
    
    # Ensure valid time range
    t_max = min(t_max, T - 1)
    t_min = min(t_min, t_max)
    
    B = labels.shape[0]
    targets = torch.zeros(T, B, O, device=device)
    
    # Map each class to a specific time step
    # Distribute classes evenly across the time range
    if O > 1:
        time_span = max(1, t_max - t_min)
        time_per_class = time_span / max(1, O - 1)
        for b in range(B):
            class_idx = labels[b].item()
            spike_time = int(t_min + class_idx * time_per_class)
            spike_time = max(0, min(spike_time, T - 1))  # Clamp to valid range [0, T-1]
            targets[spike_time, b, class_idx] = 1.0
    else:
        # Single output (binary or regression)
        spike_time = max(0, min((t_min + t_max) // 2, T - 1))
        for b in range(B):
            targets[spike_time, b, 0] = 1.0
    
    return targets

def evaluate_snn_span(
    model, model_name, epoch, encoder, cfg, data_loader,
    criterion, optimizer, device, snn_type, is_eval, task_type, num_classes,
    gradient_clip,
    # NEW:
    decoder_train: str = "ce",        # "ce" | "span"
    span_obj=None,                    # SPANOptimizer or None
    decoder=None,                     # persistent nn.Linear readout for SPAN or CE
    span_tmin: int = 4,               # latency coding params
    span_tmax_cap: int = 24
):
    """
    If decoder_train == "ce":
        - behaves like your original evaluate_snn (CE/MSE with backprop)
        - (keeps creating a fresh Decoder per batch like before)

    If decoder_train == "span":
        - trains/uses a persistent linear decoder via SPAN temporal rule
        - no CE/MSE; loss is reported as 0.0 (metric computed from decoder output)
        - returns the 'decoder' so you can keep it across epochs
    """
    use_span = (decoder_train.lower() == "span")
    is_classification = (task_type == "classification")

    if is_eval:
        model.eval()
        if encoder: encoder.eval()
    else:
        model.train()
        if encoder: encoder.train()

    running_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.no_grad() if is_eval else nullcontext()
    with ctx:
        for batch in data_loader:
            raw_inputs = batch['tensor'].to(device)
            labels = batch['label'].to(device)
            inputs = spike_encoder(cfg, raw_inputs, encoder)   # [T,B,*,C] -> model expects this

            # Forward through SNN
            if model_name[0:8] == "STDP_LSM":
                spk_rec = model(inputs, epoch, is_eval)        # shape [T,B,F]
            else:
                spk_rec = model(inputs)                        # shape [T,B,F]
            # make sure it's float for downstream ops
            spk_rec = spk_rec.float()

            if not use_span:
                # -------- CE/MSE path (original behavior) --------
                if task_type == "classification":
                    # NOTE: your original code re-creates Decoder each batch – kept identical
                    dec = Decoder(spk_rec.shape[-1], num_classes).to(device)
                    logits = dec(spk_rec)                      # CE expects logits from time-collapsed decoder
                else:
                    dec = Decoder(spk_rec.shape[-1], 1).to(device)
                    logits = dec(spk_rec)                      # regression

                loss = criterion(logits, labels)

                if not is_eval:
                    optimizer.zero_grad()
                    loss.backward()
                    if gradient_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                if task_type == "classification":
                    _, predicted = logits.max(dim=1)
                else:
                    predicted = logits.squeeze()

            else:
                # -------- SPAN path (temporal decoder learning) --------
                # persistent decoder (lazy init once we know feature dim)
                # Handle different spike recording shapes
                if spk_rec.dim() == 2:
                    # Shape is [B, F], add time dimension
                    spk_rec = spk_rec.unsqueeze(0)  # [1, B, F]
                elif spk_rec.dim() == 4:
                    # Shape is [T, B, ?, F], collapse extra dimension
                    T, B, _, F = spk_rec.shape
                    spk_rec = spk_rec.squeeze(2)  # [T, B, F]
                
                T, B, F = spk_rec.shape
                O = num_classes if is_classification else 1
                if decoder is None:
                    # simple linear readout trained by SPAN (no backprop)
                    import torch.nn as nn
                    decoder = nn.Linear(F, O, bias=True).to(device)

                # TRAIN step only if not eval
                if not is_eval:
                    if is_classification:
                        # Latency-coded targets for classification
                        t_max = min(span_tmax_cap, max(1, T - 1))
                        targets = build_latency_targets(labels, T=T, O=O,
                                                        t_min=span_tmin, t_max=t_max, device=device)  # [T,B,O]
                    else:
                        # Regression: spread the target value across time as a smooth temporal trace
                        # This gives SPAN more temporal signal to learn from
                        targets = torch.zeros(T, B, O, device=device, dtype=spk_rec.dtype)
                        
                        # Normalize labels to a reasonable range for temporal coding
                        labels_norm = labels.float()
                        
                        # Create a temporal Gaussian-like trace centered in time
                        # This spreads the target signal across multiple time steps
                        t_center = T // 2
                        sigma = max(1.0, T / 6.0)  # Width of the temporal window
                        
                        for t in range(T):
                            # Gaussian weighting: stronger signal near center, decaying at edges
                            weight = torch.exp(torch.tensor(-((t - t_center) ** 2) / (2 * sigma ** 2), device=device))
                            targets[t, :, 0] = labels_norm * weight

                    # One SPAN update on readout
                    span_obj.step(pre_spikes=spk_rec, targets=targets,
                                  W=decoder.weight, b=decoder.bias)

                # Inference: collapse spikes over time (sum) then decode
                X = spk_rec.sum(dim=0)         # [B,F]
                logits = decoder(X)            # [B,O]
                
                # Compute loss for monitoring (not used for gradient updates with SPAN)
                if is_classification:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                else:
                    loss = torch.nn.functional.mse_loss(logits.squeeze(), labels.float())

                # Metrics
                if is_classification:
                    predicted = logits.argmax(dim=1)
                else:
                    predicted = logits.squeeze()

                # Track loss for monitoring
                running_loss += loss.item() * inputs.size(0)

            # Collect preds/labels (handles both paths)
            pred_np = predicted.detach().cpu().numpy()
            label_np = labels.detach().cpu().numpy()

            if pred_np.ndim == 0:
                all_preds.append(pred_np.item())
                all_labels.append(label_np.item())
            else:
                all_preds.extend(pred_np.tolist())
                all_labels.extend(label_np.tolist())

    # Aggregate metrics
    loss = running_loss / len(data_loader.dataset)
    if task_type == "classification":
        from sklearn.metrics import accuracy_score
        metric = accuracy_score(all_labels, all_preds)
        metric_name = "Accuracy"
    else:
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(all_labels, all_preds)
        metric = np.sqrt(mse)
        metric_name = "RMSE"

    # Return decoder so SPAN readout persists across calls
    return loss, metric, all_preds, all_labels, metric_name, decoder









def plot_training_history(
    train_losses, val_losses,
    train_metrics, val_metrics,
    title, fold_suffix="",
    metric_name="Accuracy",
    model_obj=None,
    skip_save=False
):
    # Skip plotting during hyperparameter tuning to avoid disk clutter
    if skip_save:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Metric plot (Accuracy or RMSE)
    ax2.plot(train_metrics, label=f'Train {metric_name}')
    ax2.plot(val_metrics, label=f'Validation {metric_name}')
    ax2.set_title(f'{metric_name} History')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True)

    # If a model object is provided, use its actual class name in the title/filename
    model_cls_name = None
    if model_obj is not None:
        try:
            model_cls_name = model_obj.__class__.__name__
        except Exception:
            model_cls_name = None

    # Add timestamp to the figure title and filename for traceability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_title = f"{model_cls_name} Training History" if model_cls_name else title
    fig.suptitle(f"{display_title} — {timestamp}")
    plt.tight_layout()
    # Save with timestamp to avoid overwriting and to keep track of runs
    # Use model class name in filename if available to ensure it reflects the actual class
    base_name = model_cls_name if model_cls_name else title
    filename = f"{base_name.replace(' ', '_')}_{timestamp}{fold_suffix}.png"
    save_path = os.path.join("cv_results", filename) if fold_suffix else filename
    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', fold_suffix=""):
    """Plot a confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save with fold suffix if applicable
    filename = f"{title.replace(' ', '_')}{fold_suffix}.png"
    plt.savefig(os.path.join("cv_results", filename) if fold_suffix else filename)
    plt.close()

# ================ DATA PREPARATION UTILITIES ================
def get_sklearn_data(dataset, indices):
    """Extract data for scikit-learn models"""
    features = []
    labels = []
    
    # Debug: Check first sample to understand data structure
    if len(indices) > 0:
        first_sample = dataset[indices[0]]
        print(f"DEBUG - get_sklearn_data: Sample type = {type(first_sample)}")
        
    
    for idx in indices:
        sample = dataset[idx]
        # Handle both dict and object formats
        if isinstance(sample, dict):
           
            features.append(sample['features'].numpy())
            labels.append(sample['label'])
        else:
           
            # Assume it's an object with attributes
            features.append(sample.features.numpy())
            labels.append(sample.label)
    
    print(f"DEBUG - Extracted {len(features)} samples, feature shape: {np.array(features).shape}")
    return np.array(features), np.array(labels)

# NOTE: create_data_loaders is defined above with extended, deterministic-friendly
# signature (supports worker_init_fn, generator, num_workers, pin_memory).
# Remove older duplicate to avoid shadowing and signature mismatches.
# ================ OVERSAMPLING UTILITIES ================
def apply_random_oversampling(X, y, random_state=42, strategy='auto'):
    """
    Apply random oversampling to balance class distribution
    
    Args:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        random_state: Random seed for reproducibility
        strategy: Sampling strategy ('auto', 'minority', 'not majority', or dict)
    
    Returns:
        X_resampled: Oversampled features
        y_resampled: Oversampled labels
    """
    print(f"Original class distribution: {Counter(y)}")
    
    ros = RandomOverSampler(random_state=random_state, sampling_strategy=strategy)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    print(f"Oversampled class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def apply_oversampling_to_indices(raw_dataset, indices, random_state=42, strategy='auto'):
    """
    Apply oversampling to dataset indices (for PyTorch datasets)
    
    Args:
        raw_dataset: The raw EEG dataset
        indices: List of indices to oversample
        random_state: Random seed
        strategy: Sampling strategy
    
    Returns:
        oversampled_indices: New list of indices after oversampling
    """
    # Get labels for the given indices
    labels = [raw_dataset[i]['label'] for i in indices]
    
    print(f"Original class distribution: {Counter(labels)}")
    
    # Convert indices to 2D array for RandomOverSampler
    X_indices = np.array(indices).reshape(-1, 1)
    y_labels = np.array(labels)
    
    # Apply oversampling
    ros = RandomOverSampler(random_state=random_state, sampling_strategy=strategy)
    X_resampled, y_resampled = ros.fit_resample(X_indices, y_labels)
    
    # Extract the oversampled indices
    oversampled_indices = X_resampled.flatten().tolist()
    
    print(f"Oversampled class distribution: {Counter(y_resampled)}")
    print(f"Original samples: {len(indices)}, Oversampled samples: {len(oversampled_indices)}")
    
    return oversampled_indices




# Example Grids

# HYPERPARAM_GRIDS = {
#     'logistic_regression': {
#         'C': [0.01, 0.1, 1, 10, 100],
#         'penalty': ['l1', 'l2'],
#         'solver': ['saga']
#     },
#     'svm': {
#         'C': [0.1, 1, 10],
#         'gamma': ['scale', 'auto', 0.1, 1],
#         'kernel': ['rbf', 'poly', 'sigmoid']
#     },
#     'mlp': {
#         'input_size': [input_size],  # Fixed
#         'hidden_size': [64, 128, 256],
#         'num_classes': [num_classes]  # Fixed
#     },
#    'relu_kan' = {
#        "input_dim": [input_size],  # Fixed
#        "num_classes": [num_classes],  # Usually fixed
#        "width": [[64, 128], [128, 256]],
#        "grid": [5, 10],
#        "k": [2, 3],
#        "lr": [0.001, 0.0005]
#    }
# }

# # How to run:

# results = run_experiment(
#     raw_dataset,
#     selection_method='mrmr',
#     models_to_run=['logistic_regression', 'svm', 'mlp'],
#     n_features=30,
#     epochs=100,
#     hyperparam_grids=HYPERPARAM_GRIDS
# )

# # Run cross-validated with tuning
# cv_results = run_cv_experiment(
#     raw_dataset,
#     selection_method='anova',
#     models_to_run=['svm', 'mlp'],
#     n_features=20,
#     epochs=50,
#     n_splits=5,
#     hyperparam_grids=HYPERPARAM_GRIDS
# )


# ============================================================================
# Standardized Tuning Results Saving
# ============================================================================

def save_final_results(
    out_dir,
    *,
    model: str,
    dataset: str,
    best_params: dict,
    task: str = "regression",
    test_results: Optional[dict] = None,
    val_metric: Optional[float] = None,
    val_metric_name: str = "rmse",
    n_trials: Optional[int] = None,
    cv_folds: Optional[int] = None,
    target: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    filename: str = "final_results.json",
) -> Path:
    """
    Save tuning results in a standardized JSON format across all tuning scripts.
    
    This utility ensures consistent structure for final_results.json files,
    making it easy to compare results across different models and experiments.
    
    Parameters
    ----------
    out_dir : Path or str
        Output directory for the results file.
    model : str
        Model name (e.g., 'snn', 'eegnet', 'ctm', 'relukan').
    dataset : str
        Dataset name (e.g., 'mocas', 'htc', 'nback').
    best_params : dict
        Dictionary of best hyperparameters found during tuning.
    task : str, optional
        Task type: 'regression' or 'classification'. Default: 'regression'.
    test_results : dict, optional
        Dictionary of test set metrics (e.g., {'rmse': 0.5, 'mse': 0.25}).
    val_metric : float, optional
        Best validation metric value achieved during tuning.
    val_metric_name : str, optional
        Name of the validation metric (e.g., 'rmse', 'accuracy'). Default: 'rmse'.
    n_trials : int, optional
        Number of optimization trials run.
    cv_folds : int, optional
        Number of cross-validation folds used.
    target : str, optional
        Target variable name (e.g., 'arousal', 'valence').
    extra_metadata : dict, optional
        Any additional metadata to include in the results.
    filename : str, optional
        Output filename. Default: 'final_results.json'.
    
    Returns
    -------
    Path
        Path to the saved results file.
    
    Example
    -------
    >>> from core.utils import save_final_results
    >>> save_final_results(
    ...     out_dir="./results/snn/mocas",
    ...     model="tcn",
    ...     dataset="mocas",
    ...     best_params={'lr': 0.001, 'hidden_size': 128},
    ...     test_results={'rmse': 0.45, 'mse': 0.2025},
    ...     val_metric=0.48,
    ...     n_trials=100,
    ...     cv_folds=5
    ... )
    """
    from pathlib import Path
    import json
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Safely convert numpy types to Python native types
    def _convert_value(v):
        if hasattr(v, 'item'):  # numpy scalar
            return v.item()
        elif isinstance(v, (list, tuple)):
            return [_convert_value(x) for x in v]
        elif isinstance(v, dict):
            return {k: _convert_value(val) for k, val in v.items()}
        return v
    
    # Build standardized result structure
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    final_data = {
        # Core identification
        'model': str(model),
        'dataset': str(dataset),
        'task': str(task),
        'timestamp': timestamp,
        
        # Results
        'best_params': {k: _convert_value(v) for k, v in best_params.items()},
    }
    
    # Add validation metric with consistent naming
    if val_metric is not None:
        final_data[f'best_val_{val_metric_name}'] = float(val_metric)
    
    # Add test results
    if test_results is not None:
        final_data['test_results'] = {k: _convert_value(v) for k, v in test_results.items()}
    
    # Add optional fields if provided
    if n_trials is not None:
        final_data['n_trials'] = int(n_trials)
    if cv_folds is not None:
        final_data['cv_folds'] = int(cv_folds)
    if target is not None:
        final_data['target'] = str(target)
    
    # Merge extra metadata
    if extra_metadata:
        for k, v in extra_metadata.items():
            if k not in final_data:  # Don't overwrite core fields
                final_data[k] = _convert_value(v)
    
    # Save with error handling
    results_file = out_dir / filename
    try:
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
    except Exception as e:
        # Emergency fallback
        emergency_file = out_dir / f"{filename.replace('.json', '')}_emergency.txt"
        with open(emergency_file, 'w') as f:
            f.write(f"Error saving JSON: {e}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Best params: {best_params}\n")
            if val_metric is not None:
                f.write(f"Best val {val_metric_name}: {val_metric}\n")
            if test_results is not None:
                f.write(f"Test results: {test_results}\n")
        raise RuntimeError(f"Failed to save final_results.json, emergency file saved to {emergency_file}") from e
    
    return results_file


def load_final_results(results_path) -> dict:
    """
    Load a standardized final_results.json file.
    
    Parameters
    ----------
    results_path : Path or str
        Path to the final_results.json file or directory containing it.
    
    Returns
    -------
    dict
        Loaded results dictionary.
    """
    from pathlib import Path
    import json
    
    results_path = Path(results_path)
    if results_path.is_dir():
        results_path = results_path / "final_results.json"
    
    with open(results_path, 'r') as f:
        return json.load(f)


#