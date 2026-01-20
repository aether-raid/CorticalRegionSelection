import re
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
import numpy as np
import numpy.typing as npt
import pandas as pd
from torch.utils.data import Dataset


@dataclass
class EEGRawDatasetEntry:
    raw_features: npt.NDArray[np.float32]
    label: int
    file_id: str


#Raw Datasets are for classification only
class EEGRawDataset(Dataset):
    def __init__(self, data_dir: str, *, target_suffix: str = None, verbose: bool = True):
        """
        EEG classification dataset with multi-subscale support.
        
        Args:
            data_dir: Path to classification dataset directory
            target_suffix: (keyword-only) TLX subscale to use for classification labels. Options:
                          None (default) - uses 'combined' TLX
                          'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration'
            verbose: (keyword-only) Whether to print loading information
        
        Directory structure:
            OLD (folder-based): data_dir/low/*.parquet, data_dir/medium/*.parquet, data_dir/high/*.parquet
            NEW (metadata-based): data_dir/all/*.parquet + data_dir/classification_metadata.json
        """
        super().__init__()
        # Initialize data directory
        self.data_dir = Path(data_dir)
        self.target_suffix = 'combined' if target_suffix is None else target_suffix
        self.verbose = verbose
        
        exclude_metric_patterns = [
            r"theta/beta",
            r"\(theta \+ alpha\)/\(alpha \+ beta\)",
            r"gamma/delta",
            r"\(gamma \+ beta\)/\(delta \+ alpha\)",
            r"brain_rate"
        ]
        exclude_bands_for_patterns = ['alpha', 'gamma', 'beta', 'theta', 'delta']

        exclude_metric_patterns_2 = [
            r"absolute_power", r"relative_power"
        ]
        exclude_bands_for_patterns_2 = ['Overall']

        # Check for metadata-based format (new multi-subscale approach)
        metadata_file = self.data_dir / 'classification_metadata.json'
        
        if metadata_file.exists():
            # NEW FORMAT: Load from 'all' folder using metadata JSON
            import json
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            # Collect files from 'all' directory
            all_dir = self.data_dir / 'all'
            if not all_dir.exists():
                raise ValueError(f"Metadata file found but 'all' directory missing: {all_dir}")
            
            self.file_list: list[tuple[Path, int]] = []
            
            for parquet_file in all_dir.glob('*.parquet'):
                filename = parquet_file.name
                
                if filename not in self.metadata:
                    if self.verbose:
                        print(f"Warning: {filename} not in metadata, skipping")
                    continue
                
                # Get label for selected subscale
                file_metadata = self.metadata[filename]
                
                if self.target_suffix not in file_metadata:
                    if self.verbose:
                        print(f"Warning: {filename} missing {self.target_suffix} label, skipping")
                    continue
                
                label = file_metadata[self.target_suffix]
                self.file_list.append((parquet_file, label))
            
            # Create reverse label mapping for display
            self.label_map = {0: 'low', 1: 'medium', 2: 'high'}
            
            if self.verbose:
                print(f"Loaded classification dataset from metadata")
                print(f"  Target subscale: {self.target_suffix}")
                print(f"  Label mapping: {self.label_map}")
                print(f"  Total files: {len(self.file_list)}")
        
        else:
            # OLD FORMAT: Folder-based structure (backward compatibility)
            if self.target_suffix != 'combined':
                raise ValueError(
                    f"Subscale '{self.target_suffix}' requires metadata-based classification dataset. "
                    f"Old folder-based datasets (low/medium/high/) only support combined TLX. "
                    f"To use subscales, regenerate the dataset with the updated load_htc.py. "
                    f"See MULTI_SUBSCALE_CLASSIFICATION_GUIDE.md for details."
                )
            
            # Find all label folders (classes)
            self.label_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
            
            if not self.label_folders:
                raise ValueError(f"No label folders found in {data_dir}")
            
            # Create label mapping (folder name -> integer)
            self.label_map = {folder.name: idx for idx, folder in enumerate(self.label_folders)}
            if self.verbose:
                print(f"Loaded classification dataset from folders (old format)")
                print(f"  Label mapping: {self.label_map}")
            
            # Collect all parquet files with their labels
            self.file_list: list[tuple[Path, int]] = []
            for folder in self.label_folders:
                label = self.label_map[folder.name]
                for parquet_file in folder.glob('*.parquet'):
                    self.file_list.append((parquet_file, label))
            
            if self.verbose:
                print(f"  Total files: {len(self.file_list)}")

        # Process first file to determine feature structure
        sample_df = pd.read_parquet(self.file_list[0][0], engine="fastparquet")
        self.bands = sample_df.columns.get_level_values(0).unique().tolist()
        self.channels = sample_df.columns.get_level_values(1).unique().tolist()
        

        self.feature_names = sample_df.index.tolist()

        # Create flattened feature names: [feature]_[band]_[channel]
        self.flattened_features = []
        for feature in self.feature_names:
            # Convert feature to string to ensure it works with re.fullmatch()
            feature_str = str(feature)
            for band in self.bands:
                for channel in self.channels:
                    should_exclude = False
                    for pattern in exclude_metric_patterns:
                        if re.fullmatch(pattern, feature_str) and band in exclude_bands_for_patterns:
                            should_exclude = False
                            break
                    for pattern in exclude_metric_patterns_2:
                        if re.fullmatch(pattern, feature_str) and band in exclude_bands_for_patterns_2:
                            should_exclude = False
                            break
                    if not should_exclude:
                        self.flattened_features.append(f"{feature_str}_{band}_{channel}")
        
        # Feature index mapping
        self.feature_indices = self._get_feature_indices()
        #print("Feature indices:", self.feature_indices)
    
    def _get_feature_indices(self):
        return {col: idx for idx, col in enumerate(self.flattened_features)}
    
    def categorize_electrode(self, channel_name: str) -> Literal['Fp', 'F', 'T', 'O', 'P'] | None:
        # channel_name example: 'EEG.AF3'
        
        # Remove 'EEG.' prefix if present
        if channel_name.startswith("EEG."):
            label = channel_name[4:]  # e.g. 'AF3'
        else:
            label = channel_name

        # Now check categories
        if label.startswith(("AF", "Fp", "FP")):  # add 'Fp' or 'FP' just in case
            return "Fp"
        elif label.startswith(("F", "FC")):
            return "F"
        elif label.startswith("T"):
            return "T"
        elif label.startswith("O"):
            return "O"
        elif label.startswith("P"):
            return "P"
        else:
            return None

    def filter_by_electrodes(self, electrode_categories: list[Literal['Fp', 'F', 'T', 'O', 'P']]):
        """
        Keep only features whose EEG channel belongs to one of the electrode_categories.
        electrode_categories: list of ['Fp', 'F', 'T', 'O', 'P']
        """
        print(f"🔎 Filtering electrodes: {electrode_categories}")

        matched_features = []
        matched_channels = set()

        for feat in self.flattened_features:
            try:
                # Extract the channel name (last part after '_')
                channel = feat.split('_')[-1]  # e.g. 'EEG.AF3'
                category = self.categorize_electrode(channel)
                if category and category in electrode_categories:
                    matched_features.append(feat)
                    matched_channels.add(channel)
            except Exception as e:
                print(f"⚠️ Skipping feature '{feat}' due to error: {e}")
                continue

        self.flattened_features = matched_features
        self.feature_indices = self._get_feature_indices()

        print(f"✅ Filtered down to {len(self.flattened_features)} features.")
        print(f"✅ Matched channels (sample): {sorted(list(matched_channels))[:10]}")

    def _process_file(self, file_path: Path):
        # Load and process a single parquet file
        #df = pd.read_parquet(file_path)
        df = pd.read_parquet(file_path, engine="fastparquet")
        
        feature_vector = np.zeros(len(self.flattened_features), dtype=np.float32)
        
        # Populate feature vector
        for feature in self.feature_names:
            for band in self.bands:
                for channel in self.channels:
                    col_name = f"{feature}_{band}_{channel}"
                    if col_name in self.flattened_features:
                        pos = self.feature_indices[col_name]
                        try:
                            value = df.loc[feature, (band, channel)]
                        except KeyError:
                            value = np.nan
                        feature_vector[pos] = value

        # # Handle missing values
        # if np.isnan(feature_vector).any():
        #     col_mean = np.nanmean(feature_vector)
        #     feature_vector = np.nan_to_num(feature_vector, nan=col_mean)
        
        return feature_vector

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> EEGRawDatasetEntry:
        file, label = self.file_list[idx]
        feature_vector = self._process_file(file)
        
        return EEGRawDatasetEntry(
            raw_features=feature_vector,
            label=label,
            file_id=file.stem
        )
