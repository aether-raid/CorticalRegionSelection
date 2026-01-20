from dataclasses import dataclass
from typing import Callable
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler,
    RobustScaler
)
import torch
from torch.utils.data import Dataset
from datasets.raw_eeg import EEGRawDataset


@dataclass
class EEGProcessedDatasetEntry:
    features: torch.Tensor
    label: torch.Tensor
    file_id: str


class EEGProcessedDataset(Dataset):
    def __init__(self,
                 raw_dataset: EEGRawDataset,
                
                 scaler: StandardScaler | MinMaxScaler | MaxAbsScaler | RobustScaler,
                 feature_indices : list[int] | None=None, 
                 transform: Callable | None = None):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.feature_indices = feature_indices  # Selected feature indices
        self.scaler = scaler  # Fitted scaler
        self.transform = transform

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx: int):
        sample = self.raw_dataset[idx]
        # lil fix, was trying to access attribute of instead of dict key
        features = sample['raw_features'] if isinstance(sample, dict) else sample.raw_features
        
        # Apply feature selection
        if self.feature_indices is not None:
            # Ensure feature_indices is a numpy array for proper indexing
            import numpy as np
            if isinstance(self.feature_indices, list):
                indices = np.array(self.feature_indices)
            elif hasattr(self.feature_indices, 'tolist'):  # Already numpy array
                indices = self.feature_indices
            else:
                indices = np.array(list(self.feature_indices))
            features = features[indices]
        
        # Scale features manually to avoid sklearn's validation that rejects inf values
        # StandardScaler formula: (X - mean) / std
        # We need to bypass .transform() because it has internal validation that rejects infinities
        # BEFORE we can apply nan_to_num
        import numpy as np
        features_reshaped = features.reshape(1, -1)
        scaled_features = (features_reshaped - self.scaler.mean_) / self.scaler.scale_
        scaled_features = scaled_features[0]
        
        # Handle infinities and extreme values from scaling features with near-zero std
        # StandardScaler creates inf when dividing by std~0 (even after variance filtering)
        # Replace inf/nan with zero (these features have no discriminative power anyway)
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        if self.transform:
            features_tensor = self.transform(features_tensor)
        # another lil fix, was trying to access attribute of instead of dict key
        return EEGProcessedDatasetEntry(
            features=features_tensor,
            label=torch.tensor(sample['label'] if isinstance(sample, dict) else sample.label, dtype=torch.long),
            file_id=sample['file_id'] if isinstance(sample, dict) else sample.file_id
        )
