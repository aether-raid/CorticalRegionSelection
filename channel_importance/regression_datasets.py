from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import re

# EEG Raw Regression Dataset
class EEGRawRegressionDataset(Dataset):
    def __init__(self, data_dir, target_suffix=None):
        """
        Initialize EEG Raw Regression Dataset.
        
        Args:
            data_dir (str or Path): Directory containing .parquet feature files
            target_suffix (str, optional): Suffix for target filename selection.
                If None (default): looks for {basename}.txt (backward compatible)
                If provided: looks for {basename}_{target_suffix}.txt
                
                Example:
                    target_suffix=None    -> S01_1_features.txt
                    target_suffix='mental' -> S01_1_features_mental.txt
                    target_suffix='combined' -> S01_1_features_combined.txt
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.target_suffix = target_suffix

        exclude_metric_patterns = [
            r"theta/beta",
            r"\(theta \+ alpha\)/\(alpha \+ beta\)",
            r"gamma/delta",
            r"\(gamma \+ beta\)/\(delta \+ alpha\)"
        ]
        exclude_bands_for_patterns = ['alpha', 'gamma', 'beta', 'theta', 'delta']

        exclude_metric_patterns_2 = [
            r"absolute_power", r"relative_power"
        ]
        exclude_bands_for_patterns_2 = ['Overall']

        # Collect all parquet files
        self.file_list = sorted(list(self.data_dir.glob("**/*.parquet")))
        if not self.file_list:
            raise ValueError(f"No .parquet files found in {data_dir}")

        # Determine feature structure from the first file
        sample_df = pd.read_parquet(self.file_list[0], engine="fastparquet")
        print(self.file_list[0])
        print(sample_df.head())

        self.bands = sample_df.columns.get_level_values(0).unique().tolist()
        print(self.bands)
        #changed to get level 2 for channels
        self.channels = sample_df.columns.get_level_values(1).unique().tolist()
        #self.channels = sample_df.columns.unique().tolist()
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
                            should_exclude = True
                            break
                    for pattern in exclude_metric_patterns_2:
                        if re.fullmatch(pattern, feature_str) and band in exclude_bands_for_patterns_2:
                            should_exclude = True
                            break
                    if not should_exclude:
                        self.flattened_features.append(f"{feature_str}_{band}_{channel}")

        self.feature_indices = self._get_feature_indices()

    def _get_feature_indices(self):
        return {col: idx for idx, col in enumerate(self.flattened_features)}

    def categorize_electrode(self, channel_name):
        if channel_name.startswith("EEG."):
            label = channel_name[4:]
        else:
            label = channel_name
        if label.startswith(("AF", "Fp", "FP")):
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

    def filter_by_electrodes(self, electrode_categories):
        print(f"🔎 Filtering electrodes: {electrode_categories}")
        matched_features = []
        matched_channels = set()

        for feat in self.flattened_features:
            try:
                channel = feat.split('_')[-1]
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

    def _process_file(self, file_path):
        
        #df = pd.read_parquet(file_path).reset_index()
        #df = pd.read_parquet(file_path, index_col=None)
        df = pd.read_parquet(file_path, engine='fastparquet')


        feature_vector = np.zeros(len(self.flattened_features), dtype=np.float32)
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
        return feature_vector

    def _load_target(self, file_path):
        """
        Load target value from corresponding .txt file.
        
        Args:
            file_path (Path): Path to the .parquet feature file
            
        Returns:
            float: Target value
            
        Target file naming:
            - If target_suffix is None: {basename}.txt (e.g., S01_1_features.txt)
            - If target_suffix provided: {basename}_{suffix}.txt (e.g., S01_1_features_mental.txt)
            - Special case for 'combined': tries both suffixed and non-suffixed versions
        """
        if self.target_suffix is None:
            # Default behavior: look for exact match with .txt extension
            txt_path = file_path.with_suffix(".txt")
        else:
            # Suffix-based: insert suffix before .txt extension
            # e.g., S01_1_features.parquet -> S01_1_features_mental.txt
            base_name = file_path.stem  # Get filename without extension
            txt_path = file_path.parent / f"{base_name}_{self.target_suffix}.txt"
            
            # Special case for 'combined': also check for non-suffixed version
            if self.target_suffix == 'combined' and not txt_path.exists():
                txt_path = file_path.with_suffix(".txt")
        
        if not txt_path.exists():
            raise FileNotFoundError(
                f"Missing target file: {txt_path}\n"
                f"Expected target_suffix: {self.target_suffix if self.target_suffix else '(none - default)'}"
            )
        
        with open(txt_path, "r") as f:
            return float(f.read().strip())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        features = self._process_file(file_path)
        target = self._load_target(file_path)
        return {
            'raw_features': features,
            'label': target,
            'file_id': file_path.stem
        }

class TimeRawRegressionDataset(Dataset):
    def __init__(self, data_dir, t=5.0, fs=128, target_suffix=None, cache_in_memory=True, window_position='first'):
        """
        Initialize Time-series Raw Regression Dataset.
        
        Args:
            data_dir (str or Path): Directory containing .parquet time-series files
            t (float): Time window duration in seconds (default: 5.0)
            fs (int): Sampling frequency in Hz (default: 128)
            target_suffix (str, optional): Suffix for target filename selection.
                If None (default): looks for {basename}.txt (backward compatible)
                If provided: looks for {basename}_{target_suffix}.txt
                
                Example:
                    target_suffix=None    -> S01_1_eeg_raw.txt
                    target_suffix='mental' -> S01_1_eeg_raw_mental.txt
            cache_in_memory (bool): If True (default), preload all data into memory
                during initialization for faster training. Set False for large datasets
                that don't fit in memory.
            window_position (str): 'first' to use the first t seconds of recording,
                'middle' for the middle t seconds, or 'last' for the last t seconds
                (default: 'first').
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.fs = fs
        self.t = t
        self.target_suffix = target_suffix
        self.cache_in_memory = cache_in_memory
        self.window_position = window_position
        
        if window_position not in ('first', 'middle', 'last'):
            raise ValueError(f"window_position must be 'first', 'middle', or 'last', got '{window_position}'")

        
        if self.target_suffix =="combined":
            self.target_suffix = None


        # Collect all parquet files recursively
        self.file_list = sorted(list(self.data_dir.glob("**/*.parquet")))
        if not self.file_list:
            raise ValueError(f"No .parquet files found in {data_dir}")

        # Use first file to extract shape info
        n_samples = int(self.fs * self.t)
        full_sample_df = pd.read_parquet(self.file_list[0])
        if self.window_position == 'first':
            sample_df = full_sample_df.iloc[:n_samples]
        elif self.window_position == 'middle':
            total_len = len(full_sample_df)
            start_idx = (total_len - n_samples) // 2
            sample_df = full_sample_df.iloc[start_idx:start_idx + n_samples]
        else:
            sample_df = full_sample_df.iloc[-n_samples:]
        self.bands = sample_df.columns.get_level_values(0).unique().tolist()
        self.channels = sample_df.columns.get_level_values(1).unique().tolist()
        self.time_steps = int(self.fs * self.t)

        # Flattened feature names (optional, for mapping)
        self.flattened_features = [
            f"{t}_{band}_{chan}"
            for t in range(self.time_steps)
            for band in self.bands
            for chan in self.channels
        ]

        # Cache data in memory for faster training
        self._cached_data = None
        self._cached_labels = None
        self._cached_file_ids = None
        if self.cache_in_memory:
            self._preload_data()

    def _preload_data(self):
        """Preload all samples into memory to avoid repeated file I/O during training."""
        import time
        start = time.time()
        n_samples = len(self.file_list)
        
        # Pre-allocate numpy arrays for efficiency
        sample_size = self.time_steps * len(self.bands) * len(self.channels)
        self._cached_data = np.empty((n_samples, sample_size), dtype=np.float32)
        self._cached_labels = np.empty(n_samples, dtype=np.float32)
        self._cached_file_ids = []
        n_samples = int(self.fs * self.t)
        
        for i, fpath in enumerate(self.file_list):
            full_df = pd.read_parquet(fpath)
            if self.window_position == 'first':
                df = full_df.iloc[:n_samples]
            elif self.window_position == 'middle':
                total_len = len(full_df)
                start_idx = (total_len - n_samples) // 2
                df = full_df.iloc[start_idx:start_idx + n_samples]
            else:
                df = full_df.iloc[-n_samples:]
            self._cached_data[i] = df.values.astype(np.float32).flatten()
            self._cached_labels[i] = self._load_target(fpath)
            self._cached_file_ids.append(fpath.stem)
        
        elapsed = time.time() - start
        print(f"📦 Cached {n_samples} samples in memory ({elapsed:.2f}s, "
              f"{self._cached_data.nbytes / 1024 / 1024:.1f} MB)")

    def categorize_electrode(self, channel_name):
        if channel_name.startswith("EEG."):
            label = channel_name[4:]
        else:
            label = channel_name

        if label.startswith(("AF", "Fp", "FP")):
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

    def filter_by_electrodes(self, electrode_categories):
        print(f"🔎 Filtering time-series dataset by electrodes: {electrode_categories}")

        filtered_channels = []
        for ch in self.channels:
            category = self.categorize_electrode(ch)
            if category and category in electrode_categories:
                filtered_channels.append(ch)

        if not filtered_channels:
            print("⚠️ No channels matched the filter.")
        else:
            print(f"✅ Retained channels: {filtered_channels}")

        self.channels = filtered_channels
        self.flattened_features = [
            f"{t}_{band}_{chan}"
            for t in range(self.time_steps)
            for band in self.bands
            for chan in self.channels
        ]

    def _load_target(self, parquet_path):
        """
        Load target value from corresponding .txt file.
        
        Args:
            parquet_path (Path): Path to the .parquet time-series file
            
        Returns:
            float: Target value
            
        Target file naming:
            - If target_suffix is None: {basename}.txt (e.g., S01_1_eeg_raw.txt)
            - If target_suffix provided: {basename}_{suffix}.txt (e.g., S01_1_eeg_raw_mental.txt)
            - Special case for 'combined': tries both suffixed and non-suffixed versions
        """
        if self.target_suffix is None:
            # Default behavior: look for exact match with .txt extension
            txt_path = parquet_path.with_suffix(".txt")
        else:
            # Suffix-based: insert suffix before .txt extension
            base_name = parquet_path.stem  # Get filename without extension
            txt_path = parquet_path.parent / f"{base_name}_{self.target_suffix}.txt"
            
            # Special case for 'combined': also check for non-suffixed version
            if self.target_suffix == 'combined' and not txt_path.exists():
                txt_path = parquet_path.with_suffix(".txt")
        
        if not txt_path.exists():
            raise FileNotFoundError(
                f"Missing target file: {txt_path}\n"
                f"Expected target_suffix: {self.target_suffix if self.target_suffix else '(none - default)'}"
            )
        
        with open(txt_path, "r") as f:
            return float(f.read().strip())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Use cached data if available (much faster)
        if self._cached_data is not None:
            return {
                "raw_features": self._cached_data[idx],
                "label": self._cached_labels[idx],
                "file_id": self._cached_file_ids[idx]
            }
        
        # Fallback to on-demand loading (when cache_in_memory=False)
        fpath = self.file_list[idx]
        n_samples = int(self.fs * self.t)
        full_df = pd.read_parquet(fpath)
        if self.window_position == 'first':
            df = full_df.iloc[:n_samples]
        elif self.window_position == 'middle':
            total_len = len(full_df)
            start_idx = (total_len - n_samples) // 2
            df = full_df.iloc[start_idx:start_idx + n_samples]
        else:
            df = full_df.iloc[-n_samples:]
        flat_data = df.values.astype(np.float32).flatten()

        label = self._load_target(fpath)

        return {
            "raw_features": flat_data,
            "label": label,
            "file_id": fpath.stem
        }
