# WAUC Dataset Processing Pipeline

## Overview
Complete preprocessing pipeline for the WAUC (Workload Assessment Using Continuous EEG) dataset that transforms raw session-based data into processed features suitable for regression analysis.

## Directory Structure

### Input Data
```
data/wauc/wauc_by_session/          # Session-based EEG recordings
├── 01/ ... 48/                 # 48 participants
│   └── session0/ ... session5/ # 6 sessions per participant
│       ├── base1_eeg.parquet   # Baseline trial 1
│       ├── base2_eeg.parquet   # Baseline trial 2
│       └── test_eeg.parquet    # Test trial (USED for processing)
│
data/wauc/
└── ratings.parquet             # TLX workload ratings (282 entries)
```

### Output Data (All under `data/wauc/`)
```
data/wauc/
├── wauc_time_regression_dataset/       # ✅ FINAL: Time series dataset
│   ├── S1_session0_eeg_raw.parquet     # Subband-decomposed (48 cols: 8ch×6bands)
│   ├── S1_session0_target.txt          # TLX mental workload score
│   ├── S1_session1_eeg_raw.parquet
│   ├── S1_session1_target.txt
│   └── ...
│
├── wauc_feature_regression_dataset/    # ✅ FINAL: Features dataset
│   ├── S1_session0_features.parquet    # 400+ statistical features
│   ├── S1_session0_target.txt          # TLX mental workload score
│   ├── S1_session1_features.parquet
│   ├── S1_session1_target.txt
│   └── ...
│
├── raw_eeg_extracted/                  # ⚠️ TEMPORARY: Intermediate processing
└── features_extracted/                 # ⚠️ TEMPORARY: Intermediate processing
```

-## Processing Pipeline

### Step 1: EEG Extraction & Subband Decomposition
- **Input**: `data/wauc/wauc_by_session/`
- **Output**: `data/wauc/raw_eeg_extracted/`
- **Process**:
  - Uses ONLY test trial per session (excludes baseline trials)
  - Applies subband decomposition into 6 frequency bands:
    - Overall (0.5-45Hz)
    - Delta (0.5-4Hz)
    - Theta (4-8Hz)
    - Alpha (8-13Hz)
    - Beta (13-30Hz)
    - Gamma (30-45Hz)
  - Creates MultiIndex DataFrame: `(band, channel)`
  - Output: 48 columns = 8 channels × 6 bands
  - Format: `S{participant}_session{N}_eeg_raw.parquet`

### Step 2: Statistical Feature Extraction
- **Input**: `data/wauc/raw_eeg_extracted/`
- **Output**: `data/wauc/features_extracted/`
- **Process**:
  - Extracts 400+ statistical features per file
  - Features include: power bands, spectral entropy, Hjorth parameters, etc.
  - Sampling rate: 500Hz (Neuroelectrics Enobio)
  - Format: `S{participant}_session{N}_features.parquet`

### Step 3: Target File Creation
- **Input**: `data/wauc/ratings.parquet`
- **Output**: Target `.txt` files in both `raw_eeg_extracted/` and `features_extracted/`
- **Process**:
  - Matches participant-session pairs with EEG/feature files
  - Creates individual target files for TLX dimensions
  - Default target: `tlx_mental`
  - Available targets: `tlx_mental`, `tlx_physical`, `tlx_temporal`, `tlx_performance`, `tlx_effort`, `tlx_frustration`

### Step 4: Cleanup
- Removes files without corresponding targets
- Ensures paired feature-target files

### Step 5: Copy to Final Datasets
- **Organizes paired files into final regression dataset directories**:
  - `wauc_time_regression_dataset/` - Subband time series + targets
  - `wauc_feature_regression_dataset/` - Statistical features + targets

## Usage

### Run Full Pipeline
```python
from datasets.WAUC.load_wauc import run_full_pipeline

# Use default target (tlx_mental)
run_full_pipeline()

# Or specify a different TLX dimension
run_full_pipeline(target_column='tlx_physical')
run_full_pipeline(target_column='tlx_effort')
```

### Run Individual Steps
```python
from datasets.WAUC.load_wauc import *

# Step 1: Extract raw EEG with subband decomposition
reformat_raw_eeg(data_path, raw_output_path, sampling_rate=500.0)

# Step 2: Extract features
extract_features_from_wauc_files(raw_output_path, features_output_path, sampling_rate=500.0)

# Step 3: Create targets
create_target_files_from_ratings(
    ratings_path, 
    raw_output_path, 
    features_output_path,
    target_column='tlx_mental'
)

# Step 4: Cleanup
cleanup_files_without_targets(raw_output_path, features_output_path)

# Step 5: Copy to final datasets
copy_to_final_regression_datasets(
    raw_output_path,
    features_output_path,
    time_regression_path,
    feature_regression_path
)
```

## Dataset Specifications

### EEG Recording
- **Device**: Neuroelectrics Enobio
- **Sampling Rate**: 500 Hz
- **Channels**: 8 (AF8, FP2, FP1, AF7, T10, T9, P4, P3)
- **Duration**: 600 seconds per test trial
- **Participants**: 48
- **Sessions**: 6 per participant
- **Total Files**: 288 (48 × 6)

### Subband Time Series Format
- **Shape**: (300,002 samples, 48 columns)
- **Column Structure**: MultiIndex with (band, channel)
- **Bands**: 6 (Overall, delta, theta, alpha, beta, gamma)
- **Channels**: 8
- **File Format**: Parquet

### Feature Format
- **Features**: 400+ statistical features
- **Feature Types**: Power bands, spectral entropy, Hjorth parameters, etc.
- **File Format**: Parquet with MultiIndex

### Target Format
- **Format**: Plain text file (`.txt`)
- **Content**: Single floating-point value (TLX score)
- **Range**: Typically 0-100 for TLX dimensions

## Integration

Compatible with:
- `TimeRawDataset` for loading time series and targets
- `EEGRawRegressionDataset` for loading features and targets
- Existing ML model evaluation pipelines
- Channel importance analysis methods

## Notes

- **Baseline trials excluded**: Only test trials are used for consistency
- **Subband decomposition**: All raw files contain frequency-decomposed signals
- **Paired files**: Each `.parquet` file has a corresponding `.txt` target file
- **Temporary directories**: `raw_eeg_extracted/` and `features_extracted/` can be deleted after final datasets are created
