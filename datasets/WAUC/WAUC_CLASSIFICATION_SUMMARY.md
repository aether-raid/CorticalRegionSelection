# WAUC Classification Datasets - Implementation Summary

## Overview
Added classification dataset creation to the WAUC pipeline, creating organized datasets for three types of workload classification tasks.

## Classification Targets Added

### 1. **Mental Workload (MW) - Binary Classification**
- **Column**: `mw_labels`
- **Classes**: 
  - `0` = Low mental workload (141 samples)
  - `1` = High mental workload (141 samples)
- **Balance**: Perfect 50/50 split
- **Output directories**:
  - `wauc_time_mw_classification_dataset/` (subband time series)
  - `wauc_feature_mw_classification_dataset/` (400+ features)

### 2. **Physical Workload (PW) - 3-Class Classification**
- **Column**: `pw_labels`
- **Classes**:
  - `0` = Low physical workload (94 samples)
  - `1` = Medium physical workload (94 samples)
  - `2` = High physical workload (94 samples)
- **Balance**: Perfect 33.3% each class
- **Output directories**:
  - `wauc_time_pw_classification_dataset/` (subband time series)
  - `wauc_feature_pw_classification_dataset/` (400+ features)

### 3. **Combined MW+PW - 6-Class Classification**
- **Column**: `mwpw_labels`
- **Classes** (47 samples each):
  - `0` = Low MW + Low PW
  - `1` = High MW + Low PW
  - `2` = Low MW + Medium PW
  - `3` = High MW + Medium PW
  - `4` = Low MW + High PW
  - `5` = High MW + High PW
- **Balance**: Perfect 16.7% each class
- **Output directories**:
  - `wauc_time_mwpw_classification_dataset/` (subband time series)
  - `wauc_feature_mwpw_classification_dataset/` (400+ features)

## Directory Structure (Following MOCAS Convention)

```
data/wauc/
    # Binary Classification
    wauc_time_mw_classification_dataset/
        low_mw/
            S1_session0_eeg_raw.parquet
            S2_session3_eeg_raw.parquet
            ... (141 files)
        high_mw/
            S3_session1_eeg_raw.parquet
            ... (141 files)
    
    wauc_feature_mw_classification_dataset/
        low_mw/
            S1_session0_features.parquet
            ... (141 files)
        high_mw/
            ... (141 files)
    
    # 3-Class Classification
    wauc_time_pw_classification_dataset/
        low_pw/
            ... (94 files)
        medium_pw/
            ... (94 files)
        high_pw/
            ... (94 files)
    
    wauc_feature_pw_classification_dataset/
        low_pw/
            ... (94 files)
        medium_pw/
            ... (94 files)
        high_pw/
            ... (94 files)
    
    # 6-Class Classification
    wauc_time_mwpw_classification_dataset/
        low_mw_low_pw/
            ... (47 files)
        high_mw_low_pw/
            ... (47 files)
        low_mw_medium_pw/
            ... (47 files)
        high_mw_medium_pw/
            ... (47 files)
        low_mw_high_pw/
            ... (47 files)
        high_mw_high_pw/
            ... (47 files)
    
    wauc_feature_mwpw_classification_dataset/
        (same structure as time)
```

## Key Functions Added

### `organize_files_by_classification()`
```python
organize_files_by_classification(
    raw_dir,                    # Directory with raw EEG files
    features_dir,               # Directory with feature files
    ratings_file,               # Path to ratings.parquet
    output_base_dir,            # Base output directory
    classification_column,      # 'mw_labels', 'pw_labels', or 'mwpw_labels'
    class_names=None           # Optional custom class names
)
```

**Features**:
- Reads classification labels from `ratings.parquet`
- Matches participant-session pairs to files
- Copies files into class-specific folders
- Creates both time series and feature versions
- Generates processing statistics and JSON reports
- Handles missing labels gracefully

## Pipeline Integration

### Updated `run_full_pipeline()`
```python
run_full_pipeline(
    target_column='tlx_mental',          # Regression target
    create_classification_datasets=True  # Create classification datasets
)
```

**Pipeline Steps**:
1. Extract raw EEG with subband decomposition
2. Extract statistical features
3. Create regression target files
4. Cleanup unpaired files
5. Copy to regression datasets
6. **NEW**: Create classification datasets (MW, PW, MW+PW)

## Advantages Over MOCAS

1. **Better Balance**: All classes perfectly balanced
2. **Multiple Tasks**: 3 classification tasks vs MOCAS's 1
3. **More Classes**: Binary, 3-class, and 6-class options
4. **Richer Labels**: Based on validated TLX workload assessments
5. **Larger Dataset**: 282 total samples vs MOCAS's ~195

## Usage Examples

### Full Pipeline (Default)
```python
python datasets/WAUC/load_wauc.py
```
Creates both regression and all classification datasets.

### Regression Only
```python
from datasets.WAUC.load_wauc import run_full_pipeline
run_full_pipeline(target_column='tlx_mental', create_classification_datasets=False)
```

### Classification Only (After Regression Exists)
```python
from datasets.WAUC.load_wauc import organize_files_by_classification

# Binary classification
organize_files_by_classification(
    raw_output_path, features_output_path, ratings_path, 
    output_base_path, classification_column='mw_labels'
)

# 3-class classification
organize_files_by_classification(
    raw_output_path, features_output_path, ratings_path,
    output_base_path, classification_column='pw_labels'
)

# 6-class classification
organize_files_by_classification(
    raw_output_path, features_output_path, ratings_path,
    output_base_path, classification_column='mwpw_labels'
)
```

## Testing

Test script created: `test_wauc_classification.py`
- Tests all three classification tasks
- Verifies file organization
- Checks class distributions
- Validates output structure

## File Naming Convention

Follows established patterns:
- **Time series**: `S{participant}_session{N}_eeg_raw.parquet`
- **Features**: `S{participant}_session{N}_features.parquet`
- **Targets** (regression only): `S{participant}_session{N}_target.txt`

Note: Classification datasets don't need target files since the class is determined by folder location.

## Compatibility

- Compatible with existing dataset loaders
- Follows MOCAS structure for easy comparison
- Works with time-domain and feature-based models
- Supports binary, multi-class, and hierarchical classification approaches
