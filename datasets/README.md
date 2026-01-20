# EEG Dataset Processing Scripts

This directory contains preprocessing pipelines for multiple EEG cognitive workload datasets: **MOCAS**, **Heat the Chair (HTC)**, **N-Back**, **WAUC**, **SENSE-42**, and **UNIVERSE**.

## Overview

Each dataset loader script transforms raw EEG recordings into standardized, ML-ready datasets with NASA-TLX workload scores as targets. All datasets follow a consistent structure for interoperability across experiments.

### Common Pipeline (5 Steps)

All datasets share the same processing pipeline:

#### Step 1: Raw EEG Extraction & Reformatting
- **Input:** Original nested folders with raw EEG files
- **Processing:**
  1. Load raw file (parquet/bdf/pickle format)
  2. Remove non-EEG columns (timestamps, markers, battery, etc.)
  3. Standardize channel names (strip 'EEG.' and 'RAW_' prefixes)
  4. Create EEG class instance with metadata
  5. Apply basic filtering (bandpass: 0.5-45 Hz)
  6. Apply notch filter (50, 60 Hz with Q factor 30.0)
  7. Subband decomposition into 6 bands:
     - Overall (0.5-45 Hz)
     - Delta (0.5-4 Hz)
     - Theta (4-8 Hz)
     - Alpha (8-13 Hz)
     - Beta (13-30 Hz)
     - Gamma (30-45 Hz)
  8. Normalize EEG data
  9. Save to parquet with MultiIndex structure (band, channel)
- **Output:** `*_raw_eeg_extracted/` directory with `S{subject}_{task}_eeg_raw.parquet` files

#### Step 2: Statistical Feature Extraction
- **Critical Design:** Loads ORIGINAL raw files, NOT Step 1 output (prevents double-processing bug causing ~17,300x power corruption)
- **Processing:**
  1. Iterate through Step 1 file list (reference only)
  2. Load ORIGINAL raw data for each recording
  3. Create EEG class instance (with extract_time parameter for resampling)
  4. Apply filtering and subband decomposition
  5. Generate statistical features via `eeg.generate_stats()`:
     - Band Powers (absolute and relative)
     - Power Ratio Indices (alpha/theta, theta/beta, etc.)
     - Peak Statistics (number of peaks per channel/band)
     - Time-Domain Statistics (mean, median, variance, std, skewness, kurtosis)
     - Zero-Crossing Rate
     - Entropy Measures (Renyi, Differential, Fuzzy entropy)
     - Brain Rate Index (weighted average of band frequencies)
- **Output:** `*_features_extracted/` directory with `S{subject}_{task}_features.parquet` files (400+ features per recording)

#### Step 3: Target File Creation (Multi-Label TLX)
- **Input:** TLX source files (json/csv/parquet/events)
- **Processing:**
  1. Extract 6 subscale scores:
     - Mental demand
     - Physical demand
     - Temporal demand
     - Performance (often inverted: higher = better)
     - Effort
     - Frustration
  2. Rescale to 0-100 range (dataset-specific scaling):
     - HTC: score × 10 (0-10 → 0-100)
     - N-Back: score × 5 (0-20 → 0-100)
     - WAUC: (score - 1) × 5 (1-21 → 0-100)
     - SENSE-42: score × 11.11 (0-9 → 0-100)
  3. Calculate combined TLX score: mean(all_6_subscales)
  4. Create 7 target files per recording
- **Output:** 7 text files per sample in `*_features_extracted/` directory:
  - `S{X}_{Y}_features.txt` (combined TLX, 0-100)
  - `S{X}_{Y}_features_mental.txt`
  - `S{X}_{Y}_features_physical.txt`
  - `S{X}_{Y}_features_temporal.txt`
  - `S{X}_{Y}_features_performance.txt`
  - `S{X}_{Y}_features_effort.txt`
  - `S{X}_{Y}_features_frustration.txt`

#### Step 4: Create Regression Datasets
- **Processing:**
  1. Create output directories
  2. For each recording with valid targets:
     - Copy time-series EEG + 7 renamed target files → `*_time_regression_dataset/`
     - Copy features + 7 target files → `*_feature_regression_dataset/`
- **Output:**
  ```
  *_time_regression_dataset/
      S01_1_eeg_raw.parquet
      S01_1_eeg_raw.txt              # Combined TLX
      S01_1_eeg_raw_mental.txt
      S01_1_eeg_raw_physical.txt
      ...
      
  *_feature_regression_dataset/
      S01_1_features.parquet
      S01_1_features.txt             # Combined TLX
      S01_1_features_mental.txt
      ...
  ```

#### Step 5: Create Classification Datasets
- **Processing:**
  1. Collect all combined TLX scores across dataset
  2. Calculate tertile boundaries (3-class):
     - 33rd percentile → low/medium boundary
     - 67th percentile → medium/high boundary
  3. Assign class labels per subscale:
     - Class 0 (Low): score < 33rd percentile
     - Class 1 (Medium): 33rd ≤ score < 67th percentile
     - Class 2 (High): score ≥ 67th percentile
  4. Create `classification_metadata.json` with subscale labels
- **Output:**
  ```
  *_time_classification_dataset/
      all/
          S01_1_eeg_raw.parquet
      classification_metadata.json
      
  *_feature_classification_dataset/
      all/
          S01_1_features.parquet
      classification_metadata.json
  ```

### Dataset Input Formats

| Dataset | Source Format | Rate | Channels | Source File |
|---------|--------------|------|----------|-------------|
| HTC | subject_XX/taskY/phase2.parquet | 128 Hz | 14 (EPOC+) | phase2.parquet |
| N-Back | subject_XX/taskY/phase2.parquet | 128 Hz | 14 (EPOC+) | phase2.parquet |
| MOCAS | participantX/taskY/raw_eeg.parquet | 128 Hz | 14 (EPOC+) | raw_eeg.parquet |
| WAUC | XX/sessionY/test_eeg.parquet | 500 Hz | 8 (Muse) | test_eeg.parquet |
| SENSE-42 | p0XX.bdf (BioSemi) | 1024 Hz | 32 EEG | .bdf files |
| UNIVERSE | EEG_filtered.pickle (pre-processed) | ~256 Hz | 4 (Muse) | .pickle files |

---

## Standard Dataset Format

All datasets follow a consistent structure with four main subdirectories for different modalities and tasks:

### 1. Feature Regression Dataset (`{dataset}_feature_regression_dataset`)
- **Features:** `{prefix}{id}_{session}_features.parquet`
  - MOCAS: `P{participant}_{session}_features.parquet`
  - HTC/N-back/WAUC: `S{subject}_{session}_features.parquet`
  - Contains DataFrame with EEG-derived features (power bands, connectivity, etc.)
- **Targets:** Text files for each subscale
  - Combined: `{base}_features.txt`
  - Subscales: `{base}_features_{subscale}.txt` (mental, physical, temporal, performance, effort, frustration)
  - Each file contains a single float value (0-100 scale)

### 2. Feature Classification Dataset (`{dataset}_feature_classification`)
- **Features:** Parquet files stored in `all/` subdirectory (same format as regression)
- **Labels:** Single `classification_metadata.json` file
  - Format: `{filename: {subscale: label}}`
  - Labels are strings: "low", "medium", "high"
  - Covers all 7 subscales per sample

### 3. Time Regression Dataset (`{dataset}_time_regression_dataset`)
- **Features:** Time-series Parquet files with sequential EEG windows (MultiIndex columns: band, channel)
- **Targets:** Time-aligned regression scores (same 7-file structure as feature regression)

### 4. Time Classification Dataset (`{dataset}_time_classification`)
- **Features:** Time-series Parquet files
- **Labels:** JSON metadata with time-windowed classification labels

### File Naming Conventions
- **Features:** `{prefix}{id}_{session}_features.parquet`
- **Regression Targets:** `{base}_features{subscale}.txt` (subscale empty for combined)
- **Classification Labels:** `classification_metadata.json`

### Data Types
- **Features:** Pandas DataFrame in Parquet format
- **Regression Targets:** Plain text files with float values
- **Classification Labels:** JSON dictionary with string labels

### Subscales
All datasets use exactly 7 subscales:
- combined
- mental
- physical
- temporal
- performance
- effort
- frustration

### Consistency Requirements
- Sample counts must match between regression and classification datasets
- All subscales must be present for every sample
- File naming must follow patterns for automated processing

---

## Complete Output Structure

```
data/{DATASET}/
|
+-- *_raw_eeg_extracted/              # Step 1: Decomposed time series
|   +-- S01_1_eeg_raw.parquet
|   +-- extraction_summary.json
|
+-- *_features_extracted/             # Step 2-3: Features + targets
|   +-- S01_1_features.parquet
|   +-- S01_1_features.txt            # Combined TLX
|   +-- S01_1_features_mental.txt
|   +-- ...
|   +-- feature_extraction_summary.json
|
+-- *_time_regression_dataset/        # Step 4: Time regression
|   +-- S01_1_eeg_raw.parquet
|   +-- S01_1_eeg_raw.txt
|   +-- S01_1_eeg_raw_mental.txt
|
+-- *_feature_regression_dataset/     # Step 4: Feature regression
|   +-- S01_1_features.parquet
|   +-- S01_1_features.txt
|   +-- S01_1_features_mental.txt
|
+-- *_time_classification_dataset/    # Step 5: Time classification
|   +-- all/
|   |   +-- S01_1_eeg_raw.parquet
|   +-- classification_metadata.json
|
+-- *_feature_classification_dataset/ # Step 5: Feature classification
    +-- all/
    |   +-- S01_1_features.parquet
    +-- classification_metadata.json
```

---

---

## Key Design Patterns Across All Datasets

1. **Double-Processing Prevention:** Load original raw files for feature extraction (prevents double-processing bug causing ~17,300x power corruption)
2. **EEG Class:** Handles all signal processing (filtering, decomposition, normalization)
3. **MultiIndex Columns:** (band, channel) structure for all processed data
4. **Multi-Label Targets:** 7 files per sample (combined + 6 subscales)
5. **Metadata-Based Classification:** Zero file duplication via JSON labels
6. **Consistent Scaling:** 0-100 scale for all TLX targets (regardless of original scale)
7. **Parallel Processing:** Support via multiprocessing.Pool

---

## Dataset Details

### MOCAS Dataset
**Script:** `MOCAS/load_mocas.py`  
**Source:** `data/MOCAS/` (30 subjects, up to 9 tasks each)  
**TLX Scores:** All 7 subscales (combined, mental, physical, temporal, performance, effort, frustration)

### Output: 4 Datasets (2 regression + 2 classification)
```
data/MOCAS/
├── mocas_time_regression_dataset/              # Raw EEG + subscale targets
├── mocas_feature_regression_dataset/           # Features + subscale targets
├── mocas_time_classification_dataset/          # Raw EEG by metadata workload (low/medium/high)
└── mocas_feature_classification_dataset/       # Features by metadata workload (low/medium/high)
```

**Regression Targets:** 7 text files per sample: `{base}_features.txt` (combined) + `{base}_features_{subscale}.txt`  
**Classification Labels:** `classification_metadata.json` with subscale labels ("low"/"medium"/"high")  
**TLX Source:** metadata.json + scores.csv (0-100 scale)

---

## Heat the Chair (HTC) Dataset
**Script:** `HTC/load_htc.py`  
**Source:** `data/heat_the_chair/` (17 subjects, 2 tasks each)  
**TLX Scores:** All 7 subscales (combined, mental, physical, temporal, performance, effort, frustration)

### Output: 4 Datasets (2 regression + 2 classification)
```
data/heat_the_chair/
├── htc_time_regression/              # Raw EEG + subscale targets
├── htc_feature_regression/           # Features + subscale targets
├── htc_time_classification/          # Raw EEG classification (low/medium/high)
└── htc_feature_classification/       # Features classification (low/medium/high)
```

**Regression Targets:** 7 text files per sample: `{base}_features.txt` (combined) + `{base}_features_{subscale}.txt`  
**Classification Labels:** `classification_metadata.json` with subscale labels ("low"/"medium"/"high")  
**TLX Source:** tlx.json (0-10 scale → scaled to 0-100)

---

## N-Back Dataset
**Script:** `NBACK/load_nback.py`  
**Source:** `data/n_back/n_back/` (16 subjects, 3 tasks each)  
**TLX Scores:** All 7 subscales (combined, mental, physical, temporal, performance, effort, frustration)

### Output: 4 Datasets (2 regression + 2 classification)
```
data/n_back/
├── nback_time_regression/            # Raw EEG + subscale targets
├── nback_feature_regression/         # Features + subscale targets
├── nback_time_classification/        # Raw EEG classification (low/medium/high)
└── nback_feature_classification/     # Features classification (low/medium/high)
```

**Regression Targets:** 7 text files per sample: `{base}_features.txt` (combined) + `{base}_features_{subscale}.txt`  
**Classification Labels:** `classification_metadata.json` with subscale labels ("low"/"medium"/"high")  
**TLX Source:** tlx.json (0-20 scale → scaled to 0-100)

---

## WAUC Dataset
**Script:** `WAUC/load_wauc.py`  
**Source:** `data/wauc/wauc_by_session/` (48 participants, 6 sessions each)  
**Sampling Rate:** 500 Hz → resampled to 128 Hz for consistency  
**Channels:** 8 EEG channels (AF8, FP2, FP1, AF7, T10, T9, P4, P3)

### Output: 4 Datasets (2 regression + 2 classification)
```
data/wauc/
├── wauc_time_regression_dataset/              # Raw EEG + subscale targets
├── wauc_feature_regression_dataset/           # Features + subscale targets
├── wauc_time_classification_dataset/          # Raw EEG by subscale labels (low/medium/high)
└── wauc_feature_classification_dataset/       # Features by subscale labels (low/medium/high)
```

**Regression Targets:** 7 text files per sample: `{base}_features.txt` (combined) + `{base}_features_{subscale}.txt`  
**Classification Labels:** `classification_metadata.json` with subscale labels ("low"/"medium"/"high")  
**TLX Source:** ratings.parquet (1-21 scale → scaled to 0-100)

### Known Issues

**⚠️ Data Quality Issues - 11 Participants Affected (66 sessions excluded)**

Comprehensive analysis identified two types of failures causing feature extraction to fail:

**Type 1: Sensor Disconnections (Zero Variance) - 9 participants, 54 sessions**

All affected channels show constant value of **-400105.13 µV** (likely sentinel value for disconnection):

- **P02:** P4 flat (6 sessions)
- **P03:** P3 + P4 flat (6 sessions)
- **P09:** T10 flat (6 sessions)
- **P23:** P4 flat (6 sessions)
- **P26:** P3 + P4 flat (6 sessions)
- **P28:** T10 flat (6 sessions)
- **P39:** P3 + P4 flat (6 sessions)
- **P47:** T9 + T10 flat (6 sessions)
- **P48:** P3 + P4 flat (6 sessions)

**Technical Details:**
- Zero variance (std = 0.00, range = 0.00) across all 300,002 samples
- Error: "Maximum allowed size exceeded" during entropy calculation
- Flat signals cause division by zero in variance-based features
- FFT of constant values produces degenerate results

**Type 2: Extreme Value Ranges - 2 participants, 12 sessions**

- **P17:** T9 (299,694 µV) + T10 (318,276 µV) extreme ranges (6 sessions)
  - 20x normal range of 8,000-15,000 µV
  - Error: "Unable to allocate 14.0 TiB" (histogram binning creates ~1.9 trillion bins)
  
- **P35:** P3 (391,323 µV) extreme range (6 sessions)
  - 26x normal range
  - Similar memory allocation failure

**Technical Details:**
- Failure in `calc_renyi_entropy()` function
- scipy.stats.entropy() automatic binning via np.histogram
- Extreme ranges cause np.linspace to create massive arrays

**Impact Summary:**
- **Total Excluded:** 66 sessions (54 zero-variance + 12 extreme-range)
- **Affected Participants:** 11 of 48 (22.9%)
- **Final Dataset Size:** 222 of 288 recordings (77.1% success rate)
- Failed extractions are logged but do not halt pipeline
- Orphaned target files are automatically excluded during final dataset creation
- Pipeline completes successfully with remaining recordings

**Graceful Failure Handling:**
1. Failed files tracked in `failed_files_info` list
2. `copy_to_final_regression_datasets()` only copies files with both data and target
3. Final datasets contain only successfully processed recordings
4. JSON summary reports include both successful and failed file counts

---

## Dataset Types Explained

### Regression Datasets
- **Time Regression:** Raw decomposed EEG time-series + continuous subscale targets (7 per sample)
- **Feature Regression:** Statistical features + continuous subscale targets (7 per sample)
- **Use Case:** Predicting exact workload scores for each subscale

### Classification Datasets
- **Time Classification:** Raw decomposed EEG organized by subscale workload class (low/medium/high)
- **Feature Classification:** Statistical features organized by subscale workload class
- **Binning:** 33rd percentile (low/medium boundary), 67th percentile (medium/high boundary) per subscale
- **Use Case:** Classifying workload levels for each subscale

---

## Key Differences

| Dataset | Subjects | Tasks/Subject | Total Recordings | TLX Format | Total Datasets |
|---------|----------|---------------|------------------|------------|----------------|
| **MOCAS** | 30 | 9 (variable) | ~270 | 7 subscales (combined + 6) | **4** |
| **HTC** | 17 | 2 | 34 | 7 subscales (combined + 6) | **4** |
| **N-Back** | 16 | 3 | 48 | 7 subscales (combined + 6) | **4** |
| **WAUC** | 48 | 6 | 222* | 7 subscales (combined + 6) | **4** |

\* *222 of 288 total sessions (66 excluded: 11 participants with sensor disconnections or extreme values)*

---

## Usage

### Running Individual Dataset Pipelines

Process individual datasets:
```bash
python datasets/MOCAS/load_mocas.py
python datasets/HTC/load_htc.py
python datasets/NBACK/load_nback.py
python datasets/WAUC/load_wauc.py
```

Each script includes:
- ✅ Double-processing prevention (loads original raw data for feature extraction)
- ✅ Automatic cleanup of unpaired files
- ✅ Progress tracking with detailed logging
- ✅ JSON summaries of processing statistics
- ✅ Graceful failure handling (WAUC: excludes problematic recordings)

### Core Processing Functions

All datasets implement these core functions:
- `reformat_raw_eeg()` - Step 1: Extract and decompose raw EEG
- `extract_features_from_*_files()` - Step 2: Generate statistical features
- `create_target_files_from_tlx()` - Step 3: Create TLX target files
- `cleanup_parquet_files_without_targets()` - Remove unpaired files
- `create_regression_datasets()` - Step 4: Organize regression datasets
- `create_classification_datasets()` - Step 5: Organize classification datasets

### Load Scripts
- `datasets/HTC/load_htc.py`
- `datasets/NBACK/load_nback.py`
- `datasets/MOCAS/load_mocas.py`
- `datasets/WAUC/load_wauc.py`
- `datasets/sense-42/load_sense42.py`
- `datasets/universe/load_universe.py`

Core EEG processing:
- `channel_importance/eeg.py` (EEG class)

---

## Validation

After running dataset loaders, validate the output format using:
```bash
python channel_importance/feature_tests/test_datasets_subscales_ready.py
```

This script checks:
- Directory existence and structure
- Complete subscale targets (7 per regression sample)
- Classification metadata completeness
- Sample count consistency across datasets
- All subscales present for every sample

If all datasets pass validation, you can proceed with multi-subscale experiments.

---

## Technical Details

### EEG Processing

**Hardware:**
- **EPOC+:** 14 channels, 128 Hz (HTC, N-Back, MOCAS)
- **Muse:** 4-8 channels, 256-500 Hz (WAUC, UNIVERSE)
- **BioSemi:** 32 channels, 1024 Hz (SENSE-42)

**Signal Processing (via `channel_importance/eeg.py`):**
- Bandpass filtering (0.5-45 Hz)
- Notch filtering (50, 60 Hz, Q=30.0)
- Frequency decomposition into 6 subbands
- Normalization
- MultiIndex DataFrame structure: (band, channel)

**Feature Extraction:**
- 400+ statistical features per recording
- Band Powers (absolute and relative)
- Power Ratio Indices
- Peak Statistics
- Time-Domain Statistics
- Zero-Crossing Rate
- Entropy Measures (Renyi, Differential, Fuzzy)
- Brain Rate Index

### File Naming Conventions

**Raw EEG:** `S{subject}_{task}_eeg_raw.parquet` (or `P{X}_{Y}_eeg_raw.parquet` for MOCAS)  
**Features:** `S{subject}_{task}_features.parquet`  
**Regression Targets:** `{base}_features.txt` (combined) + `{base}_features_{subscale}.txt` (6 subscales)  
**Classification Labels:** `classification_metadata.json` with subscale labels

### Dependencies

- `channel_importance.eeg.EEG` - Signal processing and feature extraction
- `pandas`, `numpy` - Data manipulation
- `fastparquet` - Efficient parquet I/O
- `scipy` - Signal processing and statistics
- `multiprocessing` - Parallel processing

---

## Summary Table

| Dataset | Subjects | Tasks/Subject | Total Recordings | TLX Format | Hardware | Sampling Rate | Total Datasets |
|---------|----------|---------------|------------------|------------|----------|---------------|----------------|
| **MOCAS** | 30 | 9 (variable) | ~270 | 0-100 (metadata.json) | EPOC+ (14ch) | 128 Hz | 4 |
| **HTC** | 17 | 2 | 34 | 0-10 → 0-100 (tlx.json) | EPOC+ (14ch) | 128 Hz | 4 |
| **N-Back** | 16 | 3 | 48 | 0-20 → 0-100 (tlx.json) | EPOC+ (14ch) | 128 Hz | 4 |
| **WAUC** | 48 | 6 | 222* | 1-21 → 0-100 (ratings.parquet) | Muse (8ch) | 500 Hz | 4 |
| **SENSE-42** | ~42 | variable | variable | 0-9 → 0-100 (events) | BioSemi (32ch) | 1024 Hz | 4 |
| **UNIVERSE** | variable | variable | variable | 0-100 (Task_Labels.csv) | Muse (4ch) | ~256 Hz | 4 |

\* *222 of 288 total sessions (66 excluded: 11 participants with sensor disconnections or extreme values)*

---

*Last updated: January 20, 2026*
