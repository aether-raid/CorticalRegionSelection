# EEG Channel Importance Analysis - Aggregated Model Ensemble

## Overview

This suite analyzes EEG channel importance using an **aggregated ensemble of 5 models** to provide robust, consensus-based channel rankings. Instead of relying on a single model, it combines insights from multiple classifiers.

## Files

### Main Script
- **`test_feature_classification_agg.py`** - Core analysis script with 5-model ensemble

### Automation Scripts
- **`run_all_datasets.py`** - Python script to run all 4 datasets (cross-platform)
- **`run_all_datasets.bat`** - Windows batch script to run all 4 datasets

## Models Used

1. **LogisticRegression** - Linear model, interpretable
2. **RandomForest** - Ensemble trees, handles non-linearity
3. **GradientBoosting** - Sequential boosting, strong performance
4. **SVM** - Kernel-based, high-dimensional data
5. **XGBoost** - Advanced boosting (optional, falls back to 4 models if not installed)

## Usage

### Quick Test (Recommended First)

Test that everything works with a quick 2-3 minute validation:

```bash
# Test single dataset
python test_feature_classification_agg.py --mocas --test

# Test all datasets (8-12 minutes)
python run_all_datasets.py --test
# or
run_all_datasets.bat test
```

### Single Dataset

```bash
# MOCAS (default)
python test_feature_classification_agg.py
python test_feature_classification_agg.py --mocas

# Heat the Chair
python test_feature_classification_agg.py --htc

# N-Back
python test_feature_classification_agg.py --nback

# WAUC
python test_feature_classification_agg.py --wauc
```

**Runtime:** ~15-25 minutes per dataset

### All Datasets

```bash
# Python (cross-platform)
python run_all_datasets.py

# Windows batch
run_all_datasets.bat
```

**Runtime:** ~60-100 minutes total

## Output Files

### 1. Aggregated Rankings CSV
`channel_importance_aggregated_{dataset}_classification_{timestamp}.csv`

**Columns:**
- `Rank` - Overall rank (1 = most important)
- `Channel_Category` - Channel group name
- `Aggregated_Accuracy` - Average accuracy across all models
- `Delta_Accuracy` - Difference from baseline
- `Average_Rank` - Average rank across all models
- `{Model}_Rank` - Individual rank from each model
- `{Model}_Accuracy` - Test accuracy for each model
- `{Model}_F1_Score` - F1 score for each model
- `{Model}_CV_Mean` - Cross-validation mean for each model
- `{Model}_CV_Std` - Cross-validation std for each model
- `{Model}_CV_Fold1/2/3` - Individual fold scores
- `{Model}_Delta_Accuracy` - Model-specific delta from baseline

### 2. Detailed Run Metrics CSV
`run_details_aggregated_{dataset}_classification_{timestamp}.csv`

**Comprehensive per-model, per-channel metrics:**
- `Dataset` - Dataset name
- `Test_Mode` - Whether run in test mode
- `Channel_Category` - Channel group
- `Model` - Model name
- `Test_Accuracy` - Hold-out test accuracy
- `F1_Score` - Weighted F1 score
- `CV_Mean` - 3-fold CV mean accuracy
- `CV_Std` - 3-fold CV standard deviation
- `CV_Fold_1/2/3` - Individual fold accuracies
- `Baseline_Accuracy` - Model's all-channels baseline
- `Delta_Accuracy` - Difference from baseline
- `N_Train_Samples` - Training set size
- `N_Test_Samples` - Test set size
- `N_Features` - Number of features used
- `Random_State` - Random seed (42)
- `Timestamp` - Run timestamp

## Methodology

### Aggregation Process

1. **Same Data Split** - All 5 models use identical train/test/CV splits
2. **Feature Selection** - ANOVA with k=24 features
3. **Per-Model Ranking** - Each model ranks channels 1-N by ΔAccuracy
4. **Average Rank** - Final ranking = average rank across all 5 models
5. **Lower Rank = More Important**

### Channel Categories Tested

- **All_Channels** - All electrodes (baseline)
- **Frontal** - Fp, AF, F, FC (attention, working memory)
- **Central** - C, CP (motor control, sensorimotor)
- **Parietal** - P, PO (spatial processing, attention)
- **Occipital** - O, OZ (visual processing)
- **Temporal** - T, TP, FT (auditory, memory)

## Configuration

- **Random State:** 42 (controls all randomization)
- **Train/Test Split:** 70/30
- **Cross-Validation:** 3-fold stratified
- **Feature Selection:** ANOVA, k=24
- **Evaluation Metric:** Classification accuracy

## Example Output

```
📊 AGGREGATED Channel Importance Ranking (by average rank across 5 models):
Rank   Channel              Agg Acc    ΔAcc       Avg Rank    Model Ranks
----------------------------------------------------------------------------------
1      🟢 Frontal           0.847      +0.023     1.80        [2, 1, 2, 1, 3]
2      🟢 Central           0.839      +0.015     2.40        [1, 3, 1, 3, 4]
3      🟢 Parietal          0.831      +0.007     3.20        [3, 2, 3, 4, 2]
4      🟡 Temporal          0.825      +0.001     3.80        [4, 4, 4, 2, 5]
5      🔴 Occipital         0.818      -0.006     4.80        [5, 5, 5, 5, 1]
```

## Test Mode

Test mode (`--test` flag) runs a quick validation:
- Only tests 3 channel groups: All_Channels, Frontal, Central
- Verifies all components work correctly
- Runtime: ~2-3 minutes per dataset
- Output files include `_TEST` suffix

**Always run test mode first** to ensure everything works before starting full runs.

## Datasets

- **MOCAS** - `data/MOCAS/mocas_feature_tlx_classification_dataset`
- **HTC** - `data/heat_the_chair/htc_feature_classification`
- **N-Back** - `data/n_back/nback_feature_classification`
- **WAUC** - `data/wauc/wauc_feature_tlx_classification_dataset`

## Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- xgboost (optional, will fall back to 4 models if not available)

## Tips

1. **Start with test mode** - Verify setup works correctly
2. **Single dataset first** - Test one dataset before running all
3. **Check CSV outputs** - Both aggregated and detailed files contain complete metrics
4. **Monitor progress** - Script shows real-time progress for each channel group
5. **Disk space** - Ensure adequate space in `experiment_results/` directory

## Troubleshooting

**"XGBoost not available"** - Normal warning, will use 4 models instead of 5

**Long runtime** - Expected, full runs take 60-100 minutes total

**Memory issues** - Close other applications, consider running datasets individually

**Missing data** - Verify dataset paths are correct for your setup
