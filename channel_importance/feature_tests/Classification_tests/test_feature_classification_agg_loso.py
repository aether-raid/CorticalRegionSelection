"""
EEG Channel Importance Analysis for Classification Tasks - AGGREGATED MODEL ENSEMBLE with LOSO CV
=================================================================================================

This script performs channel importance analysis by AGGREGATING rankings from 30+ different models
using Leave-N-Subjects-Out (LOSO) cross-validation for subject-independent evaluation.

CHANNEL GROUPING & FILTERING METHODOLOGY
========================================

OVERVIEW
--------
This analysis evaluates the importance of different brain regions for mental workload classification
by systematically testing 11 anatomical channel groups. Each group represents specific cortical areas
based on the International 10-20/10-10 EEG electrode placement system.

CHANNEL GROUP DEFINITIONS
-------------------------
We define 11 groups spanning individual anatomical regions and extended composite areas:

**Individual Regions** (Pure anatomical areas):
  1. All_Channels - Full feature set (baseline comparison)
  2. Frontal_Pole - Most anterior: Fp, Af electrodes
  3. Frontal - Primary frontal: F, FC electrodes  
  4. Temporal - Lateral temporal: T, TP, FT electrodes
  5. Occipital - Visual cortex: O electrodes
  6. Parietal - Sensorimotor integration: P, PO, CP electrodes
  7. Central - Primary motor/sensory: C electrodes

**Extended Regions** (Composite areas):
  8. Frontal_Extended - Entire frontal lobe: Fp, Af, F, FC
  9. Posterior - Entire posterior: P, PO, CP, O
  10. Temporal_Extended - All temporal: T, TP, FT
  11. Central_Extended - Sensorimotor + adjacent: C, CP, FC

HOW CHANNEL FILTERING WORKS
---------------------------
For each channel group, we filter features in 4 steps:

Step 1: Extract Channel Names from Features
  • Supports two feature naming formats:
    - Format 1: "feature_type_band_EEG.CHANNEL" (e.g., "absolute_power_alpha_EEG.F3")
    - Format 2: "feature_type_band_CHANNEL" (e.g., "relative_power_beta_FC1")
  • Extracts the channel identifier (F3, FC1, P4, etc.)

Step 2: Validate Extracted Channels
  • Checks against known EEG electrode patterns (F, C, P, T, O, Fp, Af, FC, CP, PO, TP, FT)
  • Excludes composite/summary features:
    - neurometric_workload (computed workload indices)
    - summary_*, global_*, composite_* (aggregated features)
  • Ensures channel names are alphanumeric, 2-4 characters, starting with a letter

Step 3: Match Channels to Groups Using Prefix Matching
  • Case-insensitive matching (F3 = f3 = EEG.F3)
  • Exact prefix boundaries to avoid substring matches:
    ✓ "F3" matches prefix "F" (next char is digit)
    ✓ "Fz" matches prefix "F" (next char is 'Z')  
    ✗ "FC1" does NOT match prefix "F" (would be substring, but FC1 belongs to FC prefix)
    ✓ "FC1" matches prefix "FC" (exact prefix match)

Step 4: Select Features Matching the Group
  • Returns indices of all features from channels in the group
  • For 'All_Channels' group: returns all features (no filtering)

EXAMPLE: Filtering for 'Frontal' Group
---------------------------------------
Input features: 
  1. 'absolute_power_alpha_EEG.F3'
  2. 'relative_power_beta_FC1'  
  3. 'spectral_entropy_theta_P4'
  4. 'neurometric_workload'

Frontal group definition: ['F', 'FC']

Processing:
  1. Extract channels: F3, FC1, P4, None
  2. Validate: F3 ✓, FC1 ✓, P4 ✓, neurometric_workload ✗ (composite)
  3. Match to prefixes:
     - F3 matches 'F' ✓ → INCLUDE
     - FC1 matches 'FC' ✓ → INCLUDE  
     - P4 matches neither 'F' nor 'FC' ✗ → EXCLUDE
     - neurometric_workload already excluded ✗
  4. Result: Features 1 and 2 selected (F3 and FC1 features)

TYPICAL FEATURE DISTRIBUTION (SENSE-42 Example)
-----------------------------------------------
Dataset: SENSE-42 (NASA-TLX workload, 42 subjects, 1,075 trials)
Total features: 3,648 (before filtering)

Channel Group           Features    Percentage    Channels
--------------------------------------------------------------------------------
All_Channels            3,648       100.0%        All available
Frontal_Extended        1,200       32.9%         Fp1/2, Af3/4, F3/4/7/8/z, FC1/2/5/6/z
Posterior               1,400       38.4%         P3/4/7/8/z, PO3/4/7/8, CP1/2/5/6/z, O1/2/z
Frontal                 900         24.7%         F3/4/7/8/z, FC1/2/5/6/z
Parietal                700         19.2%         P3/4/7/8/z, PO3/4/7/8, CP1/2/5/6/z
Temporal                650         17.8%         T7/8, TP7/8, FT7/8/9/10
Central_Extended        600         16.4%         C3/4/z, CP1/2/5/6/z, FC1/2/5/6/z
Frontal_Pole            300         8.2%          Fp1/2, Af3/4/7/8
Temporal_Extended       650         17.8%         T7/8, TP7/8, FT7/8/9/10
Occipital               350         9.6%          O1/2/z, PO3/4/7/8
Central                 400         11.0%         C3/4/z

VALIDATION CHECKS
-----------------
• Before training, the script verifies:
  1. At least 1 feature remains after filtering
  2. All class labels are present in filtered dataset
  3. No NaN/Inf values in features or labels
  4. Feature count matches expected range for the channel group

• If validation fails, the script logs an error and skips that group

NEUROLOGICAL CONTEXT
--------------------
The channel groups correspond to established neuroscience:

• Frontal regions → Executive function, working memory, decision-making
• Parietal regions → Attention, spatial processing, sensory integration  
• Temporal regions → Auditory processing, language, episodic memory
• Occipital regions → Visual processing
• Central regions → Motor control, somatosensory processing

By comparing model performance across these groups, we identify which brain regions
are most informative for mental workload classification.

See get_channel_categories() function for detailed group definitions.
"""
import sys
import os
import time

# Force unbuffered output for real-time logging
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

print(f"[{time.strftime('%H:%M:%S')}] Script starting...")
print(f"[{time.strftime('%H:%M:%S')}] Loading imports...")

# Author: EEG Biosensing Team
# Date: October 2025
# Dataset: MOCAS (Multi-Operator Cognitive Assessment)
# Task: Classification (discrete workload levels)
#
# AGGREGATION METHODOLOGY:
# - Feature Selection: MRMR only, k=24 features (fixed)
# - Models: 30+ total across 8 algorithm families
# - Evaluation: Leave-N-Subjects-Out CV
# - Metric: Accuracy (classification accuracy)

import argparse
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import re

# Force UTF-8 encoding for stdout to handle emoji characters on Windows
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# RANDOM STATE CONFIGURATION
# =============================================================================
RANDOM_STATE = 42  # For reproducibility (train/test split, CV folds, model initialization)
N_JOBS = 1  # Limit parallelism per model to avoid CPU contention when running multiple scripts

# Add parent directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import warnings
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from collections import Counter
import argparse
    
# Optional dependencies
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("âš ï¸ XGBoost not available")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("âš ï¸ CatBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("âš ï¸ LightGBM not available")

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("âš ï¸ imbalanced-learn not available")
    
from datasets.raw_eeg import EEGRawDataset
from channel_importance.preprocessing_fixed import preprocess_data
from core.utils import get_sklearn_data
from datasets.processed_eeg import EEGProcessedDataset


# =============================================================================
# LOSO CROSS-VALIDATION UTILITIES
# =============================================================================

def extract_subject_from_filename(filepath):
    """Extract subject ID from file path 
    Supports multiple formats:
    - MOCAS: P10_4_features.parquet -> P10
    - Heat-the-chair regression: S01_C1_task1_phase1_features.parquet -> S01
    - Heat-the-chair classification: 01_task1_phase1.parquet -> S01
    - N-back: 05_task1_phase2.parquet.parquet -> S05
    """
    filename = filepath.name if hasattr(filepath, 'name') else str(filepath)
    
    # Try MOCAS format: P10_4_features.parquet
    match = re.match(r'P(\d+)_', filename)
    if match:
        return f"P{match.group(1)}"
    
    # Try Heat-the-chair regression format: S01_C1_task1_phase1_features.parquet  
    match = re.match(r'S(\d+)_', filename)
    if match:
        return f"S{match.group(1)}"
    
    # Try Heat-the-chair classification and N-back format: 01_task1_phase1.parquet or 05_task1_phase2.parquet.parquet
    match = re.match(r'(\d{2,})_', filename)  # At least 2 digits to avoid single digit matches
    if match:
        return f"S{match.group(1).zfill(2)}"  # Convert "05" to "S05", "12" to "S12"
    
    return None


def get_subject_ids(dataset):
    """Extract unique subject IDs from the dataset file paths"""
    subjects = set()
    for file_entry in dataset.file_list:
        # Handle both formats: (filepath, label) for EEGRawDataset or just filepath for EEGRawRegressionDataset
        filepath = file_entry[0] if isinstance(file_entry, tuple) else file_entry
        subject_id = extract_subject_from_filename(filepath)
        if subject_id:
            subjects.add(subject_id)
    return sorted(list(subjects))


def create_subject_to_indices_map(dataset):
    """Create mapping from subject ID to list of sample indices"""
    subject_map = {}
    
    for idx, file_entry in enumerate(dataset.file_list):
        # Handle both formats: (filepath, label) for EEGRawDataset or just filepath for EEGRawRegressionDataset
        filepath = file_entry[0] if isinstance(file_entry, tuple) else file_entry
        subject_id = extract_subject_from_filename(filepath)
        if subject_id:
            if subject_id not in subject_map:
                subject_map[subject_id] = []
            subject_map[subject_id].append(idx)
    
    return subject_map


def create_lnso_splits(subjects, n_test_subjects=4, random_state=42):
    """Create Leave-N-Subjects-Out cross-validation splits"""
    np.random.seed(random_state)
    subjects = list(subjects)
    np.random.shuffle(subjects)  # Shuffle for randomness
    
    # Create folds
    folds = []
    n_subjects = len(subjects)
    
    for i in range(0, n_subjects, n_test_subjects):
        test_subjects = subjects[i:i + n_test_subjects]
        train_subjects = [s for s in subjects if s not in test_subjects]
        folds.append((train_subjects, test_subjects))
    
    return folds


# =============================================================================
# CHANNEL FILTERING AND CATEGORIES
# =============================================================================

def filter_features_by_channels(feature_names, channels_to_keep):
    """
    Filter feature names based on EEG electrode channels (case-insensitive).
    Handles both formats: 'feature_band_EEG.CHANNEL' and 'feature_band_CHANNEL'
    Enhanced with exclusion patterns and EEG channel validation.
    
    Args:
        feature_names: List of feature names (e.g., ['absolute_power_alpha_EEG.AF3', 'absolute_power_alpha_Af3'])
        channels_to_keep: List of channel prefixes to keep (e.g., ['Fp', 'F'])
    
    Returns:
        List of indices of features that match the specified channels
    """
    # Define exclusion patterns for composite/summary features (not spatial channels)
    exclude_patterns = [
        'neurometric_workload',  # Composite workload features
        'overall_cv',           # Cross-validation summary features
        'summary_',            # Summary statistics
        'global_',             # Global features
        'composite_'           # Composite features
    ]
    
    # Valid EEG channel patterns (alphanumeric, 2-4 chars, starts with letter)
    valid_channel_patterns = [
        'F', 'C', 'P', 'T', 'O',           # Standard 10-20 system
        'Fp', 'Af', 'FC', 'CP', 'PO',      # Extended channels
        'TP', 'FT'                         # Additional temporal combinations
    ]
    
    filtered_indices = []
    identified_channels = set()
    
    for i, feature_name in enumerate(feature_names):
        # First check: Skip features matching exclusion patterns
        if any(pattern.lower() in feature_name.lower() for pattern in exclude_patterns):
            continue
            
        channel = None
        
        # Try format 1: feature_type_band_EEG.CHANNEL (e.g., "absolute_power_alpha_EEG.F3")
        if '_EEG.' in feature_name:
            channel = feature_name.split('_EEG.')[-1]  # Gets "F3"
        
        # Try format 2: feature_type_band_CHANNEL (e.g., "absolute_power_alpha_Af3")
        else:
            parts = feature_name.split('_')
            if len(parts) >= 3:
                potential_channel = parts[-1]  # Gets last part
                
                # Enhanced validation: Check if it's a valid EEG channel
                if (len(potential_channel) >= 2 and len(potential_channel) <= 4 and
                    potential_channel[0].isalpha() and potential_channel.isalnum()):
                    
                    # Check if it matches known EEG channel patterns
                    channel_prefix = potential_channel.rstrip('0123456789')  # Remove numbers
                    if any(channel_prefix.upper().startswith(pattern.upper()) 
                           for pattern in valid_channel_patterns):
                        channel = potential_channel
        
        # Check if we found a valid channel and it matches our criteria
        if channel:
            # Check if channel starts with any of the specified prefixes (case-insensitive)
            # Use exact prefix matching to avoid 'C' matching 'CP' or 'FC'
            for channel_prefix in channels_to_keep:
                channel_upper = channel.upper()
                prefix_upper = channel_prefix.upper()
                # Match if channel starts with prefix AND next char (if exists) is a digit or 'Z'
                if channel_upper.startswith(prefix_upper):
                    # Ensure it's an exact prefix match (not a substring of a longer prefix)
                    if len(channel_upper) == len(prefix_upper) or channel_upper[len(prefix_upper)].isdigit() or channel_upper[len(prefix_upper)] == 'Z':
                        filtered_indices.append(i)
                        identified_channels.add(channel_upper)  # Store in uppercase for consistency
                        break
    
    # Summary of channel filtering
    if identified_channels:
        print(f"  â†’ Found {len(filtered_indices)} features from {len(identified_channels)} channels")
    else:
        print(f"  âš ï¸ No channels found for prefixes: {channels_to_keep}")
        print(f"  â†’ 0 features selected from {len(feature_names)} total features")
    return filtered_indices


def get_channel_categories():
    """
    Get standard EEG electrode categories for brain region analysis.
    
    CHANNEL GROUPING METHODOLOGY
    ============================
    This function defines 11 anatomical channel groups based on the International 10-20 
    system and its extended variants (10-10, 10-5). Channels are grouped by:
    
    1. **Anatomical Location**: Standard brain regions (Frontal, Temporal, Parietal, etc.)
    2. **Electrode Prefix Matching**: Case-insensitive prefix matching with exact boundary checks
    3. **Extended Combinations**: Composite regions combining adjacent areas
    
    HOW CHANNEL MATCHING WORKS
    ==========================
    Each group specifies electrode prefixes (e.g., ['F', 'FC'] for Frontal).
    The filter_features_by_channels() function:
    
    1. Extracts channel names from feature names in two formats:
       - Format 1: "feature_type_band_EEG.CHANNEL" → extracts "CHANNEL"
       - Format 2: "feature_type_band_CHANNEL" → extracts last part as channel
    
    2. Validates extracted channels against known EEG patterns:
       - Valid patterns: F, C, P, T, O, Fp, Af, FC, CP, PO, TP, FT
       - Excludes composite features: neurometric_workload, summary_, global_
    
    3. Matches channels to groups using prefix matching with boundary checks:
       - Channel "F3" matches prefix "F" (next char is digit)
       - Channel "Fz" matches prefix "F" (next char is 'Z')
       - Channel "FC1" does NOT match prefix "F" (would be substring match)
       - Channel "FC1" matches prefix "FC" (exact prefix match)
    
    4. All matching is case-insensitive (F3 = f3 = EEG.F3)
    
    CHANNEL GROUP DEFINITIONS
    =========================
    
    **Individual Regions** (Pure anatomical areas):
    -----------------------------------------------
    • All_Channels: [None] - Uses all available features (no filtering)
      Purpose: Baseline comparison with full feature set
    
    • Frontal_Pole: ['Fp', 'Af'] - Most anterior electrodes
      Channels: Fp1, Fp2, Fpz, Af3, Af4, Af7, Af8, Afz
      Function: Executive control, working memory, attention
      
    • Frontal: ['F', 'FC'] - Primary frontal cortex
      Channels: F3, F4, F7, F8, Fz, FC1, FC2, FC5, FC6, FCz
      Function: Motor planning, decision-making, cognitive control
      Note: Excludes Fp/Af (separate Frontal_Pole group)
      
    • Temporal: ['T', 'TP', 'FT'] - Lateral temporal regions
      Channels: T3/T7, T4/T8, T5/P7, T6/P8, TP7, TP8, FT7, FT8, FT9, FT10
      Function: Auditory processing, language, memory
      Note: Includes fronto-temporal (FT) and temporo-parietal (TP) junctions
      
    • Occipital: ['O'] - Visual cortex (posterior)
      Channels: O1, O2, Oz
      Function: Visual processing
      
    • Parietal: ['P', 'PO', 'CP'] - Sensorimotor integration
      Channels: P3, P4, P7, P8, Pz, PO3, PO4, PO7, PO8, CP1, CP2, CP5, CP6, CPz
      Function: Spatial processing, sensory integration, attention
      Note: Includes parieto-occipital (PO) and centro-parietal (CP) junctions
      
    • Central: ['C'] - Primary sensorimotor cortex
      Channels: C3, C4, Cz
      Function: Motor control, sensory processing
      Note: Pure central electrodes only (excludes FC, CP)
    
    **Extended Regions** (Composite areas combining adjacent regions):
    -----------------------------------------------------------------
    • Frontal_Extended: ['Fp', 'Af', 'F', 'FC'] - Entire frontal lobe
      Combines: Frontal_Pole + Frontal
      Use case: Broad executive function analysis
      
    • Posterior: ['P', 'PO', 'CP', 'O'] - Entire posterior cortex
      Combines: Parietal + Occipital
      Use case: Sensory processing, visual-spatial attention
      
    • Temporal_Extended: ['T', 'TP', 'FT'] - All temporal regions
      Same as Temporal (already includes junctions)
      Use case: Comprehensive language/auditory analysis
      
    • Central_Extended: ['C', 'CP', 'FC'] - Sensorimotor + adjacent
      Combines: Central + adjacent parietal/frontal junctions
      Use case: Motor planning to execution pipeline
    
    EXAMPLE FEATURE FILTERING
    =========================
    Given features: ['absolute_power_alpha_EEG.F3', 'relative_power_beta_FC1', 
                     'spectral_entropy_theta_P4', 'neurometric_workload']
    
    For 'Frontal' group (['F', 'FC']):
      ✓ 'absolute_power_alpha_EEG.F3' → channel 'F3' → matches 'F' prefix → INCLUDED
      ✓ 'relative_power_beta_FC1' → channel 'FC1' → matches 'FC' prefix → INCLUDED
      ✗ 'spectral_entropy_theta_P4' → channel 'P4' → no match → EXCLUDED
      ✗ 'neurometric_workload' → composite feature → EXCLUDED (no spatial channel)
    
    TYPICAL FEATURE COUNTS (SENSE-42 dataset example)
    ================================================
    • All_Channels: 3,648 features (100%)
    • Frontal: ~1,200 features (33%) - largest group
    • Central: ~400 features (11%) - smallest pure group
    • Posterior: ~1,400 features (38%) - largest extended group
    
    Returns:
        Dictionary mapping category names to electrode prefix lists
        Format: {category_name: [prefix1, prefix2, ...] or None}
    """
    return {
        'All_Channels': None,  # Use all channels
        'Frontal_Pole': ['Fp', 'Af'],           # Frontal pole: Fp1/Fp2 (traditional) or Af3/Af4 (alternative)
        'Frontal': ['F', 'FC'],                 # Frontal: F3/F4/F7/F8/Fz + Fronto-Central FC1/FC2/FC5/FC6/FCz
        'Temporal': ['T', 'TP', 'FT'],          # Temporal: T3/T4/T5/T6/T7/T8 + Temporo-Parietal TP + Fronto-Temporal FT
        'Occipital': ['O'],                     # Occipital: O1/O2/Oz (case-insensitive matching)
        'Parietal': ['P', 'PO', 'CP'],          # Parietal: P3/P4/P7/P8/Pz + Parieto-Occipital PO + Centro-Parietal CP
        'Central': ['C'],                       # Central: C3/C4/Cz (case-insensitive matching)
        'Frontal_Extended': ['Fp', 'Af', 'F', 'FC'],        # All frontal regions combined
        'Posterior': ['P', 'PO', 'CP', 'O'],                # All posterior regions (parietal + occipital)
        'Temporal_Extended': ['T', 'TP', 'FT'],             # All temporal regions
        'Central_Extended': ['C', 'CP', 'FC'],              # Central + adjacent regions
    }


# =============================================================================
# MODEL ENSEMBLE DEFINITION
# =============================================================================

def get_model_ensemble2():
    """
    Get ensemble of 30+ diverse classification models spanning multiple algorithm families.
    
    Model families included:
    - Linear: Logistic Regression, Ridge, SGD, Passive Aggressive, Perceptron
    - Tree-based: Random Forest, Extra Trees, Decision Tree, Gradient Boosting
    - Boosting: AdaBoost, Hist Gradient Boosting, XGBoost, CatBoost, LightGBM
    - SVM: RBF SVM, Linear SVC, Nu-SVC
    - Neural: MLP (multiple architectures)
    - Probabilistic: Naive Bayes (Gaussian, Bernoulli), Gaussian Process
    - Distance-based: KNN, Nearest Centroid
    - Discriminant: LDA, QDA
    - Ensemble: Bagging
    
    Returns:
        Dictionary mapping model names to model constructors
    """
    models = {
        # ===== LINEAR MODELS (5) =====
        'LogisticRegression': lambda: LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        
        'RidgeClassifier': lambda: RidgeClassifier(
            alpha=10.0,
            solver='auto',
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        'SGDClassifier': lambda: SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=0.001,
            l1_ratio=0.15,
            max_iter=2000,
            tol=1e-3,
            learning_rate='optimal',
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        'PassiveAggressive': lambda: PassiveAggressiveClassifier(
            C=1.0,
            max_iter=2000,
            tol=1e-3,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        'Perceptron': lambda: Perceptron(
            penalty='l2',
            alpha=0.001,
            max_iter=2000,
            tol=1e-3,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        # ===== TREE-BASED MODELS (4) =====
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        
        'ExtraTrees': lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight="balanced",
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        
        'DecisionTree': lambda: DecisionTreeClassifier(
            criterion='gini',
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=RANDOM_STATE
        ),
        
        # ===== BOOSTING MODELS (2) =====
        'HistGradientBoosting': lambda: HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_depth=5,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE
        ),
        
        'AdaBoost': lambda: AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.8,
            algorithm='SAMME',
            random_state=RANDOM_STATE
        ),
        
        # ===== SVM MODELS (3) =====
        'SVM_RBF': lambda: SVC(
            C=10.0,
            kernel='rbf',
            gamma='auto',
            class_weight='balanced',
            cache_size=500,
            random_state=RANDOM_STATE
        ),
        
        'LinearSVC': lambda: LinearSVC(
            C=1.0,
            penalty='l2',
            loss='squared_hinge',
            dual=True,
            max_iter=2000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        'NuSVC': lambda: NuSVC(
            nu=0.5,
            kernel='rbf',
            gamma='auto',
            class_weight='balanced',
            cache_size=500,
            random_state=RANDOM_STATE
        ),
        
        # ===== NEURAL NETWORKS (3 architectures) =====
        'MLP_Large': lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (HTC has only 23 training samples)
            random_state=RANDOM_STATE
        ),
        
        'MLP_Medium': lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (HTC has only 23 training samples)
            random_state=RANDOM_STATE
        ),
        
        'MLP_Small': lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=0.02,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (HTC has only 23 training samples)
            random_state=RANDOM_STATE
        ),
        
        # ===== PROBABILISTIC MODELS (3) =====
        'GaussianNB': lambda: GaussianNB(
            var_smoothing=1e-8
        ),
        
        'BernoulliNB': lambda: BernoulliNB(
            alpha=1.0,
            binarize=0.0
        ),
        
        'GaussianProcess': lambda: GaussianProcessClassifier(
            kernel=1.0 * RBF(1.0),
            random_state=RANDOM_STATE,
            n_restarts_optimizer=2,
            max_iter_predict=100
        ),
        
        # ===== DISTANCE-BASED MODELS (3) =====
        'KNeighbors_7': lambda: KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2,
            n_jobs=N_JOBS
        ),
        
        'KNeighbors_5': lambda: KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=N_JOBS
        ),
        
        'NearestCentroid': lambda: NearestCentroid(
            metric='euclidean',
            shrink_threshold=None
        ),
        
        # ===== DISCRIMINANT ANALYSIS (2) =====
        'LDA': lambda: LinearDiscriminantAnalysis(
            solver='svd',
            shrinkage=None
        ),
        
        'QDA': lambda: QuadraticDiscriminantAnalysis(
            reg_param=0.1
        ),
        
        # ===== ENSEMBLE META-LEARNERS (1) =====
        'Bagging': lambda: BaggingClassifier(
            n_estimators=50,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
    }
    
    # Add optional advanced boosting models if available
    if HAS_XGBOOST:
        models['XGBoost'] = lambda: XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=1.0,
            scale_pos_weight=1,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    if HAS_CATBOOST:
        models['CatBoost'] = lambda: CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.1,
            l2_leaf_reg=3.0,
            border_count=64,
            bootstrap_type='Bayesian',
            bagging_temperature=1.0,
            random_strength=1.0,
            min_data_in_leaf=5,
            random_state=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False
        )
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = lambda: LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=20,
            min_child_samples=10,
            min_child_weight=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=-1,
            force_col_wise=True
        )
    
    return models



def get_model_ensemble():
    """
    Expanded ensemble of CLASSIFICATION models for tabular EEG statistical features.
    Returns:
        dict[str, callable]: model name -> zero-arg constructor
    """
    models = {
        # ========== LINEAR FAMILY ==========
        'LogisticRegression': lambda: LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs', max_iter=4000,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'LogisticRegressionCV': lambda: LogisticRegressionCV(
            Cs=10, cv=5, solver='lbfgs', max_iter=4000,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'RidgeClassifier': lambda: RidgeClassifier(
            alpha=6.0, class_weight='balanced', random_state=RANDOM_STATE
        ),
        'RidgeClassifierCV': lambda: RidgeClassifierCV(
            alphas=(0.5, 1.0, 2.0, 6.0, 10.0), class_weight='balanced'
        ),
        'SGD_Logistic': lambda: SGDClassifier(
            loss='log_loss', penalty='elasticnet', alpha=1e-3, l1_ratio=0.15,
            max_iter=4000, tol=1e-3, learning_rate='optimal',
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'SGD_Hinge': lambda: SGDClassifier(
            loss='hinge', penalty='elasticnet', alpha=1e-3, l1_ratio=0.15,
            max_iter=4000, tol=1e-3, learning_rate='optimal',
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'PassiveAggressive': lambda: PassiveAggressiveClassifier(
            C=0.5, max_iter=4000, tol=1e-3, class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'Perceptron': lambda: Perceptron(
            penalty='l2', alpha=1e-3, max_iter=4000, tol=1e-3,
            class_weight='balanced', random_state=RANDOM_STATE
        ),

        # ========== TREE-BASED ==========
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_split=10, min_samples_leaf=3,
            max_features='sqrt', class_weight='balanced_subsample',
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        ),
        'ExtraTrees': lambda: ExtraTreesClassifier(
            n_estimators=400, max_depth=10, min_samples_split=10, min_samples_leaf=3,
            max_features='sqrt', bootstrap=True, class_weight='balanced',
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        ),
        'DecisionTree': lambda: DecisionTreeClassifier(
            criterion='gini', max_depth=6, min_samples_split=10, min_samples_leaf=4,
            max_features='sqrt', class_weight='balanced', random_state=RANDOM_STATE
        ),
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=3,
            min_samples_split=10, min_samples_leaf=5, subsample=0.9,
            random_state=RANDOM_STATE
        ),

        # ========== HIST-GRAD / ADA ==========
        'HistGradientBoosting': lambda: HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=250, max_depth=5, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE
        ),
        'AdaBoost': lambda: AdaBoostClassifier(
            n_estimators=200, learning_rate=0.6, algorithm='SAMME',
            random_state=RANDOM_STATE
        ),
        # AdaBoost with slightly deeper base stumps
        'AdaBoost_TreeDepth2': lambda: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
            n_estimators=200, learning_rate=0.4, algorithm='SAMME',
            random_state=RANDOM_STATE
        ),

        # ========== SVM (remember to scale features) ==========
        'SVM_RBF': lambda: SVC(
            C=5.0, kernel='rbf', gamma='scale', probability=True,
            class_weight='balanced', cache_size=500, random_state=RANDOM_STATE
        ),
        'SVM_Poly': lambda: SVC(
            C=3.0, kernel='poly', degree=3, coef0=1.0, gamma='scale', probability=True,
            class_weight='balanced', cache_size=500, random_state=RANDOM_STATE
        ),
        'SVM_Sigmoid': lambda: SVC(
            C=3.0, kernel='sigmoid', coef0=0.5, gamma='scale', probability=True,
            class_weight='balanced', cache_size=500, random_state=RANDOM_STATE
        ),
        'LinearSVC': lambda: LinearSVC(
            C=1.0, loss='squared_hinge', dual=True, max_iter=4000,
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'NuSVC': lambda: NuSVC(
            nu=0.4, kernel='rbf', gamma='scale', probability=True,
            class_weight='balanced', cache_size=500, random_state=RANDOM_STATE
        ),
        # Calibrated LinearSVC for better probabilities
        'Calibrated_LinearSVC': lambda: CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0, loss='squared_hinge', class_weight='balanced',
                max_iter=4000, random_state=RANDOM_STATE
            ),
            cv=5, method='sigmoid'
        ),

        # ========== MLPs ==========
        'MLP_Large': lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            alpha=1e-2, learning_rate='adaptive', max_iter=1000,
            early_stopping=False, random_state=RANDOM_STATE  # Disabled for small datasets
        ),
        'MLP_Medium': lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            alpha=1e-2, learning_rate='adaptive', max_iter=1000,
            early_stopping=False, random_state=RANDOM_STATE  # Disabled for small datasets
        ),
        'MLP_Small': lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
            alpha=2e-2, learning_rate='adaptive', max_iter=1000,
            early_stopping=False, random_state=RANDOM_STATE  # Disabled for small datasets
        ),

        # ========== PROBABILISTIC ==========
        'GaussianNB': lambda: GaussianNB(var_smoothing=1e-8),
        'BernoulliNB': lambda: BernoulliNB(alpha=1.0, binarize=0.0),
        # 'GaussianProcess': lambda: GaussianProcessClassifier(
        #     kernel=C(1.0) * (RBF(1.0) + 0.3*Matern(nu=1.5) + 0.3*RationalQuadratic()),
        #     random_state=RANDOM_STATE, n_restarts_optimizer=2, max_iter_predict=200
        # ),  # DISABLED: O(n³) too slow for high-dimensional data (30-60+ min per model)
        # 'GaussianProcess_RBF': lambda: GaussianProcessClassifier(
        #     kernel=C(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3),
        #     random_state=RANDOM_STATE, n_restarts_optimizer=2, max_iter_predict=200
        # ),  # DISABLED: O(n³) too slow for high-dimensional data

        # ========== DISTANCE-BASED ==========
        'KNeighbors_7': lambda: KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='minkowski', p=2, n_jobs=N_JOBS
        ),
        'KNeighbors_5': lambda: KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='euclidean', n_jobs=N_JOBS
        ),
        'KNeighbors_Manhattan_9': lambda: KNeighborsClassifier(
            n_neighbors=9, weights='distance', metric='manhattan', n_jobs=N_JOBS
        ),
        'RadiusNeighbors': lambda: RadiusNeighborsClassifier(
            radius=10.0, weights='distance', outlier_label=None
        ),
        'NearestCentroid': lambda: NearestCentroid(metric='euclidean', shrink_threshold=None),

        # ========== DISCRIMINANT ==========
        'LDA_LSQR_Shrink': lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'LDA_SVD': lambda: LinearDiscriminantAnalysis(solver='svd'),
        'QDA_Reg': lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),

        # ========== PIPELINES (Dimensionality / Feature Maps) ==========
        'LogReg_PCA95': lambda: Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(n_components=0.95, svd_solver='full')),
            ('clf', LogisticRegression(max_iter=4000, class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'LinearSVC_PCA95': lambda: Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(n_components=0.95, svd_solver='full')),
            ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=RANDOM_STATE, max_iter=4000))
        ]),
        'SVM_RBF_PCA95': lambda: Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(n_components=0.95, svd_solver='full')),
            ('clf', SVC(C=5.0, kernel='rbf', gamma='scale', probability=True,
                        class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'RBFSampler_LinearSVC': lambda: Pipeline([
            ('scale', StandardScaler()),
            ('rbf', RBFSampler(gamma=0.5, n_components=500, random_state=RANDOM_STATE)),
            ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=RANDOM_STATE, max_iter=4000))
        ]),
        'Nystroem_LogReg': lambda: Pipeline([
            ('scale', StandardScaler()),
            ('nys', Nystroem(kernel='rbf', gamma=0.5, n_components=500, random_state=RANDOM_STATE)),
            ('clf', LogisticRegression(max_iter=4000, class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'PolyFeatures_LogReg': lambda: Pipeline([
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('clf', LogisticRegression(max_iter=4000, class_weight='balanced', random_state=RANDOM_STATE))
        ]),

        # ========== ENSEMBLES ==========
        'Bagging': lambda: BaggingClassifier(
            n_estimators=80, max_samples=0.8, max_features=0.8,
            bootstrap=True, bootstrap_features=False,
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        ),
        'Bagging_LogReg': lambda: BaggingClassifier(
            estimator=LogisticRegression(max_iter=4000, class_weight='balanced', random_state=RANDOM_STATE),
            n_estimators=50, max_samples=0.9, max_features=0.9, bootstrap=True,
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        ),
        'VotingSoft': lambda: VotingClassifier(
            estimators=[
                ('logreg', LogisticRegression(max_iter=4000, class_weight='balanced', random_state=RANDOM_STATE)),
                ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced_subsample', n_jobs=N_JOBS, random_state=RANDOM_STATE)),
                ('svm', SVC(C=5.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_STATE))
            ],
            voting='soft', n_jobs=N_JOBS
        ),
        'Stacking_LogRegMeta': lambda: StackingClassifier(
            estimators=[
                ('hgb', HistGradientBoostingClassifier(max_depth=5, random_state=RANDOM_STATE)),
                ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced_subsample', n_jobs=N_JOBS, random_state=RANDOM_STATE)),
                ('svm', SVC(C=5.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_STATE)),
            ],
            final_estimator=LogisticRegression(max_iter=4000, class_weight='balanced', random_state=RANDOM_STATE),
            passthrough=True, n_jobs=N_JOBS
        ),
    }

    # ===== OPTIONAL BOOSTERS =====
    if HAS_XGBOOST:
        models['XGBoost'] = lambda: XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.06, min_child_weight=3,
            subsample=0.9, colsample_bytree=0.8, gamma=0.0,
            reg_alpha=0.2, reg_lambda=1.0, scale_pos_weight=1.0,
            random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False,
        )
    if HAS_CATBOOST:
        models['CatBoost'] = lambda: CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.06, l2_leaf_reg=3.0,
            border_count=64, bootstrap_type='Bayesian', bagging_temperature=1.0,
            random_strength=1.0, min_data_in_leaf=10,
            random_state=RANDOM_STATE, verbose=False, allow_writing_files=False
        )
    if HAS_LIGHTGBM:
        models['LightGBM'] = lambda: LGBMClassifier(
            n_estimators=500, max_depth=-1, learning_rate=0.05, num_leaves=31,
            min_child_samples=15, min_child_weight=1e-3, subsample=0.9,
            colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.0,
            class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, force_col_wise=True
        )

    # ===== OPTIONAL IMBALANCED ENSEMBLE =====
    if HAS_IMBLEARN:
        models['BalancedRandomForest'] = lambda: BalancedRandomForestClassifier(
            n_estimators=400, max_depth=10, sampling_strategy='auto',
            class_weight=None, n_jobs=N_JOBS, random_state=RANDOM_STATE
        )

    return models

# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def train_evaluate_sklearn_classification(model_func, X_train, X_test, y_train, y_test, model_name="Model"):
    """Train and evaluate a scikit-learn classification model."""
    model = model_func()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'model': model,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'f1_score': f1
    }


def train_ensemble_models_classification_fold(raw_dataset, train_indices, test_indices, channel_filter_indices=None, selection_method='rf', no_feature_selection=False):
    """
    Train and evaluate ALL models in the ensemble with feature selection (k=24) for a single LOSO fold.
    
    Args:
        raw_dataset: EEGRawDataset instance
        train_indices: List of training sample indices
        test_indices: List of testing sample indices  
        channel_filter_indices: Indices of features to filter by channel
        no_feature_selection: If True, skip feature selection and use all features
        
    Returns:
        dict: Results containing all model performances
    """
    
    try:
        # Apply channel filtering if specified
        pre_filtered_indices = channel_filter_indices if channel_filter_indices is not None else None
        
        if no_feature_selection:
            # No feature selection - use all (pre-filtered) features
            if pre_filtered_indices is None:
                feat_idx = list(range(len(raw_dataset.flattened_features)))
            else:
                feat_idx = pre_filtered_indices
            
            # Create scaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_all = np.array([raw_dataset[i].raw_features if hasattr(raw_dataset[i], 'raw_features') else raw_dataset[i]['raw_features'] for i in range(len(raw_dataset))])
            scaler.fit(X_all[:, feat_idx])
            
            feature_names = [raw_dataset.flattened_features[i] for i in feat_idx] if hasattr(raw_dataset, 'flattened_features') else [f"feature_{i}" for i in feat_idx]
            selection_scores = None
        else:
            # Use feature selection with fixed k=24 features
            feat_idx, scaler, feature_names, selection_scores = preprocess_data(
                raw_dataset, 
                train_indices, 
                24,  # Fixed k=24 features
                task_type='classification',
                selection_method=selection_method,
                pre_filtered_indices=pre_filtered_indices
            )
        
        # Create processed dataset (scaler, feat_idx order matches constructor signature)
        full_processed = EEGProcessedDataset(raw_dataset, scaler, feat_idx)
        
        # Get sklearn-compatible data
        X_train, y_train = get_sklearn_data(full_processed, train_indices)
        X_test, y_test = get_sklearn_data(full_processed, test_indices)
        
        # Handle NaN and infinity values if present
        train_nans = np.isnan(X_train).sum()
        test_nans = np.isnan(X_test).sum()
        train_infs = np.isinf(X_train).sum()
        test_infs = np.isinf(X_test).sum()
        
        if train_nans > 0 or test_nans > 0 or train_infs > 0 or test_infs > 0:
            # Suppress sklearn warnings about inf values during replacement
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Get model ensemble
        models = get_model_ensemble()
        
        # Train and evaluate each model
        model_results = {}
        n_models = len(models)
        
        for model_idx, (model_name, model_func) in enumerate(models.items(), 1):
            try:
                print(f"\r      Model {model_idx}/{n_models}: {model_name[:20]:<20}", end="")
                sys.stdout.flush()
                
                # Clean data immediately before training to catch any infinities
                X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Train and test with error suppression for numerical issues
                with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                    result = train_evaluate_sklearn_classification(model_func, X_train_clean, X_test_clean, y_train, y_test, model_name)
                
                # Store results
                model_results[model_name] = {
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score']
                }
            except Exception as e:
                continue
        
        print()  # Newline after model progress
        return {
            'model_results': model_results,
            'feature_names': feature_names,
            'selection_scores': selection_scores
        }
        
    except Exception as e:
        print(f"  ERROR in fold: {str(e)}")
        traceback.print_exc()
        return {}


def train_ensemble_models_lnso_cv(raw_dataset, cv_splits, subject_to_indices, channel_filter_indices=None, selection_method='rf', no_feature_selection=False):
    """
    Train ensemble models using Leave-N-Subjects-Out cross-validation
    
    Args:
        raw_dataset: EEGRawDataset instance
        cv_splits: List of (train_subjects, test_subjects) tuples
        subject_to_indices: Dict mapping subject_id -> list of sample indices
        channel_filter_indices: List of feature indices to use (None = all features)
    
    Returns:
        dict: Results with accuracy scores for each model across all folds
    """
    # Store fold results for each model - all metrics
    model_fold_scores = {}
    feature_names_final = []
    selection_scores_final = {}
    
    for fold_idx, (train_subjects, test_subjects) in enumerate(cv_splits):
        print(f"   Fold {fold_idx + 1}/{len(cv_splits)}: Train subjects={len(train_subjects)}, Test subjects={len(test_subjects)}")
        
        # Convert subjects to sample indices
        train_indices = []
        for subject in train_subjects:
            if subject in subject_to_indices:
                train_indices.extend(subject_to_indices[subject])
        
        test_indices = []
        for subject in test_subjects:
            if subject in subject_to_indices:
                test_indices.extend(subject_to_indices[subject])
        
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"     âš ï¸ Skipping fold due to empty train/test set")
            continue
            
        # Get fold results for all models
        fold_results = train_ensemble_models_classification_fold(
            raw_dataset, train_indices, test_indices, channel_filter_indices, selection_method, no_feature_selection
        )
        
        # Store feature names from first successful fold (they're consistent across folds)
        if fold_idx == 0 and 'feature_names' in fold_results:
            feature_names_final = fold_results.get('feature_names', [])
            selection_scores_final = fold_results.get('selection_scores', {})
        
        # Store results for each model
        if 'model_results' in fold_results:
            for model_name, model_res in fold_results['model_results'].items():
                if model_name not in model_fold_scores:
                    model_fold_scores[model_name] = {'accuracy': [], 'f1_score': []}
                model_fold_scores[model_name]['accuracy'].append(model_res['accuracy'])
                model_fold_scores[model_name]['f1_score'].append(model_res['f1_score'])
        else:
            # Old format - direct model results
            for model_name, model_res in fold_results.items():
                if isinstance(model_res, dict) and 'accuracy' in model_res:
                    if model_name not in model_fold_scores:
                        model_fold_scores[model_name] = {'accuracy': [], 'f1_score': []}
                    model_fold_scores[model_name]['accuracy'].append(model_res['accuracy'])
                    if 'f1_score' in model_res:
                        model_fold_scores[model_name]['f1_score'].append(model_res['f1_score'])
    
    # Aggregate results across folds for each model
    aggregated_results = {}
    for model_name, fold_scores_dict in model_fold_scores.items():
        if fold_scores_dict and len(fold_scores_dict['accuracy']) > 0:
            aggregated_results[model_name] = {
                'fold_scores': fold_scores_dict,
                'accuracy': np.mean(fold_scores_dict['accuracy']),
                'accuracy_std': np.std(fold_scores_dict['accuracy']),
                'f1_score': np.mean(fold_scores_dict['f1_score']) if fold_scores_dict['f1_score'] else 0.0,
                'f1_score_std': np.std(fold_scores_dict['f1_score']) if fold_scores_dict['f1_score'] else 0.0,
                'n_folds': len(fold_scores_dict['accuracy'])
            }
            f1_str = f", F1={np.mean(fold_scores_dict['f1_score']):.3f} Â± {np.std(fold_scores_dict['f1_score']):.3f}" if fold_scores_dict['f1_score'] else ""
            print(f"     {model_name}: Acc={np.mean(fold_scores_dict['accuracy']):.3f} Â± {np.std(fold_scores_dict['accuracy']):.3f}{f1_str}")
    
    # Calculate overall aggregated accuracy (median for robustness)
    if aggregated_results:
        all_accuracies = [res['accuracy'] for res in aggregated_results.values()]
        
        # Filter out catastrophically bad models (accuracy < 0.1)
        valid_accuracies = [res['accuracy'] for res in aggregated_results.values() if res['accuracy'] >= 0.1]
        
        if valid_accuracies:
            agg_accuracy = np.median(valid_accuracies)
            agg_std = np.std(valid_accuracies)
            n_valid_models = len(valid_accuracies)
            n_failed_models = len(all_accuracies) - n_valid_models
        else:
            agg_accuracy = 0.0
            agg_std = 0.0
            n_valid_models = 0
            n_failed_models = len(all_accuracies)
        
        print(f"  âœ… Aggregated: Acc={agg_accuracy:.3f}Â±{agg_std:.3f} across {n_valid_models} valid models")
        if n_failed_models > 0:
            print(f"  âš ï¸ Excluded {n_failed_models} models with catastrophic failures (accuracy < 0.1)")
        
        return {
            'model_results': aggregated_results,
            'feature_names': feature_names_final,
            'selection_scores': selection_scores_final,
            'accuracy_score': agg_accuracy,
            'accuracy_std': agg_std,
            'n_models': len(aggregated_results)
        }
    else:
        return None


def get_dataset_path(dataset_name):
    """
    Get the dataset path based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset ('mocas', 'htc', 'nback', 'wauc')
    
    Returns:
        str: Absolute path to the dataset
    """
    # Relative paths from project root
    dataset_paths = {
        'mocas': os.path.join('data', 'MOCAS', 'mocas_feature_classification'),
        'htc': os.path.join('data', 'heat_the_chair', 'htc_feature_classification'),
        'nback': os.path.join('data', 'n_back', 'nback_feature_classification'),
        'wauc': os.path.join('data', 'wauc', 'wauc_feature_classification'),
        'sense42': os.path.join('data', 'SENSE-42', 'sense42_feature_classification')
    }
    
    if dataset_name.lower() not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(dataset_paths.keys())}")
    
    # project_root is already absolute, just join the relative path
    relative_path = dataset_paths[dataset_name.lower()]
    absolute_path = os.path.join(project_root, relative_path)
    
    return absolute_path


# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='EEG Channel Importance Analysis - Aggregated Model Ensemble with LOSO CV',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python test_feature_classification_agg_loso.py --mocas
  python test_feature_classification_agg_loso.py --htc
  python test_feature_classification_agg_loso.py --nback
  python test_feature_classification_agg_loso.py --wauc
  python test_feature_classification_agg_loso.py --mocas --test
    '''
)

# Add mutually exclusive group for dataset selection
dataset_group = parser.add_mutually_exclusive_group()
dataset_group.add_argument('--mocas', action='store_true', help='Use MOCAS dataset')
dataset_group.add_argument('--htc', action='store_true', help='Use Heat the Chair (HTC) dataset')
dataset_group.add_argument('--nback', action='store_true', help='Use N-Back dataset')
dataset_group.add_argument('--wauc', action='store_true', help='Use WAUC dataset')
dataset_group.add_argument('--sense42', action='store_true', help='Use SENSE-42 dataset')
dataset_group.add_argument('--dataset', type=str, choices=['mocas', 'htc', 'nback', 'wauc', 'sense42'],
                          help='Specify dataset by name')

# Add subscale selection
parser.add_argument('--subscale', type=str, default=None,
                   choices=['combined', 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration'],
                   help='TLX subscale to use for classification labels (default: combined TLX, use "combined" or omit for combined)')

# Add test mode
parser.add_argument('--test', action='store_true', 
                   help='Run in test mode (quick validation with 3 channel groups only)')

# Feature selection method
parser.add_argument('--selection-method', type=str,
                   choices=['rf', 'mrmr', 'anova'], default='rf',
                   help='Feature selection method: rf (Random Forest), mrmr, or anova (default: rf)')

# Add all-channels-only flag
parser.add_argument('--all-channels-only', action='store_true',
                   help='Only test All_Channels group (skip other channel groups)')

# Add no-feature-selection flag
parser.add_argument('--no-feature-selection', action='store_true',
                   help='Skip feature selection and use all features')

args = parser.parse_args()

# Normalize 'combined' to None for backward compatibility
if args.subscale == 'combined':
    args.subscale = None

# Determine which dataset to use
if args.mocas or (args.dataset and args.dataset.lower() == 'mocas'):
    dataset_name = 'mocas'
elif args.htc or (args.dataset and args.dataset.lower() == 'htc'):
    dataset_name = 'htc'
elif args.nback or (args.dataset and args.dataset.lower() == 'nback'):
    dataset_name = 'nback'
elif args.wauc or (args.dataset and args.dataset.lower() == 'wauc'):
    dataset_name = 'wauc'
elif args.sense42 or (args.dataset and args.dataset.lower() == 'sense42'):
    dataset_name = 'sense42'
else:
    # Default to MOCAS if no argument provided
    dataset_name = 'mocas'
    print("â„¹ï¸ No dataset specified, defaulting to MOCAS")
    print("   Use --mocas, --htc, --nback, --wauc, or --sense42 to specify a dataset\n")

# Get dataset path
data_path = get_dataset_path(dataset_name)

# Check if test mode
test_mode = args.test
all_channels_only = args.all_channels_only
no_feature_selection = args.no_feature_selection

if test_mode:
    print("\n[TEST MODE] Running quick validation")
    print("   Only testing 3 channel groups: All_Channels, Frontal, Central")
    print("   Estimated time: ~15-30 minutes\n")

if all_channels_only:
    print("\n[ALL-CHANNELS-ONLY] Testing only All_Channels group\n")

if no_feature_selection:
    print("\n[NO FEATURE SELECTION] Using all features\n")

# Get feature selection method
selection_method = getattr(args, 'selection_method', 'rf')

# Load dataset
print(f"\n{'='*80}")
print(f"EEG CHANNEL IMPORTANCE ANALYSIS - AGGREGATED MODEL ENSEMBLE with LOSO CV")
if test_mode:
    print(f"                      *** TEST MODE ***")
print(f"{'='*80}")
print(f"\nDataset: {dataset_name.upper()}")
if args.subscale:
    print(f"Subscale: {args.subscale.upper()}")
print(f"Feature Selection: {selection_method.upper() if selection_method else 'NONE (all features)'}")
print(f"Loading from: {data_path}")

try:
    raw_dataset = EEGRawDataset(data_path, target_suffix=args.subscale)
    
    # Get dataset info - handle both dataclass (EEGRawDatasetEntry) and dict formats
    first_sample = raw_dataset[0]
    feature_vector = first_sample.raw_features if hasattr(first_sample, 'raw_features') else first_sample['raw_features']
    # Optimized: get labels from file_list directly instead of reading all files
    all_labels = [label for _, label in raw_dataset.file_list]
    class_counts = Counter(all_labels)
    
    print(f"OK Dataset loaded: {len(raw_dataset)} samples, {len(raw_dataset.flattened_features)} features")
    print(f"   Classes: {dict(class_counts)}")
    
except Exception as e:
    print(f"ERROR: Error loading dataset: {e}")
    exit(1)

# Setup Leave-N-Subjects-Out Cross-Validation
print(f"\nðŸ“Š Setting up Leave-N-Subjects-Out Cross-Validation...")

# Get subject information
subjects = get_subject_ids(raw_dataset)
subject_to_indices = create_subject_to_indices_map(raw_dataset)
n_subjects = len(subjects)
n_test_subjects = max(1, n_subjects // 5)  # N = total_subjects / 5

print(f"   Total subjects: {n_subjects}")
print(f"   Test subjects per fold: {n_test_subjects}")
print(f"   Subjects: {subjects}")

# Create cross-validation splits
cv_splits = create_lnso_splits(subjects, n_test_subjects=n_test_subjects, random_state=RANDOM_STATE)
print(f"   Number of CV folds: {len(cv_splits)}")

# Display model ensemble
models = get_model_ensemble()
print(f"\nðŸ¤– Model Ensemble ({len(models)} models):")
for i, model_name in enumerate(models.keys(), 1):
    print(f"   {i}. {model_name}")

# Channel categories
channel_categories = get_channel_categories()

# In all-channels-only mode, only use All_Channels
if all_channels_only:
    channel_categories = {
        'All_Channels': channel_categories['All_Channels']
    }
    print(f"\n[ALL-CHANNELS-ONLY] Using only All_Channels group")
# In test mode, only use subset of channels
elif test_mode:
    test_channels = {
        'All_Channels': channel_categories['All_Channels'],
        'Frontal': channel_categories['Frontal'],
        'Central': channel_categories['Central']
    }
    channel_categories = test_channels
    print(f"\nðŸ§ª Test mode: Using only {len(channel_categories)} channel groups")

# Results structure: results[channel_category] = model ensemble results with LOSO CV
results = {}
all_channels_baseline = None  # Store baseline performance for Î”Accuracy calculation
model_specific_baselines = {}  # Store baseline for each model

# Get feature names for channel filtering
first_sample = raw_dataset[0]
all_feature_names = raw_dataset.flattened_features

print(f"\nðŸ§  Starting aggregated channel importance analysis with LOSO CV...")
print(f"   Method: ANOVA (k=24) + {len(models)} Models + Leave-{n_test_subjects}-Subjects-Out CV")
print(f"   Testing {len(channel_categories)} channel groups")
sys.stdout.flush()

channel_start_time = time.time()
for ch_idx, (channel_name, channel_prefixes) in enumerate(channel_categories.items(), 1):
    print(f"\n[{time.strftime('%H:%M:%S')}] ðŸ“ Channel Group {ch_idx}/{len(channel_categories)}: {channel_name}", end=" ")
    sys.stdout.flush()
    # Get channel filter indices
    if channel_prefixes is None:
        channel_filter_indices = None
        print(f"all channels")
    else:
        channel_filter_indices = filter_features_by_channels(all_feature_names, channel_prefixes)
        if len(channel_filter_indices) == 0:
            print(f"âš ï¸ No features found, skipping")
            continue
    
    # Train ensemble models with LOSO CV
    try:
        ensemble_results = train_ensemble_models_lnso_cv(
            raw_dataset, cv_splits, subject_to_indices, channel_filter_indices, selection_method, no_feature_selection
        )
        
        if ensemble_results and 'accuracy_score' in ensemble_results:
            results[channel_name] = ensemble_results
            
            # Store all-channels baseline for Î”Accuracy calculation
            if channel_name == 'All_Channels':
                all_channels_baseline = ensemble_results['accuracy_score']
                # Also store individual model baselines
                if 'model_results' in ensemble_results:
                    model_specific_baselines = {
                        name: res['accuracy'] 
                        for name, res in ensemble_results['model_results'].items()
                    }
            
        else:
            print(f"â†’ Failed")
            
    except Exception as e:
        print(f"â†’ Error: {str(e)}")
        traceback.print_exc()

# Calculate aggregated rankings
print(f"\n{'='*80}")
print(f"AGGREGATED CHANNEL IMPORTANCE RESULTS (LOSO CV)")
print(f"{'='*80}")

if all_channels_baseline is not None and len(model_specific_baselines) > 0:
    print(f"\nBaseline (All Channels):")
    print(f"   Aggregated Accuracy = {all_channels_baseline:.3f}")
    for model_name, baseline_acc in model_specific_baselines.items():
        print(f"   {model_name}: {baseline_acc:.3f}")
    
    # Calculate Î”Accuracy for each model and each channel group
    channel_rankings_per_model = {model_name: [] for model_name in model_specific_baselines.keys()}
    
    for channel_name, channel_results in results.items():
        if channel_name != 'All_Channels' and 'model_results' in channel_results:
            for model_name, model_res in channel_results['model_results'].items():
                if model_name in model_specific_baselines:
                    delta_accuracy = model_res['accuracy'] - model_specific_baselines[model_name]
                    channel_rankings_per_model[model_name].append(
                        (channel_name, model_res['accuracy'], delta_accuracy)
                    )
    
    # Rank channels for each model independently
    model_ranks = {model_name: {} for model_name in model_specific_baselines.keys()}
    
    for model_name, channel_deltas in channel_rankings_per_model.items():
        # Sort by Î”Accuracy descending
        channel_deltas.sort(key=lambda x: x[2], reverse=True)
        
        # Assign ranks (1 = best)
        for rank, (channel_name, acc, delta) in enumerate(channel_deltas, 1):
            model_ranks[model_name][channel_name] = rank
    
    # Calculate average rank for each channel
    all_channel_names = set()
    for ranks in model_ranks.values():
        all_channel_names.update(ranks.keys())
    
    aggregated_rankings = []
    for channel_name in all_channel_names:
        ranks = [model_ranks[model_name].get(channel_name, 999) 
                 for model_name in model_specific_baselines.keys()]
        avg_rank = np.mean(ranks)
        
        # Get aggregated accuracy for this channel
        channel_acc = results[channel_name]['accuracy_score'] if channel_name in results else 0.0
        delta_acc = channel_acc - all_channels_baseline
        
        aggregated_rankings.append((channel_name, channel_acc, delta_acc, avg_rank, ranks))
    
    # Sort by average rank (lower is better)
    aggregated_rankings.sort(key=lambda x: x[3])
    
    print(f"\nðŸ“Š AGGREGATED Channel Importance Ranking (by average rank across {len(models)} models):")
    print(f"{'Rank':<6} {'Channel':<20} {'Agg Acc':<10} {'Î”Acc':<10} {'Avg Rank':<12} {'Model Ranks'}")
    print(f"{'-'*90}")
    
    for i, (channel_name, acc, delta, avg_rank, individual_ranks) in enumerate(aggregated_rankings):
        rank = i + 1
        delta_sign = "+" if delta >= 0 else ""
        status = "ðŸŸ¢" if delta > 0 else "ðŸ”´" if delta < -0.01 else "ðŸŸ¡"
        ranks_str = ', '.join([f'{r}' for r in individual_ranks])
        print(f"{rank:<6} {status} {channel_name:<18} {acc:.3f}      {delta_sign}{delta:>6.3f}    {avg_rank:>6.2f}      [{ranks_str}]")
    
    # Performance summary
    if aggregated_rankings:
        best_channel = aggregated_rankings[0]
        worst_channel = aggregated_rankings[-1]
        positive_count = len([ch for ch in aggregated_rankings if ch[2] > 0])
        
        print(f"\nðŸ“ˆ Summary:")
        print(f"   â€¢ {positive_count}/{len(aggregated_rankings)} groups outperform baseline")
        print(f"   â€¢ Best: {best_channel[0]} (avg rank {best_channel[3]:.2f}, Î”Acc {best_channel[2]:+.3f})")
        print(f"   â€¢ Worst: {worst_channel[0]} (avg rank {worst_channel[3]:.2f}, Î”Acc {worst_channel[2]:+.3f})")
        print(f"   â€¢ Aggregation Method: Average rank across {len(models)} models with LOSO CV")
    
    # Export results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_TEST" if test_mode else ""
    subscale_suffix = f"_{args.subscale}" if args.subscale else ""
    output_file = f"channel_importance_aggregated_{dataset_name}_LOSO{subscale_suffix}{mode_suffix}_{timestamp}.csv"
    output_dir = os.path.join(project_root, 'channel_importance', 'feature_tests', 'tests', 'classification', 'loso')
    output_path = os.path.join(output_dir, output_file)
    
    # Prepare comprehensive DataFrame with all metrics
    export_data = []
    for rank, (channel_name, acc, delta, avg_rank, individual_ranks) in enumerate(aggregated_rankings, 1):
        row = {
            'Rank': rank,
            'Channel_Category': channel_name,
            'Aggregated_Accuracy': acc,
            'Delta_Accuracy': delta,
            'Average_Rank': avg_rank,
            'N_Features': len(results[channel_name].get('feature_names', [])) if channel_name in results else 0,
            'Feature_Names': '|'.join(results[channel_name].get('feature_names', [])) if channel_name in results else '',
        }
        
        # Add feature selection scores (pipe-separated: feature=score)
        if channel_name in results and results[channel_name].get('selection_scores'):
            scores_dict = results[channel_name]['selection_scores']
            # Sort by score descending for readability
            sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            row['Selection_Scores'] = '|'.join([f"{feat}={score:.4f}" for feat, score in sorted_scores])
        else:
            row['Selection_Scores'] = ''
        
        # Add individual model ranks
        for model_name, model_rank in zip(model_specific_baselines.keys(), individual_ranks):
            row[f'{model_name}_Rank'] = model_rank
        
        # Add individual model detailed metrics
        if channel_name in results and 'model_results' in results[channel_name]:
            for model_name, model_res in results[channel_name]['model_results'].items():
                row[f'{model_name}_Accuracy'] = model_res['accuracy']
                row[f'{model_name}_Accuracy_Std'] = model_res.get('accuracy_std', 0.0)
                row[f'{model_name}_F1_Score'] = model_res.get('f1_score', 0.0)
                row[f'{model_name}_F1_Score_Std'] = model_res.get('f1_score_std', 0.0)
                row[f'{model_name}_N_Folds'] = model_res.get('n_folds', 0)
                row[f'{model_name}_Delta_Accuracy'] = model_res['accuracy'] - model_specific_baselines.get(model_name, 0)
        
        export_data.append(row)
    
    df_results = pd.DataFrame(export_data)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nðŸ’¾ Results exported to: {output_path}")
    
    # Also save a detailed run info file
    detailed_output_file = f"run_details_aggregated_{dataset_name}_classification_loso{subscale_suffix}{mode_suffix}_{timestamp}.csv"
    detailed_output_path = os.path.join(output_dir, detailed_output_file)
    
    # Create detailed records for each channel-model combination
    detailed_data = []
    for channel_name, channel_results in results.items():
        if 'model_results' in channel_results:
            # Get feature info once per channel
            feature_names_str = '|'.join(channel_results.get('feature_names', []))
            selection_scores = channel_results.get('selection_scores', {})
            if selection_scores:
                sorted_scores = sorted(selection_scores.items(), key=lambda x: x[1], reverse=True)
                selection_scores_str = '|'.join([f"{feat}={score:.4f}" for feat, score in sorted_scores])
            else:
                selection_scores_str = ''
            
            for model_name, model_res in channel_results['model_results'].items():
                baseline_acc = model_specific_baselines.get(model_name, 0)
                detailed_data.append({
                    'Dataset': dataset_name.upper(),
                    'Test_Mode': test_mode,
                    'Channel_Category': channel_name,
                    'Model': model_name,
                    'Accuracy_Mean': model_res['accuracy'],
                    'Accuracy_Std': model_res.get('accuracy_std', 0.0),
                    'F1_Score_Mean': model_res.get('f1_score', 0.0),
                    'F1_Score_Std': model_res.get('f1_score_std', 0.0),
                    'N_Folds': model_res.get('n_folds', 0),
                    'Baseline_Accuracy': baseline_acc,
                    'Delta_Accuracy': model_res['accuracy'] - baseline_acc,
                    'N_LNSO_Folds': len(cv_splits),
                    'N_Test_Subjects_Per_Fold': n_test_subjects,
                    'N_Total_Subjects': n_subjects,
                    'N_Features': len(channel_results.get('feature_names', [])),
                    'Feature_Names': feature_names_str,
                    'Selection_Scores': selection_scores_str,
                    'Random_State': RANDOM_STATE,
                    'Timestamp': timestamp
                })
    
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(detailed_output_path, index=False)
    print(f"ðŸ’¾ Detailed run data exported to: {detailed_output_path}")
    
    # Print test mode summary
    if test_mode:
        print(f"\n{'='*80}")
        print(f"ðŸ§ª TEST MODE SUMMARY")
        print(f"{'='*80}")
        print(f"âœ… All components working correctly:")
        print(f"   âœ“ Dataset loading: {len(raw_dataset)} samples")
        print(f"   âœ“ LOSO CV setup: {len(cv_splits)} folds with {n_test_subjects} test subjects")
        print(f"   âœ“ Model ensemble: {len(models)} models trained")
        print(f"   âœ“ Channel groups: {len(results)} tested")
        print(f"   âœ“ Ranking aggregation: {len(aggregated_rankings)} channels ranked")
        print(f"   âœ“ CSV export: 1 file saved")
        print(f"\nðŸ’¡ Test passed! Ready for full run without --test flag")
        print(f"{'='*80}\n")

else:
    print("   âš ï¸ No baseline found - cannot calculate aggregated rankings")

print(f"\nâœ… Aggregated LOSO analysis complete - tested {len(results)} channel groups with {len(models)} models")
if test_mode:
    print(f"   ðŸ§ª TEST MODE - Use without --test flag for full analysis")
print(f"{'='*80}\n")

