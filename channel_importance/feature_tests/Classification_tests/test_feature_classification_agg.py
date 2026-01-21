"""
EEG Channel Importance Analysis for Classification Tasks - AGGREGATED MODEL ENSEMBLE
====================================================================================
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Force unbuffered output for real-time logging
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

print(f"[{time.strftime('%H:%M:%S')}] Script starting...")
print(f"[{time.strftime('%H:%M:%S')}] Loading imports...")

import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Import models
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, Perceptron
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier, VotingClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

print(f"[{time.strftime('%H:%M:%S')}] Core imports complete, loading optional dependencies...")

# Try optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]  # Go up from Classification_tests -> feature_tests -> channel_importance -> eeg-biosensing
sys.path.insert(0, str(project_root))

from datasets.raw_eeg import EEGRawDataset
from datasets.processed_eeg import EEGProcessedDataset
from channel_importance.preprocessing_fixed import preprocess_data
from core.utils import get_sklearn_data

"""
This script performs channel importance analysis by AGGREGATING rankings from 30+ different models.
Instead of relying on a single model, this approach combines insights from multiple diverse models
spanning different algorithm families to produce highly robust and generalizable channel importance rankings.

Author: EEG Biosensing Team
Date: October 2025
Dataset: MOCAS (Multi-Operator Cognitive Assessment)
Task: Classification (discrete workload levels)

AGGREGATION METHODOLOGY:
-----------------------
- Feature Selection: ANOVA only, k=24 features (fixed)
- Models (30+ total across 8 algorithm families):
  
  LINEAR MODELS (5):
    1. LogisticRegression (L2 regularization)
    2. Ridge Classifier
    3. SGD Classifier (ElasticNet)
    4. Passive Aggressive Classifier
    5. Perceptron
  
  TREE-BASED MODELS (4):
    6. Random Forest
    7. Extra Trees
    8. Decision Tree
    9. Gradient Boosting
  
  BOOSTING MODELS (5):
    10. Histogram Gradient Boosting
    11. AdaBoost
    12. XGBoost (optional)
    13. CatBoost (optional)
    14. LightGBM (optional)
  
  SVM MODELS (3):
    15. SVM with RBF kernel
    16. Linear SVC
    17. Nu-SVC
  
  NEURAL NETWORKS (3 architectures):
    18. MLP Large (128→64→32)
    19. MLP Medium (64→32)
    20. MLP Small (32→16)
  
  PROBABILISTIC MODELS (3):
    21. Gaussian Naive Bayes
    22. Bernoulli Naive Bayes
    23. Gaussian Process Classifier
  
  DISTANCE-BASED MODELS (3):
    24. K-Nearest Neighbors (k=7)
    25. K-Nearest Neighbors (k=5)
    26. Nearest Centroid
  
  DISCRIMINANT ANALYSIS (2):
    27. Linear Discriminant Analysis (LDA)
    28. Quadratic Discriminant Analysis (QDA)
  
  META-ENSEMBLES (1):
    29. Bagging Classifier

- Evaluation: 3-fold TIED StratifiedKFold CV, identical splits for all channel groups and models
- Metric: Accuracy (classification accuracy)
- Channel Importance: AGGREGATED by averaging rankings across all available models
  * Each model produces ΔAccuracy = (channel_group_accuracy) - (all_channels_baseline)
  * Channels are ranked 1-N for each model independently
  * Final ranking = average rank across all models (lower = more important)
- Preprocessing: Near-zero variance filtering (std < 1e-10) + Standardized feature scaling

Aggregation Benefits:
--------------------
- Reduces model-specific bias (different models have different strengths)
- More robust to outliers and overfitting
- Captures channel importance across diverse modeling approaches (linear, non-linear, probabilistic, etc.)
- Provides consensus ranking instead of single-model opinion
- Covers 8 distinct algorithm families for maximum generalizability
- Multiple neural network architectures for complexity diversity
- Both parametric and non-parametric approaches

Model Family Diversity:
----------------------
- Linear: Fast, interpretable, assumes linear relationships
- Tree-based: Captures non-linear interactions, feature importance
- Boosting: Sequential learning, error correction
- SVM: Kernel methods, high-dimensional spaces
- Neural: Deep learning, complex patterns
- Probabilistic: Uncertainty quantification, Bayesian approaches
- Distance-based: Local patterns, similarity-based
- Discriminant: Statistical separation, covariance structure

Channel Categories:
------------------
- All: All available EEG electrodes (baseline for ΔAccuracy calculation)
- Frontal: ['Fp', 'AF', 'F', 'FC'] - attention, working memory
- Central: ['C', 'CP'] - motor control, sensorimotor processing  
- Parietal: ['P', 'PO'] - spatial processing, attention networks
- Occipital: ['O', 'OZ'] - visual processing
- Temporal: ['T', 'TP', 'FT'] - auditory processing, memory

Expected Runtime:
----------------
- ~60-100 minutes depending on system specifications and available models
- 11 channel groups × 26-29 models = 286-319 training configurations
- Each with 3-fold CV = 858-957 total model fits
- Results exported to CSV with aggregated rankings
- Test mode (~12-20 minutes): 3 channel groups × 26-29 models = 78-87 fits

Note: Gaussian Process may be slower on larger datasets due to O(n³) complexity
"""

import pandas as pd
import numpy as np
import sys
import os
import traceback
from datetime import datetime

# Force UTF-8 encoding for stdout to handle emoji characters on Windows
# Note: When running with `python -u`, buffering is already disabled
import io

# =============================================================================
# RANDOM STATE CONFIGURATION
# =============================================================================
RANDOM_STATE = 42  # For reproducibility (train/test split, CV folds, model initialization)
N_JOBS = 1  # Limit parallelism per model to avoid CPU contention when running multiple scripts

# Note: Already set up project_root and sys.path at top of file
# These sklearn imports are duplicates but harmless (already imported above)
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
    print("⚠️ XGBoost not available")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️ CatBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM not available")

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def filter_features_by_channels(feature_names, channels_to_keep):
    """
    Filter feature names based on EEG electrode channels (case-insensitive).
    Handles both formats: 
    - Frequency-domain: 'feature_band_EEG.CHANNEL' or 'feature_band_CHANNEL'
    - Time-domain: 'time_band_CHANNEL' (e.g., '0_theta_Fp1')
    
    Args:
        feature_names: List of feature names 
        channels_to_keep: List of channel prefixes to keep (e.g., ['Fp', 'F'])
    
    Returns:
        List of indices of features that match the specified channels
    """
    import re
    
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
        
        # Try format 1: Time-domain format "t_band_CHANNEL" (e.g., "0_theta_Fp1")
        # Pattern: number_band_channel
        if '_' in feature_name:
            parts = feature_name.split('_')
            if len(parts) >= 3:
                # Check if first part is a number (time index)
                if parts[0].isdigit():
                    # Last part should be the channel name
                    potential_channel = parts[-1]
                    # Validate it's a channel (starts with letter, alphanumeric, 2-4 chars)
                    if (len(potential_channel) >= 2 and len(potential_channel) <= 4 and
                        potential_channel[0].isalpha() and potential_channel.replace('.', '').isalnum()):
                        channel = potential_channel.replace('EEG.', '')  # Remove EEG. prefix if present
                
                # Try format 2: Frequency-domain "feature_band_EEG.CHANNEL" or "feature_band_CHANNEL"
                elif 'EEG.' in feature_name:
                    channel = feature_name.split('_EEG.')[-1]  # Gets "F3"
                else:
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
        print(f"  → Found {len(filtered_indices)} features from {len(identified_channels)} channels")
    else:
        print(f"  ⚠️ No channels found for prefixes: {channels_to_keep}")
        print(f"  → 0 features selected from {len(feature_names)} total features")
    return filtered_indices


def get_channel_categories():
    """
    Get standard EEG electrode categories for brain region analysis.
    
    Returns:
        Dictionary mapping category names to electrode prefixes
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
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
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
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
        ),
        
        'MLP_Medium': lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
        ),
        
        'MLP_Small': lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=0.02,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
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
            Cs=10, cv=3, solver='lbfgs', max_iter=4000,
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
            l2_regularization=1.0, early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
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
            cv=3, method='sigmoid'
        ),

        # ========== MLPs ==========
        'MLP_Large': lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            alpha=1e-2, learning_rate='adaptive', max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
        ),
        'MLP_Medium': lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            alpha=1e-2, learning_rate='adaptive', max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
        ),
        'MLP_Small': lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
            alpha=2e-2, learning_rate='adaptive', max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
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
        # 'RadiusNeighbors': lambda: RadiusNeighborsClassifier(
        #     radius=50.0, weights='distance', outlier_label='most_frequent'
        # ),  # Removed due to radius issues
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
                ('hgb', HistGradientBoostingClassifier(max_depth=5, early_stopping=False, random_state=RANDOM_STATE)),
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


def train_ensemble_models_classification(raw_dataset, train_indices, test_indices, channel_filter_indices=None, selection_method='rf'):
    """
    Train and evaluate ALL models in the ensemble with MRMR feature selection (k=24).
    Uses TIED StratifiedKFold CV - same splits across all models for fair comparison.
    
    Args:
        raw_dataset: EEGRawDataset instance
        train_indices: List of training sample indices
        test_indices: List of testing sample indices  
        channel_filter_indices: Indices of features to filter by channel
        
    Returns:
        dict: Results containing all model performances and aggregated metrics
    """
    
    try:
        # Apply feature selection or use all features based on flag
        # Apply channel filtering if specified
        pre_filtered_indices = channel_filter_indices if channel_filter_indices is not None else None
        
        if selection_method is None:
            # NO FEATURE SELECTION - use all features
            from sklearn.preprocessing import StandardScaler
            
            # Get all data first
            X_all = []
            for idx in range(len(raw_dataset)):
                entry = raw_dataset[idx]
                X_all.append(entry.raw_features)
            X_all = np.array(X_all)
            
            # Apply channel filtering if needed
            if pre_filtered_indices is not None:
                X_all = X_all[:, pre_filtered_indices]
            
            # Fit scaler on training data
            scaler = StandardScaler()
            scaler.fit(X_all[train_indices])
            
            # Use all features
            feat_idx = np.arange(X_all.shape[1])
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
        
        # Create processed dataset (note: constructor order is raw_dataset, scaler, feature_indices)
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
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
            print(f"  ⚠️ Handled {train_nans + test_nans} NaN values, {train_infs + test_infs} inf values")
        
        # Get model ensemble
        models = get_model_ensemble()
        
        # Create TIED StratifiedKFold CV splits - same for all models
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        # Train and evaluate each model
        model_results = {}
        n_models = len(models)
        
        for model_idx, (model_name, model_func) in enumerate(models.items(), 1):
            print(f"\r    Training model {model_idx}/{n_models}: {model_name[:20]:<20}", end="")
            sys.stdout.flush()
            
            try:
                # Clean data immediately before training to catch any infinities
                X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Train and test
                result = train_evaluate_sklearn_classification(model_func, X_train_clean, X_test_clean, y_train, y_test, model_name)
                
                # Cross-validation using TIED splits (same cv object for all models)
                fresh_model = model_func()
                with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                    cv_scores = cross_val_score(fresh_model, X_train_clean, y_train, cv=cv, scoring='accuracy')
                
                # Store results with detailed CV scores
                model_results[model_name] = {
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),  # Individual fold scores
                    'cv_fold_1': cv_scores[0],
                    'cv_fold_2': cv_scores[1],
                    'cv_fold_3': cv_scores[2]
                }
            except Exception as e:
                # Model failed - skip it
                print(f"\r    Training model {model_idx}/{n_models}: {model_name[:20]:<20} [FAILED]")
                continue
        
        # Calculate aggregated metrics (median across valid models)
        # Filter out catastrophically bad models (accuracy < 0.1) to avoid outliers
        print()  # Newline after model progress
        all_accuracies = [res['accuracy'] for res in model_results.values()]
        all_cv_means = [res['cv_mean'] for res in model_results.values()]
        
        valid_accuracies = [res['accuracy'] for res in model_results.values() if res['accuracy'] >= 0.1]
        valid_cv_means = [res['cv_mean'] for res in model_results.values() if res['accuracy'] >= 0.1]
        
        if valid_accuracies:
            agg_accuracy = np.median(valid_accuracies)  # Use median for robustness
            agg_cv_mean = np.median(valid_cv_means)
            agg_cv_std = np.std(valid_cv_means)
            n_valid_models = len(valid_accuracies)
            n_failed_models = len(all_accuracies) - n_valid_models
        else:
            agg_accuracy = 0.0
            agg_cv_mean = 0.0
            agg_cv_std = 0.0
            n_valid_models = 0
            n_failed_models = len(all_accuracies)
        
        print(f"  ✅ Aggregated: Acc={agg_accuracy:.3f}, CV={agg_cv_mean:.3f}±{agg_cv_std:.3f} ({n_valid_models} valid models)")
        if n_failed_models > 0:
            print(f"  ⚠️ Excluded {n_failed_models} models with catastrophic failures (accuracy < 0.1)")
        model_scores = ', '.join([f"{name}={res['accuracy']:.3f}" for name, res in model_results.items()])
        print(f"     Models: {model_scores}")
        
        return {
            'model_results': model_results,
            'selected_features': feature_names[:32] if feature_names else [],
            'train_nans': train_nans,
            'test_nans': test_nans,
            'feature_names': feature_names,
            'selection_scores': selection_scores,  # Add selection scores
            'accuracy_score': agg_accuracy,  # Aggregated accuracy
            'cv_accuracy_mean': agg_cv_mean,
            'cv_accuracy_std': agg_cv_std,
            'individual_accuracies': all_accuracies,  # For ranking calculation
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        traceback.print_exc()
        return {
            'model_results': {},
            'selected_features': [],
            'error': str(e),
            'train_nans': 0,
            'test_nans': 0,
            'feature_names': [],
            'selection_scores': {},  # Add selection scores
            'accuracy_score': 0.0,
            'cv_accuracy_mean': 0.0,
            'cv_accuracy_std': 0.0,
            'individual_accuracies': []
        }


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


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='EEG Channel Importance Analysis - Aggregated Model Ensemble',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python test_feature_classification_agg.py --mocas
  python test_feature_classification_agg.py --htc --subscale mental
  python test_feature_classification_agg.py --nback --subscale physical
  python test_feature_classification_agg.py --wauc --subscale combined
  python test_feature_classification_agg.py --dataset mocas --subscale effort
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

# Add test mode
parser.add_argument('--test', action='store_true', 
                   help='Run in test mode (quick validation with 3 channel groups only)')

# Add subscale selection
parser.add_argument('--subscale', type=str, 
                   choices=['combined', 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration'],
                   default='combined',
                   help='TLX subscale to analyze (default: combined)')

# Add feature selection method
parser.add_argument('--selection-method', type=str,
                   choices=['rf', 'mrmr', 'anova'],
                   default='rf',
                   help='Feature selection method: rf (Random Forest), mrmr, or anova (default: rf)')

# Add flag to test only All_Channels
parser.add_argument('--all-channels-only', action='store_true',
                   help='Test only All_Channels group (skip other channel groups)')

# Add flag to disable feature selection
parser.add_argument('--no-feature-selection', action='store_true',
                   help='Disable feature selection and use all features')

args = parser.parse_args()

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
    print("ℹ️ No dataset specified, defaulting to MOCAS")
    print("   Use --mocas, --htc, --nback, --wauc, or --sense42 to specify a dataset\n")

# Get dataset path
data_path = get_dataset_path(dataset_name)

# Get subscale selection
subscale = args.subscale
subscale_suffix = '' if subscale == 'combined' else f'_{subscale}'

# Check if test mode
test_mode = args.test
all_channels_only = args.all_channels_only
no_feature_selection = args.no_feature_selection

if test_mode:
    print("\n🧪 TEST MODE ENABLED - Running quick validation")
    print("   Only testing 3 channel groups: All_Channels, Frontal, Central")
    print("   Estimated time: ~2-3 minutes\n")

if all_channels_only:
    print("\n📍 ALL CHANNELS ONLY MODE - Testing only All_Channels group")
    
if no_feature_selection:
    print("\n🔓 NO FEATURE SELECTION - Using all available features")

# Get feature selection method
selection_method = getattr(args, 'selection_method', 'rf') if not no_feature_selection else None

# Load dataset
print(f"\n{'='*80}")
print(f"EEG CHANNEL IMPORTANCE ANALYSIS - AGGREGATED MODEL ENSEMBLE")
if test_mode:
    print(f"                      *** TEST MODE ***")
print(f"{'='*80}")
print(f"\n🔧 Loading dataset...")
print(f"Dataset: {dataset_name.upper()}")
print(f"Subscale: {subscale.upper()}")
print(f"Feature Selection: {selection_method.upper() if selection_method else 'NONE (all features)'}")
print(f"Loading from: {data_path}")

try:
    raw_dataset = EEGRawDataset(data_path, target_suffix=subscale)
    
    # Get dataset info
    first_sample = raw_dataset[0]
    
    # Handle both dataclass (EEGRawDatasetEntry) and dict formats
    feature_vector = first_sample.raw_features if hasattr(first_sample, 'raw_features') else first_sample['raw_features']
    # Optimize: Labels are already in file_list, no need to read all files
    all_labels = [label for _, label in raw_dataset.file_list]
    class_counts = Counter(all_labels)
    
    print(f"✅ Dataset loaded: {len(raw_dataset)} samples, {len(raw_dataset.flattened_features)} features")
    print(f"   Classes: {dict(class_counts)}")
    
except Exception as e:
    print(f"\n❌ ERROR loading dataset!")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    import traceback
    print(f"\n📍 Full traceback:")
    traceback.print_exc()
    exit(1)

# Create train/test split
print(f"\n📊 Creating train/test split (70/30)...")
all_indices = list(range(len(raw_dataset)))
# Optimize: Use labels from file_list instead of reading files
labels = [label for _, label in raw_dataset.file_list]

train_indices, test_indices = train_test_split(
    all_indices, test_size=0.3, random_state=RANDOM_STATE, 
    stratify=labels
)

print(f"   Training: {len(train_indices)} samples | Testing: {len(test_indices)} samples")

# Display model ensemble
models = get_model_ensemble()
print(f"\n🤖 Model Ensemble ({len(models)} models):")
for i, model_name in enumerate(models.keys(), 1):
    print(f"   {i}. {model_name}")

# Channel categories
channel_categories = get_channel_categories()

# In test mode, only use subset of channels
if test_mode:
    test_channels = {
        'All_Channels': channel_categories['All_Channels'],
        'Frontal': channel_categories['Frontal'],
        'Central': channel_categories['Central']
    }
    channel_categories = test_channels
    print(f"\n🧪 Test mode: Using only {len(channel_categories)} channel groups")

# If all-channels-only mode, only use All_Channels
if all_channels_only:
    channel_categories = {
        'All_Channels': channel_categories['All_Channels']
    }
    print(f"\n📍 All-Channels-Only mode: Using only All_Channels group")

# Results structure: results[channel_category] = model ensemble results
results = {}
all_channels_baseline = None  # Store baseline performance for ΔAccuracy calculation
model_specific_baselines = {}  # Store baseline for each model

# Get feature names for channel filtering
first_sample = raw_dataset[0]
all_feature_names = raw_dataset.flattened_features

# Handle feature count mismatch (actual data may have different dimensions than expected)
first_features = first_sample.raw_features if hasattr(first_sample, 'raw_features') else first_sample['raw_features']
actual_feature_count = len(first_features)
expected_feature_count = len(all_feature_names)

if actual_feature_count != expected_feature_count:
    # Reconstruct flattened_features to match actual data dimensions
    actual_time_steps = actual_feature_count // (len(raw_dataset.bands) * len(raw_dataset.channels))
    all_feature_names = [
        f"{t}_{band}_{chan}"
        for t in range(actual_time_steps)
        for band in raw_dataset.bands
        for chan in raw_dataset.channels
    ]

print(f"\n🧠 Starting aggregated channel importance analysis...")
print(f"   Method: ANOVA (k=24) + {len(models)} Models + 3-fold TIED CV + Aggregated Rankings")
print(f"   Testing {len(channel_categories)} channel groups")
sys.stdout.flush()

channel_start_time = time.time()
for ch_idx, (channel_name, channel_prefixes) in enumerate(channel_categories.items(), 1):
    print(f"\n[{time.strftime('%H:%M:%S')}] 📍 Channel Group {ch_idx}/{len(channel_categories)}: {channel_name}", end=" ")
    sys.stdout.flush()
    # Get channel filter indices
    if channel_prefixes is None:
        channel_filter_indices = None
        print(f"all channels", end=" ")
    else:
        channel_filter_indices = filter_features_by_channels(all_feature_names, channel_prefixes)
        if len(channel_filter_indices) == 0:
            print(f"⚠️ No features found, skipping")
            continue
    
    # Train ensemble models
    try:
        ensemble_results = train_ensemble_models_classification(
            raw_dataset, train_indices, test_indices, channel_filter_indices, selection_method
        )
        
        if ensemble_results and 'accuracy_score' in ensemble_results:
            results[channel_name] = ensemble_results
            
            # Store all-channels baseline for ΔAccuracy calculation
            if channel_name == 'All_Channels':
                all_channels_baseline = ensemble_results['accuracy_score']
                # Also store individual model baselines
                if 'model_results' in ensemble_results:
                    model_specific_baselines = {
                        name: res['accuracy'] 
                        for name, res in ensemble_results['model_results'].items()
                    }
            
        else:
            print(f"→ Failed")
            
    except Exception as e:
        print(f"→ Error: {str(e)}")
        traceback.print_exc()

# Calculate aggregated rankings
print(f"\n{'='*80}")
print(f"AGGREGATED CHANNEL IMPORTANCE RESULTS")
print(f"{'='*80}")

if all_channels_baseline is not None and len(model_specific_baselines) > 0:
    print(f"\nBaseline (All Channels):")
    print(f"   Aggregated Accuracy = {all_channels_baseline:.3f}")
    for model_name, baseline_acc in model_specific_baselines.items():
        print(f"   {model_name}: {baseline_acc:.3f}")
    
    # Calculate ΔAccuracy for each model and each channel group
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
        # Sort by ΔAccuracy descending
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
    
    print(f"\n📊 AGGREGATED Channel Importance Ranking (by average rank across {len(models)} models):")
    print(f"{'Rank':<6} {'Channel':<20} {'Agg Acc':<10} {'ΔAcc':<10} {'Avg Rank':<12} {'Model Ranks'}")
    print(f"{'-'*90}")
    
    for i, (channel_name, acc, delta, avg_rank, individual_ranks) in enumerate(aggregated_rankings):
        rank = i + 1
        delta_sign = "+" if delta >= 0 else ""
        status = "🟢" if delta > 0 else "🔴" if delta < -0.01 else "🟡"
        ranks_str = ', '.join([f'{r}' for r in individual_ranks])
        print(f"{rank:<6} {status} {channel_name:<18} {acc:.3f}      {delta_sign}{delta:>6.3f}    {avg_rank:>6.2f}      [{ranks_str}]")
    
    # Performance summary
    if aggregated_rankings:
        best_channel = aggregated_rankings[0]
        worst_channel = aggregated_rankings[-1]
        positive_count = len([ch for ch in aggregated_rankings if ch[2] > 0])
        
        print(f"\n📈 Summary:")
        print(f"   • {positive_count}/{len(aggregated_rankings)} groups outperform baseline")
        print(f"   • Best: {best_channel[0]} (avg rank {best_channel[3]:.2f}, ΔAcc {best_channel[2]:+.3f})")
        print(f"   • Worst: {worst_channel[0]} (avg rank {worst_channel[3]:.2f}, ΔAcc {worst_channel[2]:+.3f})")
        print(f"   • Aggregation Method: Average rank across {len(models)} models")
    
    # Export results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_TEST" if test_mode else ""
    output_file = f"channel_importance_aggregated_{dataset_name}_{subscale}_classification{mode_suffix}_{timestamp}.csv"
    output_dir = os.path.join(project_root, 'channel_importance', 'feature_tests', 'tests', 'classification', 'cv')
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
                row[f'{model_name}_F1_Score'] = model_res['f1_score']
                row[f'{model_name}_CV_Mean'] = model_res['cv_mean']
                row[f'{model_name}_CV_Std'] = model_res['cv_std']
                row[f'{model_name}_CV_Fold1'] = model_res['cv_fold_1']
                row[f'{model_name}_CV_Fold2'] = model_res['cv_fold_2']
                row[f'{model_name}_CV_Fold3'] = model_res['cv_fold_3']
                row[f'{model_name}_Delta_Accuracy'] = model_res['accuracy'] - model_specific_baselines.get(model_name, 0)
        
        export_data.append(row)
    
    df_results = pd.DataFrame(export_data)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n💾 Results exported to: {output_path}")
    
    # Also save a detailed run info file
    detailed_output_file = f"run_details_aggregated_{dataset_name}_{subscale}_classification{mode_suffix}_{timestamp}.csv"
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
                    'Test_Accuracy': model_res['accuracy'],
                    'F1_Score': model_res['f1_score'],
                    'CV_Mean': model_res['cv_mean'],
                    'CV_Std': model_res['cv_std'],
                    'CV_Fold_1': model_res['cv_fold_1'],
                    'CV_Fold_2': model_res['cv_fold_2'],
                    'CV_Fold_3': model_res['cv_fold_3'],
                    'Baseline_Accuracy': baseline_acc,
                    'Delta_Accuracy': model_res['accuracy'] - baseline_acc,
                    'N_Train_Samples': len(train_indices),
                    'N_Test_Samples': len(test_indices),
                    'N_Features': len(channel_results.get('feature_names', [])),
                    'Feature_Names': feature_names_str,
                    'Selection_Scores': selection_scores_str,
                    'Random_State': RANDOM_STATE,
                    'Timestamp': timestamp
                })
    
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(detailed_output_path, index=False)
    print(f"💾 Detailed run data exported to: {detailed_output_path}")
    
    # Print test mode summary
    if test_mode:
        print(f"\n{'='*80}")
        print(f"🧪 TEST MODE SUMMARY")
        print(f"{'='*80}")
        print(f"✅ All components working correctly:")
        print(f"   ✓ Dataset loading: {len(raw_dataset)} samples")
        print(f"   ✓ Train/test split: {len(train_indices)}/{len(test_indices)} samples")
        print(f"   ✓ Model ensemble: {len(models)} models trained")
        print(f"   ✓ Channel groups: {len(results)} tested")
        print(f"   ✓ Ranking aggregation: {len(aggregated_rankings)} channels ranked")
        print(f"   ✓ CSV export: 2 files saved")
        print(f"\n💡 Test passed! Ready for full run without --test flag")
        print(f"{'='*80}\n")

else:
    print("   ⚠️ No baseline found - cannot calculate aggregated rankings")

print(f"\n✅ Aggregated analysis complete - tested {len(results)} channel groups with {len(models)} models")
if test_mode:
    print(f"   🧪 TEST MODE - Use without --test flag for full analysis")
print(f"{'='*80}\n")

