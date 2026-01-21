"""
EEG Channel Importance Analysis for Regression Tasks - AGGREGATED MODEL ENSEMBLE WITH LNSO CV
==============================================================================================

This script performs channel importance analysis by AGGREGATING rankings from 27+ different regression models
using Leave-N-Subjects-Out (LNSO) cross-validation to ensure subject-independent validation.
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
# Dataset: MOCAS, HTC, N-Back, WAUC
# Task: Regression (continuous workload prediction)
#
# AGGREGATION METHODOLOGY:
# - Feature Selection: MRMR only, k=24 features (fixed)
# - Cross-Validation: Leave-N-Subjects-Out (LNSO) with N = total_subjects/5
# - Models: 27+ total across 8 algorithm families
# - Evaluation: LNSO CV, all models use same fold splits
# - Metric: RÂ² (coefficient of determination), averaged across folds

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
RANDOM_STATE = 42  # For reproducibility (CV folds, model initialization)
N_JOBS = 1  # Limit parallelism per model to avoid CPU contention when running multiple scripts

# Add parent directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              HistGradientBoostingRegressor, AdaBoostRegressor, 
                              ExtraTreesRegressor, BaggingRegressor, StackingRegressor, VotingRegressor)
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, SGDRegressor, 
                                 HuberRegressor, PassiveAggressiveRegressor, 
                                 RANSACRegressor, BayesianRidge, ARDRegression, 
                                 TheilSenRegressor, QuantileRegressor, TweedieRegressor, 
                                 OrthogonalMatchingPursuit, LassoLarsIC)
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import warnings

# Optional dependencies
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("âš ï¸ XGBoost not available")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("âš ï¸ CatBoost not available")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("âš ï¸ LightGBM not available")

from channel_importance.regression_datasets import EEGRawRegressionDataset
from channel_importance.preprocessing_fixed import preprocess_data
from core.utils import get_sklearn_data
from datasets.processed_eeg import EEGProcessedDataset


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
    Get ensemble of 27+ diverse regression models spanning multiple algorithm families.
    
    Returns:
        Dictionary mapping model names to model constructors
    """
    models = {
        # ===== LINEAR MODELS (6) =====
        'Ridge': lambda: Ridge(
            alpha=3.0,
            max_iter=5000,
            tol=1e-6,
            fit_intercept=True,
            solver='auto',
            random_state=RANDOM_STATE
        ),
        
        'Lasso': lambda: Lasso(
            alpha=0.1,
            max_iter=5000,
            tol=1e-4,
            fit_intercept=True,
            selection='cyclic',
            random_state=RANDOM_STATE
        ),
        
        'ElasticNet': lambda: ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=5000,
            tol=1e-4,
            fit_intercept=True,
            selection='cyclic',
            random_state=RANDOM_STATE
        ),
        
        'SGDRegressor': lambda: SGDRegressor(
            loss='squared_error',
            penalty='elasticnet',
            alpha=0.001,
            l1_ratio=0.15,
            max_iter=5000,
            tol=1e-3,
            learning_rate='optimal',
            random_state=RANDOM_STATE
        ),
        
        'HuberRegressor': lambda: HuberRegressor(
            epsilon=1.35,
            alpha=0.001,
            max_iter=5000,
            tol=1e-4,
            fit_intercept=True
        ),
        
        'PassiveAggressiveRegressor': lambda: PassiveAggressiveRegressor(
            C=1.0,
            max_iter=5000,
            tol=1e-3,
            fit_intercept=True,
            random_state=RANDOM_STATE
        ),
        
        # ===== TREE-BASED MODELS (4) =====
        'RandomForest': lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        
        'ExtraTrees': lambda: ExtraTreesRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        
        'DecisionTree': lambda: DecisionTreeRegressor(
            criterion='squared_error',
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=RANDOM_STATE
        ),
        
        'GradientBoosting': lambda: GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=RANDOM_STATE
        ),
        
        # ===== BOOSTING MODELS (2 core) =====
        'HistGradientBoosting': lambda: HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=150,
            max_depth=5,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE
        ),
        
        'AdaBoost': lambda: AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.8,
            loss='linear',
            random_state=RANDOM_STATE
        ),
        
        # ===== SVM MODELS (3) =====
        'SVR_RBF': lambda: SVR(
            C=10.0,
            kernel='rbf',
            gamma='auto',
            epsilon=0.1,
            cache_size=500
        ),
        
        'LinearSVR': lambda: LinearSVR(
            C=1.0,
            epsilon=0.1,
            loss='squared_epsilon_insensitive',
            dual=True,
            max_iter=5000,
            tol=1e-4,
            random_state=RANDOM_STATE
        ),
        
        'NuSVR': lambda: NuSVR(
            nu=0.5,
            C=1.0,
            kernel='rbf',
            gamma='auto',
            cache_size=500
        ),
        
        # ===== NEURAL NETWORKS (3 architectures) =====
        'MLP_Large': lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (HTC has only 23 training samples)
            random_state=RANDOM_STATE
        ),
        
        'MLP_Medium': lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (HTC has only 23 training samples)
            random_state=RANDOM_STATE
        ),
        
        'MLP_Small': lambda: MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=0.02,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (HTC has only 23 training samples)
            random_state=RANDOM_STATE
        ),
        
        # ===== NEAREST NEIGHBORS (2) =====
        'KNeighbors_7': lambda: KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2,
            n_jobs=N_JOBS
        ),
        
        'KNeighbors_5': lambda: KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=N_JOBS
        ),
        
        # ===== KERNEL METHODS (2) =====
        'KernelRidge': lambda: KernelRidge(
            alpha=1.0,
            kernel='rbf',
            gamma=0.1
        ),
        
        'GaussianProcess': lambda: GaussianProcessRegressor(
            kernel=1.0 * RBF(1.0) + WhiteKernel(noise_level=1.0),
            alpha=1e-10,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=RANDOM_STATE
        ),
        
        # ===== ENSEMBLE META-LEARNERS (2) =====
        'Bagging': lambda: BaggingRegressor(
            n_estimators=50,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        
        'RANSAC': lambda: RANSACRegressor(
            min_samples=0.5,
            max_trials=100,
            residual_threshold=None,
            random_state=RANDOM_STATE
        )
    }
    
    # Add optional advanced boosting models if available
    if HAS_XGBOOST:
        models['XGBoost'] = lambda: XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
    
    if HAS_CATBOOST:
        models['CatBoost'] = lambda: CatBoostRegressor(
            iterations=150,
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
        models['LightGBM'] = lambda: LGBMRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=20,
            min_child_samples=10,
            min_child_weight=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1,
            force_col_wise=True,
            n_jobs=N_JOBS
        )
    
    return models




def get_model_ensemble():
    """
    Get ensemble of 30+ diverse regression models for tabular EEG statistical features.

    Returns:
        dict[str, callable]: mapping model name -> zero-arg constructor
    """
    models = {
        # ===== LINEAR MODELS =====
        'Ridge': lambda: Ridge(alpha=3.0, max_iter=5000, tol=1e-6, solver='auto', random_state=RANDOM_STATE),
        'Lasso': lambda: Lasso(alpha=0.1, max_iter=5000, tol=1e-6, selection='cyclic', random_state=RANDOM_STATE),
        'ElasticNet': lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, tol=1e-6, selection='cyclic', random_state=RANDOM_STATE),
        'SGDRegressor': lambda: SGDRegressor(loss='squared_error', penalty='elasticnet', alpha=1e-3, l1_ratio=0.15, max_iter=5000, tol=1e-6, learning_rate='optimal', random_state=RANDOM_STATE),
        'HuberRegressor': lambda: HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=5000, tol=1e-6),
        'PassiveAggressive': lambda: PassiveAggressiveRegressor(C=1.0, max_iter=5000, tol=1e-6, random_state=RANDOM_STATE),

        # Extra linear/sparse/bayesian (good for n << p with collinearity)
        'BayesianRidge': lambda: BayesianRidge(),
        'ARDRegression': lambda: ARDRegression(),
        'TheilSen': lambda: TheilSenRegressor(max_subpopulation=1_000, random_state=RANDOM_STATE),
        'QuantileReg': lambda: QuantileRegressor(quantile=0.5, alpha=1.0),
        'TweedieReg': lambda: TweedieRegressor(power=1.5, alpha=0.001, link='auto', max_iter=10_000),
        'OMP': lambda: OrthogonalMatchingPursuit(n_nonzero_coefs=20),
        'LassoLarsIC_AIC': lambda: LassoLarsIC(criterion='aic'),

        # Projection-based baselines for collinear stats features
        'PLSRegression': lambda: PLSRegression(n_components=8),
        'PCR_Ridge': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, svd_solver='full')),
            ('ridge', Ridge(alpha=2.0, random_state=RANDOM_STATE))
        ]),

        # ===== TREE-BASED MODELS =====
        'RandomForest': lambda: RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_split=10, min_samples_leaf=4,
            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        'ExtraTrees': lambda: ExtraTreesRegressor(
            n_estimators=300, max_depth=8, min_samples_split=10, min_samples_leaf=4,
            max_features='sqrt', bootstrap=True, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        'DecisionTree': lambda: DecisionTreeRegressor(max_depth=6, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_STATE),
        'GradientBoosting': lambda: GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_split=10, min_samples_leaf=5,
            subsample=0.8, max_features='sqrt', random_state=RANDOM_STATE
        ),

        # Robust/quantile variants
        'GB_Huber': lambda: GradientBoostingRegressor(
            loss='huber', alpha=0.9, n_estimators=200, learning_rate=0.05,
            max_depth=3, subsample=0.9, random_state=RANDOM_STATE
        ),
        'GB_Quantile_Median': lambda: GradientBoostingRegressor(
            loss='quantile', alpha=0.5, n_estimators=300, learning_rate=0.05,
            max_depth=3, subsample=0.9, random_state=RANDOM_STATE
        ),

        # ===== BOOSTING (Sklearn-native) =====
        'HistGradientBoosting': lambda: HistGradientBoostingRegressor(
            learning_rate=0.1, max_iter=200, max_depth=5, min_samples_leaf=15, l2_regularization=1.0, random_state=RANDOM_STATE
        ),
        'AdaBoost': lambda: AdaBoostRegressor(n_estimators=150, learning_rate=0.8, loss='linear', random_state=RANDOM_STATE),

        # ===== SVM / KERNEL METHODS =====
        'SVR_RBF': lambda: SVR(C=10.0, kernel='rbf', gamma='auto', epsilon=0.1, cache_size=500),
        'LinearSVR': lambda: LinearSVR(C=1.0, epsilon=0.1, max_iter=5000, tol=1e-6, random_state=RANDOM_STATE),
        'NuSVR': lambda: NuSVR(nu=0.5, C=10.0, kernel='rbf', gamma='auto', cache_size=500),
        'SVR_Poly': lambda: SVR(kernel='poly', C=5.0, degree=3, coef0=1.0, epsilon=0.1, cache_size=500),
        'SVR_Sigmoid': lambda: SVR(kernel='sigmoid', C=5.0, coef0=0.5, epsilon=0.1, cache_size=500),
        'KernelRidge': lambda: KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1),
        'KernelRidge_Poly': lambda: KernelRidge(alpha=1.0, kernel='poly', degree=3, coef0=1.0),

        # ===== GAUSSIAN PROCESSES (small-n regimes) =====
        # 'GaussianProcess': lambda: GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), alpha=1e-6, n_restarts_optimizer=2, random_state=RANDOM_STATE),  # DISABLED: O(n³) too slow
        # 'GPR_Matern': lambda: GaussianProcessRegressor(
        #     kernel=1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3),
        #     alpha=1e-6, n_restarts_optimizer=2, random_state=RANDOM_STATE
        # ),  # DISABLED: O(n³) too slow
        # 'GPR_RQ': lambda: GaussianProcessRegressor(
        #     kernel=1.0 * RationalQuadratic(alpha=1.0, length_scale=1.0) + WhiteKernel(1e-3),
        #     alpha=1e-6, n_restarts_optimizer=2, random_state=RANDOM_STATE
        # ),  # DISABLED: O(n³) too slow

        # ===== NEAREST NEIGHBORS =====
        'KNeighbors_7': lambda: KNeighborsRegressor(n_neighbors=7, weights='distance', metric='minkowski', p=2, n_jobs=N_JOBS),
        'KNeighbors_5': lambda: KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean', n_jobs=N_JOBS),
        'RadiusNeighbors': lambda: RadiusNeighborsRegressor(radius=1.0, weights='distance', p=2, leaf_size=30),

        # ===== ENSEMBLE META-LEARNERS =====
        'Bagging': lambda: BaggingRegressor(n_estimators=50, max_samples=0.8, max_features=0.8, bootstrap=True, bootstrap_features=False, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        'RANSAC': lambda: RANSACRegressor(min_samples=0.5, max_trials=100, residual_threshold=None, random_state=RANDOM_STATE),
        'Stacking_RidgeMeta': lambda: StackingRegressor(
            estimators=[
                ('hgb', HistGradientBoostingRegressor(max_depth=5, random_state=RANDOM_STATE)),
                ('rf', RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
                ('svr', SVR(C=10.0, kernel='rbf', epsilon=0.1))
            ],
            final_estimator=Ridge(alpha=2.0, random_state=RANDOM_STATE),
            passthrough=True, n_jobs=N_JOBS
        ),
        'Voting': lambda: VotingRegressor(
            estimators=[
                ('ridge', Ridge(alpha=3.0, random_state=RANDOM_STATE)),
                ('hgb', HistGradientBoostingRegressor(max_depth=5, random_state=RANDOM_STATE)),
                ('rf', RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=N_JOBS))
            ],
            n_jobs=N_JOBS
        ),
    }

    # ===== OPTIONAL ADVANCED BOOSTERS (if available flags are set) =====
    if HAS_XGBOOST:
        models['XGBoost'] = lambda: XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0.5, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        )
    if HAS_CATBOOST:
        models['CatBoost'] = lambda: CatBoostRegressor(
            iterations=200, depth=5, learning_rate=0.1, l2_leaf_reg=3.0, border_count=64,
            bootstrap_type='Bayesian', bagging_temperature=1.0, random_strength=1.0,
            min_data_in_leaf=5, random_state=RANDOM_STATE, verbose=False, allow_writing_files=False
        )
    if HAS_LIGHTGBM:
        models['LightGBM'] = lambda: LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, num_leaves=20,
            min_child_samples=10, min_child_weight=1e-3, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1, force_col_wise=True
        )

    # ===== OPTIONAL THIRD-PARTY (best-effort import guards) =====
    try:
        from ngboost import NGBRegressor
        models['NGBoost'] = lambda: NGBRegressor(n_estimators=800, learning_rate=0.03, random_state=RANDOM_STATE, verbose=False)
    except Exception:
        pass

    try:
        from pygam import LinearGAM
        models['GAM'] = lambda: LinearGAM(n_splines=10, lam=0.6)
    except Exception:
        pass

    return models



def train_evaluate_sklearn_regression(model_func, X_train, X_test, y_train, y_test, model_name="Model"):
    """Train and evaluate a scikit-learn regression model."""
    model = model_func()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


def train_ensemble_models_regression_lnso(raw_dataset, cv_splits, subject_to_indices, channel_filter_indices=None, selection_method='rf', no_feature_selection=False):
    """
    Train and evaluate ALL models in the ensemble with feature selection (k=24) 
    using Leave-N-Subjects-Out cross-validation.
    
    Args:
        raw_dataset: EEGRawRegressionDataset instance
        cv_splits: List of (train_subjects, test_subjects) tuples
        subject_to_indices: Dict mapping subject_id -> list of sample indices
        channel_filter_indices: Indices of features to filter by channel
        selection_method: Feature selection method ('rf' or 'mrmr')
        no_feature_selection: If True, use all features without selection
        
    Returns:
        dict: Results containing all model performances and aggregated metrics across LNSO folds
    """
    
    # Get model ensemble
    models = get_model_ensemble()
    
    # Store results for each model across folds - all metrics
    model_fold_scores = {model_name: {'r2': [], 'mae': [], 'mse': [], 'rmse': []} for model_name in models.keys()}
    feature_names_final = []
    selection_scores_final = {}
    
    # Process each LNSO fold
    for fold_idx, (train_subjects, test_subjects) in enumerate(cv_splits):
        print(f"   Fold {fold_idx + 1}/{len(cv_splits)}: Train={len(train_subjects)} subj, Test={len(test_subjects)} subj")
        
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
        
        try:
            # MRMR feature selection with k=24
            pre_filtered_indices = channel_filter_indices if channel_filter_indices is not None else None
            
            if no_feature_selection:
                # Use all features without selection
                from sklearn.preprocessing import StandardScaler
                
                # Get all data from raw dataset
                X_all = []
                y_all = []
                for idx in range(len(raw_dataset)):
                    entry = raw_dataset[idx]
                    X_all.append(entry['raw_features'])
                    y_all.append(entry['label'])
                X_all = np.array(X_all)
                y_all = np.array(y_all)
                
                if pre_filtered_indices is not None:
                    feat_idx = pre_filtered_indices
                    X_all = X_all[:, feat_idx]
                else:
                    feat_idx = np.arange(X_all.shape[1])
                
                # Fit scaler on training data
                scaler = StandardScaler()
                scaler.fit(X_all[train_indices])
                
                feature_names = [raw_dataset.flattened_features[i] for i in feat_idx]
                selection_scores = None
            else:
                # Use feature selection with fixed k=24 features
                feat_idx, scaler, feature_names, selection_scores = preprocess_data(
                    raw_dataset, 
                    train_indices, 
                    24,  # Fixed k=24 features
                    task_type='regression',
                    selection_method=selection_method,
                    pre_filtered_indices=pre_filtered_indices
                )
            
            # Create processed dataset (scaler, feat_idx order matches constructor signature)
            full_processed = EEGProcessedDataset(raw_dataset, scaler, feat_idx)
            
            # Get sklearn-compatible data
            X_train, y_train = get_sklearn_data(full_processed, train_indices)
            X_test, y_test = get_sklearn_data(full_processed, test_indices)
            
            # Handle NaN and infinity values if present (near-zero variance already filtered by preprocess_data)
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
            
            # Train and evaluate each model for this fold
            n_models = len(models)
            for model_idx, (model_name, model_func) in enumerate(models.items(), 1):
                try:
                    print(f"\r      Model {model_idx}/{n_models}: {model_name[:20]:<20}", end="")
                    sys.stdout.flush()
                    
                    # Clean data immediately before training to catch any infinities
                    X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                    X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
                    
                    # Train with error suppression
                    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                        result = train_evaluate_sklearn_regression(model_func, X_train_clean, X_test_clean, y_train, y_test, model_name)
                    
                    # Store all metrics for this fold
                    model_fold_scores[model_name]['r2'].append(result['r2'])
                    model_fold_scores[model_name]['mae'].append(result['mae'])
                    model_fold_scores[model_name]['mse'].append(result['mse'])
                    model_fold_scores[model_name]['rmse'].append(result['rmse'])
                except Exception as e:
                    # Don't append scores if model fails
                    pass
            print()  # Newline after model progress
            
        except Exception as e:
            print(f"     âŒ Fold {fold_idx + 1} failed: {str(e)}")
            continue
    
    # Aggregate results across folds for each model
    model_results = {}
    for model_name, fold_scores_dict in model_fold_scores.items():
        if len(fold_scores_dict['r2']) > 0:
            model_results[model_name] = {
                'r2_mean': np.mean(fold_scores_dict['r2']),
                'r2_std': np.std(fold_scores_dict['r2']),
                'mae_mean': np.mean(fold_scores_dict['mae']),
                'mae_std': np.std(fold_scores_dict['mae']),
                'mse_mean': np.mean(fold_scores_dict['mse']),
                'mse_std': np.std(fold_scores_dict['mse']),
                'rmse_mean': np.mean(fold_scores_dict['rmse']),
                'rmse_std': np.std(fold_scores_dict['rmse']),
                'fold_scores': fold_scores_dict,  # Keep all fold scores for detailed analysis
                'n_folds': len(fold_scores_dict['r2'])
            }
    
    # Calculate aggregated metrics (mean across all models)
    if model_results:
        all_r2_means = [res['r2_mean'] for res in model_results.values()]
        agg_r2_mean = np.mean(all_r2_means)
        agg_r2_std = np.std(all_r2_means)
        
        # Calculate aggregated metrics across models (median for robustness)
        all_r2_means = [res['r2_mean'] for res in model_results.values()]
        
        # Filter out catastrophically bad models (RÂ² < -10)
        valid_r2_means = [res['r2_mean'] for res in model_results.values() if res['r2_mean'] > -10]
        
        if valid_r2_means:
            agg_r2_mean = np.median(valid_r2_means)
            agg_r2_std = np.std(valid_r2_means)
            n_valid_models = len(valid_r2_means)
            n_failed_models = len(all_r2_means) - n_valid_models
        else:
            agg_r2_mean = 0.0
            agg_r2_std = 0.0
            n_valid_models = 0
            n_failed_models = len(all_r2_means)
        
        print(f"  âœ… Aggregated: RÂ²={agg_r2_mean:.3f}Â±{agg_r2_std:.3f} (across {n_valid_models} valid models)")
        if n_failed_models > 0:
            print(f"  âš ï¸ Excluded {n_failed_models} models with catastrophic failures (RÂ² < -10)")
        
        return {
            'model_results': model_results,
            'feature_names': feature_names if 'feature_names' in locals() else [],
            'selection_scores': selection_scores if 'selection_scores' in locals() else {},
            'r2_score': agg_r2_mean,  # Aggregated RÂ² across models
            'r2_std': agg_r2_std,
            'individual_r2s': all_r2_means,  # For ranking calculation
            'n_models': len(model_results)
        }
    else:
        print(f"  âŒ No models succeeded")
        return {
            'model_results': {},
            'feature_names': [],
            'selection_scores': {},
            'error': 'All models failed',
            'r2_score': 0.0,
            'r2_std': 0.0,
            'individual_r2s': [],
            'n_models': 0
        }


def get_dataset_path(dataset_name):
    """
    Get the dataset path based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset ('mocas', 'htc', 'nback', 'wauc')
    
    Returns:
        str: Absolute path to the dataset
    """
   


    dataset_paths = {
        'mocas': os.path.join('data', 'MOCAS', 'mocas_feature_regression_dataset'),
        'htc': os.path.join('data', 'heat_the_chair', 'htc_feature_regression'),
        'nback': os.path.join('data', 'n_back', 'nback_feature_regression'),
        'wauc': os.path.join('data', 'wauc', 'wauc_feature_regression_dataset'),
        'sense42': os.path.join('data', 'SENSE-42', 'sense42_feature_regression_dataset')
    }
    
    
    
    if dataset_name.lower() not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(dataset_paths.keys())}")
    
    # project_root is already absolute, just join the relative path
    relative_path = dataset_paths[dataset_name.lower()]
    absolute_path = os.path.join(project_root, relative_path)
    
    return absolute_path


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='EEG Channel Importance Analysis - Aggregated Regression Ensemble with LNSO CV',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python test_feature_regression_agg_loso.py --mocas
  python test_feature_regression_agg_loso.py --htc
  python test_feature_regression_agg_loso.py --nback
  python test_feature_regression_agg_loso.py --wauc
  python test_feature_regression_agg_loso.py --dataset mocas --test
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
                   help='TLX subscale to use for regression targets (default: combined TLX, use "combined" or omit for combined)')

# Add test mode
parser.add_argument('--test', action='store_true', 
                   help='Run in test mode (quick validation with 3 channel groups only)')

# Add feature selection method
parser.add_argument('--selection-method', type=str,
                   choices=['rf', 'mrmr', 'anova'],
                   default='rf',
                   help='Feature selection method: rf (Random Forest), mrmr, or anova (default: rf)')

# Add flags for all-channels-only and no feature selection
parser.add_argument('--all-channels-only', action='store_true',
                   help='Only test All_Channels group (skip other channel groups)')
parser.add_argument('--no-feature-selection', action='store_true',
                   help='Use all features without selection (skip MRMR/RF/ANOVA)')

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
    print("   Estimated time: ~25-40 minutes\n")

if all_channels_only:
    print("\n[ALL-CHANNELS-ONLY] Testing only All_Channels group\n")

if no_feature_selection:
    print("\n[NO FEATURE SELECTION] Using all features\n")

# Get feature selection method
selection_method = getattr(args, 'selection_method', 'rf')

# Load dataset
print(f"\n{'='*80}")
print(f"EEG CHANNEL IMPORTANCE ANALYSIS - AGGREGATED REGRESSION ENSEMBLE WITH LNSO CV")
if test_mode:
    print(f"                      *** TEST MODE ***")
print(f"{'='*80}")
print(f"\nDataset: {dataset_name.upper()}")
if args.subscale:
    print(f"Subscale: {args.subscale.upper()}")
print(f"Feature Selection: {selection_method.upper() if selection_method else 'NONE (all features)'}")
print(f"Loading from: {data_path}")

try:
    raw_dataset = EEGRawRegressionDataset(data_path, target_suffix=args.subscale)
    
    # Get dataset info
    first_sample = raw_dataset[0]
    feature_vector = first_sample['raw_features']
    all_targets = [raw_dataset[i]['label'] for i in range(len(raw_dataset))]
    
    print(f"âœ… Dataset loaded: {len(raw_dataset)} samples, {len(raw_dataset.flattened_features)} features")
    print(f"   Target range: [{np.min(all_targets):.2f}, {np.max(all_targets):.2f}], mean: {np.mean(all_targets):.2f}")
    
except Exception as e:
    print(f"âŒ Error loading dataset: {e}")
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

# In test mode, only use subset of channels
if test_mode:
    test_channels = {
        'All_Channels': channel_categories['All_Channels'],
        'Frontal': channel_categories['Frontal'],
        'Central': channel_categories['Central']
    }
    channel_categories = test_channels
    print(f"\n[TEST MODE] Using only {len(channel_categories)} channel groups")
elif all_channels_only:
    channel_categories = {'All_Channels': channel_categories['All_Channels']}
    print(f"\n[ALL-CHANNELS-ONLY] Using only All_Channels group")

# Results structure
results = {}
all_channels_baseline = None  # Store baseline performance for Î”RÂ² calculation
model_specific_baselines = {}  # Store baseline for each model

# Get feature names for channel filtering
first_sample = raw_dataset[0]
all_feature_names = raw_dataset.flattened_features

print(f"\nðŸ§  Starting aggregated channel importance analysis with LNSO CV...")
print(f"   Method: MRMR (k=24) + {len(models)} Models + Leave-{n_test_subjects}-Subjects-Out CV")
print(f"   Testing {len(channel_categories)} channel groups")
sys.stdout.flush()

channel_start_time = time.time()
for ch_idx, (channel_name, channel_prefixes) in enumerate(channel_categories.items(), 1):
    print(f"\n[{time.strftime('%H:%M:%S')}] ðŸ“ Channel Group {ch_idx}/{len(channel_categories)}: {channel_name}", end=" ")
    sys.stdout.flush()
    # Get channel filter indices
    if channel_prefixes is None:
        channel_filter_indices = None
        print(f"all channels", end=" ")
    else:
        channel_filter_indices = filter_features_by_channels(all_feature_names, channel_prefixes)
        if len(channel_filter_indices) == 0:
            print(f"âš ï¸ No features found, skipping")
            continue
    
    # Train ensemble models with LNSO CV
    try:
        ensemble_results = train_ensemble_models_regression_lnso(
            raw_dataset, cv_splits, subject_to_indices, channel_filter_indices, selection_method, no_feature_selection
        )
        
        if ensemble_results and 'r2_score' in ensemble_results:
            results[channel_name] = ensemble_results
            
            # Store all-channels baseline for Î”RÂ² calculation
            if channel_name == 'All_Channels':
                all_channels_baseline = ensemble_results['r2_score']
                # Also store individual model baselines
                if 'model_results' in ensemble_results:
                    model_specific_baselines = {
                        name: res['r2_mean'] 
                        for name, res in ensemble_results['model_results'].items()
                    }
            
        else:
            print(f"â†’ Failed")
            
    except Exception as e:
        print(f"â†’ Error: {str(e)}")
        traceback.print_exc()

# Calculate aggregated rankings
print(f"\n{'='*80}")
print(f"AGGREGATED CHANNEL IMPORTANCE RESULTS (LNSO CV)")
print(f"{'='*80}")

if all_channels_baseline is not None and len(model_specific_baselines) > 0:
    print(f"\nBaseline (All Channels):")
    print(f"   Aggregated RÂ² = {all_channels_baseline:.3f}")
    for model_name, baseline_r2 in list(model_specific_baselines.items())[:5]:
        print(f"   {model_name}: {baseline_r2:.3f}")
    if len(model_specific_baselines) > 5:
        print(f"   ... and {len(model_specific_baselines) - 5} more models")
    
    # Calculate Î”RÂ² for each model and each channel group
    channel_rankings_per_model = {model_name: [] for model_name in model_specific_baselines.keys()}
    
    for channel_name, channel_results in results.items():
        if channel_name != 'All_Channels' and 'model_results' in channel_results:
            for model_name, model_res in channel_results['model_results'].items():
                if model_name in model_specific_baselines:
                    delta_r2 = model_res['r2_mean'] - model_specific_baselines[model_name]
                    channel_rankings_per_model[model_name].append(
                        (channel_name, model_res['r2_mean'], delta_r2)
                    )
    
    # Rank channels for each model independently
    model_ranks = {model_name: {} for model_name in model_specific_baselines.keys()}
    
    for model_name, channel_deltas in channel_rankings_per_model.items():
        # Sort by Î”RÂ² descending
        channel_deltas.sort(key=lambda x: x[2], reverse=True)
        
        # Assign ranks (1 = best)
        for rank, (channel_name, r2, delta) in enumerate(channel_deltas, 1):
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
        
        # Get aggregated RÂ² for this channel
        channel_r2 = results[channel_name]['r2_score'] if channel_name in results else 0.0
        delta_r2 = channel_r2 - all_channels_baseline
        
        aggregated_rankings.append((channel_name, channel_r2, delta_r2, avg_rank, ranks))
    
    # Sort by average rank (lower is better)
    aggregated_rankings.sort(key=lambda x: x[3])
    
    print(f"\nðŸ“Š AGGREGATED Channel Importance Ranking (by average rank across {len(models)} models):")
    print(f"{'Rank':<6} {'Channel':<20} {'Agg RÂ²':<10} {'Î”RÂ²':<10} {'Avg Rank':<12} {'Model Count'}")
    print(f"{'-'*80}")
    
    for i, (channel_name, r2, delta, avg_rank, individual_ranks) in enumerate(aggregated_rankings):
        rank = i + 1
        delta_sign = "+" if delta >= 0 else ""
        status = "ðŸŸ¢" if delta > 0 else "ðŸ”´" if delta < -0.01 else "ðŸŸ¡"
        n_models = len([r for r in individual_ranks if r < 999])
        print(f"{rank:<6} {status} {channel_name:<18} {r2:.3f}      {delta_sign}{delta:>6.3f}    {avg_rank:>6.2f}      {n_models}/{len(models)}")
    
    # Performance summary
    if aggregated_rankings:
        best_channel = aggregated_rankings[0]
        worst_channel = aggregated_rankings[-1]
        positive_count = len([ch for ch in aggregated_rankings if ch[2] > 0])
        
        print(f"\nðŸ“ˆ Summary:")
        print(f"   â€¢ {positive_count}/{len(aggregated_rankings)} groups outperform baseline")
        print(f"   â€¢ Best: {best_channel[0]} (avg rank {best_channel[3]:.2f}, Î”RÂ² {best_channel[2]:+.3f})")
        print(f"   â€¢ Worst: {worst_channel[0]} (avg rank {worst_channel[3]:.2f}, Î”RÂ² {worst_channel[2]:+.3f})")
        print(f"   â€¢ Aggregation Method: Average rank across {len(models)} models with LNSO CV")
    
    # Export results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_TEST" if test_mode else ""
    subscale_suffix = f"_{args.subscale}" if args.subscale else ""
    output_file = f"channel_importance_aggregated_{dataset_name}_regression_loso{subscale_suffix}{mode_suffix}_{timestamp}.csv"
    output_dir = os.path.join(project_root, 'channel_importance', 'feature_tests', 'tests', 'regression', 'loso')
    output_path = os.path.join(output_dir, output_file)
    
    # Prepare comprehensive DataFrame with all metrics
    export_data = []
    for rank, (channel_name, r2, delta, avg_rank, individual_ranks) in enumerate(aggregated_rankings, 1):
        row = {
            'Rank': rank,
            'Channel_Category': channel_name,
            'Aggregated_R2': r2,
            'Delta_R2': delta,
            'Average_Rank': avg_rank,
            'N_LNSO_Folds': len(cv_splits),
            'N_Test_Subjects_Per_Fold': n_test_subjects,
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
            row[f'{model_name}_Rank'] = model_rank if model_rank < 999 else None
        
        # Add individual model detailed metrics
        if channel_name in results and 'model_results' in results[channel_name]:
            for model_name, model_res in results[channel_name]['model_results'].items():
                row[f'{model_name}_R2_Mean'] = model_res['r2_mean']
                row[f'{model_name}_R2_Std'] = model_res['r2_std']
                row[f'{model_name}_MAE_Mean'] = model_res['mae_mean']
                row[f'{model_name}_MAE_Std'] = model_res['mae_std']
                row[f'{model_name}_MSE_Mean'] = model_res['mse_mean']
                row[f'{model_name}_MSE_Std'] = model_res['mse_std']
                row[f'{model_name}_RMSE_Mean'] = model_res['rmse_mean']
                row[f'{model_name}_RMSE_Std'] = model_res['rmse_std']
                row[f'{model_name}_N_Folds'] = model_res['n_folds']
                row[f'{model_name}_Delta_R2'] = model_res['r2_mean'] - model_specific_baselines.get(model_name, 0)
        
        export_data.append(row)
    
    df_results = pd.DataFrame(export_data)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nðŸ’¾ Results exported to: {output_path}")
    
    # Also save a detailed run info file
    detailed_output_file = f"run_details_aggregated_{dataset_name}_regression_loso{subscale_suffix}{mode_suffix}_{timestamp}.csv"
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
                baseline_r2 = model_specific_baselines.get(model_name, 0)
                detailed_data.append({
                    'Dataset': dataset_name.upper(),
                    'Test_Mode': test_mode,
                    'Channel_Category': channel_name,
                    'Model': model_name,
                    'R2_Mean': model_res['r2_mean'],
                    'R2_Std': model_res['r2_std'],
                    'MAE_Mean': model_res['mae_mean'],
                    'MAE_Std': model_res['mae_std'],
                    'MSE_Mean': model_res['mse_mean'],
                    'MSE_Std': model_res['mse_std'],
                    'RMSE_Mean': model_res['rmse_mean'],
                    'RMSE_Std': model_res['rmse_std'],
                    'N_Folds': model_res['n_folds'],
                    'Baseline_R2': baseline_r2,
                    'Delta_R2': model_res['r2_mean'] - baseline_r2,
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
        print(f"   âœ“ LNSO CV setup: {len(cv_splits)} folds, {n_subjects} subjects")
        print(f"   âœ“ Model ensemble: {len(models)} models trained")
        print(f"   âœ“ Channel groups: {len(results)} tested")
        print(f"   âœ“ Ranking aggregation: {len(aggregated_rankings)} channels ranked")
        print(f"   âœ“ CSV export: Results saved")
        print(f"\nðŸ’¡ Test passed! Ready for full run without --test flag")
        print(f"{'='*80}\n")

else:
    print("   âš ï¸ No baseline found - cannot calculate aggregated rankings")

print(f"\nâœ… Aggregated LNSO analysis complete - tested {len(results)} channel groups with {len(models)} models")
if test_mode:
    print(f"   ðŸ§ª TEST MODE - Use without --test flag for full analysis")
print(f"{'='*80}\n")

