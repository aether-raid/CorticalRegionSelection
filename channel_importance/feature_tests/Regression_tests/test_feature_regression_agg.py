"""
EEG Channel Importance Analysis for Regression Tasks - AGGREGATED MODEL ENSEMBLE
================================================================================

This script performs channel importance analysis by AGGREGATING rankings from 30+ different regression models.
Instead of relying on a single model, this approach combines insights from multiple diverse models
spanning different algorithm families to produce highly robust and generalizable channel importance rankings.
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
# Task: Regression (continuous workload prediction)
#
# AGGREGATION METHODOLOGY:
# - Feature Selection: MRMR only, k=24 features (fixed)
# - Models: 30+ total across 8 algorithm families
# - Evaluation: 3-fold CV, same splits for all channel groups and models
# - Metric: RÂ² (coefficient of determination)

import argparse
import pandas as pd
import numpy as np
import sys
import os
import traceback
from datetime import datetime

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

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor, PassiveAggressiveRegressor, RANSACRegressor, BayesianRidge, ARDRegression, TheilSenRegressor, QuantileRegressor, TweedieRegressor, OrthogonalMatchingPursuit, LassoLarsIC
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from collections import Counter
import argparse
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
    Get ensemble of 30+ diverse regression models spanning multiple algorithm families.
    
    Model families included:
    - Linear: Ridge, Lasso, ElasticNet, SGD, Huber, Passive Aggressive
    - Tree-based: Random Forest, Extra Trees, Decision Tree, Gradient Boosting
    - Boosting: AdaBoost, Hist Gradient Boosting, XGBoost, CatBoost, LightGBM
    - SVM: RBF SVR, Linear SVR, Nu-SVR
    - Neural: MLP (multiple architectures)
    - Nearest Neighbors: KNN (different k values)
    - Kernel Methods: Kernel Ridge, Gaussian Process
    - Ensemble: Bagging, RANSAC
    
    Returns:
        Dictionary mapping model names to model constructors
    """
    models = {
        # ===== LINEAR MODELS (6) =====
        'Ridge': lambda: Ridge(
            alpha=3.0,
            max_iter=5000,
            tol=1e-6,
            solver='auto',
            random_state=RANDOM_STATE
        ),
        
        'Lasso': lambda: Lasso(
            alpha=0.1,
            max_iter=5000,
            tol=1e-6,
            selection='cyclic',
            random_state=RANDOM_STATE
        ),
        
        'ElasticNet': lambda: ElasticNet(
            alpha=0.5,
            l1_ratio=0.5,
            max_iter=5000,
            tol=1e-6,
            selection='cyclic',
            random_state=RANDOM_STATE
        ),
        
        'SGDRegressor': lambda: SGDRegressor(
            loss='squared_error',
            penalty='elasticnet',
            alpha=0.001,
            l1_ratio=0.15,
            max_iter=5000,
            tol=1e-6,
            learning_rate='optimal',
            random_state=RANDOM_STATE
        ),
        
        'HuberRegressor': lambda: HuberRegressor(
            epsilon=1.35,
            alpha=1.0,
            max_iter=5000,
            tol=1e-6
        ),
        
        'PassiveAggressive': lambda: PassiveAggressiveRegressor(
            C=1.0,
            max_iter=5000,
            tol=1e-6,
            random_state=RANDOM_STATE
        ),
        
        # ===== TREE-BASED MODELS (4) =====
        'RandomForest': lambda: RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        
        'ExtraTrees': lambda: ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        
        'DecisionTree': lambda: DecisionTreeRegressor(
            max_depth=6,
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
        
        # ===== BOOSTING MODELS (2) =====
        'HistGradientBoosting': lambda: HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=200,
            max_depth=5,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
        ),
        
        'AdaBoost': lambda: AdaBoostRegressor(
            n_estimators=150,
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
            max_iter=5000,
            tol=1e-6,
            random_state=RANDOM_STATE
        ),
        
        'NuSVR': lambda: NuSVR(
            nu=0.5,
            C=10.0,
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
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
        ),
        
        'MLP_Medium': lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
        ),
        
        'MLP_Small': lambda: MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=0.02,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
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
            kernel=C(1.0) * RBF(1.0),
            alpha=1e-6,
            n_restarts_optimizer=2,
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
            n_estimators=200,
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
        models['LightGBM'] = lambda: LGBMRegressor(
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
            random_state=RANDOM_STATE,
            verbose=-1,
            force_col_wise=True
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
        # 'LassoLarsIC_AIC': lambda: LassoLarsIC(criterion='aic'),  # Requires n_samples > n_features

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
            learning_rate=0.1, max_iter=200, max_depth=5, min_samples_leaf=15, l2_regularization=1.0, 
            early_stopping=False,  # Disabled for small datasets (avoids internal validation split errors)
            random_state=RANDOM_STATE
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
        # 'RadiusNeighbors': lambda: RadiusNeighborsRegressor(radius=3.0, weights='distance', p=2, leaf_size=30),  # Removed due to radius issues

        # ===== ENSEMBLE META-LEARNERS =====
        'Bagging': lambda: BaggingRegressor(n_estimators=50, max_samples=0.8, max_features=0.8, bootstrap=True, bootstrap_features=False, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        'RANSAC': lambda: RANSACRegressor(min_samples=0.5, max_trials=100, residual_threshold=None, random_state=RANDOM_STATE),
        'Stacking_RidgeMeta': lambda: StackingRegressor(
            estimators=[
                ('hgb', HistGradientBoostingRegressor(max_depth=5, early_stopping=False, random_state=RANDOM_STATE)),
                ('rf', RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
                ('svr', SVR(C=10.0, kernel='rbf', epsilon=0.1))
            ],
            final_estimator=Ridge(alpha=2.0, random_state=RANDOM_STATE),
            passthrough=True, n_jobs=N_JOBS
        ),
        'Voting': lambda: VotingRegressor(
            estimators=[
                ('ridge', Ridge(alpha=3.0, random_state=RANDOM_STATE)),
                ('hgb', HistGradientBoostingRegressor(max_depth=5, early_stopping=False, random_state=RANDOM_STATE)),
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
        # models['GAM'] = lambda: LinearGAM(n_splines=10, lam=0.6, callbacks=[])
        pass  # LinearGAM incompatible with sklearn cloning for CV
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
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def train_ensemble_models_regression(raw_dataset, train_indices, test_indices, channel_filter_indices=None, selection_method='rf', no_feature_selection=False):
    """
    Train and evaluate ALL models in the ensemble with feature selection (k=24).
    
    Args:
        raw_dataset: EEGRawRegressionDataset instance
        
        train_indices: List of training sample indices
        test_indices: List of testing sample indices  
        channel_filter_indices: Indices of features to filter by channel
        selection_method: Feature selection method ('rf' or 'mrmr')
        no_feature_selection: If True, use all features without selection
        
    Returns:
        dict: Results containing all model performances and aggregated metrics
    """
    
    try:
    # MRMR feature selection with k=24
        # Apply channel filtering if specified
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
        
        # Remove features with near-zero variance to prevent numerical instability
        # These features (like mean_* from bandpass-filtered signals) have std~1e-18
        # StandardScaler divides by near-zero std, creating extreme values that break models
        feature_stds = np.std(X_train, axis=0)
        variance_threshold = 1e-10
        valid_features = feature_stds >= variance_threshold
        
        n_removed = (~valid_features).sum()
        if n_removed > 0:
            print(f"  Removing {n_removed} features with near-zero variance (<{variance_threshold})")
            
            # Filter features
            X_train = X_train[:, valid_features]
            X_test = X_test[:, valid_features]
            feature_names = [feature_names[i] for i in np.where(valid_features)[0]]
        
        # Handle NaN and infinity values if present
        train_nans = np.isnan(X_train).sum()
        test_nans = np.isnan(X_test).sum()
        train_infs = np.isinf(X_train).sum()
        test_infs = np.isinf(X_test).sum()
        
        if train_nans > 0 or test_nans > 0 or train_infs > 0 or test_infs > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
            print(f"   Handled {train_nans + test_nans} NaN values, {train_infs + test_infs} inf values")
        
        # Get model ensemble
        models = get_model_ensemble()
        
        # Train and evaluate each model
        model_results = {}
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        n_models = len(models)
        
        for model_idx, (model_name, model_func) in enumerate(models.items(), 1):
            print(f"\r    Training model {model_idx}/{n_models}: {model_name[:20]:<20}", end="")
            sys.stdout.flush()
            
            try:
                # Clean data immediately before training to catch any infinities
                X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
                X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Train and test
                result = train_evaluate_sklearn_regression(model_func, X_train_clean, X_test_clean, y_train, y_test, model_name)
                
                # Cross-validation with error suppression
                fresh_model = model_func()
                with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                    cv_scores = cross_val_score(fresh_model, X_train_clean, y_train, cv=cv, scoring='r2')
                
                # Store results with detailed CV scores
                model_results[model_name] = {
                    'r2': result['r2'],
                    'mae': result['mae'],
                    'mse': result['mse'],
                    'rmse': result['rmse'],
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'cv_fold_1': cv_scores[0],
                    'cv_fold_2': cv_scores[1],
                    'cv_fold_3': cv_scores[2]
                }
            except Exception as e:
                # Model failed - skip it
                print(f"\r    Training model {model_idx}/{n_models}: {model_name[:20]:<20} [FAILED]")
                continue
        
        # Calculate aggregated metrics (median across valid models)
        # Filter out catastrophically bad models (RÂ² < -10) to avoid outliers like SGDRegressor
        print()  # Newline after model progress
        all_r2_scores = [res['r2'] for res in model_results.values()]
        all_cv_means = [res['cv_mean'] for res in model_results.values()]
        
        valid_r2_scores = [res['r2'] for res in model_results.values() if res['r2'] > -10]
        valid_cv_means = [res['cv_mean'] for res in model_results.values() if res['r2'] > -10]
        
        if valid_r2_scores:
            agg_r2 = np.median(valid_r2_scores)  # Use median for robustness
            agg_cv_mean = np.median(valid_cv_means)
            agg_cv_std = np.std(valid_cv_means)
            n_valid_models = len(valid_r2_scores)
            n_failed_models = len(all_r2_scores) - n_valid_models
        else:
            agg_r2 = 0.0
            agg_cv_mean = 0.0
            agg_cv_std = 0.0
            n_valid_models = 0
            n_failed_models = len(all_r2_scores)
        
        print(f"  âœ… Aggregated: RÂ²={agg_r2:.3f}, CV={agg_cv_mean:.3f}Â±{agg_cv_std:.3f} ({n_valid_models} valid models)")
        if n_failed_models > 0:
            print(f"  âš ï¸ Excluded {n_failed_models} models with catastrophic failures (RÂ² < -10)")
        model_scores = ', '.join([f"{name}={res['r2']:.3f}" for name, res in list(model_results.items())[:5]])
        print(f"     Models (sample): {model_scores}...")
        
        return {
            'model_results': model_results,
            'selected_features': feature_names[:32] if feature_names else [],
            'train_nans': train_nans,
            'test_nans': test_nans,
            'feature_names': feature_names,
            'selection_scores': selection_scores,  # Add selection scores
            'r2_score': agg_r2,  # Aggregated RÂ²
            'cv_r2_mean': agg_cv_mean,
            'cv_r2_std': agg_cv_std,
            'individual_r2_scores': all_r2_scores,  # For ranking calculation
        }
        
    except Exception as e:
        print(f"  âŒ ERROR: {str(e)}")
        traceback.print_exc()
        return {
            'model_results': {},
            'selected_features': [],
            'error': str(e),
            'train_nans': 0,
            'test_nans': 0,
            'feature_names': [],
            'selection_scores': {},  # Add selection scores
            'r2_score': 0.0,
            'cv_r2_mean': 0.0,
            'cv_r2_std': 0.0,
            'individual_r2_scores': []
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
    description='EEG Channel Importance Analysis - Aggregated Model Ensemble (Regression)',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  # Default (combined TLX)
  python test_feature_regression_agg.py --mocas
  python test_feature_regression_agg.py --htc
  
  # Specific TLX subscale
  python test_feature_regression_agg.py --htc --subscale mental
  python test_feature_regression_agg.py --htc --subscale effort
  python test_feature_regression_agg.py --htc --subscale frustration
  
  # Test mode (quick validation)
  python test_feature_regression_agg.py --htc --test
  python test_feature_regression_agg.py --htc --subscale mental --test
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
                   help='TLX subscale to analyze (default: combined TLX). Options: combined, mental, physical, temporal, performance, effort, frustration')

# Add feature selection method
parser.add_argument('--selection-method', type=str,
                   choices=['rf', 'mrmr', 'anova'],
                   default='rf',
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
    print("   Estimated time: ~2-3 minutes\n")

if all_channels_only:
    print("\n[ALL-CHANNELS-ONLY] Testing only All_Channels group\n")

if no_feature_selection:
    print("\n[NO FEATURE SELECTION] Using all features\n")

# Get subscale parameter
subscale = args.subscale if hasattr(args, 'subscale') else None

# Get feature selection method
selection_method = getattr(args, 'selection_method', 'rf')

# Load dataset
print(f"\n{'='*80}")
print(f"EEG CHANNEL IMPORTANCE ANALYSIS - AGGREGATED MODEL ENSEMBLE (REGRESSION)")
if test_mode:
    print(f"                      *** TEST MODE ***")
if subscale:
    print(f"                   TLX SUBSCALE: {subscale.upper()}")
print(f"{'='*80}")
print(f"\nDataset: {dataset_name.upper()}")
if subscale:
    print(f"Target: TLX {subscale.capitalize()} Demand Subscale")
else:
    print(f"Target: Combined TLX Score (default)")
print(f"Feature Selection: {selection_method.upper() if selection_method else 'NONE (all features)'}")
print(f"Loading from: {data_path}")
print(f"Path exists: {os.path.exists(data_path)}")

try:
    raw_dataset = EEGRawRegressionDataset(data_path, target_suffix=subscale)
    
    # Get dataset info
    first_sample = raw_dataset[0]
    feature_vector = first_sample['raw_features']
    all_targets = [raw_dataset[i]['label'] for i in range(len(raw_dataset))]
    
    print(f"âœ… Dataset loaded: {len(raw_dataset)} samples, {len(raw_dataset.flattened_features)} features")
    print(f"   Target range: [{np.min(all_targets):.2f}, {np.max(all_targets):.2f}], mean: {np.mean(all_targets):.2f}")
    
except Exception as e:
    print(f"âŒ Error loading dataset: {e}")
    exit(1)

# Create train/test split
print(f"\nðŸ“Š Creating train/test split (70/30)...")
all_indices = list(range(len(raw_dataset)))
targets = [raw_dataset[i]['label'] for i in all_indices]

train_indices, test_indices = train_test_split(
    all_indices, test_size=0.3, random_state=RANDOM_STATE
)

print(f"   Training: {len(train_indices)} samples | Testing: {len(test_indices)} samples")

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

# Results structure: results[channel_category] = model ensemble results
results = {}
all_channels_baseline = None  # Store baseline performance for Î”RÂ² calculation
model_specific_baselines = {}  # Store baseline for each model

# Get feature names for channel filtering
first_sample = raw_dataset[0]
all_feature_names = raw_dataset.flattened_features

print(f"\nðŸ§  Starting aggregated channel importance analysis...")
print(f"   Method: MRMR (k=24) + {len(models)} Models + 3-fold CV + Aggregated Rankings")
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
    
    # Train ensemble models
    try:
        ensemble_results = train_ensemble_models_regression(
            raw_dataset, train_indices, test_indices, channel_filter_indices, selection_method, no_feature_selection
        )
        
        if ensemble_results and 'r2_score' in ensemble_results:
            results[channel_name] = ensemble_results
            
            # Store all-channels baseline for Î”RÂ² calculation
            if channel_name == 'All_Channels':
                all_channels_baseline = ensemble_results['r2_score']
                # Also store individual model baselines
                if 'model_results' in ensemble_results:
                    model_specific_baselines = {
                        name: res['r2'] 
                        for name, res in ensemble_results['model_results'].items()
                    }
            
        else:
            print(f"â†’ Failed")
            
    except Exception as e:
        print(f"â†’ Error: {str(e)}")
        traceback.print_exc()

# Calculate aggregated rankings
print(f"\n{'='*80}")
print(f"AGGREGATED CHANNEL IMPORTANCE RESULTS (REGRESSION)")
print(f"{'='*80}")

if all_channels_baseline is not None and len(model_specific_baselines) > 0:
    print(f"\nBaseline (All Channels):")
    print(f"   Aggregated RÂ² = {all_channels_baseline:.3f}")
    baseline_sample = list(model_specific_baselines.items())[:5]
    for model_name, baseline_r2 in baseline_sample:
        print(f"   {model_name}: {baseline_r2:.3f}")
    print(f"   ... and {len(model_specific_baselines) - 5} more models")
    
    # Calculate Î”RÂ² for each model and each channel group
    channel_rankings_per_model = {model_name: [] for model_name in model_specific_baselines.keys()}
    
    for channel_name, channel_results in results.items():
        if channel_name != 'All_Channels' and 'model_results' in channel_results:
            for model_name, model_res in channel_results['model_results'].items():
                if model_name in model_specific_baselines:
                    delta_r2 = model_res['r2'] - model_specific_baselines[model_name]
                    channel_rankings_per_model[model_name].append(
                        (channel_name, model_res['r2'], delta_r2)
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
    print(f"{'Rank':<6} {'Channel':<20} {'Agg RÂ²':<10} {'Î”RÂ²':<10} {'Avg Rank':<12}")
    print(f"{'-'*60}")
    
    for i, (channel_name, r2, delta, avg_rank, individual_ranks) in enumerate(aggregated_rankings):
        rank = i + 1
        delta_sign = "+" if delta >= 0 else ""
        status = "ðŸŸ¢" if delta > 0 else "ðŸ”´" if delta < -0.01 else "ðŸŸ¡"
        print(f"{rank:<6} {status} {channel_name:<18} {r2:.3f}      {delta_sign}{delta:>6.3f}    {avg_rank:>6.2f}")
    
    # Performance summary
    if aggregated_rankings:
        best_channel = aggregated_rankings[0]
        worst_channel = aggregated_rankings[-1]
        positive_count = len([ch for ch in aggregated_rankings if ch[2] > 0])
        
        print(f"\nðŸ“ˆ Summary:")
        print(f"   â€¢ {positive_count}/{len(aggregated_rankings)} groups outperform baseline")
        print(f"   â€¢ Best: {best_channel[0]} (avg rank {best_channel[3]:.2f}, Î”RÂ² {best_channel[2]:+.3f})")
        print(f"   â€¢ Worst: {worst_channel[0]} (avg rank {worst_channel[3]:.2f}, Î”RÂ² {worst_channel[2]:+.3f})")
        print(f"   â€¢ Aggregation Method: Average rank across {len(models)} models")
    
    # Export results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_TEST" if test_mode else ""
    subscale_suffix = f"_{subscale}" if subscale else ""
    output_file = f"channel_importance_aggregated_{dataset_name}_regression{subscale_suffix}{mode_suffix}_{timestamp}.csv"
    output_dir = os.path.join(project_root, 'channel_importance', 'feature_tests', 'tests', 'regression', 'cv')
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
            'N_Features': len(results[channel_name].get('feature_names', [])) if channel_name in results else 0,
            'Feature_Names': '|'.join(results[channel_name].get('feature_names', [])) if channel_name in results else '',
        }
        
        # Add feature selection scores (comma-separated: feature=score)
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
                row[f'{model_name}_R2'] = model_res['r2']
                row[f'{model_name}_MAE'] = model_res['mae']
                row[f'{model_name}_MSE'] = model_res['mse']
                row[f'{model_name}_RMSE'] = model_res['rmse']
                row[f'{model_name}_CV_Mean'] = model_res['cv_mean']
                row[f'{model_name}_CV_Std'] = model_res['cv_std']
                row[f'{model_name}_CV_Fold1'] = model_res['cv_fold_1']
                row[f'{model_name}_CV_Fold2'] = model_res['cv_fold_2']
                row[f'{model_name}_CV_Fold3'] = model_res['cv_fold_3']
                row[f'{model_name}_Delta_R2'] = model_res['r2'] - model_specific_baselines.get(model_name, 0)
        
        export_data.append(row)
    
    df_results = pd.DataFrame(export_data)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nðŸ’¾ Results exported to: {output_path}")
    
    # Also save a detailed run info file
    detailed_output_file = f"run_details_aggregated_{dataset_name}_regression{subscale_suffix}{mode_suffix}_{timestamp}.csv"
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
                    'Target_Subscale': subscale if subscale else 'combined',
                    'Test_Mode': test_mode,
                    'Channel_Category': channel_name,
                    'Model': model_name,
                    'Test_R2': model_res['r2'],
                    'MAE': model_res['mae'],
                    'MSE': model_res['mse'],
                    'RMSE': model_res['rmse'],
                    'CV_Mean': model_res['cv_mean'],
                    'CV_Std': model_res['cv_std'],
                    'CV_Fold_1': model_res['cv_fold_1'],
                    'CV_Fold_2': model_res['cv_fold_2'],
                    'CV_Fold_3': model_res['cv_fold_3'],
                    'Baseline_R2': baseline_r2,
                    'Delta_R2': model_res['r2'] - baseline_r2,
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
    print(f"ðŸ’¾ Detailed run data exported to: {detailed_output_path}")
    
    # Print test mode summary
    if test_mode:
        print(f"\n{'='*80}")
        print(f"ðŸ§ª TEST MODE SUMMARY")
        print(f"{'='*80}")
        print(f"âœ… All components working correctly:")
        print(f"   âœ“ Dataset loading: {len(raw_dataset)} samples")
        print(f"   âœ“ Train/test split: {len(train_indices)}/{len(test_indices)} samples")
        print(f"   âœ“ Model ensemble: {len(models)} models trained")
        print(f"   âœ“ Channel groups: {len(results)} tested")
        print(f"   âœ“ Ranking aggregation: {len(aggregated_rankings)} channels ranked")
        print(f"   âœ“ CSV export: 2 files saved")
        print(f"\nðŸ’¡ Test passed! Ready for full run without --test flag")
        print(f"{'='*80}\n")

else:
    print("   âš ï¸ No baseline found - cannot calculate aggregated rankings")

print(f"\nâœ… Aggregated analysis complete - tested {len(results)} channel groups with {len(models)} models")
if test_mode:
    print(f"   ðŸ§ª TEST MODE - Use without --test flag for full analysis")
print(f"{'='*80}\n")

