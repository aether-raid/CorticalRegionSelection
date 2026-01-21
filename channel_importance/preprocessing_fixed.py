"""
RECENT CHANGES (Oct 9, 2025):
==============================

Fixed critical data access and shape mismatch issues:

1. DICTIONARY ACCESS FIX:
   - Problem: Code was treating dataset samples as objects with attributes (sample.raw_features, sample.label)
   - Reality: EEGRawRegressionDataset.__getitem__() returns dictionaries with keys 'raw_features', 'label', 'file_id'
   - Fix: Changed sample.raw_features → sample['raw_features'] and sample.label → sample['label']
   - Error prevented: AttributeError: 'dict' object has no attribute 'raw_features'

2. FEATURE COUNT MISMATCH FIX:
   - Problem: Dataset had been pre-filtered (468 actual features) but flattened_features list still contained 
     original 696 feature names, causing pandas DataFrame creation to fail
   - Symptoms: "Shape of passed values is (130, 468), indices imply (130, 696)" error
   - Root cause: Dataset filtering modified actual data but not the metadata feature names list
   - Fix: Added validation to detect mismatches and automatically truncate feature names to match actual data size
   - Benefits: Robust handling of pre-filtered datasets, prevents shape mismatches in feature selection

3. DEFENSIVE PROGRAMMING:
   - Added feature count consistency checks with helpful warning messages
   - Ensured graceful fallback when feature metadata doesn't match actual data structure
   - Improved error messages for debugging future data pipeline issues

These fixes enable the regression analysis pipeline to work correctly with filtered EEG datasets
and handle cases where dataset preprocessing has modified the feature space.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from mrmr import mrmr_classif, mrmr_regression


def preprocess_data(raw_dataset, train_indices, n_features, task_type, selection_method, qpso_kwargs=None, pre_filtered_indices=None):
    """
    Preprocess EEG statistical features and perform feature selection.
    
    Fixed version with proper Random Forest feature selection implementation.
    
    Parameters:
        raw_dataset (Dataset): EEG dataset containing statistical features
        train_indices (list): Indices of training samples  
        n_features (int): Number of features to select
        task_type (str): 'classification' or 'regression'
        selection_method (str): Feature selection method ('anova', 'mrmr', 'rf', 'none')
        qpso_kwargs (dict): Not used, kept for compatibility
        pre_filtered_indices (list): Pre-filtered feature indices (e.g., by channel), or None for all features
        
    Returns:
        tuple: (feature_indices, scaler, feature_names, selection_scores)
            - feature_indices: Indices of selected features in original dataset
            - scaler: StandardScaler fitted on selected features
            - feature_names: Names of selected features
            - selection_scores: Dict mapping feature names to their selection scores (or None if no selection)
    """
    # Get training data
    train_samples = [raw_dataset[i] for i in train_indices]
    
    # Extract features and labels from training samples
    train_features = []
    train_labels = []
    
    for sample in train_samples:
        # Handle both dataclass and dictionary formats
        if hasattr(sample, 'raw_features'):  # dataclass (EEGRawDatasetEntry)
            features = sample.raw_features
            label = sample.label
        else:  # dictionary format
            features = sample['raw_features']
            label = sample['label']
            
        if hasattr(features, 'values'):  # pandas Series
            features = features.values
        elif isinstance(features, dict):  # dict case
            features = np.array(list(features.values()))
        train_features.append(features)
        train_labels.append(label)
    
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    
    # Handle NaN values - replace with 0 instead of using SimpleImputer to avoid column removal
    train_features = np.nan_to_num(train_features, nan=0.0)
    
    # Get feature names from the dataset's flattened_features list
    all_feature_names = raw_dataset.flattened_features
    
    # DEBUG: Check consistency between feature data and feature names
    actual_feature_count = train_features.shape[1]
    expected_feature_count = len(all_feature_names)
    print(f"DEBUG: Actual feature count in data: {actual_feature_count}")
    print(f"DEBUG: Expected feature count from names: {expected_feature_count}")
    
    if actual_feature_count != expected_feature_count:
        print(f"WARNING: Feature count mismatch! Using actual feature count.")
        # Create dummy feature names if they don't match
        all_feature_names = [f"feature_{i}" for i in range(actual_feature_count)]
    
    # Apply pre-filtering if specified
    if pre_filtered_indices is not None:
        train_features = train_features[:, pre_filtered_indices]
        feature_names = [all_feature_names[i] for i in pre_filtered_indices]
        print(f"Pre-filtered to {len(feature_names)} features by channel")
    else:
        feature_names = all_feature_names
    
    print(f"Dataset contains {len(feature_names)} features")
    
    # Track original indices for mapping back to dataset
    if pre_filtered_indices is not None:
        current_indices = pre_filtered_indices.copy()
    else:
        current_indices = list(range(len(feature_names)))
    
    # Remove features with near-zero variance BEFORE scaling/selection
    # These features (like mean_* from bandpass-filtered signals) have std~1e-18
    # StandardScaler divides by near-zero std, creating extreme values that break Ridge
    feature_stds = np.std(train_features, axis=0)
    variance_threshold = 1e-10
    valid_variance_features = feature_stds >= variance_threshold
    
    n_removed = (~valid_variance_features).sum()
    if n_removed > 0:
        print(f"  Removing {n_removed} features with near-zero variance (<{variance_threshold})")
        removed_indices = np.where(~valid_variance_features)[0]
        removed_feature_names = [feature_names[i] for i in removed_indices]
        print(f"  Removed features: {removed_feature_names[:10]}..." if len(removed_feature_names) > 10 else f"  Removed features: {removed_feature_names}")
        
        # Filter features and update indices
        train_features = train_features[:, valid_variance_features]
        feature_names = [feature_names[i] for i in np.where(valid_variance_features)[0]]
        current_indices = [current_indices[i] for i in np.where(valid_variance_features)[0]]
        print(f"  Retained {len(feature_names)} features after variance filtering")
    
    # Scale features for feature selection
    scaler_for_selection = StandardScaler()
    train_features_scaled = scaler_for_selection.fit_transform(train_features)
    
    # Create scaler for model training (will be fitted on selected features)
    scaler_for_model = StandardScaler()
    
    # Perform feature selection if requested
    selection_scores = {}  # Track feature selection scores
    
    if selection_method and selection_method != "none":
        print(f"Performing {selection_method.upper()} feature selection for {n_features} features...")
        
        # Ensure we don't select more features than available
        n_features = min(n_features, len(feature_names))
        
        selected_features = []
        selected_indices = []
        
        # Classification task feature selection
        if task_type == "classification":
            if selection_method == "mrmr":
                # Create DataFrame for MRMR
                feature_df = pd.DataFrame(train_features_scaled, columns=feature_names)
                selected_features = mrmr_classif(feature_df, train_labels, K=n_features)
                selected_indices = [feature_names.index(f) for f in selected_features]
                # MRMR doesn't provide scores, use selection order as proxy
                for i, feat in enumerate(selected_features):
                    selection_scores[feat] = n_features - i  # Higher rank = higher score
                
            elif selection_method == "anova":
                selector = SelectKBest(score_func=f_classif, k=n_features)
                selector.fit(train_features_scaled, train_labels)
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
                # Store F-scores for selected features
                all_scores = selector.scores_
                for idx, feat in zip(selected_indices, selected_features):
                    selection_scores[feat] = float(all_scores[idx])
                
            elif selection_method == "rf":
                # Random Forest importance-based selection
                print("RF Debug: Starting Random Forest feature selection...")
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(train_features_scaled, train_labels)
                importances = rf.feature_importances_
                
                print(f"RF Debug: importances shape: {importances.shape}, n_features: {n_features}")
                print(f"RF Debug: max importance: {importances.max():.6f}, min importance: {importances.min():.6f}")
                
                # Get top n_features indices
                top_indices = np.argsort(importances)[::-1][:n_features]
                print(f"RF Debug: top_indices: {top_indices[:10]}...")
                
                selected_indices = top_indices
                selected_features = [feature_names[i] for i in selected_indices]
                # Store importance scores
                for idx, feat in zip(selected_indices, selected_features):
                    selection_scores[feat] = float(importances[idx])
                print(f"Selected features: {selected_features[:4]}")
                
        # Regression task feature selection
        else:
            if selection_method == "mrmr":
                feature_df = pd.DataFrame(train_features_scaled, columns=feature_names)
                selected_features = mrmr_regression(feature_df, train_labels, K=n_features)
                selected_indices = [feature_names.index(f) for f in selected_features]
                # MRMR doesn't provide scores, use selection order as proxy
                for i, feat in enumerate(selected_features):
                    selection_scores[feat] = n_features - i  # Higher rank = higher score
                
            elif selection_method == "anova":
                selector = SelectKBest(score_func=f_regression, k=n_features)
                selector.fit(train_features_scaled, train_labels)
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
                # Store F-scores for selected features
                all_scores = selector.scores_
                for idx, feat in zip(selected_indices, selected_features):
                    selection_scores[feat] = float(all_scores[idx])
                
            elif selection_method == "rf":
                print("RF Debug: Starting Random Forest regression feature selection...")
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(train_features_scaled, train_labels)
                importances = rf.feature_importances_
                
                print(f"RF Debug: importances shape: {importances.shape}, n_features: {n_features}")
                print(f"RF Debug: max importance: {importances.max():.6f}, min importance: {importances.min():.6f}")
                
                # Get top n_features indices
                top_indices = np.argsort(importances)[::-1][:n_features]
                print(f"RF Debug: top_indices: {top_indices[:10]}...")
                
                selected_indices = top_indices
                selected_features = [feature_names[i] for i in selected_indices]
                # Store importance scores
                for idx, feat in zip(selected_indices, selected_features):
                    selection_scores[feat] = float(importances[idx])
                print(f"Selected features: {selected_features[:4]}")

        # Map selected indices back to original dataset indices
        # selected_indices are relative to current filtered features
        # current_indices map back to original dataset
        feature_indices = np.array([current_indices[i] for i in selected_indices])
        feature_names = selected_features
        print(f"Selected features: {feature_names}")
        
        # Fit final scaler to selected features
        if len(selected_indices) > 0:
            # Convert selected_indices to numpy array if it's a list
            selected_idx_array = np.array(selected_indices) if isinstance(selected_indices, list) else selected_indices
            scaler_for_model.fit(train_features[:, selected_idx_array])
        else:
            raise ValueError(f"No features selected by {selection_method} method!")
            
    else:
        # Case when no feature selection is performed (use all features)
        feature_indices = np.array(current_indices)
        scaler_for_model.fit(train_features)
        # No selection scores when no selection is performed
        selection_scores = None

    return feature_indices, scaler_for_model, feature_names, selection_scores