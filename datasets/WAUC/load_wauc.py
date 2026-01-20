import os 
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directories to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from channel_importance.eeg import EEG


"""
WAUC Dataset Processing Pipeline for EEG Cognitive Workload Analysis
====================================================================

Complete preprocessing pipeline for the WAUC (Workload Assessment Using Continuous EEG) dataset 
that transforms raw session-based data into processed features suitable for regression analysis.

DATASET SOURCE:
    WAUC dataset with EEG data for cognitive workload assessment with TLX scores.
    Data organized by participant sessions with physiological measurements.

EXPECTED INPUT STRUCTURE:
    data/
        wauc_by_session/    # Session-based EEG data directory
            01/                  # Participant 1
                session0/
                    base1_eeg.parquet    # Baseline trial 1 (500Hz, 8 channels)
                    base2_eeg.parquet    # Baseline trial 2
                    test_eeg.parquet     # Test trial (used for processing)
                session1/
                    base1_eeg.parquet
                    base2_eeg.parquet
                    test_eeg.parquet
                ...session0-5
            02/                  # Participant 2
                session0/
                    ...
                ...
            ...01-48
        wauc/               # Main WAUC directory (output location)
            ratings.parquet      # TLX workload ratings (282 entries with participant-session mappings)
            S01/
                metadata.json
            S02/
                metadata.json
            ...S01-S48

PROCESSING PIPELINE (5 STEPS):
    
    STEP 1: EEG File Extraction & Reformatting (with Subband Decomposition)
        • Flattens session-based directory structure
        • Uses ONLY test trial per session (excludes baseline trials)
        • Applies subband decomposition (Overall, delta, theta, alpha, beta, gamma)
        • Creates MultiIndex DataFrame with (band, channel) structure
        • Standardizes filenames: S{participant}_session{N}_eeg_raw.parquet
        • Output: data/wauc/raw_eeg_extracted/ (temporary)
    
    STEP 2: Statistical Feature Extraction
        • Applies signal processing: bandpass filtering, frequency decomposition
        • Extracts 400+ statistical features per file using EEG class
        • Features: power bands, spectral entropy, hjorth parameters, etc.
        • Adapts to 500Hz sampling rate (vs 128Hz in MOCAS)
        • Output: data/wauc/features_extracted/S{X}_session{Y}_features.parquet (temporary)
        
    STEP 3: Target File Creation (Regression)
        • Loads ratings.parquet with TLX scores
        • Matches participant-session pairs with EEG files
        • Creates individual target files for each TLX dimension
        • Handles missing session data gracefully
        • Output: data/wauc/raw_eeg_extracted/S{X}_session{Y}_eeg_raw.txt
        • Output: data/wauc/features_extracted/S{X}_session{Y}_features.txt
        
    STEP 4: Data Consistency Cleanup
        • Removes feature files without corresponding targets
        • Removes raw files without corresponding targets
        • Ensures paired feature-target files for regression analysis
    
    STEP 5: Copy to Final Regression Datasets
        • Organizes paired files into final dataset directories
        • Separates time series and feature datasets
        • Output: data/wauc/wauc_time_regression_dataset/
        • Output: data/wauc/wauc_feature_regression_dataset/

FINAL OUTPUT STRUCTURE:
    data/wauc/
        # REGRESSION DATASETS (continuous TLX targets)
        wauc_time_regression_dataset/      # Subband-decomposed time series with TLX targets
            S1_session0_eeg_raw.parquet    # MultiIndex (band, channel) - 48 columns (8ch × 6bands)
            S1_session0_eeg_raw.txt        # TLX mental workload score (combined)
            S1_session1_eeg_raw.parquet
            S1_session1_eeg_raw.txt
            ...
        wauc_feature_regression_dataset/   # Statistical features with TLX targets
            S1_session0_features.parquet   # 400+ statistical features
            S1_session0_features.txt       # TLX mental workload score (combined)
            S1_session1_features.parquet
            S1_session1_features.txt
            ...
        
        # CLASSIFICATION DATASETS (categorical workload labels)
        wauc_time_mw_classification_dataset/    # Mental Workload - Binary (0=low, 1=high)
            low_mw/                        # 141 samples
                S1_session0_eeg_raw.parquet
                S2_session1_eeg_raw.parquet
                ...
            high_mw/                       # 141 samples
                S3_session2_eeg_raw.parquet
                ...
        wauc_feature_mw_classification_dataset/  # Mental Workload features
            low_mw/
                S1_session0_features.parquet
                ...
            high_mw/
                ...
        
        wauc_time_pw_classification_dataset/    # Physical Workload - 3-class (0=low, 1=med, 2=high)
            low_pw/                        # 94 samples each
                S1_session0_eeg_raw.parquet
                ...
            medium_pw/
                ...
            high_pw/
                ...
        wauc_feature_pw_classification_dataset/  # Physical Workload features
            low_pw/
                ...
            medium_pw/
                ...
            high_pw/
                ...
        
        wauc_time_mwpw_classification_dataset/  # Combined MW+PW - 6-class
            low_mw_low_pw/                 # 47 samples each
                ...
            high_mw_low_pw/
                ...
            low_mw_medium_pw/
                ...
            high_mw_medium_pw/
                ...
            low_mw_high_pw/
                ...
            high_mw_high_pw/
                ...
        wauc_feature_mwpw_classification_dataset/  # Combined workload features
            low_mw_low_pw/
                ...
            (same structure as time dataset)
        
        wauc_time_tlx_classification_dataset/   # TLX-Based - 3-class (low/medium/high)
            low/                           # TLX score < 33
                S1_session0_eeg_raw.parquet
                ...
            medium/                        # 33 <= TLX score < 67
                S2_session1_eeg_raw.parquet
                ...
            high/                          # TLX score >= 67
                S3_session2_eeg_raw.parquet
                ...
        wauc_feature_tlx_classification_dataset/  # TLX-Based features
            low/
                S1_session0_features.parquet
                ...
            medium/
                ...
            high/
                ...
        
        raw_eeg_extracted/                 # Temporary directory (intermediate processing)
        features_extracted/                # Temporary directory (intermediate processing)

INTEGRATION:
    The processed data is compatible with:
    • TimeRawDataset for loading time series and targets
    • EEGRawRegressionDataset for loading features and targets
    • Existing ML model evaluation pipelines
    • Channel importance analysis and feature selection methods

USAGE:
    python datasets/WAUC/load_wauc.py
    
    Or call individual functions:
    • reformat_raw_eeg()
    • extract_features_from_wauc_files()
    • create_target_files_from_ratings()
    • cleanup_files_without_targets()
"""

# Configuration paths - Use relative paths from project root
data_path = "data/wauc/wauc_by_session"  # Input: session-based structure
ratings_path = "data/wauc/wauc/ratings.parquet"  # Target values (correct location)
output_base_path = "data/wauc"  # Base output directory (parent wauc folder)
raw_output_path = os.path.join(output_base_path, "raw_eeg_extracted")  # Step 1 output (temp)
features_output_path = os.path.join(output_base_path, "features_extracted")  # Step 2 output (temp)
time_regression_path = os.path.join(output_base_path, "wauc_time_regression_dataset")  # Final time series
feature_regression_path = os.path.join(output_base_path, "wauc_feature_regression_dataset")  # Final features


def reformat_raw_eeg(source_dir, output_dir, sampling_rate=500.0, decompose=True, raw_only=False):
    """
    Reformat and extract raw EEG files from WAUC dataset.
    
    Processes session-based EEG data by:
    - Using ONLY test trial per session (excludes baseline trials)
    - Removing non-EEG columns (timestamps, time_vct, markers)
    - Creating proper time index from sampling rate
    - Standardizing to S{participant}_session{N}_eeg_raw.parquet format

    Args:
        source_dir (str): Path to wauc_by_session directory
        output_dir (str): Path to output directory for reformatted files
        sampling_rate (float): EEG sampling frequency in Hz (default 500.0)
        decompose (bool): Whether to perform subband decomposition (default True).
                         If False, keeps raw filtered data without frequency band separation.
                         Ignored if raw_only=True.
        raw_only (bool): If True, skip ALL signal processing (no filtering, no decomposition)
                        and just copy raw EEG channels. Use this for LaBraM embeddings.
                        Default: False.

    Returns:
        List of dictionaries with information about each processed file
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Append _rawonly to output directory if raw_only mode
    if raw_only:
        output_path = Path(str(output_path) + "_rawonly")
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_path}")
        return None
        
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_path}: {e}")
        return None

    print(f"{'='*60}")
    if raw_only:
        print(f"WAUC EEG EXTRACTION - RAW ONLY")
    else:
        print(f"WAUC EEG EXTRACTION AND REFORMATTING")
    print(f"{'='*60}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Raw only (no processing): {raw_only}")

    processed_files = 0
    failed_files = 0
    file_info = []
    failed_info = []
    
    # Get all participant folders (01, 02, ..., 48)
    participant_folders = sorted([d for d in source_path.iterdir() 
                                 if d.is_dir() and d.name.isdigit()])
    
    print(f"\nFound {len(participant_folders)} participant folders")
    print(f"{'='*60}\n")
    
    for participant_folder in participant_folders:
        participant_num = participant_folder.name  # "01", "02", etc.
        print(f"Processing Participant {participant_num}")
        
        # Get all session folders (session0, session1, ..., session5)
        session_folders = sorted([d for d in participant_folder.iterdir() 
                                 if d.is_dir() and d.name.startswith('session')])
        
        print(f"  Found {len(session_folders)} session folders")
        
        for session_folder in session_folders:
            session_num = session_folder.name.replace('session', '')  # "0", "1", etc.
            
            try:
                # Load only the test trial file (not baselines)
                test_file = session_folder / 'test_eeg.parquet'
                
                # Check if test file exists
                if not test_file.exists():
                    error_msg = f"Missing test_eeg.parquet in {session_folder.name}"
                    print(f"  WARNING: {error_msg}")
                    failed_info.append({
                        'participant': participant_num,
                        'session': session_num,
                        'error': error_msg,
                        'missing_files': ['test_eeg.parquet']
                    })
                    failed_files += 1
                    continue
                
                # Load only test trial
                df_test = pd.read_parquet(test_file)
                
                # EEG channel columns (exclude metadata columns)
                eeg_channels = ['AF8', 'FP2', 'FP1', 'AF7', 'T10', 'T9', 'P4', 'P3']
                
                # Extract only EEG channels from test trial
                df_eeg = df_test[eeg_channels].copy()
                
                # Create time array
                num_samples = len(df_eeg)
                time_seconds = np.arange(num_samples) / sampling_rate
                duration_seconds = time_seconds[-1] if num_samples > 0 else 0
                
                # Create output filename
                new_filename = f"S{participant_num}_session{session_num}_eeg_raw.parquet"
                output_file = output_path / new_filename
                
                if raw_only:
                    # Just save raw EEG without any processing
                    df_eeg.to_parquet(output_file)
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    
                    print(f"  ✓ Session {session_num}: {num_samples} samples, {duration_seconds:.1f}s, {file_size_mb:.2f} MB (raw)")
                    
                    file_info.append({
                        'participant': participant_num,
                        'session': session_num,
                        'filename': new_filename,
                        'output_path': str(output_file),
                        'num_samples': num_samples,
                        'num_channels': len(eeg_channels),
                        'channel_names': eeg_channels,
                        'duration_seconds': float(duration_seconds),
                        'sampling_rate': sampling_rate,
                        'file_size_mb': file_size_mb,
                        'trial_used': 'test',
                        'test_samples': len(df_test),
                        'raw_only': True
                    })
                else:
                    # Prepare channel data dictionary
                    sample_numbers = np.arange(num_samples)
                    channels_dict = {col: df_eeg[col].values for col in eeg_channels}
                    
                    # Create EEG instance to decompose into subbands
                    eeg = EEG(
                        s_n=sample_numbers,
                        t=time_seconds,
                        channels=channels_dict,
                        frequency=sampling_rate,
                        extract_time=False,  # Don't resample, just decompose
                        apply_notch=(50,60),
                        decompose=decompose
                    )
                    
                    # Get the subband-decomposed time series data
                    # eeg.data has MultiIndex columns: (band, channel)
                    # Format: (Overall, AF8), (delta, AF8), (theta, AF8), etc.
                    df_combined = eeg.data.copy()
                    df_combined.index.name = 'time'
                    
                    # Save combined EEG data
                    df_combined.to_parquet(output_file)
                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    
                    print(f"  ✓ Session {session_num}: {num_samples} samples, {duration_seconds:.1f}s, {file_size_mb:.2f} MB")
                    
                    file_info.append({
                        'participant': participant_num,
                        'session': session_num,
                        'filename': new_filename,
                        'output_path': str(output_file),
                        'num_samples': num_samples,
                        'num_channels': len(eeg_channels),
                        'channel_names': eeg_channels,
                        'duration_seconds': float(duration_seconds),
                        'sampling_rate': sampling_rate,
                        'file_size_mb': file_size_mb,
                        'trial_used': 'test',
                        'test_samples': len(df_test),
                        'raw_only': False
                    })
                
                processed_files += 1
                
            except Exception as e:
                error_msg = f"Error processing session {session_num}: {str(e)}"
                print(f"  ERROR: {error_msg}")
                failed_info.append({
                    'participant': participant_num,
                    'session': session_num,
                    'error': error_msg
                })
                failed_files += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_files} files")
    print(f"Failed: {failed_files} files")
    
    if processed_files > 0:
        total_samples = sum(f['num_samples'] for f in file_info)
        total_duration = sum(f['duration_seconds'] for f in file_info)
        avg_duration = total_duration / processed_files
        total_size_mb = sum(f['file_size_mb'] for f in file_info)
        
        print(f"\nStatistics:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total duration: {total_duration/60:.1f} minutes")
        print(f"  Average duration per session: {avg_duration:.1f} seconds")
        print(f"  Total data size: {total_size_mb:.1f} MB")
    
    # Save processing results
    results_summary = {
        'processing_stats': {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'sampling_rate': sampling_rate
        },
        'successful_files': file_info,
        'failed_files': failed_info
    }
    
    summary_file = output_path / 'extraction_summary.json'
    try:
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"\nProcessing summary saved to: {summary_file}")
    except Exception as e:
        print(f"Warning: Could not save summary file: {e}")
    
    return file_info



def _process_single_file_features_from_tuple(file_info_tuple, output_path, original_data_path, sampling_rate):
    """
    Process a single EEG file for feature extraction from tuple info (parallelizable).
    
    This version accepts a simple tuple (participant_num, session_num) instead of file object,
    making it compatible with multiprocessing pickle serialization.
    
    Args:
        file_info_tuple: Tuple of (participant_num, session_num) as strings
        output_path: Path object for output directory
        original_data_path: Path object to original wauc_by_session directory
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with processing results or error info
    """
    try:
        participant_num, session_num = file_info_tuple
        participant_id = f"S{participant_num}"
        session_id = f"session{session_num}"
        
        print(f"  → Processing {participant_id}_{session_id}...")
        
        # Construct path to original raw data
        original_file = original_data_path / participant_num / f"session{session_num}" / "test_eeg.parquet"
        
        if not original_file.exists():
            raise FileNotFoundError(f"Original raw file not found: {original_file}")
        
        print(f"    ✓ Loading ORIGINAL raw data from: {original_file.name}")
        
        # Load the ORIGINAL raw EEG data (not the decomposed MultiIndex version)
        df = pd.read_parquet(original_file)
        print(f"    ✓ Loaded {len(df)} samples")
        
        if df.empty:
            raise ValueError("Empty dataframe")
        
        # Get EEG channels (exclude metadata columns)
        eeg_channels = ['AF8', 'FP2', 'FP1', 'AF7', 'T10', 'T9', 'P4', 'P3']
        
        # Create time array
        num_samples = len(df)
        time_seconds = np.arange(num_samples) / sampling_rate
        sample_numbers = np.arange(num_samples)
        
        # Prepare channel data dictionary
        channels_dict = {col: df[col].values for col in eeg_channels}
        
        print(f"    ✓ Creating EEG instance with {len(eeg_channels)} channels, {num_samples} samples")
        print(f"    ✓ Resampling from {sampling_rate} Hz to 128 Hz...")
        
        # Create EEG instance from ORIGINAL raw data (processes ONCE correctly)
        # extract_time=True triggers resampling from 500Hz to 128Hz
        eeg = EEG(
            s_n=sample_numbers,
            t=time_seconds,
            channels=channels_dict,
            frequency=sampling_rate,
            extract_time=True,  # Enable resampling to 128Hz
            apply_notch=(50, 60)
        )
        print(f"    ✓ EEG object created and resampled, extracting features...")
        
        # Generate statistical features
        eeg.generate_stats()
        features = eeg.stats
        print(f"    ✓ Generated {features.shape[1]} features")
        
        # FIX: Ensure proper column structure and handle None column names
        if not isinstance(features.columns, pd.MultiIndex):
            # Replace None with empty string in column names
            clean_columns = [col if col is not None else '' for col in features.columns]
            features.columns = pd.MultiIndex.from_product([clean_columns, ['']])
            features.columns.names = ['band', 'channel']
        else:
            # Clean existing MultiIndex of None values
            new_columns = []
            for col in features.columns:
                if isinstance(col, tuple):
                    new_col = tuple(c if c is not None else '' for c in col)
                    new_columns.append(new_col)
                else:
                    new_columns.append(col if col is not None else '')
            features.columns = pd.MultiIndex.from_tuples(new_columns, names=['band', 'channel'])
        
        # Save features
        output_filename = f"{participant_id}_{session_id}_features.parquet"
        output_file = output_path / output_filename
        features.to_parquet(output_file, engine="fastparquet")
        print(f"    ✓ Saved to {output_filename}")
        
        return {
            'participant': participant_id,
            'session': session_id,
            'input_file': str(original_file),
            'output_file': str(output_file),
            'samples': num_samples,
            'channels': len(eeg_channels),
            'num_features': features.shape[1],
            'feature_types': list(features.index),
            'duration_seconds': float(time_seconds[-1]),
            'status': 'success'
        }
            
    except Exception as e:
        error_msg = f"{participant_id}_{session_id}: {str(e)}"
        print(f"    ✗ FAILED: {error_msg}")
        return {
            'file': f"{participant_id}_{session_id}_features.parquet",
            'error': str(e),
            'status': 'failed'
        }


def extract_features_from_wauc_files(input_dir, output_dir, original_data_dir, sampling_rate=500.0):
    """
    Extract statistical features from cleaned WAUC EEG files using the EEG class.
    
    FIXED VERSION: Loads ORIGINAL raw data instead of decomposed MultiIndex data
    to avoid double-processing (re-filtering and re-decomposing already processed data).
    
    Args:
        input_dir: Directory containing the cleaned EEG files (S{X}_session{Y}_eeg_raw.parquet)
        output_dir: Directory to save the extracted features
        original_data_dir: Directory containing ORIGINAL raw_eeg.parquet files from source
        sampling_rate: Sampling frequency in Hz (default 500.0 for WAUC)
    
    Returns:
        Dictionary with processing results and statistics
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    original_data_path = Path(original_data_dir)
    
    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return None
    
    # Validate original data directory
    if not original_data_path.exists():
        print(f"Error: Original data directory does not exist: {original_data_path}")
        return None
    
    # Create output directory
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_path}")
    except Exception as e:
        print(f"Error: Cannot create output directory {output_path}: {e}")
        return None
    
    # Find all ORIGINAL raw EEG files from source directory
    # Scan the original wauc_by_session directory to discover all test_eeg.parquet files
    print(f"Scanning original data directory: {original_data_path}")
    
    # Store as tuples (participant_num, session_num) - easily picklable for multiprocessing
    file_info_list = []
    participant_folders = sorted([d for d in original_data_path.iterdir() 
                                 if d.is_dir() and d.name.isdigit()])
    
    for participant_folder in participant_folders:
        participant_num = participant_folder.name
        session_folders = sorted([d for d in participant_folder.iterdir() 
                                 if d.is_dir() and d.name.startswith('session')])
        
        for session_folder in session_folders:
            session_num = session_folder.name.replace('session', '')
            test_file = session_folder / 'test_eeg.parquet'
            
            if test_file.exists():
                # Store simple tuple (easily picklable)
                file_info_list.append((participant_num, session_num))
    
    if not file_info_list:
        print(f"No test_eeg.parquet files found in {original_data_path}")
        return None
    
    print(f"{'='*60}")
    print(f"WAUC FEATURE EXTRACTION")
    print(f"{'='*60}")
    print(f"Found {len(file_info_list)} EEG files to process")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"{'='*60}\n")
    
    # Processing statistics
    processing_results = []
    failed_files_info = []
    
    # Set up parallel processing
    n_workers = min(cpu_count(), 8)  # Use up to 8 workers
    print(f"Using {n_workers} parallel workers\n")
    
    # Create partial function with fixed parameters
    process_func = partial(_process_single_file_features_from_tuple, 
                          output_path=output_path,
                          original_data_path=original_data_path,
                          sampling_rate=sampling_rate)
    
    # Process files in parallel
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, file_info_list),
            total=len(file_info_list),
            desc="Extracting features"
        ))
    
    # Aggregate results
    for result in results:
        if result['status'] == 'success':
            processing_results.append(result)
        else:
            failed_files_info.append(result)
    
    processed_files = len(processing_results)
    failed_files = len(failed_files_info)
    
    # Generate summary
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files found: {len(file_info_list)}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {failed_files}")
    print(f"Success rate: {processed_files/len(file_info_list)*100:.1f}%")
    
    if processing_results:
        # Feature statistics
        total_features = sum(r['num_features'] for r in processing_results)
        avg_features = total_features / len(processing_results)
        
        total_samples = sum(r['samples'] for r in processing_results)
        avg_duration = sum(r['duration_seconds'] for r in processing_results) / len(processing_results)
        
        print(f"\nFeature Statistics:")
        print(f"  Average features per file: {avg_features:.0f}")
        print(f"  Total samples processed: {total_samples:,}")
        print(f"  Average recording duration: {avg_duration:.1f} seconds")
        
        # Show feature types from first successful file
        first_success = processing_results[0]
        print(f"\nFeature types generated ({len(first_success['feature_types'])}):")
        for i, feat_type in enumerate(first_success['feature_types'][:10], 1):  # Show first 10
            print(f"  {i:2d}. {feat_type}")
        if len(first_success['feature_types']) > 10:
            print(f"  ... and {len(first_success['feature_types']) - 10} more")
    
    if failed_files_info:
        print(f"\nFailed Files:")
        for fail_info in failed_files_info[:5]:  # Show first 5 failures
            print(f"  - {Path(fail_info['file']).name}: {fail_info['error']}")
        if len(failed_files_info) > 5:
            print(f"  ... and {len(failed_files_info) - 5} more")
    
    # Save processing results
    results_summary = {
        'processing_stats': {
            'total_files': len(file_info_list),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'success_rate': processed_files/len(file_info_list)*100 if len(file_info_list) > 0 else 0
        },
        'successful_files': processing_results,
        'failed_files': failed_files_info,
        'output_directory': str(output_path),
        'sampling_rate': sampling_rate
    }
    
    # Save summary to JSON
    summary_file = output_path / 'feature_extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nProcessing summary saved to: {summary_file}")
    
    return results_summary


def create_target_files_from_ratings(ratings_file, raw_eeg_dir, features_dir, 
                                     target_column='tlx_mental', 
                                     create_for_raw=True, 
                                     create_for_features=True):
    """
    Create .txt target files from the WAUC ratings.parquet file.
    
    NOW CREATES MULTI-SUBSCALE TARGETS:
    - Creates 7 target files per recording (combined + 6 subscales)
    - Combined file contains AVERAGE of all 6 TLX subscales
    - Files: S{X}_session{Y}_eeg_raw.txt (combined TLX - average of 6 subscales)
             S{X}_session{Y}_eeg_raw_mental.txt
             S{X}_session{Y}_eeg_raw_physical.txt
             S{X}_session{Y}_eeg_raw_temporal.txt
             S{X}_session{Y}_eeg_raw_performance.txt
             S{X}_session{Y}_eeg_raw_effort.txt
             S{X}_session{Y}_eeg_raw_frustration.txt
    - For features: S{X}_session{Y}_features.txt (combined - average of 6 subscales)
                    S{X}_session{Y}_features_mental.txt
                    S{X}_session{Y}_features_physical.txt
                    ... (etc.)
    - Rescales from 1-21 scale to 0-100 scale: (value - 1) * 5
    
    Args:
        ratings_file: Path to the ratings.parquet file
        raw_eeg_dir: Directory containing raw EEG files (for time regression)
        features_dir: Directory containing feature files (for feature regression)
        target_column: DEPRECATED - Now always uses combined TLX (average of 6 subscales)
        create_for_raw: Whether to create targets for raw EEG files
        create_for_features: Whether to create targets for feature files
    
    Returns:
        Dictionary with processing results
    """
    
    ratings_path = Path(ratings_file)
    raw_path = Path(raw_eeg_dir) if create_for_raw else None
    features_path = Path(features_dir) if create_for_features else None
    
    # Validate inputs
    if not ratings_path.exists():
        print(f"Error: Ratings file does not exist: {ratings_path}")
        return None
    
    # Load ratings parquet
    try:
        ratings_df = pd.read_parquet(ratings_path)
        print(f"{'='*60}")
        print(f"CREATING MULTI-SUBSCALE TARGET FILES FROM RATINGS")
        print(f"{'='*60}")
        print(f"Loaded {ratings_df.shape[0]} ratings from {ratings_path.name}")
        print(f"Primary target column: '{target_column}'")
        print(f"Available columns: {list(ratings_df.columns)}")
    except Exception as e:
        print(f"Error loading ratings file: {e}")
        return None
    
    # Validate target column
    if target_column not in ratings_df.columns:
        print(f"Error: Target column '{target_column}' not found in ratings file")
        return None
    
    # Rescale TLX scores from 1-21 to 0-100 scale (matching MOCAS/HTC/N-Back)
    tlx_columns = ['tlx_mental', 'tlx_physical', 'tlx_temporal', 
                   'tlx_performance', 'tlx_effort', 'tlx_frustration']
    
    print(f"\nRescaling TLX scores from 1-21 scale to 0-100 scale...")
    for col in tlx_columns:
        if col in ratings_df.columns:
            original_mean = ratings_df[col].mean()
            # Rescale: (value - 1) / 20 * 100 = (value - 1) * 5
            ratings_df[col] = (ratings_df[col] - 1) * 5
            rescaled_mean = ratings_df[col].mean()
            print(f"  {col}: {original_mean:.2f} → {rescaled_mean:.2f}")
    
    # Calculate combined TLX score (average of all 6 subscales)
    ratings_df['tlx_combined'] = ratings_df[tlx_columns].mean(axis=1)
    
    # Print ratings summary for combined TLX
    print(f"\nCombined TLX statistics (average of 6 subscales, 0-100 scale):")
    print(f"  Mean: {ratings_df['tlx_combined'].mean():.2f}")
    print(f"  Std: {ratings_df['tlx_combined'].std():.2f}")
    print(f"  Min: {ratings_df['tlx_combined'].min():.2f}")
    print(f"  Max: {ratings_df['tlx_combined'].max():.2f}")
    print(f"  Missing: {ratings_df['tlx_combined'].isnull().sum()}")
    
    # Processing statistics
    created_raw = 0
    created_raw_subscales = {subscale: 0 for subscale in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']}
    created_features = 0
    created_features_subscales = {subscale: 0 for subscale in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']}
    missing_raw = 0
    missing_features = 0
    processing_results = []
    missing_info = []
    
    print(f"\n{'='*60}")
    print(f"Creating 7 target files per recording (combined + 6 subscales)...")
    
    # Process each rating entry
    for idx, row in ratings_df.iterrows():
        participant = int(row['participant'])
        session = int(row['session'])
        combined_tlx_value = row['tlx_combined']  # Use combined TLX (average of 6 subscales)
        
        # Format participant as zero-padded string (1 -> "01")
        participant_str = f"{participant:02d}"
        # Session in ratings is 1-6, but files use session0-session5
        session_str = f"{session - 1}"
        
        # Create base filename pattern
        base_filename = f"S{participant_str}_session{session_str}"
        
        # Track if we found files for this rating
        found_raw = False
        found_features = False
        
        # Create targets for raw EEG file (combined + 6 subscales)
        if create_for_raw and raw_path and raw_path.exists():
            raw_eeg_file = raw_path / f"{base_filename}_eeg_raw.parquet"
            
            if raw_eeg_file.exists():
                # Create combined target file (average of all 6 subscales)
                target_file = raw_path / f"{base_filename}_eeg_raw.txt"
                try:
                    with open(target_file, 'w') as f:
                        f.write(f"{combined_tlx_value:.2f}")
                    created_raw += 1
                    found_raw = True
                except Exception as e:
                    print(f"Error creating target for {base_filename}: {e}")
                
                # Create subscale target files
                for subscale in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']:
                    subscale_col = f'tlx_{subscale}'
                    if subscale_col in ratings_df.columns:
                        subscale_value = row[subscale_col]
                        subscale_target_file = raw_path / f"{base_filename}_eeg_raw_{subscale}.txt"
                        try:
                            with open(subscale_target_file, 'w') as f:
                                f.write(str(subscale_value))
                            created_raw_subscales[subscale] += 1
                        except Exception as e:
                            print(f"Error creating {subscale} target for {base_filename}: {e}")
            else:
                missing_raw += 1
        
        # Create targets for features file (combined + 6 subscales)
        if create_for_features and features_path and features_path.exists():
            features_file = features_path / f"{base_filename}_features.parquet"
            
            if features_file.exists():
                # Create combined target file (average of all 6 subscales)
                target_file = features_path / f"{base_filename}_features.txt"
                try:
                    with open(target_file, 'w') as f:
                        f.write(f"{combined_tlx_value:.2f}")
                    created_features += 1
                    found_features = True
                except Exception as e:
                    print(f"Error creating target for {base_filename}: {e}")
                
                # Create subscale target files
                for subscale in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']:
                    subscale_col = f'tlx_{subscale}'
                    if subscale_col in ratings_df.columns:
                        subscale_value = row[subscale_col]
                        subscale_target_file = features_path / f"{base_filename}_features_{subscale}.txt"
                        try:
                            with open(subscale_target_file, 'w') as f:
                                f.write(str(subscale_value))
                            created_features_subscales[subscale] += 1
                        except Exception as e:
                            print(f"Error creating {subscale} target for {base_filename}: {e}")
            else:
                missing_features += 1
        
        # Track results
        processing_results.append({
            'participant': participant_str,
            'session': session_str,
            'combined_tlx': combined_tlx_value,
            'found_raw': found_raw,
            'found_features': found_features
        })
        
        if not found_raw and not found_features:
            missing_info.append({
                'participant': participant_str,
                'session': session_str,
                'reason': 'No matching EEG or feature files found'
            })
    
    # Print summary
    print(f"TARGET FILE CREATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total ratings processed: {len(ratings_df)}")
    
    if create_for_raw:
        print(f"\nRaw EEG targets:")
        print(f"  Combined targets created: {created_raw}")
        print(f"  Subscale targets created:")
        for subscale, count in created_raw_subscales.items():
            print(f"    {subscale}: {count}")
        print(f"  Missing EEG files: {missing_raw}")
    
    if create_for_features:
        print(f"\nFeature targets:")
        print(f"  Combined targets created: {created_features}")
        print(f"  Subscale targets created:")
        for subscale, count in created_features_subscales.items():
            print(f"    {subscale}: {count}")
        print(f"  Missing feature files: {missing_features}")
    
    if missing_info:
        print(f"\nMissing data (first 10):")
        for info in missing_info[:10]:
            print(f"  - Participant {info['participant']}, Session {info['session']}: {info['reason']}")
        if len(missing_info) > 10:
            print(f"  ... and {len(missing_info) - 10} more")
    
    # Save processing results
    results_summary = {
        'target_column': target_column,
        'total_ratings': len(ratings_df),
        'created_raw_targets': created_raw,
        'created_feature_targets': created_features,
        'missing_raw_files': missing_raw,
        'missing_feature_files': missing_features,
        'processing_details': processing_results,
        'missing_files': missing_info
    }
    
    return results_summary


def create_metadata_based_classification_datasets(raw_eeg_dir, features_dir, output_base_path, ratings_file):
    """
    Create metadata-based classification datasets with tertile binning for all subscales.
    
    This function creates classification datasets matching the HTC/N-Back/MOCAS format:
    - Single data copy in 'all/' directory
    - classification_metadata.json with class labels for all subscales
    - SHARED tertile binning per TLX subscale (33rd and 67th percentiles)
      * Uses combined scores from BOTH time and feature datasets
      * Ensures consistent class labels across time and feature datasets
    - Pre-defined labels from ratings: mw_labels, pw_labels, mwpw_labels
    
    SUBSCALES (tertile binning):
    - combined: Primary target (from _target.txt files) - Classes: 0=low, 1=medium, 2=high
    - mental: Mental demand - Classes: 0=low, 1=medium, 2=high
    - physical: Physical demand - Classes: 0=low, 1=medium, 2=high
    - temporal: Temporal demand - Classes: 0=low, 1=medium, 2=high
    - performance: Performance - Classes: 0=low, 1=medium, 2=high
    - effort: Effort - Classes: 0=low, 1=medium, 2=high
    - frustration: Frustration - Classes: 0=low, 1=medium, 2=high
    
    PRE-DEFINED LABELS (from ratings):
    - mw_labels: Mental workload - Classes: 0=low, 1=high (binary)
    - pw_labels: Physical workload - Classes: 0=low, 1=medium, 2=high (3-class)
    - mwpw_labels: Combined MW+PW - Classes: 0-5 (6-class)
    
    OUTPUT STRUCTURE:
    wauc_time_classification/
        all/
            S01_session0_eeg_raw.parquet
            S01_session1_eeg_raw.parquet
            ...
        classification_metadata.json
    
    wauc_feature_classification/
        all/
            S01_session0_features.parquet
            S01_session1_features.parquet
            ...
        classification_metadata.json
    
    METADATA FORMAT:
    {
        "S01_session0_eeg_raw.parquet": {
            "combined": 1,
            "mental": 2,
            "physical": 0,
            ...
        },
        ...
    }
    
    Args:
        raw_eeg_dir: Directory containing S{X}_session{Y}_eeg_raw.parquet files with targets
        features_dir: Directory containing S{X}_session{Y}_features.parquet files with targets
        output_base_path: Base path for output (e.g., 'data/wauc')
        ratings_file: Path to ratings.parquet with mw_labels, pw_labels, mwpw_labels
    
    Returns:
        Dictionary with statistics about created datasets
    """
    
    raw_path = Path(raw_eeg_dir)
    features_path = Path(features_dir)
    base_path = Path(output_base_path)
    ratings_path = Path(ratings_file)
    
    print(f"{'='*60}")
    print(f"CREATING METADATA-BASED CLASSIFICATION DATASETS")
    print(f"{'='*60}")
    print(f"Input directories:")
    print(f"  Raw EEG: {raw_path}")
    print(f"  Features: {features_path}")
    print(f"  Ratings: {ratings_path}")
    print(f"Output base: {base_path}")
    
    # Load ratings for mw/pw/mwpw labels
    try:
        ratings_df = pd.read_parquet(ratings_path)
        print(f"\nLoaded {len(ratings_df)} ratings from {ratings_path.name}")
        print(f"Columns: {list(ratings_df.columns)}")
    except Exception as e:
        print(f"Error loading ratings file: {e}")
        return None
    
    # Define subscales (TLX dimensions with tertile binning)
    tlx_subscales = ['combined', 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']
    
    # Define pre-existing labels from ratings
    predefined_labels = ['mw_labels', 'pw_labels', 'mwpw_labels']
    
    # All subscales to include in metadata
    all_subscales = tlx_subscales + predefined_labels
    
    # Create output directories
    time_class_dir = base_path / "wauc_time_classification"
    feature_class_dir = base_path / "wauc_feature_classification"
    
    time_all_dir = time_class_dir / "all"
    feature_all_dir = feature_class_dir / "all"
    
    time_all_dir.mkdir(parents=True, exist_ok=True)
    feature_all_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreated output directories:")
    print(f"  {time_class_dir}")
    print(f"  {feature_class_dir}")
    
    # Step 1: Read all target files and organize scores by subscale
    print(f"\n{'='*60}")
    print(f"STEP 1: Reading target files and organizing scores")
    print(f"{'='*60}")
    
    # Dictionary to store scores for TLX subscales: {subscale: {filename: score}}
    time_scores = {subscale: {} for subscale in tlx_subscales}
    feature_scores = {subscale: {} for subscale in tlx_subscales}
    
    # Dictionary to store predefined labels: {subscale: {filename: label}}
    time_predefined = {label: {} for label in predefined_labels}
    feature_predefined = {label: {} for label in predefined_labels}
    
    # Process time series (raw EEG) files
    if raw_path.exists():
        eeg_files = sorted(raw_path.glob("*_eeg_raw.parquet"))
        print(f"Found {len(eeg_files)} EEG files")
        
        for eeg_file in eeg_files:
            base_name = eeg_file.stem.replace('_eeg_raw', '')
            
            # Parse participant and session from filename: S{XX}_session{Y}
            try:
                parts = base_name.split('_')
                participant_str = parts[0].replace('S', '')
                session_str = parts[1].replace('session', '')
                participant_num = int(participant_str)
                session_num = int(session_str)
                
                # Look up predefined labels from ratings (session in ratings is 1-indexed)
                rating_row = ratings_df[
                    (ratings_df['participant'] == participant_num) & 
                    (ratings_df['session'] == session_num + 1)
                ]
                
                if len(rating_row) > 0:
                    for label in predefined_labels:
                        if label in ratings_df.columns:
                            time_predefined[label][eeg_file.name] = int(rating_row[label].iloc[0])
            except Exception as e:
                print(f"Warning: Could not parse participant/session from {eeg_file.name}: {e}")
            
            # Read combined target
            combined_target = raw_path / f"{base_name}_eeg_raw.txt"
            if combined_target.exists():
                try:
                    with open(combined_target, 'r') as f:
                        score = float(f.read().strip())
                    time_scores['combined'][eeg_file.name] = score
                except Exception as e:
                    print(f"Error reading {combined_target.name}: {e}")
            
            # Read subscale targets
            for subscale in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']:
                subscale_target = raw_path / f"{base_name}_eeg_raw_{subscale}.txt"
                if subscale_target.exists():
                    try:
                        with open(subscale_target, 'r') as f:
                            score = float(f.read().strip())
                        time_scores[subscale][eeg_file.name] = score
                    except Exception as e:
                        print(f"Error reading {subscale_target.name}: {e}")
    
    # Process feature files
    if features_path.exists():
        feature_files = sorted(features_path.glob("*_features.parquet"))
        print(f"Found {len(feature_files)} feature files")
        
        for feature_file in feature_files:
            base_name = feature_file.stem.replace('_features', '')
            
            # Parse participant and session from filename: S{XX}_session{Y}
            try:
                parts = base_name.split('_')
                participant_str = parts[0].replace('S', '')
                session_str = parts[1].replace('session', '')
                participant_num = int(participant_str)
                session_num = int(session_str)
                
                # Look up predefined labels from ratings (session in ratings is 1-indexed)
                rating_row = ratings_df[
                    (ratings_df['participant'] == participant_num) & 
                    (ratings_df['session'] == session_num + 1)
                ]
                
                if len(rating_row) > 0:
                    for label in predefined_labels:
                        if label in ratings_df.columns:
                            feature_predefined[label][feature_file.name] = int(rating_row[label].iloc[0])
            except Exception as e:
                print(f"Warning: Could not parse participant/session from {feature_file.name}: {e}")
            
            # Read combined target
            combined_target = features_path / f"{base_name}_features.txt"
            if combined_target.exists():
                try:
                    with open(combined_target, 'r') as f:
                        score = float(f.read().strip())
                    feature_scores['combined'][feature_file.name] = score
                except Exception as e:
                    print(f"Error reading {combined_target.name}: {e}")
            
            # Read subscale targets
            for subscale in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']:
                subscale_target = features_path / f"{base_name}_features_{subscale}.txt"
                if subscale_target.exists():
                    try:
                        with open(subscale_target, 'r') as f:
                            score = float(f.read().strip())
                        feature_scores[subscale][feature_file.name] = score
                    except Exception as e:
                        print(f"Error reading {subscale_target.name}: {e}")
    
    print(f"\nTLX Scores collected (for tertile binning):")
    for subscale in tlx_subscales:
        print(f"  {subscale}: {len(time_scores[subscale])} time, {len(feature_scores[subscale])} feature")
    
    print(f"\nPredefined labels collected:")
    for label in predefined_labels:
        print(f"  {label}: {len(time_predefined[label])} time, {len(feature_predefined[label])} feature")
    
    # Step 2: Calculate SHARED tertile bins for each TLX subscale using ALL scores (time + feature)
    # This ensures consistent class labels across time and feature datasets
    print(f"\n{'='*60}")
    print(f"STEP 2: Calculating SHARED tertile bins for TLX subscales (33rd, 67th percentiles)")
    print(f"  Using combined scores from both time and feature datasets")
    print(f"{'='*60}")
    
    shared_bins = {}
    
    for subscale in tlx_subscales:
        # Combine ALL scores from both time and feature datasets
        all_scores = []
        
        if time_scores[subscale]:
            all_scores.extend(list(time_scores[subscale].values()))
        
        if feature_scores[subscale]:
            all_scores.extend(list(feature_scores[subscale].values()))
        
        if all_scores:
            scores_array = np.array(all_scores)
            p33 = np.percentile(scores_array, 33.33)
            p67 = np.percentile(scores_array, 66.67)
            shared_bins[subscale] = (p33, p67)
            
            print(f"\n{subscale} (shared bins for time + feature):")
            print(f"  Combined samples: {len(all_scores)} (time: {len(time_scores[subscale])}, feature: {len(feature_scores[subscale])})")
            print(f"  Scores: min={scores_array.min():.2f}, max={scores_array.max():.2f}, mean={scores_array.mean():.2f}")
            print(f"  Tertiles: p33={p33:.2f}, p67={p67:.2f}")
            print(f"  Bins: [0, {p33:.2f}), [{p33:.2f}, {p67:.2f}), [{p67:.2f}, inf)")
    
    # Step 3: Assign class labels using tertile bins for TLX, use predefined for MW/PW
    print(f"\n{'='*60}")
    print(f"STEP 3: Assigning class labels")
    print(f"  TLX subscales: tertile binning (0=low, 1=medium, 2=high)")
    print(f"  MW/PW labels: from ratings (mw_labels, pw_labels, mwpw_labels)")
    print(f"{'='*60}")
    
    def assign_class(score, bins):
        """Assign class label based on tertile bins"""
        p33, p67 = bins
        if score < p33:
            return 0  # low
        elif score < p67:
            return 1  # medium
        else:
            return 2  # high
    
    # Build metadata dictionaries
    time_metadata = {}
    feature_metadata = {}
    
    # Assign classes for time series
    all_time_files = set()
    for subscale in tlx_subscales:
        all_time_files.update(time_scores[subscale].keys())
    for label in predefined_labels:
        all_time_files.update(time_predefined[label].keys())
    
    for filename in sorted(all_time_files):
        time_metadata[filename] = {}
        
        # Add TLX subscale classes (tertile binning using SHARED bins)
        for subscale in tlx_subscales:
            if filename in time_scores[subscale] and subscale in shared_bins:
                score = time_scores[subscale][filename]
                class_label = assign_class(score, shared_bins[subscale])
                time_metadata[filename][subscale] = int(class_label)
        
        # Add predefined labels (from ratings)
        for label in predefined_labels:
            if filename in time_predefined[label]:
                time_metadata[filename][label] = time_predefined[label][filename]
    
    # Assign classes for features
    all_feature_files = set()
    for subscale in tlx_subscales:
        all_feature_files.update(feature_scores[subscale].keys())
    for label in predefined_labels:
        all_feature_files.update(feature_predefined[label].keys())
    
    for filename in sorted(all_feature_files):
        feature_metadata[filename] = {}
        
        # Add TLX subscale classes (tertile binning using SHARED bins)
        for subscale in tlx_subscales:
            if filename in feature_scores[subscale] and subscale in shared_bins:
                score = feature_scores[subscale][filename]
                class_label = assign_class(score, shared_bins[subscale])
                feature_metadata[filename][subscale] = int(class_label)
        
        # Add predefined labels (from ratings)
        for label in predefined_labels:
            if filename in feature_predefined[label]:
                feature_metadata[filename][label] = feature_predefined[label][filename]
    
    print(f"Assigned class labels:")
    print(f"  Time series: {len(time_metadata)} files")
    print(f"  Features: {len(feature_metadata)} files")
    
    # Step 4: Copy data files to all/ directories
    print(f"\n{'='*60}")
    print(f"STEP 4: Copying data files to all/ directories")
    print(f"{'='*60}")
    
    copied_time = 0
    copied_features = 0
    
    # Copy time series files
    for filename in time_metadata.keys():
        src = raw_path / filename
        dst = time_all_dir / filename
        if src.exists():
            try:
                shutil.copy2(src, dst)
                copied_time += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")
    
    # Copy feature files
    for filename in feature_metadata.keys():
        src = features_path / filename
        dst = feature_all_dir / filename
        if src.exists():
            try:
                shutil.copy2(src, dst)
                copied_features += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")
    
    print(f"Copied files:")
    print(f"  Time series: {copied_time}")
    print(f"  Features: {copied_features}")
    
    # Step 5: Save metadata JSON files
    print(f"\n{'='*60}")
    print(f"STEP 5: Saving classification metadata")
    print(f"{'='*60}")
    
    time_metadata_file = time_class_dir / "classification_metadata.json"
    feature_metadata_file = feature_class_dir / "classification_metadata.json"
    
    with open(time_metadata_file, 'w') as f:
        json.dump(time_metadata, f, indent=2)
    print(f"Saved: {time_metadata_file}")
    
    with open(feature_metadata_file, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"Saved: {feature_metadata_file}")
    
    # Step 6: Print class distribution statistics
    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION STATISTICS")
    print(f"{'='*60}")
    
    def print_distribution(metadata_dict, dataset_type):
        print(f"\n{dataset_type}:")
        
        # TLX subscales (3-class)
        for subscale in tlx_subscales:
            class_counts = {0: 0, 1: 0, 2: 0}
            for file_metadata in metadata_dict.values():
                if subscale in file_metadata:
                    class_label = file_metadata[subscale]
                    class_counts[class_label] += 1
            
            total = sum(class_counts.values())
            if total > 0:
                print(f"  {subscale} (TLX tertiles):")
                print(f"    Class 0 (low):    {class_counts[0]:3d} ({class_counts[0]/total*100:5.1f}%)")
                print(f"    Class 1 (medium): {class_counts[1]:3d} ({class_counts[1]/total*100:5.1f}%)")
                print(f"    Class 2 (high):   {class_counts[2]:3d} ({class_counts[2]/total*100:5.1f}%)")
                print(f"    Total:            {total:3d}")
        
        # Predefined labels (variable classes)
        for label in predefined_labels:
            class_counts = {}
            for file_metadata in metadata_dict.values():
                if label in file_metadata:
                    class_label = file_metadata[label]
                    class_counts[class_label] = class_counts.get(class_label, 0) + 1
            
            total = sum(class_counts.values())
            if total > 0:
                print(f"  {label} (from ratings):")
                for cls in sorted(class_counts.keys()):
                    print(f"    Class {cls}: {class_counts[cls]:3d} ({class_counts[cls]/total*100:5.1f}%)")
                print(f"    Total:   {total:3d}")
    
    print_distribution(time_metadata, "TIME SERIES CLASSIFICATION")
    print_distribution(feature_metadata, "FEATURE CLASSIFICATION")
    
    # Return summary
    return {
        'time_classification': {
            'directory': str(time_class_dir),
            'files_copied': copied_time,
            'metadata_file': str(time_metadata_file),
            'subscales': all_subscales
        },
        'feature_classification': {
            'directory': str(feature_class_dir),
            'files_copied': copied_features,
            'metadata_file': str(feature_metadata_file),
            'subscales': all_subscales
        }
    }


def organize_raw_eeg_by_classification(extracted_eeg_dir, ratings_file, output_base_dir, target_column='mw_labels'):
    """
    Organize extracted EEG files by classification labels.
    
    Creates classification datasets with class-specific subdirectories:
    - mw_labels -> wauc_time_classification_mw/{class_0, class_1}/
    - pw_labels -> wauc_time_classification_pw/{class_0, class_1, class_2}/
    - mwpw_labels -> wauc_time_classification_mwpw/{class_0...class_5}/
    
    Args:
        extracted_eeg_dir: Directory containing S{X}_session{Y}_eeg_raw.parquet files
        ratings_file: Path to ratings.parquet with classification labels
        output_base_dir: Base directory for classification datasets
        target_column: Classification target ('mw_labels', 'pw_labels', or 'mwpw_labels')
    
    Returns:
        Dictionary with processing statistics
    """
    
    extracted_path = Path(extracted_eeg_dir)
    ratings_path = Path(ratings_file)
    output_path = Path(output_base_dir)
    
    # Validate inputs
    if not extracted_path.exists():
        print(f"Error: Extracted EEG directory does not exist: {extracted_path}")
        return None
        
    if not ratings_path.exists():
        print(f"Error: Ratings file does not exist: {ratings_path}")
        return None
    
    # Load ratings
    try:
        ratings_df = pd.read_parquet(ratings_path)
    except Exception as e:
        print(f"Error loading ratings file: {e}")
        return None
    
    # Validate target column
    if target_column not in ratings_df.columns:
        print(f"Error: Target column '{target_column}' not found in ratings")
        print(f"Available columns: {list(ratings_df.columns)}")
        return None
    
    # Create dataset name based on target column
    dataset_name_map = {
        'mw_labels': 'wauc_time_classification_mw',
        'pw_labels': 'wauc_time_classification_pw',
        'mwpw_labels': 'wauc_time_classification_mwpw'
    }
    
    dataset_name = dataset_name_map.get(target_column, f'wauc_time_classification_{target_column}')
    dataset_path = output_path / dataset_name
    
    # Get unique classes
    unique_classes = sorted(ratings_df[target_column].unique())
    
    # Create class directories
    class_dirs = {}
    for class_label in unique_classes:
        class_dir = dataset_path / f'class_{class_label}'
        class_dir.mkdir(parents=True, exist_ok=True)
        class_dirs[class_label] = class_dir
    
    print(f"\n{'='*60}")
    print(f"ORGANIZING RAW EEG BY {target_column.upper()}")
    print(f"{'='*60}")
    print(f"Source: {extracted_path}")
    print(f"Output: {dataset_path}")
    print(f"Classes: {unique_classes}")
    print(f"{'='*60}\n")
    
    # Find all EEG files
    eeg_files = list(extracted_path.glob("S*_session*_eeg_raw.parquet"))
    
    if not eeg_files:
        print(f"No EEG files found in {extracted_path}")
        return None
    
    processed_files = 0
    missing_labels = 0
    class_counts = {cls: 0 for cls in unique_classes}
    processing_results = []
    
    # Process each file
    for eeg_file in eeg_files:
        try:
            # Parse filename: S{participant}_session{N}_eeg_raw.parquet
            filename = eeg_file.stem
            parts = filename.replace('_eeg_raw', '').split('_')
            
            if len(parts) < 2:
                print(f"Warning: Cannot parse filename {eeg_file.name}")
                continue
            
            participant_str = parts[0].replace('S', '')  # Remove 'S' prefix
            session_str = parts[1].replace('session', '')  # Remove 'session' prefix
            
            try:
                participant_num = int(participant_str)
                session_num = int(session_str)
            except ValueError:
                print(f"Warning: Invalid participant/session numbers in {eeg_file.name}")
                continue
            
            # Note: ratings.parquet uses session 1-6, but our files use session 0-5
            # So we need to add 1 to match
            session_for_lookup = session_num + 1
            
            # Look up classification label in ratings
            label_row = ratings_df[(ratings_df['participant'] == participant_num) & 
                                  (ratings_df['session'] == session_for_lookup)]
            
            if len(label_row) == 0:
                print(f"Warning: No label found for participant {participant_num}, session {session_for_lookup}")
                missing_labels += 1
                continue
            
            class_label = label_row[target_column].iloc[0]
            
            # Copy file to appropriate class directory
            output_file = class_dirs[class_label] / eeg_file.name
            
            try:
                shutil.copy2(eeg_file, output_file)
                
                class_counts[class_label] += 1
                processed_files += 1
                
                processing_results.append({
                    'participant': participant_num,
                    'session': session_num,
                    'class': int(class_label),
                    'original_path': str(eeg_file),
                    'new_path': str(output_file),
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"Error copying {eeg_file.name}: {e}")
                
        except Exception as e:
            print(f"Error processing {eeg_file.name}: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ORGANIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed files: {processed_files}")
    print(f"Missing labels: {missing_labels}")
    print(f"\nClass distribution:")
    
    
    for class_label in unique_classes:
        count = class_counts[class_label]
        percentage = (count / processed_files * 100) if processed_files > 0 else 0
        print(f"  Class {class_label}: {count} files ({percentage:.1f}%)")
    
    # Save results
    results_file = dataset_path / 'organization_summary.json'
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'target_column': target_column,
                'processed_files': processed_files,
                'missing_labels': missing_labels,
                'class_distribution': {int(k): v for k, v in class_counts.items()},
                'processed_files_details': processing_results
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Warning: Could not save results file: {e}")
    
    return {
        'processed_files': processed_files,
        'missing_labels': missing_labels,
        'class_distribution': class_counts,
        'dataset_path': str(dataset_path)
    }








def organize_features_by_classification(features_dir, ratings_file, output_base_dir, target_column='mw_labels'):
    """
    Organize extracted feature files by classification labels.
    
    Creates classification datasets with class-specific subdirectories:
    - mw_labels -> wauc_feature_classification_mw/{class_0, class_1}/
    - pw_labels -> wauc_feature_classification_pw/{class_0, class_1, class_2}/
    - mwpw_labels -> wauc_feature_classification_mwpw/{class_0...class_5}/
    
    Args:
        features_dir: Directory containing S{X}_session{Y}_features.parquet files
        ratings_file: Path to ratings.parquet with classification labels
        output_base_dir: Base directory for classification datasets
        target_column: Classification target ('mw_labels', 'pw_labels', or 'mwpw_labels')
    
    Returns:
        Dictionary with processing statistics
    """
    
    features_path = Path(features_dir)
    ratings_path = Path(ratings_file)
    output_path = Path(output_base_dir)
    
    # Validate inputs
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None
        
    if not ratings_path.exists():
        print(f"Error: Ratings file does not exist: {ratings_path}")
        return None
    
    # Load ratings
    try:
        ratings_df = pd.read_parquet(ratings_path)
    except Exception as e:
        print(f"Error loading ratings file: {e}")
        return None
    
    # Validate target column
    if target_column not in ratings_df.columns:
        print(f"Error: Target column '{target_column}' not found in ratings")
        print(f"Available columns: {list(ratings_df.columns)}")
        return None
    
    # Create dataset name based on target column
    dataset_name_map = {
        'mw_labels': 'wauc_feature_classification_mw',
        'pw_labels': 'wauc_feature_classification_pw',
        'mwpw_labels': 'wauc_feature_classification_mwpw'
    }
    
    dataset_name = dataset_name_map.get(target_column, f'wauc_feature_classification_{target_column}')
    dataset_path = output_path / dataset_name
    
    # Get unique classes
    unique_classes = sorted(ratings_df[target_column].unique())
    
    # Create class directories
    class_dirs = {}
    for class_label in unique_classes:
        class_dir = dataset_path / f'class_{class_label}'
        class_dir.mkdir(parents=True, exist_ok=True)
        class_dirs[class_label] = class_dir
    
    print(f"\n{'='*60}")
    print(f"ORGANIZING FEATURES BY {target_column.upper()}")
    print(f"{'='*60}")
    print(f"Source: {features_path}")
    print(f"Output: {dataset_path}")
    print(f"Classes: {unique_classes}")
    print(f"{'='*60}\n")
    
    # Find all feature files
    feature_files = list(features_path.glob("S*_session*_features.parquet"))
    
    if not feature_files:
        print(f"No feature files found in {features_path}")
        return None
    
    processed_files = 0
    missing_labels = 0
    class_counts = {cls: 0 for cls in unique_classes}
    processing_results = []
    
    # Process each file
    for feature_file in feature_files:
        try:
            # Parse filename: S{participant}_session{N}_features.parquet
            filename = feature_file.stem
            parts = filename.replace('_features', '').split('_')
            
            if len(parts) < 2:
                print(f"Warning: Cannot parse filename {feature_file.name}")
                continue
            
            participant_str = parts[0].replace('S', '')  # Remove 'S' prefix
            session_str = parts[1].replace('session', '')  # Remove 'session' prefix
            
            try:
                participant_num = int(participant_str)
                session_num = int(session_str)
            except ValueError:
                print(f"Warning: Invalid participant/session numbers in {feature_file.name}")
                continue
            
            # Note: ratings.parquet uses session 1-6, but our files use session 0-5
            session_for_lookup = session_num + 1
            
            # Look up classification label in ratings
            label_row = ratings_df[(ratings_df['participant'] == participant_num) & 
                                  (ratings_df['session'] == session_for_lookup)]
            
            if len(label_row) == 0:
                print(f"Warning: No label found for participant {participant_num}, session {session_for_lookup}")
                missing_labels += 1
                continue
            
            class_label = label_row[target_column].iloc[0]
            
            # Copy file to appropriate class directory
            output_file = class_dirs[class_label] / feature_file.name
            
            try:
                shutil.copy2(feature_file, output_file)
                
                class_counts[class_label] += 1
                processed_files += 1
                
                processing_results.append({
                    'participant': participant_num,
                    'session': session_num,
                    'class': int(class_label),
                    'original_path': str(feature_file),
                    'new_path': str(output_file),
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"Error copying {feature_file.name}: {e}")
                
        except Exception as e:
            print(f"Error processing {feature_file.name}: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ORGANIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed files: {processed_files}")
    print(f"Missing labels: {missing_labels}")
    print(f"\nClass distribution:")
    for class_label in unique_classes:
        count = class_counts[class_label]
        percentage = (count / processed_files * 100) if processed_files > 0 else 0
        print(f"  Class {class_label}: {count} files ({percentage:.1f}%)")
    
    # Save results
    results_file = dataset_path / 'organization_summary.json'
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'target_column': target_column,
                'processed_files': processed_files,
                'missing_labels': missing_labels,
                'class_distribution': {int(k): v for k, v in class_counts.items()},
                'processed_files_details': processing_results
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save results file: {e}")
    
    return {
        'processed_files': processed_files,
        'missing_labels': missing_labels,
        'class_distribution': class_counts,
        'dataset_path': str(dataset_path)
    }


def cleanup_files_without_targets(raw_dir, features_dir):
    """
    Remove EEG and feature files that don't have corresponding target files.
    
    Args:
        raw_dir: Directory containing raw EEG files
        features_dir: Directory containing feature files
    
    Returns:
        Dictionary with cleanup statistics
    """
    
    raw_path = Path(raw_dir)
    features_path = Path(features_dir)
    
    print(f"\n{'='*60}")
    print(f"CLEANING UP FILES WITHOUT TARGETS")
    print(f"{'='*60}")
    
    removed_raw = 0
    removed_features = 0
    
    # Cleanup raw EEG files
    if raw_path.exists():
        raw_files = list(raw_path.glob("S*_session*_eeg_raw.parquet"))
        print(f"\nChecking {len(raw_files)} raw EEG files...")
        
        for raw_file in raw_files:
            # Target file is named: S01_session0_eeg_raw.txt (matching the data file)
            base_name = raw_file.stem
            target_file = raw_file.parent / f"{base_name}.txt"
            
            if not target_file.exists():
                try:
                    raw_file.unlink()
                    removed_raw += 1
                    print(f"  Removed: {raw_file.name} (no target)")
                except Exception as e:
                    print(f"  Error removing {raw_file.name}: {e}")
    
    # Cleanup feature files
    if features_path.exists():
        feature_files = list(features_path.glob("S*_session*_features.parquet"))
        print(f"\nChecking {len(feature_files)} feature files...")
        
        for feature_file in feature_files:
            # Target file is named: S01_session0_features.txt (matching the data file)
            base_name = feature_file.stem
            target_file = feature_file.parent / f"{base_name}.txt"
            
            if not target_file.exists():
                try:
                    feature_file.unlink()
                    removed_features += 1
                    print(f"  Removed: {feature_file.name} (no target)")
                except Exception as e:
                    print(f"  Error removing {feature_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"Removed raw EEG files: {removed_raw}")
    print(f"Removed feature files: {removed_features}")
    
    return {
        'removed_raw': removed_raw,
        'removed_features': removed_features
    }


def copy_to_final_regression_datasets(raw_source, features_source, 
                                      time_regression_dest, feature_regression_dest):
    """
    Copy paired EEG/feature files with targets to final regression dataset directories.
    
    Args:
        raw_source: Source directory with raw EEG files and targets
        features_source: Source directory with feature files and targets
        time_regression_dest: Destination for time regression dataset
        feature_regression_dest: Destination for feature regression dataset
    
    Returns:
        Dictionary with copy statistics
    """
    
    raw_src = Path(raw_source)
    features_src = Path(features_source)
    time_dest = Path(time_regression_dest)
    feature_dest = Path(feature_regression_dest)
    
    print(f"\n{'='*60}")
    print(f"COPYING TO FINAL REGRESSION DATASETS")
    print(f"{'='*60}")
    
    # Create destination directories
    time_dest.mkdir(parents=True, exist_ok=True)
    feature_dest.mkdir(parents=True, exist_ok=True)
    
    copied_raw = 0
    copied_features = 0
    skipped_raw = 0
    skipped_features = 0
    
    # Copy raw EEG files with targets
    if raw_src.exists():
        raw_pairs = list(raw_src.glob("S*_session*_eeg_raw.parquet"))
        print(f"\nChecking {len(raw_pairs)} raw EEG files...")
        
        for raw_file in raw_pairs:
            # Target file is named: S01_session0_eeg_raw.txt (matching the data file)
            base_name = raw_file.stem
            target_file = raw_file.parent / f"{base_name}.txt"
            
            if target_file.exists():
                try:
                    # Copy data file
                    shutil.copy2(raw_file, time_dest / raw_file.name)
                    
                    # Copy combined target file
                    shutil.copy2(target_file, time_dest / target_file.name)
                    
                    # Copy all subscale target files
                    for subscale in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']:
                        subscale_target = raw_file.parent / f"{base_name}_{subscale}.txt"
                        if subscale_target.exists():
                            shutil.copy2(subscale_target, time_dest / subscale_target.name)
                    
                    copied_raw += 1
                except Exception as e:
                    print(f"  Error copying {raw_file.name}: {e}")
            else:
                skipped_raw += 1
                print(f"  Skipped {raw_file.name} (no target file)")
    
    # Copy feature files with targets
    if features_src.exists():
        feature_pairs = list(features_src.glob("S*_session*_features.parquet"))
        print(f"\nChecking {len(feature_pairs)} feature files...")
        
        for feature_file in feature_pairs:
            # Target file is named: S01_session0_features.txt (matching the data file)
            base_name = feature_file.stem
            target_file = feature_file.parent / f"{base_name}.txt"
            
            if target_file.exists():
                try:
                    # Copy data file
                    shutil.copy2(feature_file, feature_dest / feature_file.name)
                    
                    # Copy combined target file
                    shutil.copy2(target_file, feature_dest / target_file.name)
                    
                    # Copy all subscale target files
                    for subscale in ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']:
                        subscale_target = feature_file.parent / f"{base_name}_{subscale}.txt"
                        if subscale_target.exists():
                            shutil.copy2(subscale_target, feature_dest / subscale_target.name)
                    
                    copied_features += 1
                except Exception as e:
                    print(f"  Error copying {feature_file.name}: {e}")
            else:
                skipped_features += 1
                print(f"  Skipped {feature_file.name} (no target file)")
    
    print(f"\n{'='*60}")
    print(f"COPY SUMMARY")
    print(f"{'='*60}")
    print(f"Time regression:")
    print(f"  Copied: {copied_raw}")
    print(f"  Skipped (no target): {skipped_raw}")
    print(f"  Destination: {time_dest}")
    print(f"Feature regression:")
    print(f"  Copied: {copied_features}")
    print(f"  Skipped (no target): {skipped_features}")
    print(f"  Destination: {feature_dest}")
    
    return {
        'copied_raw': copied_raw,
        'copied_features': copied_features,
        'skipped_raw': skipped_raw,
        'skipped_features': skipped_features,
        'time_regression_path': str(time_dest),
        'feature_regression_path': str(feature_dest)
    }


def organize_files_by_classification(raw_dir, features_dir, ratings_file, output_base_dir, 
                                    classification_column='mw_labels',
                                    class_names=None):
    """
    Organize raw EEG and feature files by classification labels.
    
    Creates classification datasets with files organized into class-specific folders,
    following the MOCAS dataset structure convention.
    
    Args:
        raw_dir: Directory containing raw EEG files (S{X}_session{Y}_eeg_raw.parquet)
        features_dir: Directory containing feature files (S{X}_session{Y}_features.parquet)
        ratings_file: Path to ratings.parquet with classification labels
        output_base_dir: Base directory for classification datasets
        classification_column: Column name for classification labels
                              Options: 'mw_labels' (binary: 0/1)
                                      'pw_labels' (3-class: 0/1/2)
                                      'mwpw_labels' (6-class: 0-5)
        class_names: Dictionary mapping class indices to names (optional)
                    If None, uses numeric labels
    
    Returns:
        Dictionary with processing statistics
    """
    
    raw_path = Path(raw_dir)
    features_path = Path(features_dir)
    ratings_path = Path(ratings_file)
    
    # Validate inputs
    if not raw_path.exists():
        print(f"Error: Raw directory does not exist: {raw_path}")
        return None
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None
    if not ratings_path.exists():
        print(f"Error: Ratings file does not exist: {ratings_path}")
        return None
    
    # Load ratings
    try:
        ratings_df = pd.read_parquet(ratings_path)
    except Exception as e:
        print(f"Error loading ratings file: {e}")
        return None
    
    # Validate classification column
    if classification_column not in ratings_df.columns:
        print(f"Error: Classification column '{classification_column}' not found in ratings")
        print(f"Available columns: {list(ratings_df.columns)}")
        return None
    
    # Get unique classes
    unique_classes = sorted(ratings_df[classification_column].unique())
    
    # Create output directories based on classification type
    if classification_column == 'mw_labels':
        time_output = Path(output_base_dir) / 'wauc_time_mw_classification_dataset'
        feature_output = Path(output_base_dir) / 'wauc_feature_mw_classification_dataset'
        default_names = {0: 'low_mw', 1: 'high_mw'}
    elif classification_column == 'pw_labels':
        time_output = Path(output_base_dir) / 'wauc_time_pw_classification_dataset'
        feature_output = Path(output_base_dir) / 'wauc_feature_pw_classification_dataset'
        default_names = {0: 'low_pw', 1: 'medium_pw', 2: 'high_pw'}
    elif classification_column == 'mwpw_labels':
        time_output = Path(output_base_dir) / 'wauc_time_mwpw_classification_dataset'
        feature_output = Path(output_base_dir) / 'wauc_feature_mwpw_classification_dataset'
        default_names = {0: 'low_mw_low_pw', 1: 'high_mw_low_pw', 2: 'low_mw_medium_pw',
                        3: 'high_mw_medium_pw', 4: 'low_mw_high_pw', 5: 'high_mw_high_pw'}
    else:
        time_output = Path(output_base_dir) / f'wauc_time_{classification_column}_classification_dataset'
        feature_output = Path(output_base_dir) / f'wauc_feature_{classification_column}_classification_dataset'
        default_names = {cls: f'class_{cls}' for cls in unique_classes}
    
    class_names = class_names or default_names
    
    # Create class directories
    time_class_dirs = {}
    feature_class_dirs = {}
    
    for cls in unique_classes:
        class_name = class_names.get(cls, f'class_{cls}')
        time_class_dirs[cls] = time_output / class_name
        feature_class_dirs[cls] = feature_output / class_name
        
        time_class_dirs[cls].mkdir(parents=True, exist_ok=True)
        feature_class_dirs[cls].mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ORGANIZING FILES BY CLASSIFICATION: {classification_column}")
    print(f"{'='*60}")
    print(f"Classes: {unique_classes}")
    print(f"Class names: {class_names}")
    print(f"Time output: {time_output}")
    print(f"Feature output: {feature_output}")
    
    # Processing statistics
    processed_raw = 0
    processed_features = 0
    missing_raw = 0
    missing_features = 0
    missing_labels = 0
    # Convert numpy/pandas int64 to native Python int for JSON serialization
    class_counts = {int(cls): {'raw': 0, 'features': 0} for cls in unique_classes}
    
    # Find all raw and feature files
    raw_files = list(raw_path.glob("S*_session*_eeg_raw.parquet"))
    feature_files = list(features_path.glob("S*_session*_features.parquet"))
    
    print(f"\nFound {len(raw_files)} raw files and {len(feature_files)} feature files")
    
    # Process raw files
    print("\nProcessing raw EEG files...")
    for raw_file in tqdm(raw_files, desc="Organizing raw files"):
        try:
            # Parse filename: S{participant}_session{session}_eeg_raw.parquet
            filename = raw_file.stem
            parts = filename.replace('_eeg_raw', '').split('_')
            
            if len(parts) >= 2:
                participant_str = parts[0].replace('S', '')
                session_str = parts[1].replace('session', '')
                
                participant_num = int(participant_str)
                session_num = int(session_str)
                
                # Look up classification label (session numbers in ratings are 1-indexed)
                label_row = ratings_df[
                    (ratings_df['participant'] == participant_num) & 
                    (ratings_df['session'] == session_num + 1)
                ]
                
                if len(label_row) == 0:
                    missing_labels += 1
                    continue
                
                class_label = label_row[classification_column].iloc[0]
                
                # Copy to appropriate class directory
                dest_file = time_class_dirs[class_label] / raw_file.name
                shutil.copy2(raw_file, dest_file)
                
                class_counts[class_label]['raw'] += 1
                processed_raw += 1
                
        except Exception as e:
            print(f"  Error processing {raw_file.name}: {e}")
            missing_raw += 1
    
    # Process feature files
    print("\nProcessing feature files...")
    for feature_file in tqdm(feature_files, desc="Organizing feature files"):
        try:
            # Parse filename: S{participant}_session{session}_features.parquet
            filename = feature_file.stem
            parts = filename.replace('_features', '').split('_')
            
            if len(parts) >= 2:
                participant_str = parts[0].replace('S', '')
                session_str = parts[1].replace('session', '')
                
                participant_num = int(participant_str)
                session_num = int(session_str)
                
                # Look up classification label (session numbers in ratings are 1-indexed)
                label_row = ratings_df[
                    (ratings_df['participant'] == participant_num) & 
                    (ratings_df['session'] == session_num + 1)
                ]
                
                if len(label_row) == 0:
                    missing_labels += 1
                    continue
                
                class_label = label_row[classification_column].iloc[0]
                
                # Copy to appropriate class directory
                dest_file = feature_class_dirs[class_label] / feature_file.name
                shutil.copy2(feature_file, dest_file)
                
                class_counts[class_label]['features'] += 1
                processed_features += 1
                
        except Exception as e:
            print(f"  Error processing {feature_file.name}: {e}")
            missing_features += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION ORGANIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Classification: {classification_column}")
    print(f"Processed raw files: {processed_raw}")
    print(f"Processed feature files: {processed_features}")
    print(f"Missing raw: {missing_raw}")
    print(f"Missing features: {missing_features}")
    print(f"Missing labels: {missing_labels}")
    
    print(f"\nClass distribution:")
    for cls in unique_classes:
        class_name = class_names.get(cls, f'class_{cls}')
        raw_count = class_counts[cls]['raw']
        feat_count = class_counts[cls]['features']
        print(f"  {class_name} (label={cls}):")
        print(f"    Raw files: {raw_count}")
        print(f"    Feature files: {feat_count}")
    
    # Save processing results
    results = {
        'classification_column': classification_column,
        'processed_raw': processed_raw,
        'processed_features': processed_features,
        'missing_raw': missing_raw,
        'missing_features': missing_features,
        'missing_labels': missing_labels,
        'class_counts': class_counts,
        'time_output_path': str(time_output),
        'feature_output_path': str(feature_output)
    }
    
    results_file = time_output.parent / f'{classification_column}_organization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def organize_files_by_tlx_bins(raw_dir, features_dir, ratings_file, output_base_dir,
                                target_column='tlx_mental', low_threshold=33, high_threshold=67):
    """
    Organize raw EEG and feature files into 3 bins based on TLX scores.
    
    Bins TLX scores into low/medium/high categories for 3-class classification.
    This is similar to MOCAS TLX binning but adapted for WAUC's ratings structure.
    
    NOTE: TLX scores are rescaled from original 1-21 scale to 0-100 scale for consistency
    with MOCAS, HTC, and N-Back datasets.
    
    TLX Bins (0-100 scale after rescaling):
        - Low: TLX < low_threshold (default: < 33)
        - Medium: low_threshold <= TLX < high_threshold (default: 33-67)
        - High: TLX >= high_threshold (default: >= 67)
    
    Args:
        raw_dir: Directory containing raw EEG files (S{X}_session{Y}_eeg_raw.parquet)
        features_dir: Directory containing feature files (S{X}_session{Y}_features.parquet)
        ratings_file: Path to ratings.parquet with TLX scores (rescaled to 0-100)
        output_base_dir: Base directory for output
        target_column: TLX column to use for binning (default: 'tlx_mental')
                      Options: 'tlx_mental', 'tlx_physical', 'tlx_temporal',
                              'tlx_performance', 'tlx_effort', 'tlx_frustration'
        low_threshold: Upper boundary for low bin (default: 33)
        high_threshold: Lower boundary for high bin (default: 67)
    
    Returns:
        Dictionary with processing statistics
    """
    
    raw_path = Path(raw_dir)
    features_path = Path(features_dir)
    ratings_path = Path(ratings_file)
    
    # Create output directories
    time_output = Path(output_base_dir) / 'wauc_time_tlx_classification_dataset'
    feature_output = Path(output_base_dir) / 'wauc_feature_tlx_classification_dataset'
    
    # Validate inputs
    if not raw_path.exists():
        print(f"Error: Raw directory does not exist: {raw_path}")
        return None
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None
    if not ratings_path.exists():
        print(f"Error: Ratings file does not exist: {ratings_path}")
        return None
    
    # Load ratings
    try:
        ratings_df = pd.read_parquet(ratings_path)
        print(f"Loaded {len(ratings_df)} ratings from {ratings_path.name}")
    except Exception as e:
        print(f"Error loading ratings file: {e}")
        return None
    
    # Validate target column
    if target_column not in ratings_df.columns:
        print(f"Error: Target column '{target_column}' not found in ratings")
        print(f"Available columns: {list(ratings_df.columns)}")
        return None
    
    # Rescale TLX scores from 1-21 to 0-100 scale (matching MOCAS/HTC/N-Back)
    tlx_columns = ['tlx_mental', 'tlx_physical', 'tlx_temporal', 
                   'tlx_performance', 'tlx_effort', 'tlx_frustration']
    
    print(f"\nRescaling TLX scores from 1-21 scale to 0-100 scale...")
    for col in tlx_columns:
        if col in ratings_df.columns:
            # Rescale: (value - 1) / 20 * 100 = (value - 1) * 5
            ratings_df[col] = (ratings_df[col] - 1) * 5
    print(f"  {target_column}: min={ratings_df[target_column].min():.1f}, max={ratings_df[target_column].max():.1f}, mean={ratings_df[target_column].mean():.1f}")
    
    # Create bin directories
    time_bins = {
        'low': time_output / 'low',
        'medium': time_output / 'medium',
        'high': time_output / 'high'
    }
    
    feature_bins = {
        'low': feature_output / 'low',
        'medium': feature_output / 'medium',
        'high': feature_output / 'high'
    }
    
    for bin_dirs in [time_bins, feature_bins]:
        for bin_dir in bin_dirs.values():
            bin_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ORGANIZING FILES BY TLX BINS")
    print(f"{'='*60}")
    print(f"Target column: {target_column}")
    print(f"Binning thresholds:")
    print(f"  Low: {target_column} < {low_threshold}")
    print(f"  Medium: {low_threshold} <= {target_column} < {high_threshold}")
    print(f"  High: {target_column} >= {high_threshold}")
    
    # Statistics
    stats = {
        'raw_processed': 0,
        'feature_processed': 0,
        'raw_bins': {'low': 0, 'medium': 0, 'high': 0},
        'feature_bins': {'low': 0, 'medium': 0, 'high': 0},
        'missing_files': 0,
        'missing_scores': 0
    }
    
    # Find all raw and feature files
    raw_files = list(raw_path.glob("S*_session*_eeg_raw.parquet"))
    feature_files = list(features_path.glob("S*_session*_features.parquet"))
    
    print(f"\nFound {len(raw_files)} raw files and {len(feature_files)} feature files")
    
    # Process raw files
    print("\nProcessing raw EEG files...")
    for raw_file in tqdm(raw_files, desc="Binning raw files"):
        try:
            # Parse filename: S{participant}_session{session}_eeg_raw.parquet
            filename = raw_file.stem
            parts = filename.replace('_eeg_raw', '').split('_')
            
            if len(parts) >= 2:
                participant_str = parts[0].replace('S', '')
                session_str = parts[1].replace('session', '')
                
                participant_num = int(participant_str)
                session_num = int(session_str)
                
                # Look up TLX score (session numbers in ratings are 1-indexed)
                score_row = ratings_df[
                    (ratings_df['participant'] == participant_num) & 
                    (ratings_df['session'] == session_num + 1)
                ]
                
                if len(score_row) == 0:
                    stats['missing_scores'] += 1
                    continue
                
                tlx_score = score_row[target_column].iloc[0]
                
                # Determine bin
                if tlx_score < low_threshold:
                    bin_class = 'low'
                elif tlx_score < high_threshold:
                    bin_class = 'medium'
                else:
                    bin_class = 'high'
                
                # Copy to appropriate bin directory
                dest_file = time_bins[bin_class] / raw_file.name
                shutil.copy2(raw_file, dest_file)
                
                stats['raw_processed'] += 1
                stats['raw_bins'][bin_class] += 1
                
        except Exception as e:
            print(f"  Error processing {raw_file.name}: {e}")
            stats['missing_files'] += 1
    
    # Process feature files
    print("\nProcessing feature files...")
    for feature_file in tqdm(feature_files, desc="Binning feature files"):
        try:
            # Parse filename: S{participant}_session{session}_features.parquet
            filename = feature_file.stem
            parts = filename.replace('_features', '').split('_')
            
            if len(parts) >= 2:
                participant_str = parts[0].replace('S', '')
                session_str = parts[1].replace('session', '')
                
                participant_num = int(participant_str)
                session_num = int(session_str)
                
                # Look up TLX score (session numbers in ratings are 1-indexed)
                score_row = ratings_df[
                    (ratings_df['participant'] == participant_num) & 
                    (ratings_df['session'] == session_num + 1)
                ]
                
                if len(score_row) == 0:
                    stats['missing_scores'] += 1
                    continue
                
                tlx_score = score_row[target_column].iloc[0]
                
                # Determine bin
                if tlx_score < low_threshold:
                    bin_class = 'low'
                elif tlx_score < high_threshold:
                    bin_class = 'medium'
                else:
                    bin_class = 'high'
                
                # Copy to appropriate bin directory
                dest_file = feature_bins[bin_class] / feature_file.name
                shutil.copy2(feature_file, dest_file)
                
                stats['feature_processed'] += 1
                stats['feature_bins'][bin_class] += 1
                
        except Exception as e:
            print(f"  Error processing {feature_file.name}: {e}")
            stats['missing_files'] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TLX BINNING SUMMARY")
    print(f"{'='*60}")
    print(f"Target column: {target_column}")
    print(f"\nRaw EEG files organized: {stats['raw_processed']}")
    print(f"  Low (< {low_threshold}): {stats['raw_bins']['low']} files")
    print(f"  Medium ({low_threshold}-{high_threshold}): {stats['raw_bins']['medium']} files")
    print(f"  High (>= {high_threshold}): {stats['raw_bins']['high']} files")
    print(f"\nFeature files organized: {stats['feature_processed']}")
    print(f"  Low (< {low_threshold}): {stats['feature_bins']['low']} files")
    print(f"  Medium ({low_threshold}-{high_threshold}): {stats['feature_bins']['medium']} files")
    print(f"  High (>= {high_threshold}): {stats['feature_bins']['high']} files")
    print(f"\nMissing scores: {stats['missing_scores']}")
    print(f"Missing files: {stats['missing_files']}")
    
    # Save results
    results = {
        'binning_method': 'tlx_based',
        'target_column': target_column,
        'thresholds': {
            'low': f'< {low_threshold}',
            'medium': f'{low_threshold} - {high_threshold}',
            'high': f'>= {high_threshold}'
        },
        'statistics': stats,
        'output_directories': {
            'time': str(time_output),
            'feature': str(feature_output)
        }
    }
    
    results_file = time_output.parent / 'tlx_binning_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


# =============================================================================
# BASELINE EEG EXTRACTION AND REGRESSION DATASETS
# =============================================================================

# Output paths for baseline datasets
base1_output_path = os.path.join(output_base_path, "base1_eeg_extracted")
base2_output_path = os.path.join(output_base_path, "base2_eeg_extracted")
base1_regression_path = os.path.join(output_base_path, "wauc_base1_time_regression")
base2_regression_path = os.path.join(output_base_path, "wauc_base2_time_regression")


def extract_baseline_eeg(source_dir, output_dir, baseline_type='base1', sampling_rate=500.0, decompose=True):
    """
    Extract and reformat baseline EEG files from WAUC dataset.
    
    Processes baseline EEG data (pre-task recordings) with subband decomposition.
    These are paired with session TLX scores to test if pre-task baseline EEG
    can predict workload experienced during the subsequent task.
    
    Args:
        source_dir (str): Path to wauc_by_session directory
        output_dir (str): Path to output directory for reformatted files
        baseline_type (str): Which baseline to extract - 'base1' or 'base2'
        sampling_rate (float): EEG sampling frequency in Hz (default 500.0)
        decompose (bool): Whether to perform subband decomposition (default True)

    Returns:
        List of dictionaries with information about each processed file
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if baseline_type not in ['base1', 'base2']:
        print(f"Error: baseline_type must be 'base1' or 'base2', got '{baseline_type}'")
        return None
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_path}")
        return None
        
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_path}: {e}")
        return None

    print(f"{'='*60}")
    print(f"WAUC BASELINE EEG EXTRACTION - {baseline_type.upper()}")
    print(f"{'='*60}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Baseline type: {baseline_type}")
    print(f"Sampling rate: {sampling_rate} Hz")

    processed_files = 0
    failed_files = 0
    file_info = []
    failed_info = []
    
    # Get all participant folders (01, 02, ..., 48)
    participant_folders = sorted([d for d in source_path.iterdir() 
                                 if d.is_dir() and d.name.isdigit()])
    
    print(f"\nFound {len(participant_folders)} participant folders")
    print(f"{'='*60}\n")
    
    for participant_folder in participant_folders:
        participant_num = participant_folder.name  # "01", "02", etc.
        print(f"Processing Participant {participant_num}")
        
        # Get all session folders (session0, session1, ..., session5)
        session_folders = sorted([d for d in participant_folder.iterdir() 
                                 if d.is_dir() and d.name.startswith('session')])
        
        print(f"  Found {len(session_folders)} session folders")
        
        for session_folder in session_folders:
            session_num = session_folder.name.replace('session', '')  # "0", "1", etc.
            
            try:
                # Load the baseline file
                baseline_file = session_folder / f'{baseline_type}_eeg.parquet'
                
                # Check if baseline file exists
                if not baseline_file.exists():
                    error_msg = f"Missing {baseline_type}_eeg.parquet in {session_folder.name}"
                    print(f"  WARNING: {error_msg}")
                    failed_info.append({
                        'participant': participant_num,
                        'session': session_num,
                        'error': error_msg,
                        'missing_files': [f'{baseline_type}_eeg.parquet']
                    })
                    failed_files += 1
                    continue
                
                # Load baseline data
                df_baseline = pd.read_parquet(baseline_file)
                
                # EEG channel columns (exclude metadata columns)
                eeg_channels = ['AF8', 'FP2', 'FP1', 'AF7', 'T10', 'T9', 'P4', 'P3']
                
                # Check if all channels exist
                missing_channels = [ch for ch in eeg_channels if ch not in df_baseline.columns]
                if missing_channels:
                    error_msg = f"Missing EEG channels: {missing_channels}"
                    print(f"  WARNING: {error_msg}")
                    failed_info.append({
                        'participant': participant_num,
                        'session': session_num,
                        'error': error_msg
                    })
                    failed_files += 1
                    continue
                
                # Extract only EEG channels
                df_eeg = df_baseline[eeg_channels].copy()
                
                # Create time array
                num_samples = len(df_eeg)
                time_seconds = np.arange(num_samples) / sampling_rate
                sample_numbers = np.arange(num_samples)
                
                # Prepare channel data dictionary
                channels_dict = {col: df_eeg[col].values for col in eeg_channels}
                
                # Create EEG instance to decompose into subbands
                eeg = EEG(
                    s_n=sample_numbers,
                    t=time_seconds,
                    channels=channels_dict,
                    frequency=sampling_rate,
                    extract_time=False,  # Don't resample, just decompose
                    apply_notch=(50,60),
                    decompose=decompose
                )
                
                # Get the subband-decomposed time series data
                df_combined = eeg.data.copy()
                df_combined.index.name = 'time'
                
                # Create output filename (e.g., S01_session0_base1.parquet)
                new_filename = f"S{participant_num}_session{session_num}_{baseline_type}.parquet"
                output_file = output_path / new_filename
                
                # Save combined EEG data
                df_combined.to_parquet(output_file)
                
                # Calculate statistics
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                duration_seconds = time_seconds[-1] if len(time_seconds) > 0 else 0
                
                print(f"  ✓ Session {session_num}: {num_samples} samples, {duration_seconds:.1f}s, {file_size_mb:.2f} MB")
                
                file_info.append({
                    'participant': participant_num,
                    'session': session_num,
                    'filename': new_filename,
                    'output_path': str(output_file),
                    'num_samples': num_samples,
                    'num_channels': len(eeg_channels),
                    'channel_names': eeg_channels,
                    'duration_seconds': float(duration_seconds),
                    'sampling_rate': sampling_rate,
                    'file_size_mb': file_size_mb,
                    'baseline_type': baseline_type
                })
                
                processed_files += 1
                
            except Exception as e:
                error_msg = f"Error processing {baseline_type}: {str(e)}"
                print(f"  ERROR: {error_msg}")
                failed_info.append({
                    'participant': participant_num,
                    'session': session_num,
                    'error': error_msg
                })
                failed_files += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY - {baseline_type.upper()}")
    print(f"{'='*60}")
    print(f"Processed files: {processed_files}")
    print(f"Failed files: {failed_files}")
    
    if processed_files > 0:
        total_samples = sum(f['num_samples'] for f in file_info)
        total_duration = sum(f['duration_seconds'] for f in file_info)
        total_size = sum(f['file_size_mb'] for f in file_info)
        
        print(f"\nTotal statistics:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total duration: {total_duration/60:.1f} minutes")
        print(f"  Total size: {total_size:.1f} MB")
    
    # Save summary
    summary_file = output_path / f'{baseline_type}_extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'baseline_type': baseline_type,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'file_info': file_info,
            'failed_info': failed_info
        }, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return file_info


def create_baseline_target_files(ratings_file, baseline_eeg_dir, baseline_type='base1'):
    """
    Create target files for baseline EEG, paired with session TLX scores.
    
    Creates 7 target files per recording (combined + 6 subscales):
    - S{X}_session{Y}_{baseline_type}.txt (combined TLX, 0-100)
    - S{X}_session{Y}_{baseline_type}_mental.txt
    - S{X}_session{Y}_{baseline_type}_physical.txt
    - S{X}_session{Y}_{baseline_type}_temporal.txt
    - S{X}_session{Y}_{baseline_type}_performance.txt
    - S{X}_session{Y}_{baseline_type}_effort.txt
    - S{X}_session{Y}_{baseline_type}_frustration.txt
    
    Args:
        ratings_file: Path to the ratings.parquet file
        baseline_eeg_dir: Directory containing baseline EEG files
        baseline_type: 'base1' or 'base2'
    
    Returns:
        Dictionary with processing results
    """
    
    ratings_path = Path(ratings_file)
    baseline_path = Path(baseline_eeg_dir)
    
    if not ratings_path.exists():
        print(f"Error: Ratings file does not exist: {ratings_path}")
        return None
    
    if not baseline_path.exists():
        print(f"Error: Baseline EEG directory does not exist: {baseline_path}")
        return None
    
    # Load ratings
    try:
        ratings_df = pd.read_parquet(ratings_path)
        print(f"{'='*60}")
        print(f"CREATING TARGET FILES FOR {baseline_type.upper()}")
        print(f"{'='*60}")
        print(f"Loaded {ratings_df.shape[0]} ratings from {ratings_path.name}")
    except Exception as e:
        print(f"Error loading ratings file: {e}")
        return None
    
    # TLX columns and subscales
    tlx_columns = ['tlx_mental', 'tlx_physical', 'tlx_temporal', 
                   'tlx_performance', 'tlx_effort', 'tlx_frustration']
    subscales = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']
    
    # Rescale TLX scores from 1-21 to 0-100 scale
    print(f"\nRescaling TLX scores from 1-21 scale to 0-100 scale...")
    for col in tlx_columns:
        if col in ratings_df.columns:
            ratings_df[col] = (ratings_df[col] - 1) * 5
    
    # Calculate combined TLX score (average of all 6 subscales)
    ratings_df['tlx_combined'] = ratings_df[tlx_columns].mean(axis=1)
    
    print(f"\nCombined TLX statistics (0-100 scale):")
    print(f"  Mean: {ratings_df['tlx_combined'].mean():.2f}")
    print(f"  Std: {ratings_df['tlx_combined'].std():.2f}")
    print(f"  Range: [{ratings_df['tlx_combined'].min():.2f}, {ratings_df['tlx_combined'].max():.2f}]")
    
    # Processing stats
    created_combined = 0
    created_subscales = {s: 0 for s in subscales}
    missing_eeg = 0
    
    print(f"\nCreating 7 target files per baseline recording...")
    
    for idx, row in ratings_df.iterrows():
        # Handle MultiIndex (participant, session) as index
        if isinstance(idx, tuple):
            participant, session = idx
        else:
            # Fallback if flat index
            participant = row.get('participant', idx)
            session = row.get('session', 1)
        
        # Convert to int if float
        participant = int(participant)
        session = int(session)
        
        # Skip invalid participants (like -999)
        if participant < 0:
            continue
        
        # Format participant as zero-padded string (1 -> "01")
        participant_str = f"{participant:02d}"
        # Session in ratings is 1-6, but files use session0-session5
        session_str = f"{session - 1}"
        
        # Check if baseline EEG file exists
        base_filename = f"S{participant_str}_session{session_str}_{baseline_type}"
        baseline_eeg_file = baseline_path / f"{base_filename}.parquet"
        
        if not baseline_eeg_file.exists():
            missing_eeg += 1
            continue
        
        # Create combined target file
        combined_target = baseline_path / f"{base_filename}.txt"
        try:
            with open(combined_target, 'w') as f:
                f.write(f"{row['tlx_combined']:.2f}")
            created_combined += 1
        except Exception as e:
            print(f"Error creating combined target for {base_filename}: {e}")
        
        # Create subscale target files
        for subscale in subscales:
            subscale_col = f'tlx_{subscale}'
            if subscale_col in ratings_df.columns:
                subscale_value = row[subscale_col]
                subscale_target = baseline_path / f"{base_filename}_{subscale}.txt"
                try:
                    with open(subscale_target, 'w') as f:
                        f.write(f"{subscale_value:.2f}")
                    created_subscales[subscale] += 1
                except Exception as e:
                    print(f"Error creating {subscale} target for {base_filename}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TARGET FILE CREATION SUMMARY - {baseline_type.upper()}")
    print(f"{'='*60}")
    print(f"Combined targets created: {created_combined}")
    for subscale in subscales:
        print(f"  {subscale}: {created_subscales[subscale]}")
    print(f"Missing baseline EEG files: {missing_eeg}")
    
    return {
        'created_combined': created_combined,
        'created_subscales': created_subscales,
        'missing_eeg': missing_eeg
    }


def copy_baseline_to_regression_dataset(baseline_eeg_dir, output_dir, baseline_type='base1'):
    """
    Copy paired baseline EEG and target files to final regression dataset.
    
    Only copies files that have both:
    - .parquet file (baseline EEG)
    - .txt file (combined TLX target)
    
    Args:
        baseline_eeg_dir: Directory containing baseline EEG and target files
        output_dir: Final regression dataset directory
        baseline_type: 'base1' or 'base2'
    
    Returns:
        Dictionary with copy statistics
    """
    
    baseline_path = Path(baseline_eeg_dir)
    output_path = Path(output_dir)
    
    if not baseline_path.exists():
        print(f"Error: Baseline directory does not exist: {baseline_path}")
        return None
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return None
    
    print(f"{'='*60}")
    print(f"COPYING TO FINAL REGRESSION DATASET - {baseline_type.upper()}")
    print(f"{'='*60}")
    print(f"Source: {baseline_path}")
    print(f"Output: {output_path}")
    
    # Find all parquet files
    parquet_files = list(baseline_path.glob(f"*_{baseline_type}.parquet"))
    
    copied_parquet = 0
    copied_targets = 0
    skipped = 0
    
    subscales = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']
    
    for parquet_file in sorted(parquet_files):
        base_name = parquet_file.stem  # e.g., S01_session0_base1
        
        # Check if combined target exists
        combined_target = baseline_path / f"{base_name}.txt"
        if not combined_target.exists():
            skipped += 1
            continue
        
        # Copy parquet file
        shutil.copy2(parquet_file, output_path / parquet_file.name)
        copied_parquet += 1
        
        # Copy combined target
        shutil.copy2(combined_target, output_path / combined_target.name)
        copied_targets += 1
        
        # Copy subscale targets
        for subscale in subscales:
            subscale_target = baseline_path / f"{base_name}_{subscale}.txt"
            if subscale_target.exists():
                shutil.copy2(subscale_target, output_path / subscale_target.name)
                copied_targets += 1
    
    print(f"\nCopied {copied_parquet} parquet files")
    print(f"Copied {copied_targets} target files")
    print(f"Skipped {skipped} files (missing targets)")
    
    return {
        'copied_parquet': copied_parquet,
        'copied_targets': copied_targets,
        'skipped': skipped
    }


def run_baseline_pipeline(baseline_type='base1', sampling_rate=500.0, decompose=True):
    """
    Run the complete baseline extraction and regression dataset creation pipeline.
    
    Steps:
    1. Extract baseline EEG with subband decomposition
    2. Create target files from ratings (7 files per recording)
    3. Copy to final regression dataset
    
    Args:
        baseline_type: 'base1' or 'base2'
        sampling_rate: EEG sampling frequency in Hz
        decompose: Whether to perform subband decomposition
    
    Returns:
        Dictionary with pipeline results
    """
    
    if baseline_type == 'base1':
        extracted_dir = base1_output_path
        regression_dir = base1_regression_path
    elif baseline_type == 'base2':
        extracted_dir = base2_output_path
        regression_dir = base2_regression_path
    else:
        print(f"Error: baseline_type must be 'base1' or 'base2', got '{baseline_type}'")
        return None
    
    print(f"\n{'#'*60}")
    print(f"# WAUC BASELINE REGRESSION PIPELINE - {baseline_type.upper()}")
    print(f"{'#'*60}\n")
    
    # Step 1: Extract baseline EEG
    print(f"\n{'='*60}")
    print(f"STEP 1: EXTRACTING {baseline_type.upper()} EEG")
    print(f"{'='*60}")
    extract_baseline_eeg(
        source_dir=data_path,
        output_dir=extracted_dir,
        baseline_type=baseline_type,
        sampling_rate=sampling_rate,
        decompose=decompose
    )
    
    # Step 2: Create target files
    print(f"\n{'='*60}")
    print(f"STEP 2: CREATING TARGET FILES FOR {baseline_type.upper()}")
    print(f"{'='*60}")
    create_baseline_target_files(
        ratings_file=ratings_path,
        baseline_eeg_dir=extracted_dir,
        baseline_type=baseline_type
    )
    
    # Step 3: Copy to final regression dataset
    print(f"\n{'='*60}")
    print(f"STEP 3: COPYING TO FINAL REGRESSION DATASET")
    print(f"{'='*60}")
    copy_baseline_to_regression_dataset(
        baseline_eeg_dir=extracted_dir,
        output_dir=regression_dir,
        baseline_type=baseline_type
    )
    
    print(f"\n{'#'*60}")
    print(f"# {baseline_type.upper()} PIPELINE COMPLETE!")
    print(f"# Output: {regression_dir}")
    print(f"{'#'*60}\n")
    
    return {'output_dir': regression_dir}


def run_both_baselines_pipeline(sampling_rate=500.0, decompose=True):
    """
    Run baseline extraction pipeline for both base1 and base2.
    
    Creates:
    - wauc_base1_time_regression/ (Base1 EEG + Session TLX, 7 targets each)
    - wauc_base2_time_regression/ (Base2 EEG + Session TLX, 7 targets each)
    
    Args:
        sampling_rate: EEG sampling frequency in Hz
        decompose: Whether to perform subband decomposition
    """
    
    print(f"\n{'#'*60}")
    print(f"# WAUC BASELINE REGRESSION PIPELINES")
    print(f"# Creating datasets for BOTH base1 and base2")
    print(f"{'#'*60}\n")
    
    # Run base1 pipeline
    run_baseline_pipeline(baseline_type='base1', sampling_rate=sampling_rate, decompose=decompose)
    
    # Run base2 pipeline
    run_baseline_pipeline(baseline_type='base2', sampling_rate=sampling_rate, decompose=decompose)
    
    print(f"\n{'#'*60}")
    print(f"# BOTH BASELINE PIPELINES COMPLETE!")
    print(f"# Outputs:")
    print(f"#   {base1_regression_path}")
    print(f"#   {base2_regression_path}")
    print(f"{'#'*60}\n")


def run_full_pipeline(target_column='tlx_mental', create_old_classification_datasets=False, decompose=True):
    """
    Run the complete WAUC preprocessing pipeline.
    
    Steps:
    1. Extract and reformat raw EEG files (with subband decomposition)
    2. Extract statistical features
    3. Create multi-subscale target files from ratings (7 files per recording)
    4. Cleanup files without targets
    5. Copy to final regression dataset directories
    5.5. Create metadata-based multi-subscale classification (NEW - matches HTC/N-Back/MOCAS)
    6. (Optional) Create old folder-based classification datasets
    
    Args:
        target_column: TLX column to use as regression target (default: 'tlx_mental')
                      Options: 'tlx_mental', 'tlx_physical', 'tlx_temporal', 
                              'tlx_performance', 'tlx_effort', 'tlx_frustration'
        create_old_classification_datasets: Whether to create OLD folder-based classification 
                                           datasets (MW, PW, MWPW, TLX bins). Default: False.
                                           The NEW metadata-based classification is always created.
        decompose: Whether to perform subband decomposition (default: True).
                  If False, keeps raw filtered data without frequency band separation.
    """
    
    print(f"\n{'#'*60}")
    print(f"# WAUC DATASET PREPROCESSING PIPELINE")
    print(f"# Target: {target_column}")
    if not decompose:
        print(f"# Decompose: OFF (raw filtered data only)")
    print(f"{'#'*60}\n")
    
    # Step 1: Extract and reformat raw EEG
    print(f"\n{'='*60}")
    print(f"STEP 1: EXTRACTING AND REFORMATTING RAW EEG")
    print(f"{'='*60}")
    reformat_raw_eeg(data_path, raw_output_path, sampling_rate=500.0, decompose=decompose)
    
    # Step 2: Extract features
    print(f"\n{'='*60}")
    print(f"STEP 2: EXTRACTING STATISTICAL FEATURES")
    print(f"{'='*60}")
    # FIXED VERSION: Pass original data path to avoid double-processing
    extract_features_from_wauc_files(
        raw_output_path, 
        features_output_path, 
        original_data_dir=data_path,  # Original wauc_by_session directory
        sampling_rate=500.0
    )
    
    # Step 3: Create target files
    print(f"\n{'='*60}")
    print(f"STEP 3: CREATING TARGET FILES")
    print(f"{'='*60}")
    create_target_files_from_ratings(
        ratings_path, 
        raw_output_path, 
        features_output_path,
        target_column=target_column,
        create_for_raw=True,
        create_for_features=True
    )
    
    # Step 4: Cleanup files without targets
    print(f"\n{'='*60}")
    print(f"STEP 4: CLEANUP FILES WITHOUT TARGETS")
    print(f"{'='*60}")
    cleanup_files_without_targets(raw_output_path, features_output_path)
    
    # Step 5: Copy to final regression datasets
    print(f"\n{'='*60}")
    print(f"STEP 5: COPYING TO FINAL REGRESSION DATASETS")
    print(f"{'='*60}")
    copy_to_final_regression_datasets(
        raw_output_path,
        features_output_path,
        time_regression_path,
        feature_regression_path
    )
    
    # Step 5.5: Create metadata-based classification datasets (NEW - matches HTC/N-Back/MOCAS)
    print(f"\n{'='*60}")
    print(f"STEP 5.5: CREATING METADATA-BASED MULTI-SUBSCALE CLASSIFICATION")
    print(f"{'='*60}")
    create_metadata_based_classification_datasets(
        raw_output_path,
        features_output_path,
        output_base_path,
        ratings_path
    )
    
    # Step 6: Create OLD folder-based classification datasets (optional - legacy format)
    if create_old_classification_datasets:
        print(f"\n{'='*60}")
        print(f"STEP 6: CREATING OLD FOLDER-BASED CLASSIFICATION DATASETS (LEGACY)")
        print(f"{'='*60}")
        print(f"Note: These are the OLD format. Use metadata-based datasets for new experiments.")
        
        # Mental Workload (MW) - Binary Classification
        print(f"\n--- Mental Workload (Binary Classification) ---")
        organize_files_by_classification(
            raw_output_path,
            features_output_path,
            ratings_path,
            output_base_path,
            classification_column='mw_labels'
        )
        
        # Physical Workload (PW) - 3-Class Classification
        print(f"\n--- Physical Workload (3-Class Classification) ---")
        organize_files_by_classification(
            raw_output_path,
            features_output_path,
            ratings_path,
            output_base_path,
            classification_column='pw_labels'
        )
        
        # Combined MW+PW - 6-Class Classification
        print(f"\n--- Combined Workload (6-Class Classification) ---")
        organize_files_by_classification(
            raw_output_path,
            features_output_path,
            ratings_path,
            output_base_path,
            classification_column='mwpw_labels'
        )
        
        # TLX-Based Classification - 3-Class (Low/Medium/High)
        print(f"\n--- TLX-Based Classification (3-Class: Low/Medium/High) ---")
        organize_files_by_tlx_bins(
            raw_output_path,
            features_output_path,
            ratings_path,
            output_base_path,
            target_column=target_column,
            low_threshold=33,   # 0-100 scale after rescaling
            high_threshold=67   # 0-100 scale after rescaling
        )
    
    print(f"\n{'#'*60}")
    print(f"# PIPELINE COMPLETE!")
    print(f"# Regression Datasets:")
    print(f"#   Time regression: {time_regression_path}")
    print(f"#   Feature regression: {feature_regression_path}")
    print(f"#")
    print(f"# Classification Datasets (NEW metadata-based format):")
    print(f"#   Time: data/wauc/wauc_time_classification/")
    print(f"#   Feature: data/wauc/wauc_feature_classification/")
    if create_old_classification_datasets:
        print(f"#")
        print(f"# OLD Classification Datasets (folder-based - legacy):")
        print(f"#   MW (binary): wauc_time_mw_classification_dataset / wauc_feature_mw_classification_dataset")
        print(f"#   PW (3-class): wauc_time_pw_classification_dataset / wauc_feature_pw_classification_dataset")
        print(f"#   MW+PW (6-class): wauc_time_mwpw_classification_dataset / wauc_feature_mwpw_classification_dataset")
        print(f"#   TLX bins (3-class): wauc_time_tlx_classification_dataset / wauc_feature_tlx_classification_dataset")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='WAUC Dataset Processing Pipeline for EEG Cognitive Workload Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (all steps)
  python load_wauc.py
  
  # Only create time-series datasets (skip feature extraction)
  python load_wauc.py --time-only
  
  # Only create feature datasets (skip time-series)
  python load_wauc.py --feature-only
  
  # Only time regression (no classification)
  python load_wauc.py --time-only --skip-classification
  
  # Only feature classification (no regression)
  python load_wauc.py --feature-only --skip-regression
  
  # Run specific steps
  python load_wauc.py --steps 1 2 3 4 5
  
  # Include legacy folder-based classification datasets
  python load_wauc.py --old-classification
  
  # Use a different target column
  python load_wauc.py --target-column tlx_physical
  
  # Create baseline regression datasets (pre-task EEG + session TLX)
  python load_wauc.py --baseline base1      # Base1 only
  python load_wauc.py --baseline base2      # Base2 only
  python load_wauc.py --baseline both       # Both baselines

Steps:
  1   - Extract and reformat raw EEG (with subband decomposition)
  2   - Extract statistical features
  3   - Create target files from ratings
  4   - Cleanup files without targets
  5   - Copy to final regression datasets
  5.5 - Create metadata-based classification datasets
  6   - Create old folder-based classification datasets (legacy)

Baseline Datasets:
  --baseline base1  Creates wauc_base1_time_regression/ with:
                    - Base1 EEG (pre-task) paired with session TLX scores
                    - 7 target files per recording (combined + 6 subscales)
  --baseline base2  Creates wauc_base2_time_regression/ (same structure)
  --baseline both   Creates both base1 and base2 datasets
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--time-only',
        action='store_true',
        help='Only create time-series datasets (steps 1, 3, 4, 5, 5.5 - skip feature extraction)'
    )
    mode_group.add_argument(
        '--feature-only',
        action='store_true',
        help='Only create feature datasets (steps 1, 2, 3, 4, 5, 5.5 - skip time-series copy)'
    )
    mode_group.add_argument(
        '--steps',
        nargs='+',
        choices=['1', '2', '3', '4', '5', '5.5', '6'],
        help='Run specific steps only (e.g., --steps 1 2 3)'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default='tlx_mental',
        choices=['tlx_mental', 'tlx_physical', 'tlx_temporal', 
                 'tlx_performance', 'tlx_effort', 'tlx_frustration'],
        help='TLX column to use as regression target (default: tlx_mental)'
    )
    
    parser.add_argument(
        '--old-classification',
        action='store_true',
        help='Include legacy folder-based classification datasets (MW, PW, MWPW, TLX bins)'
    )
    
    parser.add_argument(
        '--skip-classification',
        action='store_true',
        help='Skip classification dataset creation (steps 5.5, 6)'
    )
    
    parser.add_argument(
        '--skip-regression',
        action='store_true',
        help='Skip regression dataset creation (step 5)'
    )
    
    parser.add_argument(
        '--no-decompose',
        action='store_true',
        help='Skip subband decomposition (keep raw filtered data without frequency bands)'
    )
    
    parser.add_argument(
        '--raw-only',
        action='store_true',
        help='Only extract raw EEG without any signal processing (for LaBraM embeddings). Saves to _rawonly folder.'
    )
    
    # Baseline dataset options
    parser.add_argument(
        '--baseline',
        type=str,
        choices=['base1', 'base2', 'both'],
        default=None,
        help='Create baseline regression datasets. Options: base1, base2, or both'
    )
    
    args = parser.parse_args()
    
    # Handle baseline pipeline (separate from main pipeline)
    if args.baseline:
        if args.baseline == 'both':
            run_both_baselines_pipeline(
                sampling_rate=500.0,
                decompose=not args.no_decompose
            )
        else:
            run_baseline_pipeline(
                baseline_type=args.baseline,
                sampling_rate=500.0,
                decompose=not args.no_decompose
            )
        # Exit after baseline processing
        import sys
        sys.exit(0)
    
    # Handle raw-only mode (just extract raw EEG, no processing)
    if args.raw_only:
        print("\n" + "="*60)
        print("   WAUC RAW EEG EXTRACTION (NO PROCESSING)")
        print("   For LaBraM embeddings")
        print("="*60 + "\n")
        
        # Extract raw EEG
        reformat_raw_eeg(data_path, raw_output_path, sampling_rate=500.0, raw_only=True)
        
        # Copy TLX target files from full pipeline directory to raw-only directory
        source_targets_dir = Path(raw_output_path)
        dest_targets_dir = Path(str(raw_output_path) + "_rawonly")
        
        print("\n" + "="*60)
        print("   COPYING TLX TARGET FILES")
        print("="*60)
        print(f"Source: {source_targets_dir}")
        print(f"Destination: {dest_targets_dir}\n")
        
        if not source_targets_dir.exists():
            print(f"Warning: Source directory does not exist: {source_targets_dir}")
            print("TLX target files will NOT be copied.")
            print("Run full pipeline first to create target files.\n")
        else:
            # Find all .txt target files in source directory
            txt_files = list(source_targets_dir.glob("*.txt"))
            
            if not txt_files:
                print("Warning: No .txt target files found in source directory.")
                print("Run full pipeline first to create target files.\n")
            else:
                copied_count = 0
                skipped_count = 0
                
                for txt_file in txt_files:
                    dest_file = dest_targets_dir / txt_file.name
                    
                    # Check if corresponding .parquet file exists in destination
                    parquet_name = txt_file.name.replace('.txt', '.parquet')
                    dest_parquet = dest_targets_dir / parquet_name
                    
                    if dest_parquet.exists():
                        try:
                            shutil.copy2(txt_file, dest_file)
                            copied_count += 1
                        except Exception as e:
                            print(f"Error copying {txt_file.name}: {e}")
                    else:
                        skipped_count += 1
                
                print(f"Copied {copied_count} TLX target files")
                print(f"Skipped {skipped_count} files (no matching .parquet file)\n")
        
        print("="*60)
        print("   RAW EXTRACTION COMPLETE!")
        print("="*60 + "\n")
        import sys
        sys.exit(0)
    
    # Determine which steps to run
    if args.steps:
        # Run only specified steps
        print("\n" + "="*60)
        print(f"   WAUC DATASET PROCESSING - CUSTOM STEPS: {', '.join(args.steps)}")
        print("="*60 + "\n")
        
        if '1' in args.steps:
            print(f"\n{'='*60}")
            print(f"STEP 1: EXTRACTING AND REFORMATTING RAW EEG")
            print(f"{'='*60}")
            reformat_raw_eeg(data_path, raw_output_path, sampling_rate=500.0, decompose=not args.no_decompose)
            
        if '2' in args.steps:
            print(f"\n{'='*60}")
            print(f"STEP 2: EXTRACTING STATISTICAL FEATURES")
            print(f"{'='*60}")
            extract_features_from_wauc_files(
                raw_output_path, 
                features_output_path, 
                original_data_dir=data_path,
                sampling_rate=500.0
            )
            
        if '3' in args.steps:
            print(f"\n{'='*60}")
            print(f"STEP 3: CREATING TARGET FILES")
            print(f"{'='*60}")
            create_target_files_from_ratings(
                ratings_path, 
                raw_output_path, 
                features_output_path,
                target_column=args.target_column,
                create_for_raw=True,
                create_for_features=True
            )
            
        if '4' in args.steps:
            print(f"\n{'='*60}")
            print(f"STEP 4: CLEANUP FILES WITHOUT TARGETS")
            print(f"{'='*60}")
            cleanup_files_without_targets(raw_output_path, features_output_path)
            
        if '5' in args.steps:
            print(f"\n{'='*60}")
            print(f"STEP 5: COPYING TO FINAL REGRESSION DATASETS")
            print(f"{'='*60}")
            copy_to_final_regression_datasets(
                raw_output_path,
                features_output_path,
                time_regression_path,
                feature_regression_path
            )
            
        if '5.5' in args.steps:
            print(f"\n{'='*60}")
            print(f"STEP 5.5: CREATING METADATA-BASED CLASSIFICATION")
            print(f"{'='*60}")
            create_metadata_based_classification_datasets(
                raw_output_path,
                features_output_path,
                output_base_path,
                ratings_path
            )
            
        if '6' in args.steps:
            print(f"\n{'='*60}")
            print(f"STEP 6: CREATING OLD FOLDER-BASED CLASSIFICATION (LEGACY)")
            print(f"{'='*60}")
            # MW
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='mw_labels'
            )
            # PW
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='pw_labels'
            )
            # MWPW
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='mwpw_labels'
            )
            # TLX bins
            organize_files_by_tlx_bins(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, target_column=args.target_column,
                low_threshold=33, high_threshold=67
            )
            
        print("\n" + "="*60)
        print("   CUSTOM STEPS COMPLETE!")
        print("="*60 + "\n")
        
    elif args.time_only:
        # Time-series only mode (skip step 2 - feature extraction)
        print("\n" + "="*60)
        print("   WAUC DATASET PROCESSING - TIME-SERIES ONLY")
        print("="*60)
        print("\nThis will create time-series datasets:")
        if not args.skip_regression:
            print("  - wauc_time_regression_dataset")
        if not args.skip_classification:
            print("  - wauc_time_classification (metadata-based)")
        print("  - Skipping feature extraction (step 2)")
        if args.skip_regression:
            print("  - Skipping regression (--skip-regression)")
        if args.skip_classification:
            print("  - Skipping classification (--skip-classification)")
        print("="*60 + "\n")
        
        # Step 1
        print(f"\n{'='*60}")
        print(f"STEP 1: EXTRACTING AND REFORMATTING RAW EEG")
        print(f"{'='*60}")
        reformat_raw_eeg(data_path, raw_output_path, sampling_rate=500.0, decompose=not args.no_decompose)
        
        # Step 3 (targets for raw only)
        print(f"\n{'='*60}")
        print(f"STEP 3: CREATING TARGET FILES (RAW EEG ONLY)")
        print(f"{'='*60}")
        create_target_files_from_ratings(
            ratings_path, raw_output_path, features_output_path,
            target_column=args.target_column,
            create_for_raw=True, create_for_features=False
        )
        
        # Step 4 (cleanup raw only)
        print(f"\n{'='*60}")
        print(f"STEP 4: CLEANUP FILES WITHOUT TARGETS")
        print(f"{'='*60}")
        cleanup_files_without_targets(raw_output_path, features_output_path)
        
        # Step 5 (time regression only)
        if not args.skip_regression:
            print(f"\n{'='*60}")
            print(f"STEP 5: COPYING TO TIME REGRESSION DATASET")
            print(f"{'='*60}")
            copy_to_final_regression_datasets(
                raw_output_path, features_output_path,
                time_regression_path, feature_regression_path
            )
        
        # Step 5.5 (classification - time only)
        if not args.skip_classification:
            print(f"\n{'='*60}")
            print(f"STEP 5.5: CREATING METADATA-BASED CLASSIFICATION (TIME ONLY)")
            print(f"{'='*60}")
            create_metadata_based_classification_datasets(
                raw_output_path, features_output_path,
                output_base_path, ratings_path
            )
        
        if args.old_classification and not args.skip_classification:
            print(f"\n{'='*60}")
            print(f"STEP 6: CREATING OLD FOLDER-BASED CLASSIFICATION (LEGACY)")
            print(f"{'='*60}")
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='mw_labels'
            )
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='pw_labels'
            )
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='mwpw_labels'
            )
            organize_files_by_tlx_bins(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, target_column=args.target_column,
                low_threshold=33, high_threshold=67
            )
        
        print("\n" + "="*60)
        print("   TIME-SERIES DATASETS COMPLETE!")
        if not args.skip_regression:
            print(f"   ✓ {time_regression_path}")
        if not args.skip_classification:
            print(f"   ✓ wauc_time_classification/")
        print("="*60 + "\n")
        
    elif args.feature_only:
        # Feature-only mode (skip time-series copy in step 5)
        print("\n" + "="*60)
        print("   WAUC DATASET PROCESSING - FEATURES ONLY")
        print("="*60)
        print("\nThis will create feature-based datasets:")
        if not args.skip_regression:
            print("  - wauc_feature_regression_dataset")
        if not args.skip_classification:
            print("  - wauc_feature_classification (metadata-based)")
        if args.skip_regression:
            print("  - Skipping regression (--skip-regression)")
        if args.skip_classification:
            print("  - Skipping classification (--skip-classification)")
        print("="*60 + "\n")
        
        # Step 1
        print(f"\n{'='*60}")
        print(f"STEP 1: EXTRACTING AND REFORMATTING RAW EEG")
        print(f"{'='*60}")
        reformat_raw_eeg(data_path, raw_output_path, sampling_rate=500.0, decompose=not args.no_decompose)
        
        # Step 2
        print(f"\n{'='*60}")
        print(f"STEP 2: EXTRACTING STATISTICAL FEATURES")
        print(f"{'='*60}")
        extract_features_from_wauc_files(
            raw_output_path, features_output_path,
            original_data_dir=data_path, sampling_rate=500.0
        )
        
        # Step 3 (targets for features only)
        print(f"\n{'='*60}")
        print(f"STEP 3: CREATING TARGET FILES (FEATURES ONLY)")
        print(f"{'='*60}")
        create_target_files_from_ratings(
            ratings_path, raw_output_path, features_output_path,
            target_column=args.target_column,
            create_for_raw=False, create_for_features=True
        )
        
        # Step 4
        print(f"\n{'='*60}")
        print(f"STEP 4: CLEANUP FILES WITHOUT TARGETS")
        print(f"{'='*60}")
        cleanup_files_without_targets(raw_output_path, features_output_path)
        
        # Step 5 (feature regression only)
        if not args.skip_regression:
            print(f"\n{'='*60}")
            print(f"STEP 5: COPYING TO FEATURE REGRESSION DATASET")
            print(f"{'='*60}")
            copy_to_final_regression_datasets(
                raw_output_path, features_output_path,
                time_regression_path, feature_regression_path
            )
        
        # Step 5.5 (classification - features only)
        if not args.skip_classification:
            print(f"\n{'='*60}")
            print(f"STEP 5.5: CREATING METADATA-BASED CLASSIFICATION (FEATURES ONLY)")
            print(f"{'='*60}")
            create_metadata_based_classification_datasets(
                raw_output_path, features_output_path,
                output_base_path, ratings_path
            )
        
        if args.old_classification and not args.skip_classification:
            print(f"\n{'='*60}")
            print(f"STEP 6: CREATING OLD FOLDER-BASED CLASSIFICATION (LEGACY)")
            print(f"{'='*60}")
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='mw_labels'
            )
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='pw_labels'
            )
            organize_files_by_classification(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, classification_column='mwpw_labels'
            )
            organize_files_by_tlx_bins(
                raw_output_path, features_output_path, ratings_path,
                output_base_path, target_column=args.target_column,
                low_threshold=33, high_threshold=67
            )
        
        print("\n" + "="*60)
        print("   FEATURE DATASETS COMPLETE!")
        if not args.skip_regression:
            print(f"   ✓ {feature_regression_path}")
        if not args.skip_classification:
            print(f"   ✓ wauc_feature_classification/")
        print("="*60 + "\n")
        
    else:
        # Run full pipeline (default)
        run_full_pipeline(
            target_column=args.target_column,
            create_old_classification_datasets=args.old_classification,
            decompose=not args.no_decompose
        )
