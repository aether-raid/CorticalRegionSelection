import os 
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count

# Add parent directories to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from channel_importance.eeg import EEG


"""
N-Back Dataset Processing Pipeline for EEG Cognitive Workload Analysis
====================================================================

Complete preprocessing pipeline for the N-Back dataset that transforms raw 
EEG data into processed features suitable for TLX-based regression and classification analysis.

DATASET SOURCE:
    N-Back dataset - EEG recordings during n-back cognitive tasks with NASA-TLX workload ratings
    Contains EEG data from Emotiv EPOC+ headset (14 channels, 128Hz sampling rate)

EXPECTED INPUT STRUCTURE:
    data/n_back/n_back/    # Root directory
        subject_01/
            task1/
                phase1.parquet       # Phase 1 EEG data (not used)
                phase2.parquet       # Phase 2 EEG data (USED for analysis)
                phase3.parquet       # Phase 3 EEG data (not used)
                tlx.json            # NASA-TLX scores
                scores.json         # Additional scores
            task2/
                phase2.parquet
                tlx.json
                scores.json
            task3/
                phase2.parquet
                tlx.json
                scores.json
        subject_02/
            task1/
                phase2.parquet
                tlx.json
                scores.json
            ...
        ...subject_01-16

TLX.JSON STRUCTURE:
    {
        "mental_demand": 0-20,      # Mental workload
        "physical_demand": 0-20,    # Physical workload
        "temporal_demand": 0-20,    # Time pressure
        "performance": 0-20,        # Task performance (inverted: higher = better)
        "effort": 0-20,             # Effort required
        "frustration": 0-20         # Frustration level
    }
    
    Combined TLX Score Calculation:
        • Average of all 6 subscales: (sum of all dimensions) / 6
        • Each subscale is 0-20, so average is 0-20
        • Rescaled to 0-100: (average / 20) * 100
        • Final TLX score range: 0-100

PROCESSING PIPELINE (5 STEPS):
    
    STEP 1: EEG File Extraction & Reformatting (Phase 2 only)
        • Extracts phase2.parquet files from nested directory structure
        • Removes non-EEG columns (timestamp, Counter, Interpolated, etc.)
        • Strips 'EEG.' prefix from channel names
        • Applies bandpass filtering and frequency decomposition via EEG class
        • Standardizes filenames: S{subject}_{task}_eeg_raw.parquet
        • Output: data/n_back/nback_raw_eeg_extracted/
    
    STEP 2: Statistical Feature Extraction
        • Loads ORIGINAL raw phase2.parquet files (prevents double-processing bug)
        • Applies signal processing: bandpass filtering, frequency decomposition
        • Extracts 400+ statistical features per file using EEG class
        • Features: power bands, spectral entropy, hjorth parameters, etc.
        • Output: data/n_back/nback_features_extracted/S{X}_{Y}_features.parquet
        
    STEP 3: Multi-Label TLX Target File Creation
        • Reads tlx.json scores for all 6 dimensions
        • Calculates combined TLX score (0-100 scale)
        • Creates 7 target files per recording:
            - S{X}_{Y}_features.txt (combined TLX, 0-100)
            - S{X}_{Y}_features_mental.txt (0-100)
            - S{X}_{Y}_features_physical.txt (0-100)
            - S{X}_{Y}_features_temporal.txt (0-100)
            - S{X}_{Y}_features_performance.txt (0-100)
            - S{X}_{Y}_features_effort.txt (0-100)
            - S{X}_{Y}_features_frustration.txt (0-100)
        • Output: data/n_back/nback_features_extracted/
        
    STEP 4: Copy to Multi-Label TLX Regression Datasets
        • Creates 2 regression datasets with 7 target files each:
            - nback_time_regression/    (Raw EEG + 7 TLX targets)
            - nback_feature_regression/ (Features + 7 TLX targets)
    
    STEP 5: Create Multi-Subscale Classification Datasets
        • Bins TLX scores (combined + 6 subscales) into Low/Medium/High classes
        • Uses 33rd and 67th percentiles as bin boundaries for each subscale
        • Zero-duplication strategy: single 'all' folder + metadata JSON
        • Creates 2 classification datasets:
            - nback_time_classification/all/ + classification_metadata.json
            - nback_feature_classification/all/ + classification_metadata.json

FINAL OUTPUT STRUCTURE (4 DATASETS TOTAL):
    data/n_back/
        nback_raw_eeg_extracted/              # Extracted phase2 with decomposition
            S01_1_eeg_raw.parquet
            S01_2_eeg_raw.parquet
            ...
        nback_features_extracted/             # Statistical features + multi-label targets
            S01_1_features.parquet
            S01_1_features.txt                # Combined TLX (0-100)
            S01_1_features_mental.txt         # Mental subscale (0-100)
            S01_1_features_physical.txt       # Physical subscale (0-100)
            S01_1_features_temporal.txt       # Temporal subscale (0-100)
            S01_1_features_performance.txt    # Performance subscale (0-100)
            S01_1_features_effort.txt         # Effort subscale (0-100)
            S01_1_features_frustration.txt    # Frustration subscale (0-100)
            ...
        nback_time_regression/               # Dataset 1: Raw EEG regression (7 targets each)
            S01_1_eeg_raw.parquet
            S01_1_eeg_raw.txt                 # Combined TLX (renamed to match .parquet)
            S01_1_eeg_raw_mental.txt
            S01_1_eeg_raw_physical.txt
            S01_1_eeg_raw_temporal.txt
            S01_1_eeg_raw_performance.txt
            S01_1_eeg_raw_effort.txt
            S01_1_eeg_raw_frustration.txt
            ...
        nback_feature_regression/            # Dataset 2: Feature regression (7 targets each)
            S01_1_features.parquet
            S01_1_features.txt                # Combined TLX
            S01_1_features_mental.txt
            S01_1_features_physical.txt
            S01_1_features_temporal.txt
            S01_1_features_performance.txt
            S01_1_features_effort.txt
            S01_1_features_frustration.txt
            ...
        nback_time_classification/           # Dataset 3: Raw EEG classification (metadata-based)
            all/
                S01_1_eeg_raw.parquet
                S02_1_eeg_raw.parquet
                ...
            classification_metadata.json      # {filename: {combined: 0-2, mental: 0-2, ...}}
        nback_feature_classification/        # Dataset 4: Feature classification (metadata-based)
            all/
                S01_1_features.parquet
                S02_1_features.parquet
                S03_1_eeg_raw.parquet
                ...
        nback_feature_classification/        # Dataset 4: Feature classification
            low/
                S01_1_features.parquet
                ...
            medium/
                S02_1_features.parquet
                ...
            high/
                S03_1_features.parquet
                ...

INTEGRATION:
    The processed data is compatible with:
    • EEGRawRegressionDataset for loading features and targets
    • Standard regression and classification pipelines
    • Channel importance analysis and feature selection methods

USAGE:
    python datasets/NBACK/load_nback.py
    
    Or call individual functions:
    • reformat_raw_eeg()
    • extract_features_from_nback_files()
    • create_target_files_from_tlx()
    • cleanup_parquet_files_without_targets()
    • create_regression_datasets()
    • create_classification_datasets()
"""

# Path configuration
# Use relative paths from project root
data_path = "data/n_back/n_back"  # Source data
raw_output_path = "data/n_back/nback_raw_eeg_extracted"  # Extracted raw EEG
features_output_path = "data/n_back/nback_features_extracted"  # Statistical features
time_regression_path = "data/n_back/nback_time_regression"  # Time regression dataset
feature_regression_path = "data/n_back/nback_feature_regression"  # Feature regression dataset
time_classification_path = "data/n_back/nback_time_classification"  # Time classification dataset
feature_classification_path = "data/n_back/nback_feature_classification"  # Feature classification dataset

# TLX dimensions (all 6 subscales combined into single score)
TLX_DIMENSIONS = [
    'mental',
    'physical', 
    'temporal',
    'performance',
    'effort',
    'frustration'
]

# Mapping from JSON keys (with _demand suffix) to output names (without suffix)
TLX_JSON_KEY_MAPPING = {
    'mental': 'mental_demand',
    'physical': 'physical_demand',
    'temporal': 'temporal_demand',
    'performance': 'performance',
    'effort': 'effort',
    'frustration': 'frustration'
}


def reformat_raw_eeg(source_dir, output_dir, sampling_rate=128.0, raw_only=False):
    """
    Extract and reformat phase2 EEG files from N-Back dataset.
    
    STEP 1: Processes ONLY phase2.parquet files from each subject/task combination.
    
    This function:
    1. Reads phase2.parquet from each subject/task directory
    2. Removes non-EEG columns (timestamp, Counter, Interpolated, Battery, etc.)
    3. Strips 'EEG.' prefix from channel names (EEG.AF3 → AF3)
    4. If raw_only=False: Applies EEG processing (filtering + frequency decomposition)
       If raw_only=True: Just saves cleaned raw EEG (no processing)
    5. Saves with standardized naming: S{subject}_{task}_eeg_raw.parquet
    
    Args:
        source_dir (str): Path to n_back/ directory containing subject folders
        output_dir (str): Path where extracted files will be saved
        sampling_rate (float): Sampling frequency in Hz (default: 128.0)
        raw_only (bool): If True, skip all signal processing and just copy raw EEG.
                        Use this for LaBraM embeddings. Default: False.
        
    Returns:
        List of dictionaries with processing details for each file
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

    print(f"\n{'='*60}")
    if raw_only:
        print(f"N-BACK EEG EXTRACTION - RAW ONLY (PHASE 2 ONLY)")
    else:
        print(f"N-BACK EEG EXTRACTION WITH SUBBAND DECOMPOSITION (PHASE 2 ONLY)")
    print(f"{'='*60}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Raw only (no processing): {raw_only}")
    print(f"{'='*60}\n")

    processed_files = 0
    missing_files = 0
    failed_files = 0
    file_info = []
    
    # Get all subject folders
    subject_folders = [d for d in source_path.iterdir() 
                      if d.is_dir() and d.name.startswith('subject_')]
    
    print(f"Found {len(subject_folders)} subject folders\n")
    
    for subject_folder in sorted(subject_folders):
        subject_num = subject_folder.name.replace('subject_', '')
        print(f"Processing Subject {subject_num}")
        
        # Get all task folders
        task_folders = [d for d in subject_folder.iterdir() 
                       if d.is_dir() and d.name.startswith('task')]
        
        print(f"  Found {len(task_folders)} task folders")
        
        for task_folder in sorted(task_folders):
            task_num = task_folder.name.replace('task', '')
            phase2_file = task_folder / 'phase2.parquet'
            
            # Check if phase2.parquet exists
            if not phase2_file.exists():
                print(f"  ✗ Task {task_num}: Missing phase2.parquet")
                missing_files += 1
                continue
            
            # Create new filename: S{subject}_{task}_eeg_raw.parquet
            new_filename = f"S{subject_num}_{task_num}_eeg_raw.parquet"
            output_file = output_path / new_filename
            
            try:
                # Load phase2 parquet file
                df = pd.read_parquet(phase2_file)
                
                # Get EEG channels (columns starting with 'EEG.' and containing actual signals)
                # Exclude: timestamp, Counter, Interpolated, RawCq, Battery, BatteryPercent, MarkerHardware
                eeg_columns = [col for col in df.columns 
                             if col.startswith('EEG.') 
                             and col not in ['EEG.Counter', 'EEG.Interpolated', 'EEG.RawCq', 
                                           'EEG.Battery', 'EEG.BatteryPercent', 'EEG.MarkerHardware']]
                
                if not eeg_columns:
                    raise ValueError("No valid EEG channels found")
                
                # Strip 'EEG.' prefix from channel names
                channel_map = {col: col.replace('EEG.', '') for col in eeg_columns}
                df_eeg = df[eeg_columns].rename(columns=channel_map)
                
                # Check for NaN columns and filter them out
                valid_channels = []
                for col in df_eeg.columns:
                    if df_eeg[col].isna().all():
                        print(f"  ⚠ Task {task_num}: Skipping channel {col} (all NaN)")
                    else:
                        valid_channels.append(col)
                
                if not valid_channels:
                    raise ValueError("No valid channels after filtering NaN columns")
                
                df_eeg = df_eeg[valid_channels]
                
                # Calculate duration
                time_seconds = np.arange(len(df_eeg)) / sampling_rate
                duration = time_seconds[-1]
                
                if raw_only:
                    # Just save raw EEG without any processing
                    df_eeg.to_parquet(output_file)
                    output_size = output_file.stat().st_size / (1024 * 1024)
                    
                    # Store processing info
                    file_info.append({
                        'subject': subject_num,
                        'task': task_num,
                        'original_path': str(phase2_file),
                        'new_filename': new_filename,
                        'samples': len(df_eeg),
                        'channels': len(valid_channels),
                        'bands': 0,  # No bands - raw only
                        'total_columns': df_eeg.shape[1],
                        'duration_seconds': float(duration),
                        'output_size_mb': output_size,
                        'channel_names': valid_channels,
                        'band_names': [],
                        'raw_only': True,
                        'status': 'success'
                    })
                else:
                    # Prepare data for EEG class
                    sample_numbers = np.arange(len(df_eeg))
                    channels_dict = {col: df_eeg[col].values for col in valid_channels}
                    
                    # Create EEG instance for subband decomposition
                    eeg = EEG(
                        s_n=sample_numbers,
                        t=time_seconds,
                        channels=channels_dict,
                        frequency=sampling_rate,
                        extract_time=False  # Don't resample, just decompose
                    )
                    
                    # Extract subband-decomposed time series
                    subband_df = eeg.data
                    
                    # Verify MultiIndex structure
                    if not isinstance(subband_df.columns, pd.MultiIndex):
                        raise ValueError("EEG data does not have MultiIndex structure")
                    
                    # Save subband-decomposed data
                    subband_df.to_parquet(output_file)
                    output_size = output_file.stat().st_size / (1024 * 1024)
                    
                    # Store processing info
                    file_info.append({
                        'subject': subject_num,
                        'task': task_num,
                        'original_path': str(phase2_file),
                        'new_filename': new_filename,
                        'samples': len(df_eeg),
                        'channels': len(valid_channels),
                        'bands': len(subband_df.columns.get_level_values(0).unique()),
                        'total_columns': subband_df.shape[1],
                        'duration_seconds': float(duration),
                        'output_size_mb': output_size,
                        'channel_names': valid_channels,
                        'band_names': list(subband_df.columns.get_level_values(0).unique()),
                        'raw_only': False,
                        'status': 'success'
                    })
                
                print(f"  ✓ Task {task_num}: {len(df_eeg)} samples, {duration:.1f}s, {len(valid_channels)} channels, {output_size:.2f} MB")
                processed_files += 1
                
            except Exception as e:
                print(f"  ✗ Task {task_num}: Error - {str(e)}")
                failed_files += 1
                file_info.append({
                    'subject': subject_num,
                    'task': task_num,
                    'original_path': str(phase2_file),
                    'error': str(e),
                    'status': 'failed'
                })
    
    # Summary
    total_files = processed_files + failed_files + missing_files
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_files} files")
    print(f"Failed: {failed_files} files")
    print(f"Missing: {missing_files} files")
    
    if processed_files > 0:
        total_samples = sum(f['samples'] for f in file_info if f['status'] == 'success')
        total_duration = sum(f['duration_seconds'] for f in file_info if f['status'] == 'success')
        total_size = sum(f['output_size_mb'] for f in file_info if f['status'] == 'success')
        avg_duration = total_duration / processed_files
        
        print(f"\nStatistics:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total duration: {total_duration/60:.1f} minutes")
        print(f"  Average duration per task: {avg_duration:.1f} seconds")
        print(f"  Total data size: {total_size:.1f} MB")
    
    # Save processing summary
    summary_file = output_path / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'processing_stats': {
                'total_files': total_files,
                'processed_files': processed_files,
                'failed_files': failed_files,
                'missing_files': missing_files
            },
            'sampling_rate': sampling_rate,
            'processed_files_details': file_info
        }, f, indent=2, default=str)
    
    print(f"\nProcessing summary saved to: {summary_file}")
    
    return file_info


def _process_single_nback_file(args):
    """
    Helper function to process a single N-Back EEG file for parallel execution.
    
    Args:
        args: Tuple of (eeg_file, original_path, output_path, sampling_rate)
        
    Returns:
        dict: Processing result information or error details
    """
    eeg_file, original_path, output_path, sampling_rate = args
    
    try:
        # Parse filename to get subject and task info
        filename = eeg_file.stem.replace('_eeg_raw', '')
        parts = filename.split('_')
        
        if len(parts) < 2:
            return {
                'file': str(eeg_file),
                'error': f'Cannot parse filename',
                'status': 'failed'
            }
        
        subject_num = parts[0].replace('S', '')
        task_num = parts[1]
        
        # FIXED VERSION: Load ORIGINAL raw phase2.parquet file
        original_file = original_path / f'subject_{subject_num}' / f'task{task_num}' / 'phase2.parquet'
        
        if not original_file.exists():
            return {
                'file': str(eeg_file),
                'error': f'Original phase2.parquet not found: {original_file}',
                'status': 'failed'
            }
        
        # Load ORIGINAL raw data
        df = pd.read_parquet(original_file)
        
        # Get EEG channels (same filtering as in reformat_raw_eeg)
        eeg_columns = [col for col in df.columns 
                     if col.startswith('EEG.') 
                     and col not in ['EEG.Counter', 'EEG.Interpolated', 'EEG.RawCq', 
                                   'EEG.Battery', 'EEG.BatteryPercent', 'EEG.MarkerHardware']]
        
        if not eeg_columns:
            return {
                'file': str(eeg_file),
                'error': 'No valid EEG channels found',
                'status': 'failed'
            }
        
        # Strip 'EEG.' prefix
        channel_map = {col: col.replace('EEG.', '') for col in eeg_columns}
        df_eeg = df[eeg_columns].rename(columns=channel_map)
        
        # Filter out NaN columns
        valid_channels = [col for col in df_eeg.columns if not df_eeg[col].isna().all()]
        
        if not valid_channels:
            return {
                'file': str(eeg_file),
                'error': 'No valid channels after filtering NaN columns',
                'status': 'failed'
            }
        
        df_eeg = df_eeg[valid_channels]
        
        # Prepare data for EEG class
        time_seconds = np.arange(len(df_eeg)) / sampling_rate
        sample_numbers = np.arange(len(df_eeg))
        channels_dict = {col: df_eeg[col].values for col in valid_channels}
        
        # Create EEG instance from ORIGINAL raw data (processes ONCE correctly)
        eeg = EEG(
            s_n=sample_numbers,
            t=time_seconds,
            channels=channels_dict,
            frequency=sampling_rate,
            extract_time=False
        )
        
        # Generate statistical features
        eeg.generate_stats()
        features = eeg.stats
        
        # Ensure proper column structure
        if isinstance(features.columns, pd.MultiIndex):
            features.columns.names = ['band', 'channel']
        else:
            # Convert to MultiIndex if not already
            features.columns = pd.MultiIndex.from_product([features.columns, ['']])
            features.columns.names = ['band', 'channel']
        
        # Save features
        output_filename = f"S{subject_num}_{task_num}_features.parquet"
        output_file = output_path / output_filename
        features.to_parquet(output_file)
        
        # Get output size
        output_size = output_file.stat().st_size / (1024 * 1024)
        
        return {
            'file': str(eeg_file),
            'output': str(output_file),
            'num_features': features.shape[1],
            'num_channels': len(valid_channels),
            'num_samples': len(df_eeg),
            'output_size_mb': output_size,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'file': str(eeg_file),
            'error': str(e),
            'status': 'failed'
        }


def extract_features_from_nback_files(input_dir, output_dir, original_data_dir, sampling_rate=128.0, n_jobs=None):
    """
    Extract statistical features from N-Back EEG files.
    
    STEP 2: FIXED VERSION - Loads ORIGINAL raw phase2.parquet files to avoid double-processing.
    
    This function reads the decomposed EEG files from STEP 1 to get the list of files to process,
    but then loads the ORIGINAL raw phase2.parquet files to create EEG instances. This prevents
    the double-processing bug where creating an EEG instance from already-processed data causes
    17,300x power corruption.
    
    Args:
        input_dir (str): Directory containing decomposed EEG files from STEP 1
        output_dir (str): Directory where feature files will be saved
        original_data_dir (str): Path to original n_back/ directory with phase2.parquet files
        sampling_rate (float): Sampling frequency in Hz (default: 128.0)
        
    Returns:
        List of dictionaries with feature extraction details
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    original_path = Path(original_data_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return None
    
    if not original_path.exists():
        print(f"Error: Original data directory does not exist: {original_path}")
        return None
        
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_path}: {e}")
        return None

    print(f"\n{'='*60}")
    print(f"N-BACK FEATURE EXTRACTION (FIXED VERSION)")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Original data: {original_path}")
    print(f"Output: {output_path}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"{'='*60}\n")

    # Get all extracted EEG files
    eeg_files = list(input_path.glob("S*_*_eeg_raw.parquet"))
    
    if not eeg_files:
        print(f"No EEG files found in {input_path}")
        return None
    
    print(f"Found {len(eeg_files)} EEG files to process\n")
    
    # Determine number of workers for parallel processing
    if n_jobs is None:
        n_workers = cpu_count()
    else:
        n_workers = min(n_jobs, cpu_count())
    
    print(f"Using {n_workers} parallel workers\n")
    
    # Prepare arguments for parallel processing
    process_args = [(eeg_file, original_path, output_path, sampling_rate) for eeg_file in sorted(eeg_files)]
    
    # Process files in parallel with progress bar
    file_info = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(pool.imap(_process_single_nback_file, process_args), 
                          total=len(eeg_files), 
                          desc="Processing EEG files",
                          unit="file"):
            file_info.append(result)
    
    # Count results
    processed_files = sum(1 for r in file_info if r.get('status') == 'success')
    failed_files = sum(1 for r in file_info if r.get('status') == 'failed')
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_files} files")
    print(f"Failed: {failed_files} files")
    
    if processed_files > 0:
        total_features = sum(f['num_features'] for f in file_info if f['status'] == 'success')
        avg_features = total_features / processed_files
        total_size = sum(f['output_size_mb'] for f in file_info if f['status'] == 'success')
        
        print(f"\nStatistics:")
        print(f"  Average features per file: {avg_features:.0f}")
        print(f"  Total data size: {total_size:.1f} MB")
    
    # Save processing summary
    summary_file = output_path / 'feature_extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'processing_stats': {
                'processed_files': processed_files,
                'failed_files': failed_files
            },
            'sampling_rate': sampling_rate,
            'processed_files_details': file_info
        }, f, indent=2, default=str)
    
    print(f"\nProcessing summary saved to: {summary_file}")
    
    return file_info


def create_target_files_from_tlx(source_dir, features_dir):
    """
    Create combined TLX target files for each EEG recording.
    
    STEP 3: Reads tlx.json files and creates single combined TLX score (0-100 scale).
    
    Combined TLX Calculation:
        1. Read all 6 TLX subscales (each 0-20)
        2. Calculate average: sum(all_dimensions) / 6
        3. Rescale to 0-100: (average / 20) * 100
        4. Save as single target file: S{X}_{Y}_target.txt
    
    Args:
        source_dir (str): Path to n_back/ directory containing tlx.json files
        features_dir (str): Directory where feature files are stored (target files saved here)
        
    Returns:
        Dictionary with creation statistics and TLX score distribution
    """
    
    source_path = Path(source_dir)
    features_path = Path(features_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_path}")
        return None
    
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None

    print(f"\n{'='*60}")
    print(f"CREATING COMBINED TLX TARGET FILES")
    print(f"{'='*60}")
    print(f"Source: {source_path}")
    print(f"Output: {features_path}")
    print(f"TLX Calculation: Average of 6 subscales, rescaled to 0-100")
    print(f"{'='*60}\n")

    created_targets = 0
    missing_tlx = 0
    missing_features = 0
    missing_dimensions = 0
    combined_tlx_scores = []
    individual_subscales = {dim: [] for dim in TLX_DIMENSIONS}
    
    # Get all subject folders
    subject_folders = [d for d in source_path.iterdir() 
                      if d.is_dir() and d.name.startswith('subject_')]
    
    print(f"Found {len(subject_folders)} subject folders\n")
    
    for subject_folder in sorted(subject_folders):
        subject_num = subject_folder.name.replace('subject_', '')
        
        # Get all task folders
        task_folders = [d for d in subject_folder.iterdir() 
                       if d.is_dir() and d.name.startswith('task')]
        
        for task_folder in sorted(task_folders):
            task_num = task_folder.name.replace('task', '')
            
            # Check if feature file exists
            feature_file = features_path / f"S{subject_num}_{task_num}_features.parquet"
            if not feature_file.exists():
                missing_features += 1
                continue
            
            # Read TLX scores
            tlx_file = task_folder / 'tlx.json'
            if not tlx_file.exists():
                print(f"Warning: Missing tlx.json for Subject {subject_num}, Task {task_num}")
                missing_tlx += 1
                continue
            
            try:
                with open(tlx_file, 'r') as f:
                    tlx_data = json.load(f)
                
                # Check all dimensions are present (check JSON keys with _demand suffix)
                missing_dims = [dim for dim in TLX_DIMENSIONS if TLX_JSON_KEY_MAPPING[dim] not in tlx_data]
                if missing_dims:
                    print(f"Warning: Missing dimensions {missing_dims} for S{subject_num}_T{task_num}")
                    missing_dimensions += 1
                    continue
                
                # Calculate combined TLX score
                # Step 1: Get all 6 subscale scores (0-20 each)
                subscale_scores = [tlx_data[TLX_JSON_KEY_MAPPING[dim]] for dim in TLX_DIMENSIONS]
                
                # Store individual subscales for statistics (rescaled to 0-100)
                for dim in TLX_DIMENSIONS:
                    json_key = TLX_JSON_KEY_MAPPING[dim]
                    raw_score = tlx_data[json_key]  # Expected: 0-20 scale
                    
                    # Clamp to valid range in case of data issues
                    raw_score = max(0.0, min(20.0, raw_score))
                    
                    # Rescale 0-20 → 0-100
                    rescaled_score = (raw_score / 20.0) * 100.0
                    individual_subscales[dim].append(rescaled_score)
                
                # Step 2: Calculate average (0-20 range)
                average_score = sum(subscale_scores) / len(subscale_scores)
                
                # Clamp average to valid range
                average_score = max(0.0, min(20.0, average_score))
                
                # Step 3: Rescale to 0-100
                combined_score = (average_score / 20.0) * 100.0
                
                combined_tlx_scores.append(combined_score)
                
                # Create multi-label target files
                # 1. Combined TLX (default, no suffix)
                base_name = f"S{subject_num}_{task_num}_features"
                target_file = features_path / f"{base_name}.txt"
                with open(target_file, 'w') as f:
                    f.write(f"{combined_score:.2f}")
                
                # 2. Individual subscales (with suffix)
                for dim in TLX_DIMENSIONS:
                    json_key = TLX_JSON_KEY_MAPPING[dim]
                    subscale_score_raw = tlx_data[json_key]  # Expected: 0-20 scale
                    
                    # Clamp to valid range (0-20)
                    subscale_score_raw = max(0.0, min(20.0, subscale_score_raw))
                    
                    # Rescale 0-20 → 0-100
                    subscale_score_rescaled = (subscale_score_raw / 20.0) * 100.0
                    
                    subscale_target_file = features_path / f"{base_name}_{dim}.txt"
                    with open(subscale_target_file, 'w') as f:
                        f.write(f"{subscale_score_rescaled:.2f}")
                
                created_targets += 1
                
            except Exception as e:
                print(f"Error processing TLX for Subject {subject_num}, Task {task_num}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TARGET FILE CREATION SUMMARY")
    print(f"{'='*60}")
    print(f"Recordings processed: {created_targets}")
    print(f"Target files created: {created_targets * 7} (combined + 6 subscales each)")
    print(f"Missing TLX files: {missing_tlx}")
    print(f"Missing feature files: {missing_features}")
    print(f"Missing TLX dimensions: {missing_dimensions}")
    
    if combined_tlx_scores:
        scores_array = np.array(combined_tlx_scores)
        print(f"\nCombined TLX Score Statistics (0-100 scale):")
        print(f"  Mean: {scores_array.mean():.2f}")
        print(f"  Std: {scores_array.std():.2f}")
        print(f"  Min: {scores_array.min():.2f}")
        print(f"  Max: {scores_array.max():.2f}")
        print(f"  Median: {np.median(scores_array):.2f}")
        
        print(f"\nIndividual Subscale Statistics (rescaled to 0-100):")
        for dimension in TLX_DIMENSIONS:
            if individual_subscales[dimension]:
                scores = np.array(individual_subscales[dimension])
                print(f"  {dimension}:")
                print(f"    Mean: {scores.mean():.2f}, Std: {scores.std():.2f}, Range: [{scores.min():.0f}, {scores.max():.0f}]")
    
    # Save summary
    summary = {
        'created_targets': created_targets,
        'missing_tlx': missing_tlx,
        'missing_features': missing_features,
        'missing_dimensions': missing_dimensions,
        'combined_tlx_scores': combined_tlx_scores,
        'combined_tlx_statistics': {
            'mean': float(np.mean(combined_tlx_scores)) if combined_tlx_scores else None,
            'std': float(np.std(combined_tlx_scores)) if combined_tlx_scores else None,
            'min': float(np.min(combined_tlx_scores)) if combined_tlx_scores else None,
            'max': float(np.max(combined_tlx_scores)) if combined_tlx_scores else None,
            'median': float(np.median(combined_tlx_scores)) if combined_tlx_scores else None
        },
        'individual_subscale_statistics': {
            dim: {
                'scores': individual_subscales[dim],
                'mean': float(np.mean(individual_subscales[dim])) if individual_subscales[dim] else None,
                'std': float(np.std(individual_subscales[dim])) if individual_subscales[dim] else None,
                'min': float(np.min(individual_subscales[dim])) if individual_subscales[dim] else None,
                'max': float(np.max(individual_subscales[dim])) if individual_subscales[dim] else None
            }
            for dim in TLX_DIMENSIONS
        }
    }
    
    summary_file = features_path / 'target_creation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTarget creation summary saved to: {summary_file}")
    
    return summary


def cleanup_parquet_files_without_targets(features_dir):
    """
    Remove feature files that don't have corresponding combined TLX target files.
    
    STEP 3.5: Ensures data consistency by removing unpaired files.
    
    Args:
        features_dir (str): Directory containing feature and target files
        
    Returns:
        Dictionary with cleanup statistics
    """
    
    features_path = Path(features_dir)
    
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None

    print(f"\n{'='*60}")
    print(f"CLEANING UP FILES WITHOUT TARGETS")
    print(f"{'='*60}")
    print(f"Directory: {features_path}")
    print(f"{'='*60}\n")

    # Get all feature files
    feature_files = list(features_path.glob("S*_*_features.parquet"))
    
    if not feature_files:
        print("No feature files found")
        return None
    
    print(f"Found {len(feature_files)} feature files")
    
    removed_count = 0
    kept_count = 0
    
    for feature_file in feature_files:
        # Get corresponding target file name (combined TLX)
        base_name = feature_file.stem.replace('_features', '')
        target_file = features_path / f"{base_name}_features.txt"
        
        if not target_file.exists():
            print(f"  Removing (no target): {feature_file.name}")
            feature_file.unlink()
            removed_count += 1
        else:
            kept_count += 1
    
    print(f"\n{'='*60}")
    print(f"CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"Feature files kept: {kept_count}")
    print(f"Feature files removed: {removed_count}")
    
    return {
        'kept_files': kept_count,
        'removed_files': removed_count
    }


def create_regression_datasets(raw_dir, features_dir, time_output_dir, feature_output_dir):
    """
    Create regression datasets with multi-label TLX scores.
    
    STEP 4: Copies paired raw EEG/features + multi-label TLX targets into regression directories.
            Each recording has 7 target files: combined + 6 subscales.
    
    Args:
        raw_dir (str): Directory containing raw EEG files
        features_dir (str): Directory containing feature files and targets
        time_output_dir (str): Output path for time regression dataset
        feature_output_dir (str): Output path for feature regression dataset
        
    Returns:
        Dictionary with dataset creation statistics
    """
    
    raw_path = Path(raw_dir)
    features_path = Path(features_dir)
    time_output = Path(time_output_dir)
    feature_output = Path(feature_output_dir)
    
    try:
        time_output.mkdir(parents=True, exist_ok=True)
        feature_output.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directories: {e}")
        return None

    print(f"\n{'='*60}")
    print(f"CREATING REGRESSION DATASETS (MULTI-LABEL)")
    print(f"{'='*60}")
    print(f"Time regression output: {time_output}")
    print(f"Feature regression output: {feature_output}")
    print(f"Format: Combined TLX + 6 subscales per recording")
    print(f"{'='*60}\n")

    copied_raw = 0
    copied_features = 0
    copied_targets = 0
    
    # Get all combined TLX target files (these are the base files)
    target_files = list(features_path.glob("S*_*_features.txt"))
    
    print(f"Found {len(target_files)} recordings with target files\n")
    
    for target_file in sorted(target_files):
        # Parse base filename (e.g., S01_1_features.txt -> S01_1)
        base_name = target_file.stem  # S01_1_features
        
        # Extract subject_task identifier (e.g., S01_1)
        subject_task = base_name.replace('_features', '')
        
        # Find corresponding raw and feature files
        raw_file = raw_path / f"{subject_task}_eeg_raw.parquet"
        feature_file = features_path / f"{base_name}.parquet"
        
        # Copy raw EEG + all 7 target files to time regression dataset
        if raw_file.exists():
            shutil.copy2(raw_file, time_output / raw_file.name)
            
            # Copy combined TLX target (rename to match raw file)
            target_base_raw = f"{subject_task}_eeg_raw"
            shutil.copy2(target_file, time_output / f"{target_base_raw}.txt")
            
            # Copy individual subscale targets (rename to match raw file)
            for dim in TLX_DIMENSIONS:
                subscale_target_src = features_path / f"{base_name}_{dim}.txt"
                if subscale_target_src.exists():
                    shutil.copy2(subscale_target_src, time_output / f"{target_base_raw}_{dim}.txt")
                    copied_targets += 1
            
            copied_raw += 1
        
        # Copy features + all 7 target files to feature regression dataset
        if feature_file.exists():
            shutil.copy2(feature_file, feature_output / feature_file.name)
            
            # Copy combined TLX target (keep features name)
            shutil.copy2(target_file, feature_output / f"{base_name}.txt")
            
            # Copy individual subscale targets (keep features name)
            for dim in TLX_DIMENSIONS:
                subscale_target_src = features_path / f"{base_name}_{dim}.txt"
                if subscale_target_src.exists():
                    shutil.copy2(subscale_target_src, feature_output / f"{base_name}_{dim}.txt")
            
            copied_features += 1
    
    print(f"\n{'='*60}")
    print(f"REGRESSION DATASET CREATION SUMMARY")
    print(f"{'='*60}")
    print(f"Recordings processed: {len(target_files)}")
    print(f"Raw EEG files copied: {copied_raw}")
    print(f"Feature files copied: {copied_features}")
    print(f"Target files per recording: 7 (combined + 6 subscales)")
    print(f"Total target files copied: {len(target_files) * 7}")
    
    return {
        'copied_raw': copied_raw,
        'copied_features': copied_features,
        'copied_recordings': len(target_files),
        'copied_target_files': len(target_files) * 7,
        'time_output_path': str(time_output),
        'feature_output_path': str(feature_output)
    }


def create_classification_datasets(raw_dir, features_dir, time_output_dir, feature_output_dir):
    """
    Create classification datasets by binning combined TLX scores into Low/Medium/High workload classes.
    
    STEP 5: Organizes files by workload class using 33rd and 67th percentiles as bin boundaries.
    
    Creates a SINGLE classification dataset with multi-subscale metadata (zero duplication).
    Instead of creating separate folders for each subscale, this function:
    1. Copies .parquet files to 'all' folder (single copy)
    2. Generates metadata JSON mapping: {filename: {combined: label, mental: label, ...}}
    3. Dataset loader uses target_suffix to select which label column to use
    
    Args:
        raw_dir (str): Directory containing raw EEG files
        features_dir (str): Directory containing feature files and targets
        time_output_dir (str): Output path for time classification dataset
        feature_output_dir (str): Output path for feature classification dataset
        
    Returns:
        Dictionary with classification dataset statistics
    """
    
    raw_path = Path(raw_dir)
    features_path = Path(features_dir)
    time_output = Path(time_output_dir)
    feature_output = Path(feature_output_dir)
    
    # Create 'all' directories (single copy of data files)
    (time_output / 'all').mkdir(parents=True, exist_ok=True)
    (feature_output / 'all').mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CREATING MULTI-SUBSCALE CLASSIFICATION DATASETS")
    print(f"{'='*60}")
    print(f"Time classification output: {time_output}")
    print(f"Feature classification output: {feature_output}")
    print(f"Strategy: Single data copy + metadata JSON (zero duplication)")
    print(f"Subscales: combined, mental, physical, temporal, performance, effort, frustration")
    print(f"{'='*60}\n")

    # Collect all target files (combined + 6 subscales)
    subscales = ['combined', 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']
    target_patterns = {
        'combined': "S*_*_features.txt",  # Default combined TLX
        'mental': "S*_*_features_mental.txt",
        'physical': "S*_*_features_physical.txt",
        'temporal': "S*_*_features_temporal.txt",
        'performance': "S*_*_features_performance.txt",
        'effort': "S*_*_features_effort.txt",
        'frustration': "S*_*_features_frustration.txt"
    }
    
    # Read scores for all subscales
    subscale_scores = {subscale: {} for subscale in subscales}  # {subscale: {subject_task: score}}
    all_subject_tasks = set()
    
    for subscale, pattern in target_patterns.items():
        target_files = list(features_path.glob(pattern))
        
        if not target_files:
            print(f"Warning: No {subscale} target files found (pattern: {pattern})")
            continue
        
        for target_file in target_files:
            # Extract subject_task identifier
            if subscale == 'combined':
                subject_task = target_file.stem.replace('_features', '')
            else:
                subject_task = target_file.stem.replace(f'_features_{subscale}', '')
            
            try:
                with open(target_file, 'r') as f:
                    score = float(f.read().strip())
                    subscale_scores[subscale][subject_task] = score
                    all_subject_tasks.add(subject_task)
            except Exception as e:
                print(f"Warning: Could not read {target_file.name}: {e}")
    
    if not all_subject_tasks:
        print("No valid samples found")
        return None
    
    # Calculate bin boundaries for each subscale
    subscale_bins = {}
    
    for subscale in subscales:
        scores_dict = subscale_scores[subscale]
        
        if not scores_dict:
            print(f"Skipping {subscale} - no scores found")
            continue
        
        scores_array = np.array(list(scores_dict.values()))
        p33 = np.percentile(scores_array, 33.33)
        p67 = np.percentile(scores_array, 66.67)
        
        subscale_bins[subscale] = {'p33': p33, 'p67': p67}
        
        print(f"\n{subscale.upper()} Score Distribution (0-100 scale):")
        print(f"  Min: {scores_array.min():.2f}")
        print(f"  33rd percentile: {p33:.2f}")
        print(f"  Median: {np.median(scores_array):.2f}")
        print(f"  67th percentile: {p67:.2f}")
        print(f"  Max: {scores_array.max():.2f}")
        print(f"  Bin boundaries -> Low: <{p33:.2f} | Medium: {p33:.2f}-{p67:.2f} | High: >={p67:.2f}")
    
    # Build metadata: {filename: {subscale: class_label}}
    metadata = {}
    class_counts = {subscale: {'low': 0, 'medium': 0, 'high': 0} for subscale in subscales}
    
    for subject_task in sorted(all_subject_tasks):
        sample_metadata = {}
        
        for subscale in subscales:
            if subject_task not in subscale_scores[subscale]:
                continue
            
            score = subscale_scores[subscale][subject_task]
            p33 = subscale_bins[subscale]['p33']
            p67 = subscale_bins[subscale]['p67']
            
            # Determine class
            if score < p33:
                class_label = 0  # low
                class_counts[subscale]['low'] += 1
            elif score < p67:
                class_label = 1  # medium
                class_counts[subscale]['medium'] += 1
            else:
                class_label = 2  # high
                class_counts[subscale]['high'] += 1
            
            sample_metadata[subscale] = class_label
        
        # Store metadata using base filename (without subscale suffix)
        raw_filename = f"{subject_task}_eeg_raw.parquet"
        feature_filename = f"{subject_task}_features.parquet"
        
        metadata[raw_filename] = sample_metadata
        metadata[feature_filename] = sample_metadata
        
        # Copy files to 'all' directory (single copy)
        raw_file = raw_path / raw_filename
        feature_file = features_path / feature_filename
        
        if raw_file.exists():
            shutil.copy2(raw_file, time_output / 'all' / raw_file.name)
        
        if feature_file.exists():
            shutil.copy2(feature_file, feature_output / 'all' / feature_file.name)
    
    # Save metadata JSON files
    time_metadata_path = time_output / 'classification_metadata.json'
    feature_metadata_path = feature_output / 'classification_metadata.json'
    
    import json
    with open(time_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(feature_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION DATASET CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_subject_tasks)}")
    print(f"Data files stored in: {time_output / 'all'} and {feature_output / 'all'}")
    print(f"Metadata files: classification_metadata.json")
    print(f"\nClass distributions:")
    
    for subscale in subscales:
        if subscale in class_counts and sum(class_counts[subscale].values()) > 0:
            counts = class_counts[subscale]
            total = sum(counts.values())
            print(f"  {subscale.capitalize()}:")
            print(f"    Low: {counts['low']} ({counts['low']/total*100:.1f}%)")
            print(f"    Medium: {counts['medium']} ({counts['medium']/total*100:.1f}%)")
            print(f"    High: {counts['high']} ({counts['high']/total*100:.1f}%)")
    
    return {
        'total_samples': len(all_subject_tasks),
        'subscales': subscales,
        'class_counts': class_counts,
        'subscale_bins': subscale_bins,
        'time_output_path': str(time_output),
        'feature_output_path': str(feature_output)
    }


def run_full_pipeline():
    """
    Run the complete N-Back preprocessing pipeline.
    
    Steps:
    1. Extract and reformat phase2 EEG files (with subband decomposition)
    2. Extract statistical features from ORIGINAL raw data
    3. Create combined TLX target files (0-100 scale)
    3.5. Cleanup files without targets
    4. Create regression datasets (2 datasets: time + features)
    5. Create classification datasets (2 datasets: time + features, binned into low/medium/high)
    
    Final Output: 4 datasets total
        - nback_time_regression/
        - nback_feature_regression/
        - nback_time_classification/
        - nback_feature_classification/
    """
    
    print(f"\n{'#'*60}")
    print(f"# N-BACK DATASET PREPROCESSING PIPELINE")
    print(f"# Combined TLX Score (0-100 scale)")
    print(f"# Output: 4 datasets total")
    print(f"{'#'*60}\n")
    
    # STEP 1: Extract and reformat raw EEG (Phase 2 only)
    print(f"\n{'='*60}")
    print(f"STEP 1: EXTRACTING AND REFORMATTING RAW EEG (PHASE 2)")
    print(f"{'='*60}")
    reformat_raw_eeg(data_path, raw_output_path, sampling_rate=128.0)
    
    # STEP 2: Extract features (FIXED VERSION - loads original raw data)
    print(f"\n{'='*60}")
    print(f"STEP 2: EXTRACTING STATISTICAL FEATURES (FIXED VERSION)")
    print(f"{'='*60}")
    extract_features_from_nback_files(
        raw_output_path, 
        features_output_path,
        original_data_dir=data_path,  # Pass original data path to prevent double-processing
        sampling_rate=128.0
    )
    
    # STEP 3: Create combined TLX target files
    print(f"\n{'='*60}")
    print(f"STEP 3: CREATING COMBINED TLX TARGET FILES (0-100 SCALE)")
    print(f"{'='*60}")
    create_target_files_from_tlx(data_path, features_output_path)
    
    # STEP 3.5: Cleanup files without targets
    print(f"\n{'='*60}")
    print(f"STEP 3.5: CLEANUP FILES WITHOUT TARGETS")
    print(f"{'='*60}")
    cleanup_parquet_files_without_targets(features_output_path)
    
    # STEP 4: Create regression datasets
    print(f"\n{'='*60}")
    print(f"STEP 4: CREATING REGRESSION DATASETS")
    print(f"{'='*60}")
    create_regression_datasets(
        raw_output_path,
        features_output_path,
        time_regression_path,
        feature_regression_path
    )
    
    # STEP 5: Create classification datasets
    print(f"\n{'='*60}")
    print(f"STEP 5: CREATING CLASSIFICATION DATASETS")
    print(f"{'='*60}")
    create_classification_datasets(
        raw_output_path,
        features_output_path,
        time_classification_path,
        feature_classification_path
    )
    
    print(f"\n{'#'*60}")
    print(f"# PIPELINE COMPLETE!")
    print(f"# 4 Datasets Created (Multi-Label Regression Format):")
    print(f"#   1. Time Regression: {time_regression_path}")
    print(f"#      - 7 target files per recording (combined + 6 subscales)")
    print(f"#   2. Feature Regression: {feature_regression_path}")
    print(f"#      - 7 target files per recording (combined + 6 subscales)")
    print(f"#   3. Time Classification: {time_classification_path}")
    print(f"#      - Multi-subscale metadata (zero duplication)")
    print(f"#   4. Feature Classification: {feature_classification_path}")
    print(f"#      - Multi-subscale metadata (zero duplication)")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='N-Back Dataset Processing Pipeline for EEG Cognitive Workload Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (all steps)
  python load_nback.py
  
  # Only extract raw EEG without processing (for LaBraM embeddings)
  python load_nback.py --raw-only
  
  # Only create time-series datasets (skip feature extraction)
  python load_nback.py --time-only
  
  # Only create feature datasets (skip time-series)
  python load_nback.py --feature-only

Steps:
  1   - Extract and reformat raw EEG (with subband decomposition)
  2   - Extract statistical features
  3   - Create target files from TLX
  3.5 - Cleanup files without targets
  4   - Create regression datasets
  5   - Create classification datasets
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--raw-only',
        action='store_true',
        help='Only extract raw EEG without signal processing (for LaBraM embeddings). Saves to _rawonly folder.'
    )
    mode_group.add_argument(
        '--time-only',
        action='store_true',
        help='Only create time-series datasets (skip feature extraction)'
    )
    mode_group.add_argument(
        '--feature-only',
        action='store_true',
        help='Only create feature datasets'
    )
    
    args = parser.parse_args()
    
    if args.raw_only:
        # Only extract raw EEG without processing
        print(f"\n{'#'*60}")
        print(f"# N-BACK RAW EEG EXTRACTION (NO PROCESSING)")
        print(f"# For LaBraM embeddings")
        print(f"{'#'*60}\n")
        print("Note: TLX target files should already exist from previous full pipeline run.")
        print("If missing, run without --raw-only flag to create them.\n")
        reformat_raw_eeg(data_path, raw_output_path, sampling_rate=128.0, raw_only=True)
        print(f"\n{'#'*60}")
        print(f"# RAW EXTRACTION COMPLETE!")
        print(f"{'#'*60}\n")
    elif args.time_only:
        # Time-series only mode
        print(f"\n{'#'*60}")
        print(f"# N-BACK TIME-SERIES ONLY PIPELINE")
        print(f"{'#'*60}\n")
        reformat_raw_eeg(data_path, raw_output_path, sampling_rate=128.0)
        create_target_files_from_tlx(data_path, raw_output_path)
        cleanup_parquet_files_without_targets(raw_output_path)
        create_regression_datasets(raw_output_path, features_output_path, time_regression_path, feature_regression_path)
        create_classification_datasets(raw_output_path, features_output_path, time_classification_path, feature_classification_path)
        print(f"\n{'#'*60}")
        print(f"# TIME-SERIES PIPELINE COMPLETE!")
        print(f"{'#'*60}\n")
    elif args.feature_only:
        # Feature-only mode
        print(f"\n{'#'*60}")
        print(f"# N-BACK FEATURE-ONLY PIPELINE")
        print(f"{'#'*60}\n")
        reformat_raw_eeg(data_path, raw_output_path, sampling_rate=128.0)
        extract_features_from_nback_files(raw_output_path, features_output_path, original_data_dir=data_path, sampling_rate=128.0)
        create_target_files_from_tlx(data_path, features_output_path)
        cleanup_parquet_files_without_targets(features_output_path)
        create_regression_datasets(raw_output_path, features_output_path, time_regression_path, feature_regression_path)
        create_classification_datasets(raw_output_path, features_output_path, time_classification_path, feature_classification_path)
        print(f"\n{'#'*60}")
        print(f"# FEATURE PIPELINE COMPLETE!")
        print(f"{'#'*60}\n")
    else:
        # Run full pipeline (default)
        run_full_pipeline()
