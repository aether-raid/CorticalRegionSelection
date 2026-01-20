import os 
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import json

# Add parent directories to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from channel_importance.eeg import EEG


"""
MOCAS Dataset Processing Pipeline for EEG Cognitive Workload Analysis
====================================================================

Complete preprocessing pipeline for the MOCAS (Multi-Operator Cognitive Assessment) dataset 
that transforms raw Kaggle data into processed features suitable for regression analysis.

DATASET SOURCE:
    MOCAS dataset from Kaggle: https://www.kaggle.com/datasets/prannayagupta/mocas-dataset
    Contains EEG data for cognitive workload classification with TLX scores.

EXPECTED INPUT STRUCTURE:
    data/MOCAS/MOCAS/    # Root directory from Kaggle
        participant1/
            task0/
                raw_eeg.parquet      # Raw EEG signals (Emotiv EPOC+, 128Hz)
                metadata.json        # Contains workload class: "low", "medium", "high"
            task1/
                raw_eeg.parquet
                metadata.json
            ...task0-8
        participant2/
            task0/
                raw_eeg.parquet
                metadata.json
            ...
        ...participant1-30
        scores.csv              # TLX workload scores (used for validation)

PROCESSING PIPELINE (6 STEPS):
    
    STEP 1: EEG File Extraction & Reformatting
        • Flattens nested directory structure
        • Removes RAW_CQ (contact quality) column 
        • Standardizes filenames: P{participant}_{task}_eeg_raw.parquet
        • Output: data/MOCAS/raw_eeg_extracted/
    
    STEP 2: Raw EEG Workload Organization  
        • Reads workload classification from metadata.json files
        • Organizes raw EEG files by cognitive workload class
        • Output: data/MOCAS/mocas_time_classification_dataset/{low,medium,high}/
        
    STEP 3: Statistical Feature Extraction
        • Applies signal processing: bandpass filtering, frequency decomposition
        • Extracts 400+ statistical features per file using EEG class
        • Features: power bands, spectral entropy, hjorth parameters, etc.
        • Output: data/MOCAS/features_extracted/P{X}_{Y}_features.parquet
        
    STEP 3.5: Feature Workload Organization
        • Organizes extracted feature files by cognitive workload class
        • Enables workload-specific analysis and model training
        • Output: data/MOCAS/mocas_feature_classification_dataset/{low,medium,high}/
        
    STEP 4: Target File Creation
        • Creates regression targets from TLX raw scores
        • Matches feature files with corresponding target values
        • Output: data/MOCAS/features_extracted/P{X}_{Y}_target.txt
        
    STEP 5: Data Consistency Cleanup
        • Removes feature files without corresponding targets
        • Ensures paired feature-target files for regression analysis

FINAL OUTPUT STRUCTURE:
    data/MOCAS/
        mocas_time_classification_dataset/ # Raw EEG files organized by workload
            low/
                P1_0_eeg_raw.parquet
                P3_1_eeg_raw.parquet
                ...
            medium/
                P2_0_eeg_raw.parquet
                ...
            high/
                P4_0_eeg_raw.parquet
                ...
        mocas_feature_classification_dataset/ # Feature files organized by workload
            low/
                P1_0_features.parquet
                P3_1_features.parquet
                ...
            medium/
                P2_0_features.parquet
                ...
            high/
                P4_0_features.parquet
                ...
        mocas_time_regression_dataset/      # Raw EEG files with TLX targets
            P1_0_eeg_raw.parquet
            P1_0_target.txt         # TLX workload score
            P1_1_eeg_raw.parquet
            P1_1_target.txt
            ...
        mocas_feature_regression_dataset/   # Statistical features with TLX targets
            P1_0_features.parquet   # Statistical features
            P1_0_target.txt         # TLX workload score
            P1_1_features.parquet
            P1_1_target.txt
            ...

INTEGRATION:
    The processed data is compatible with:
    • EEGRawRegressionDataset for loading features and targets
    • test_feature_regression.py for ML model evaluation
    • Channel importance analysis and feature selection methods

USAGE:
    python channel_importance/MOCAS/load_mocas.py
    
    Or call individual functions:
    • reformat_raw_eeg()
    • organize_extracted_eeg_by_workload()  
    • extract_features_from_mocas_files()
    • create_target_files_from_scores()
    • cleanup_parquet_files_without_targets()
"""

# Use relative paths from project root
data_path = "data/MOCAS/MOCAS"
output_path = "data/MOCAS/mocas_time_regression_dataset"  # Raw EEG time series data (with TLX targets)
features_output_path = "data/MOCAS/mocas_feature_regression_dataset"  # Statistical features data (with TLX targets)
workload_classes_path = "data/MOCAS"  # Base directory for workload classes

# Configuration flag: Set to False to disable old folder-based classification (recommended)
CREATE_OLD_CLASSIFICATION_DATASETS = False  # Set to True for backward compatibility


def organize_raw_eeg_by_workload(extracted_eeg_dir, metadata_source_dir, output_base_dir):
    """
    Organize already extracted EEG .parquet files by workload class based on metadata.json.
    
    This method should be called AFTER EEG extraction but BEFORE feature extraction.
    
    Args:
        extracted_eeg_dir (str): Directory containing extracted P{X}_{Y}_eeg_raw.parquet files
        metadata_source_dir (str): Directory containing original participant folders with metadata.json
        output_base_dir (str): Base path where workload class subdirectories will be created
        
    Expected structure:
        output_base_dir/
            workload_classes/
                low/
                    P1_0_eeg_raw.parquet
                    P3_1_eeg_raw.parquet
                    ...
                medium/
                    P2_0_eeg_raw.parquet
                    P5_1_eeg_raw.parquet
                    ...
                high/  
                    P4_0_eeg_raw.parquet
                    P6_1_eeg_raw.parquet
                    ...
    
    Returns:
        Dictionary with processing statistics and workload distribution
    """
    
    extracted_path = Path(extracted_eeg_dir)
    metadata_path = Path(metadata_source_dir)
    output_path = Path(output_base_dir) / "mocas_time_classification_dataset"
    
    if not extracted_path.exists():
        print(f"Error: Extracted EEG directory does not exist: {extracted_path}")
        return None
        
    if not metadata_path.exists():
        print(f"Error: Metadata source directory does not exist: {metadata_path}")
        return None
        
    # We'll read workload classes directly from metadata.json files
    # No need to load scores.csv since metadata already contains workload classification
    print("Reading workload classes directly from metadata.json files")
    
    # Create workload class directories
    workload_dirs = {
        'low': output_path / 'low',
        'medium': output_path / 'medium', 
        'high': output_path / 'high'
    }
    
    for workload_dir in workload_dirs.values():
        try:
            workload_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {workload_dir}: {e}")
            return None
    
    print("Reading workload classes from metadata.json and organizing EEG files")
    print("Created workload class directories: low, medium, high")
    
    # Find all extracted EEG files
    eeg_files = list(extracted_path.glob("P*_*_eeg_raw.parquet"))
    
    if not eeg_files:
        print(f"No extracted EEG files found in {extracted_path}")
        print("Expected format: P{{participant}}_{{task}}_eeg_raw.parquet")
        return None
    
    print(f"Found {len(eeg_files)} extracted EEG files to organize")
    print(f"{'='*60}")
    
    # Processing statistics
    processed_files = 0
    missing_metadata = 0
    missing_scores = 0
    workload_counts = {'low': 0, 'medium': 0, 'high': 0}
    processing_results = []
    
    # Process each EEG file
    for eeg_file in sorted(eeg_files):
        try:
            # Parse filename to get participant and task info
            filename = eeg_file.stem.replace('_eeg_raw', '')  # Remove _eeg_raw.parquet
            parts = filename.split('_')
            
            if len(parts) < 2:
                print(f"Warning: Cannot parse filename {eeg_file.name}")
                continue
                
            participant_num = parts[0].replace('P', '')  # Remove 'P' prefix
            task_num = parts[1]
            
            # Find corresponding metadata file
            metadata_file = metadata_path / f'participant{participant_num}' / f'task{task_num}' / 'metadata.json'
            
            if not metadata_file.exists():
                print(f"Warning: Missing metadata file: {metadata_file}")
                missing_metadata += 1
                continue
            
            # Read workload class directly from metadata.json
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Get workload class from metadata
                if 'workloads' in metadata:
                    workload_class = metadata['workloads'].lower()
                    # Validate workload class
                    if workload_class not in ['low', 'medium', 'high']:
                        print(f"Warning: Invalid workload class '{workload_class}' for P{participant_num}_T{task_num}, skipping")
                        continue
                else:
                    print(f"Warning: No 'workloads' field found in metadata for P{participant_num}_T{task_num}, skipping")
                    missing_scores += 1
                    continue
                    
            except Exception as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
                missing_metadata += 1
                continue
            
            # Copy EEG file to appropriate workload directory
            new_filename = f"P{participant_num}_{task_num}_eeg_raw.parquet"
            output_file = workload_dirs[workload_class] / new_filename
            
            try:
                # Copy EEG file to workload directory
                shutil.copy2(eeg_file, output_file)
                
                # Get file size info
                file_size_mb = eeg_file.stat().st_size / (1024 * 1024)
                
                workload_counts[workload_class] += 1
                processed_files += 1
                
                processing_results.append({
                    'participant': participant_num,
                    'task': task_num,
                    'workload_class': workload_class,
                    'original_eeg_path': str(eeg_file),
                    'new_eeg_path': str(output_file),
                    'metadata_path': str(metadata_file),
                    'file_size_mb': file_size_mb
                })
                
                print(f"Processed: P{participant_num}_T{task_num} -> {workload_class} ({file_size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"Error copying {eeg_file} -> {output_file}: {e}")
                
        except Exception as e:
            print(f"Error processing {eeg_file}: {e}")    # Print summary
    print(f"\n{'='*60}")
    print(f"EEG FILE ORGANIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed files: {processed_files}")
    print(f"Missing metadata files: {missing_metadata}")
    print(f"Missing EEG files: {missing_scores}")
    print(f"\nWorkload distribution:")
    for workload, count in workload_counts.items():
        percentage = (count / processed_files * 100) if processed_files > 0 else 0
        print(f"  {workload.capitalize()}: {count} files ({percentage:.1f}%)")
    
    # Save processing results
    results_file = output_path / 'eeg_workload_organization_results.json'
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'processing_stats': {
                    'processed_files': processed_files,
                    'missing_metadata': missing_metadata,
                    'missing_scores': missing_scores,
                    'workload_distribution': workload_counts
                },
                'classification_method': 'metadata_based',
                'processed_files_details': processing_results
            }, f, indent=4)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save results file: {e}")
    
    return {
        'processed_files': processed_files,
        'workload_distribution': workload_counts,
        'classification_method': 'metadata_based',
        'results_file': str(results_file)
    }


def reformat_raw_eeg(source_dir, output_dir, sampling_rate=128.0, raw_only=False):
    """
    Reformat and extract raw EEG files from MOCAS dataset.
    
    Extracts EEG data from the MOCAS dataset. If raw_only=False, performs subband
    decomposition (Overall, delta, theta, alpha, beta, gamma) for each EEG channel,
    creating a MultiIndex DataFrame with (band, channel) structure.
    If raw_only=True, just copies the cleaned raw EEG without any processing.
    
    Args:
        source_dir (str): Path to the source directory containing the raw EEG files.
        output_dir (str): Path to the output directory for the reformatted files.
        sampling_rate (float): Sampling rate in Hz (default: 128.0 for Emotiv EPOC+)
        raw_only (bool): If True, skip all signal processing and just copy raw EEG.
                        Use this for LaBraM embeddings. Default: False.

    Returns:
        List of dictionaries with information about each processed file.
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
        print(f"MOCAS EEG EXTRACTION - RAW ONLY")
    else:
        print(f"MOCAS EEG EXTRACTION WITH SUBBAND DECOMPOSITION")
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
    
    # Get all participant folders
    participant_folders = [d for d in source_path.iterdir() 
                          if d.is_dir() and d.name.startswith('participant')]
    
    print(f"Found {len(participant_folders)} participant folders\n")
    
    for participant_folder in sorted(participant_folders):
        participant_num = participant_folder.name.replace('participant', '')
        print(f"Processing Participant {participant_num}")
        
        # Get all task folders
        task_folders = [d for d in participant_folder.iterdir() 
                       if d.is_dir() and d.name.startswith('task')]
        
        print(f"  Found {len(task_folders)} task folders")
        
        for task_folder in sorted(task_folders):
            task_num = task_folder.name.replace('task', '')
            raw_eeg_file = task_folder / 'raw_eeg.parquet'
            
            # Check if raw_eeg.parquet exists
            if not raw_eeg_file.exists():
                print(f"  ✗ Task {task_num}: Missing raw_eeg.parquet")
                missing_files += 1
                continue
            
            # Create new filename: P{X}_{Y}_eeg_raw.parquet
            new_filename = f"P{participant_num}_{task_num}_eeg_raw.parquet"
            output_file = output_path / new_filename
            
            try:
                # Load parquet file
                df = pd.read_parquet(raw_eeg_file)
                
                # Remove RAW_CQ column if it exists
                if 'RAW_CQ' in df.columns:
                    df = df.drop(columns=['RAW_CQ'])
                
                # Get EEG channels (all columns except 't' and 'RAW_CQ')
                eeg_channels = [col for col in df.columns if col != 't']
                
                if not eeg_channels:
                    raise ValueError("No valid EEG channels found")
                
                # Calculate duration
                time_seconds = df.index.values - df.index.values[0]  # Relative time in seconds
                duration = time_seconds[-1] if len(time_seconds) > 0 else 0
                
                if raw_only:
                    # Just save raw EEG without any processing
                    df_eeg = df[eeg_channels].copy()
                    df_eeg.to_parquet(output_file)
                    output_size = output_file.stat().st_size / (1024 * 1024)
                    
                    # Store processing info
                    file_info.append({
                        'participant': participant_num,
                        'task': task_num,
                        'original_path': str(raw_eeg_file),
                        'new_filename': new_filename,
                        'samples': len(df),
                        'channels': len(eeg_channels),
                        'bands': 0,  # No bands - raw only
                        'total_columns': len(eeg_channels),
                        'duration_seconds': float(duration),
                        'output_size_mb': output_size,
                        'channel_names': eeg_channels,
                        'band_names': [],
                        'raw_only': True,
                        'status': 'success'
                    })
                else:
                    # Prepare data for EEG class
                    sample_numbers = np.arange(len(df))
                    channels_dict = {col: df[col].values for col in eeg_channels}
                    
                    # Create EEG instance for subband decomposition
                    eeg = EEG(
                        s_n=sample_numbers,
                        t=time_seconds,
                        channels=channels_dict,
                        frequency=sampling_rate,
                        extract_time=False,  # Don't resample, just decompose
                        apply_notch=(50, 60)  # Apply 60Hz notch filter
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
                        'participant': participant_num,
                        'task': task_num,
                        'original_path': str(raw_eeg_file),
                        'new_filename': new_filename,
                        'samples': len(df),
                        'channels': len(eeg_channels),
                        'bands': len(subband_df.columns.get_level_values(0).unique()),
                        'total_columns': subband_df.shape[1],
                        'duration_seconds': float(duration),
                        'output_size_mb': output_size,
                        'channel_names': eeg_channels,
                        'band_names': list(subband_df.columns.get_level_values(0).unique()),
                        'raw_only': False,
                        'status': 'success'
                    })
                
                print(f"  ✓ Task {task_num}: {len(df)} samples, {duration:.1f}s, {output_size:.2f} MB")
                processed_files += 1
                
            except Exception as e:
                print(f"  ✗ Task {task_num}: Error - {str(e)}")
                failed_files += 1
                file_info.append({
                    'participant': participant_num,
                    'task': task_num,
                    'original_path': str(raw_eeg_file),
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
    

def extract_features_from_mocas_files(input_dir, output_dir, original_data_dir, sampling_rate=128.0):
    """
    Extract statistical features from cleaned MOCAS EEG files using the EEG class.
    
    FIXED VERSION: Loads ORIGINAL raw data instead of decomposed MultiIndex data
    to avoid double-processing (re-filtering and re-decomposing already processed data).
    
    Args:
        input_dir: Directory containing the cleaned EEG files (P{X}_{Y}_eeg_raw.parquet)
        output_dir: Directory to save the extracted features
        original_data_dir: Directory containing ORIGINAL raw_eeg.parquet files from Kaggle
        sampling_rate: Sampling frequency in Hz (default 128 for Emotiv EPOC+)
    
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
    
    # Find all cleaned EEG files
    eeg_files = list(input_path.glob("P*_*_eeg_raw.parquet"))
    
    if not eeg_files:
        print(f"No EEG files found in {input_path}")
        print("Expected format: P{participant}_{task}_eeg_raw.parquet")
        return None
    
    print(f"Found {len(eeg_files)} EEG files to process")
    print(f"{'='*60}")
    
    # Processing statistics
    processed_files = 0
    failed_files = 0
    processing_results = []
    failed_files_info = []
    
    # Process each file
    for eeg_file in tqdm(eeg_files, desc="Processing EEG files"):
        try:
            # Parse filename to get participant and task info
            filename = eeg_file.stem  # Remove .parquet extension
            parts = filename.split('_')
            
            if len(parts) >= 3:
                participant_id = parts[0]  # P10, P11, etc.
                task_id = parts[1]         # 0, 1, 2, etc.
                
                print(f"\nProcessing {participant_id}_task{task_id}")
                
                # FIXED VERSION: Load ORIGINAL raw data to avoid double-processing
                # Extract participant number from ID (e.g., "P10" -> "10")
                participant_num = participant_id.replace('P', '')
                
                # Construct path to original raw data
                original_file = original_data_path / f"participant{participant_num}" / f"task{task_id}" / "raw_eeg.parquet"
                
                if not original_file.exists():
                    raise FileNotFoundError(f"Original raw file not found: {original_file}")
                
                print(f"  Loading ORIGINAL raw data from: {original_file}")
                
                # Load the ORIGINAL raw EEG data (not the decomposed MultiIndex version)
                df = pd.read_parquet(original_file)
                print(f"  Loaded raw data: {df.shape[0]} samples, {df.shape[1]} columns")
                
                # Validate data
                if df.empty:
                    raise ValueError("Empty dataframe")
                
                if df.isnull().sum().sum() > 0:
                    print(f"  Warning: {df.isnull().sum().sum()} NaN values found")
                
                # Get EEG channels (exclude time and quality columns)
                eeg_channels = [col for col in df.columns if col != 't' and 'RAW_CQ' not in col]
                print(f"  Channels: {eeg_channels}")
                
                if not eeg_channels:
                    raise ValueError("No valid EEG channels found")
                
                # Create time array from index (convert to relative seconds)
                timestamps = df.index.values
                if len(timestamps) == 0:
                    raise ValueError("No timestamp data")
                
                # Convert Unix timestamps to relative time in seconds
                # MOCAS timestamps are already in Unix seconds (not nanoseconds)
                time_seconds = timestamps - timestamps[0]
                
                # Create sample numbers
                sample_numbers = np.arange(len(df))
                
                # Prepare channel data dictionary
                channels_dict = {col: df[col].values for col in eeg_channels}
                
                print(f"  Creating EEG instance with {len(eeg_channels)} channels, {len(sample_numbers)} samples")
                print(f"  Duration: {time_seconds[-1]:.2f} seconds, Sampling rate: {sampling_rate} Hz")
                
                # Create EEG instance from ORIGINAL raw data (processes ONCE correctly)
                eeg = EEG(
                    s_n=sample_numbers,
                    t=time_seconds,
                    channels=channels_dict,
                    frequency=sampling_rate,
                    apply_notch=(50, 60)  # Apply 60Hz notch filter
                )
                
                print(f"  EEG instance created successfully")
                print(f"  Generating statistical features...")
                eeg.generate_stats()
                
                # Get the feature matrix
                features = eeg.stats
                print(f"  Generated {features.shape[1]} features across {features.shape[0]} feature types")
                
                # Create output filename
                output_filename = f"{participant_id}_{task_id}_features.parquet"
                output_file = output_path / output_filename
                
                
                if isinstance(features.columns, pd.MultiIndex):
                    features.columns.names = ['band', 'channel']
                    print(features.columns.names)
                else:
                    features.columns = pd.MultiIndex.from_product([features.columns, ['']])
                    features.columns.names = ['band', 'channel']
                print(features.columns.names)
                
                # Save features directly (numpy types now converted at source)
                features.to_parquet(output_file, engine="fastparquet")
                print(f"  ✓ Features saved: {output_filename}")
                
                ##HERE
                # Calculate file sizes
               # input_size = eeg_file.stat().st_size / (1024 * 1024)  # MB
               # print("failed herARGFGGRTHeefe lawl")
                #output_size = output_file.stat().st_size / (1024 * 1024)  # MB
                #print("failed hereefe lawl")
                
                print(f"  ✓ Features saved: {output_filename}")
                #print(f"  File sizes: {input_size:.2f} MB -> {output_size:.2f} MB")
                
                # Store processing results
                processing_results.append({
                    'participant': participant_id,
                    'task': task_id,
                    'input_file': str(eeg_file),
                    'output_file': str(output_file),
                 #   'input_size_mb': input_size,
                  #  'output_size_mb': output_size,
                    'samples': len(df),
                    'channels': len(eeg_channels),
                    'channel_names': eeg_channels,
                    'duration_seconds': float(time_seconds[-1]),
                    'sampling_rate': sampling_rate,
                    'num_features': features.shape[1],
                    'num_feature_types': features.shape[0],
                    'feature_types': list(features.index),
                    'status': 'success'
                })
                
                processed_files += 1
                
            else:
                raise ValueError(f"Invalid filename format: {filename}")
                
        except Exception as e:
            print(f"Error processing {eeg_file.name}: {str(e)}")
            
            failed_files_info.append({
                'file': str(eeg_file),
                'error': str(e),
                'status': 'failed'
            })
            failed_files += 1
            
            # Continue with next file
            continue
    
    # Generate summary
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files found: {len(eeg_files)}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {failed_files}")
    print(f"Success rate: {processed_files/len(eeg_files)*100:.1f}%")
    
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
        for i, feat_type in enumerate(first_success['feature_types'], 1):
            print(f"  {i:2d}. {feat_type}")
    
    if failed_files_info:
        print(f"\nFailed Files:")
        for fail_info in failed_files_info[:5]:  # Show first 5 failures
            print(f"  - {Path(fail_info['file']).name}: {fail_info['error']}")
        if len(failed_files_info) > 5:
            print(f"  ... and {len(failed_files_info) - 5} more")
    
    # Save processing results
    results_summary = {
        'processing_stats': {
            'total_files': len(eeg_files),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'success_rate': processed_files/len(eeg_files)*100
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
    print(f"Feature files saved to: {output_path}")
    
    return results_summary


def organize_extracted_features_by_workload(features_dir, metadata_source_dir, output_base_dir):
    """
    Organize extracted feature files by workload class based on metadata.json.
    
    This function should be called AFTER feature extraction but BEFORE target creation.
    It organizes the processed feature files into workload-specific directories for 
    easier analysis and model training on specific cognitive load levels.
    
    Args:
        features_dir (str): Directory containing extracted P{X}_{Y}_features.parquet files
        metadata_source_dir (str): Directory containing original participant folders with metadata.json
        output_base_dir (str): Base path where workload class subdirectories will be created
        
    Expected structure:
        output_base_dir/
            feature_workload_classes/
                low/
                    P1_0_features.parquet
                    P3_1_features.parquet
                    ...
                medium/
                    P2_0_features.parquet
                    P5_1_features.parquet
                    ...
                high/  
                    P4_0_features.parquet
                    P6_1_features.parquet
                    ...
    
    Returns:
        Dictionary with processing statistics and workload distribution
    """
    
    features_path = Path(features_dir)
    metadata_path = Path(metadata_source_dir)
    output_path = Path(output_base_dir) / "mocas_feature_classification_dataset"
    
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None
        
    if not metadata_path.exists():
        print(f"Error: Metadata source directory does not exist: {metadata_path}")
        return None
    
    # Create workload class directories
    workload_dirs = {
        'low': output_path / 'low',
        'medium': output_path / 'medium', 
        'high': output_path / 'high'
    }
    
    for workload_dir in workload_dirs.values():
        try:
            workload_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {workload_dir}: {e}")
            return None
    
    print("Organizing extracted feature files by workload class")
    print("Created feature workload directories: low, medium, high")
    
    # Find all extracted feature files
    feature_files = list(features_path.glob("P*_*_features.parquet"))
    
    if not feature_files:
        print(f"No feature files found in {features_path}")
        print("Expected format: P{participant}_{task}_features.parquet")
        return None
    
    print(f"Found {len(feature_files)} feature files to organize")
    print(f"{'='*60}")
    
    # Processing statistics
    processed_files = 0
    missing_metadata = 0
    missing_features = 0
    workload_counts = {'low': 0, 'medium': 0, 'high': 0}
    processing_results = []
    
    # Process each feature file
    for feature_file in sorted(feature_files):
        try:
            # Parse filename to get participant and task info
            filename = feature_file.stem.replace('_features', '')  # Remove _features.parquet
            parts = filename.split('_')
            
            if len(parts) < 2:
                print(f"Warning: Cannot parse filename {feature_file.name}")
                continue
                
            participant_num = parts[0].replace('P', '')  # Remove 'P' prefix
            task_num = parts[1]
            
            # Find corresponding metadata file
            metadata_file = metadata_path / f'participant{participant_num}' / f'task{task_num}' / 'metadata.json'
            
            if not metadata_file.exists():
                print(f"Warning: Missing metadata file: {metadata_file}")
                missing_metadata += 1
                continue
            
            # Read workload class from metadata
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                workload_class = metadata.get('workloads', '').lower()
                
                if workload_class not in ['low', 'medium', 'high']:
                    print(f"Warning: Invalid workload class '{workload_class}' in {metadata_file}")
                    continue
                    
            except Exception as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
                missing_metadata += 1
                continue
            
            # Copy feature file to appropriate workload directory
            new_filename = f"P{participant_num}_{task_num}_features.parquet"
            output_file = workload_dirs[workload_class] / new_filename
            
            try:
                # Copy feature file to workload directory
                shutil.copy2(feature_file, output_file)
                
                # Get file size info
                file_size_mb = feature_file.stat().st_size / (1024 * 1024)
                
                workload_counts[workload_class] += 1
                processed_files += 1
                
                processing_results.append({
                    'participant': participant_num,
                    'task': task_num,
                    'workload_class': workload_class,
                    'original_features_path': str(feature_file),
                    'new_features_path': str(output_file),
                    'metadata_path': str(metadata_file),
                    'file_size_mb': file_size_mb
                })
                
                print(f"Processed: P{participant_num}_T{task_num} -> {workload_class} ({file_size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"Error copying {feature_file} -> {output_file}: {e}")
                
        except Exception as e:
            print(f"Error processing {feature_file}: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"FEATURE ORGANIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed files: {processed_files}")
    print(f"Missing metadata files: {missing_metadata}")
    print(f"Missing feature files: {missing_features}")
    print(f"\nWorkload distribution:")
    for workload, count in workload_counts.items():
        percentage = (count / processed_files * 100) if processed_files > 0 else 0
        print(f"  {workload.capitalize()}: {count} files ({percentage:.1f}%)")
    
    # Save processing results
    results_file = output_path / 'feature_workload_organization_results.json'
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'processing_stats': {
                    'processed_files': processed_files,
                    'missing_metadata': missing_metadata,
                    'missing_features': missing_features,
                    'workload_distribution': workload_counts
                },
                'classification_method': 'metadata_based',
                'processed_files_details': processing_results
            }, f, indent=4)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save results file: {e}")
    
    return {
        'processed_files': processed_files,
        'workload_distribution': workload_counts,
        'classification_method': 'metadata_based',
        'results_file': str(results_file)
    }


def create_metadata_based_classification_datasets(raw_eeg_dir, features_dir, output_base_dir):
    """
    Create metadata-based multi-subscale classification datasets for MOCAS.
    
    This function creates classification datasets compatible with EEGRawDataset's
    subscale support, matching the format used by HTC and N-Back datasets.
    
    Structure created:
        mocas_time_classification/
            all/
                P10_0_eeg_raw.parquet
                P10_1_eeg_raw.parquet
                ...
            classification_metadata.json
        
        mocas_feature_classification/
            all/
                P10_0_features.parquet
                P10_1_features.parquet
                ...
            classification_metadata.json
    
    Metadata JSON format:
        {
            "P10_0_eeg_raw.parquet": {
                "combined": 0,    # 0=low, 1=medium, 2=high
                "mental": 1,
                "physical": 0,
                ...
            },
            ...
        }
    
    Binning strategy:
        - Uses tertiles (33rd, 67th percentiles) independently for each subscale
        - Low: score < 33rd percentile
        - Medium: 33rd percentile <= score < 67th percentile
        - High: score >= 67th percentile
    
    Args:
        raw_eeg_dir: Directory containing raw EEG files with target files
        features_dir: Directory containing feature files with target files
        output_base_dir: Base directory for classification outputs
    
    Returns:
        Dictionary with processing statistics
    """
    import numpy as np
    
    raw_path = Path(raw_eeg_dir)
    features_path = Path(features_dir)
    
    # Create output directories
    time_output = Path(output_base_dir) / "mocas_time_classification"
    feature_output = Path(output_base_dir) / "mocas_feature_classification"
    
    time_all = time_output / 'all'
    feature_all = feature_output / 'all'
    
    time_all.mkdir(parents=True, exist_ok=True)
    feature_all.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"CREATING METADATA-BASED CLASSIFICATION DATASETS")
    print(f"{'='*60}")
    print(f"Time classification output: {time_output}")
    print(f"Feature classification output: {feature_output}")
    print(f"Format: Single data copy + metadata JSON (zero duplication)")
    print(f"Subscales: combined, mental, physical, temporal, performance, effort, frustration")
    print(f"{'='*60}")
    
    # TLX subscales
    subscales = ['combined', 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']
    
    # Step 1: Read all target files and collect scores
    subscale_scores = {subscale: {} for subscale in subscales}
    all_subject_tasks = set()
    
    print(f"\nReading target files...")
    
    for subscale in subscales:
        if subscale == 'combined':
            pattern = 'P*_*_features.txt'
        else:
            pattern = f'P*_*_features_{subscale}.txt'
        
        target_files = list(features_path.glob(pattern))
        
        if not target_files:
            print(f"Warning: No {subscale} target files found (pattern: {pattern})")
            continue
        
        for target_file in target_files:
            # Extract subject_task identifier (e.g., "P10_0")
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
        print("❌ No valid samples found")
        return None
    
    print(f"Found {len(all_subject_tasks)} unique recordings")
    
    # Step 2: Calculate bin boundaries for each subscale (tertiles)
    subscale_bins = {}
    
    print(f"\nCalculating tertile bins for each subscale...")
    
    for subscale in subscales:
        scores_dict = subscale_scores[subscale]
        
        if not scores_dict:
            print(f"  Skipping {subscale} - no scores found")
            continue
        
        scores_array = np.array(list(scores_dict.values()))
        p33 = np.percentile(scores_array, 33.33)
        p67 = np.percentile(scores_array, 66.67)
        
        subscale_bins[subscale] = {'p33': p33, 'p67': p67}
        
        print(f"\n  {subscale.upper()} Score Distribution (0-100 scale):")
        print(f"    Min: {scores_array.min():.2f}")
        print(f"    33rd percentile (low/med boundary): {p33:.2f}")
        print(f"    Median: {np.median(scores_array):.2f}")
        print(f"    67th percentile (med/high boundary): {p67:.2f}")
        print(f"    Max: {scores_array.max():.2f}")
        print(f"    Bins -> Low: <{p33:.2f} | Medium: {p33:.2f}-{p67:.2f} | High: >={p67:.2f}")
    
    # Step 3: Build metadata and assign class labels
    metadata = {}
    class_counts = {subscale: {'low': 0, 'medium': 0, 'high': 0} for subscale in subscales}
    
    print(f"\nAssigning class labels...")
    
    for subject_task in sorted(all_subject_tasks):
        sample_metadata = {}
        
        for subscale in subscales:
            if subject_task not in subscale_scores[subscale]:
                continue
            
            score = subscale_scores[subscale][subject_task]
            p33 = subscale_bins[subscale]['p33']
            p67 = subscale_bins[subscale]['p67']
            
            # Determine class (0=low, 1=medium, 2=high)
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
        
        # Store metadata using base filenames
        raw_filename = f"{subject_task}_eeg_raw.parquet"
        feature_filename = f"{subject_task}_features.parquet"
        
        metadata[raw_filename] = sample_metadata.copy()
        metadata[feature_filename] = sample_metadata.copy()
        
        # Step 4: Copy files to 'all' directory (single copy, no duplication)
        raw_file = raw_path / raw_filename
        feature_file = features_path / feature_filename
        
        if raw_file.exists():
            shutil.copy2(raw_file, time_all / raw_file.name)
        else:
            print(f"  Warning: Raw file not found: {raw_filename}")
        
        if feature_file.exists():
            shutil.copy2(feature_file, feature_all / feature_file.name)
        else:
            print(f"  Warning: Feature file not found: {feature_filename}")
    
    # Step 5: Save metadata JSON files
    time_metadata_path = time_output / 'classification_metadata.json'
    feature_metadata_path = feature_output / 'classification_metadata.json'
    
    import json
    with open(time_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(feature_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION DATASET CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_subject_tasks)}")
    print(f"Data files stored in:")
    print(f"  - {time_all}")
    print(f"  - {feature_all}")
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
        'metadata_files': {
            'time': str(time_metadata_path),
            'feature': str(feature_metadata_path)
        }
    }


def organize_files_by_tlx_bins(scores_file, raw_eeg_dir, features_dir, output_base_dir, 
                                 low_threshold=30, high_threshold=50):
    """
    Organize raw EEG and feature files into 3 bins based on TLX scores.
    
    TLX Bins (rescaled 1-100):
        - Low: TLX < 30
        - Medium: 30 <= TLX < 50  
        - High: TLX >= 50
    
    Args:
        scores_file: Path to scores.csv with tlx_raw column
        raw_eeg_dir: Directory containing raw EEG files (P{X}_{Y}_eeg_raw.parquet)
        features_dir: Directory containing feature files (P{X}_{Y}_features.parquet)
        output_base_dir: Base directory for output
        low_threshold: Upper boundary for low bin (default: 30)
        high_threshold: Lower boundary for high bin (default: 50)
    
    Returns:
        Dictionary with processing statistics
    """
    
    scores_path = Path(scores_file)
    raw_path = Path(raw_eeg_dir)
    features_path = Path(features_dir)
    
    # Create output directories
    time_output = Path(output_base_dir) / "mocas_time_tlx_classification_dataset"
    feature_output = Path(output_base_dir) / "mocas_feature_tlx_classification_dataset"
    
    # Validate inputs
    if not scores_path.exists():
        print(f"Error: Scores file does not exist: {scores_path}")
        return None
        
    if not raw_path.exists():
        print(f"Error: Raw EEG directory does not exist: {raw_path}")
        return None
        
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None
    
    # Load scores
    try:
        scores_df = pd.read_csv(scores_path)
        print(f"Loaded {scores_df.shape[0]} TLX scores from {scores_path.name}")
    except Exception as e:
        print(f"Error loading scores file: {e}")
        return None
    
    if 'tlx_raw' not in scores_df.columns:
        print("Error: 'tlx_raw' column not found in scores.csv")
        return None
    
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
    
    print(f"\nOrganizing files by TLX bins:")
    print(f"  Low: TLX < {low_threshold}")
    print(f"  Medium: {low_threshold} <= TLX < {high_threshold}")
    print(f"  High: TLX >= {high_threshold}")
    print(f"{'='*60}")
    
    # Statistics
    stats = {
        'raw_processed': 0,
        'feature_processed': 0,
        'raw_bins': {'low': 0, 'medium': 0, 'high': 0},
        'feature_bins': {'low': 0, 'medium': 0, 'high': 0},
        'missing_scores': 0,
        'missing_files': 0
    }
    
    # Process each score entry
    for _, row in scores_df.iterrows():
        participant = str(row['P_name'])  # Column is 'P_name' not 'participant'
        task = str(row['task'])
        tlx_score = row['tlx_raw']
        
        # Determine bin
        if tlx_score < low_threshold:
            bin_class = 'low'
        elif tlx_score < high_threshold:
            bin_class = 'medium'
        else:
            bin_class = 'high'
        
        # File naming pattern (participant already has 'P' prefix)
        raw_filename = f"{participant}_{task}_eeg_raw.parquet"
        feature_filename = f"{participant}_{task}_features.parquet"
        
        # Copy raw EEG file
        raw_source = raw_path / raw_filename
        if raw_source.exists():
            raw_dest = time_bins[bin_class] / raw_filename
            try:
                shutil.copy2(raw_source, raw_dest)
                stats['raw_processed'] += 1
                stats['raw_bins'][bin_class] += 1
            except Exception as e:
                print(f"Error copying {raw_filename}: {e}")
        else:
            stats['missing_files'] += 1
        
        # Copy feature file
        feature_source = features_path / feature_filename
        if feature_source.exists():
            feature_dest = feature_bins[bin_class] / feature_filename
            try:
                shutil.copy2(feature_source, feature_dest)
                stats['feature_processed'] += 1
                stats['feature_bins'][bin_class] += 1
            except Exception as e:
                print(f"Error copying {feature_filename}: {e}")
        else:
            stats['missing_files'] += 1
    
    # Print summary
    print(f"\nTLX BINNING SUMMARY")
    print(f"{'='*60}")
    print(f"Raw EEG files organized: {stats['raw_processed']}")
    print(f"  Low (< {low_threshold}): {stats['raw_bins']['low']} files")
    print(f"  Medium ({low_threshold}-{high_threshold}): {stats['raw_bins']['medium']} files")
    print(f"  High (>= {high_threshold}): {stats['raw_bins']['high']} files")
    print(f"\nFeature files organized: {stats['feature_processed']}")
    print(f"  Low (< {low_threshold}): {stats['feature_bins']['low']} files")
    print(f"  Medium ({low_threshold}-{high_threshold}): {stats['feature_bins']['medium']} files")
    print(f"  High (>= {high_threshold}): {stats['feature_bins']['high']} files")
    print(f"\nMissing files: {stats['missing_files']}")
    
    # Save results
    results = {
        'binning_method': 'tlx_based',
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
    
    results_file = Path(output_base_dir) / 'tlx_binning_results.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save results file: {e}")
    
    return results


def create_target_files_from_scores(scores_file, features_dir, target_column='tlx_raw', version="feature"):
    """
    Create multi-label .txt target files from the MOCAS scores.csv file.
    
    Creates 7 target files per recording:
        1. P{X}_{Y}_features.txt (or _eeg_raw.txt) - Combined TLX (0-100, from tlx_raw)
        2. P{X}_{Y}_features_mental.txt - Mental demand (0-100, rescaled from 1-7)
        3. P{X}_{Y}_features_physical.txt - Physical demand (0-100, rescaled from 1-7)
        4. P{X}_{Y}_features_temporal.txt - Temporal demand (0-100, rescaled from 1-7)
        5. P{X}_{Y}_features_performance.txt - Performance (0-100, rescaled from 1-7)
        6. P{X}_{Y}_features_effort.txt - Effort (0-100, rescaled from 1-7)
        7. P{X}_{Y}_features_frustration.txt - Frustration (0-100, rescaled from 1-7)
    
    Args:
        scores_file: Path to the scores.csv file
        features_dir: Directory containing feature files (P{X}_{Y}_features.parquet)
        target_column: Column name in scores.csv to use as combined target (default: 'tlx_raw')
        version: "feature" for _features files, "raw" for _eeg_raw files
    
    Returns:
        Dictionary with processing results
    """
    
    # TLX subscale dimensions mapping
    TLX_DIMENSIONS = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']
    TLX_COLUMN_MAPPING = {
        'mental': 'tlx_mental',
        'physical': 'tlx_physical',
        'temporal': 'tlx_temporal',
        'performance': 'tlx_performance',
        'effort': 'tlx_effort',
        'frustration': 'tlx_frustration'
    }
    
    scores_path = Path(scores_file)
    features_path = Path(features_dir)
    
    # Validate inputs
    if not scores_path.exists():
        print(f"Error: Scores file does not exist: {scores_path}")
        return None
        
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None
    
    # Load scores CSV
    try:
        scores_df = pd.read_csv(scores_path)
        print(f"Loaded {scores_df.shape[0]} target scores from {scores_path.name}")
    except Exception as e:
        print(f"Error loading scores file: {e}")
        return None
    
    # Validate target column
    if target_column not in scores_df.columns:
        print(f"Error: Target column '{target_column}' not found in scores file")
        print(f"Available columns: {list(scores_df.columns)}")
        return None
    feature_files=[]
    if version=="feature":
        feature_files = list(features_path.glob("P*_*_features.parquet"))
    elif version=="raw":
        feature_files = list(features_path.glob("P*_*_eeg_raw.parquet"))
    else:
        # Find all feature files
        feature_files = list(features_path.glob("P*_*_features.parquet"))
    
    if not feature_files:
        print(f"No feature files found in {features_path}")
        if version=="feature":
            print("Expected format: P{participant}_{task}_features.parquet")
        elif version=="raw":
            print("Expected format: P{participant}_{task}_eeg_raw.parquet")
        return None
    
    print(f"Creating targets for {len(feature_files)} files using '{target_column}' column")
    
    # Processing statistics
    created_files = 0
    missing_targets = 0
    processing_results = []
    missing_targets_info = []
    
    # Process each feature file
    for feature_file in feature_files:
        try:
            # Parse filename to get participant and task info
            filename = feature_file.stem  # Remove .parquet extension
            parts = filename.split('_')
            
            if len(parts) >= 3:  # Expect P{ID}_{TASK}_features format
                participant_id = parts[0]  # P10, P11, etc.
                task_id = int(parts[1])    # 0, 1, 2, etc.
                
                # Validate participant ID format (should start with 'P' followed by numbers)
                if not participant_id.startswith('P') or not participant_id[1:].isdigit():
                    missing_targets_info.append({
                        'participant': participant_id,
                        'task': task_id,
                        'feature_file': str(feature_file),
                        'reason': f'Invalid participant ID format: {participant_id}'
                    })
                    missing_targets += 1
                    continue
                
                # Look up target value in scores CSV
                target_row = scores_df[(scores_df['P_name'] == participant_id) & 
                                     (scores_df['task'] == task_id)]
                
                if len(target_row) == 0:
                    missing_targets_info.append({
                        'participant': participant_id,
                        'task': task_id,
                        'feature_file': str(feature_file),
                        'reason': 'No matching row in scores.csv'
                    })
                    missing_targets += 1
                    continue
                elif len(target_row) > 1:
                    print(f"Warning: Multiple targets found for {participant_id} task {task_id}, using first")
                
                # Get target value and create .txt file
                target_value = target_row[target_column].iloc[0]
                
                # Determine base filename (remove _features or _eeg_raw suffix)
                if version == "feature":
                    base_name = filename.replace('_features', '')
                elif version == "raw":
                    base_name = filename.replace('_eeg_raw', '')
                else:
                    base_name = filename.replace('_features', '')
                
                # 1. Create combined TLX target file (no suffix)
                txt_file = features_path / f"{base_name}_{'features' if version == 'feature' else 'eeg_raw'}.txt"
                with open(txt_file, 'w') as f:
                    f.write(f"{target_value:.2f}")
                
                # 2. Create individual subscale target files (with suffix)
                for dim in TLX_DIMENSIONS:
                    csv_column = TLX_COLUMN_MAPPING[dim]
                    subscale_raw_value = target_row[csv_column].iloc[0]
                    
                    # Rescale from 1-7 to 0-100
                    # Formula: ((value - 1) / 6) * 100
                    subscale_rescaled = ((subscale_raw_value - 1.0) / 6.0) * 100.0
                    
                    # Clamp to valid range
                    subscale_rescaled = max(0.0, min(100.0, subscale_rescaled))
                    
                    subscale_txt_file = features_path / f"{base_name}_{'features' if version == 'feature' else 'eeg_raw'}_{dim}.txt"
                    with open(subscale_txt_file, 'w') as f:
                        f.write(f"{subscale_rescaled:.2f}")
                
                # Store processing result
                processing_results.append({
                    'participant': participant_id,
                    'task': task_id,
                    'feature_file': str(feature_file),
                    'target_file': str(txt_file),
                    'target_value': float(target_value),
                    'status': 'success'
                })
                
                created_files += 1
                
            else:
                missing_targets_info.append({
                    'participant': 'unknown',
                    'task': 'unknown',
                    'feature_file': str(feature_file),
                    'reason': f'Invalid filename format: expected P{{ID}}_{{TASK}}_features, got {filename}'
                })
                missing_targets += 1
                
        except Exception as e:
            missing_targets_info.append({
                'participant': 'unknown',
                'task': 'unknown', 
                'feature_file': str(feature_file),
                'reason': str(e)
            })
            missing_targets += 1
    
    # Generate summary
    print(f"Created {created_files * 7}/{len(feature_files) * 7} target files (combined + 6 subscales)")
    print(f"  Recordings processed: {created_files}/{len(feature_files)}")
    if missing_targets > 0:
        print(f"  Skipped {missing_targets} files (no matching target)")
    
    if processing_results:
        target_values = [r['target_value'] for r in processing_results]
        print(f"Target range: {min(target_values):.2f} to {max(target_values):.2f} (mean: {np.mean(target_values):.2f})")
    
    # Save processing results
    results_summary = {
        'processing_stats': {
            'feature_files': len(feature_files),
            'created_files': created_files,
            'skipped_files': missing_targets,
            'success_rate': created_files/len(feature_files)*100
        },
        'target_column': target_column,
        'target_statistics': {
            'min': float(min(target_values)) if target_values else None,
            'max': float(max(target_values)) if target_values else None,
            'mean': float(np.mean(target_values)) if target_values else None,
            'std': float(np.std(target_values)) if target_values else None
        },
        'successful_files': processing_results,
        'skipped_files': missing_targets_info,
        'features_directory': str(features_path),
        'scores_file': str(scores_path)
    }
    
    # Save summary to JSON
    summary_file = features_path / f'target_creation_summary_{target_column}.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nTarget creation summary saved to: {summary_file}")
    
    return results_summary


def cleanup_parquet_files_without_targets(features_dir, dry_run=True, backup_dir=None,version="feature"):
    """
    Clean up parquet files that don't have corresponding target files.
    
    Args:
        features_dir: Directory containing feature files
        dry_run: If True, only show what would be deleted (default: True for safety)
        backup_dir: Optional directory to move files to instead of deleting
        
    Returns:
        Dictionary with cleanup results
    """
    
    features_path = Path(features_dir)
    
    if not features_path.exists():
        print(f"Error: Features directory does not exist: {features_path}")
        return None
    parquet_files = []
    # Find all parquet files
    if version=="feature":
        parquet_files = list(features_path.glob("P*_*_features.parquet"))
    elif version=="raw":
        parquet_files = list(features_path.glob("P*_*_eeg_raw.parquet"))

    if not parquet_files:
        print(f"No parquet feature files found in {features_path}")
        return None
    
    # Safety checks
    files_with_targets = []
    files_without_targets = []
    
    for parquet_file in parquet_files:
        txt_file = parquet_file.with_suffix('.txt')
        
        if txt_file.exists():
            files_with_targets.append({
                'parquet': parquet_file,
                'txt': txt_file,
                'size_mb': parquet_file.stat().st_size / (1024 * 1024)
            })
        else:
            files_without_targets.append({
                'parquet': parquet_file,
                'txt': txt_file,
                'size_mb': parquet_file.stat().st_size / (1024 * 1024)
            })
    
    # Safety check: ensure we have some files with targets
    if len(files_with_targets) == 0:
        print("⚠️  SAFETY CHECK FAILED: No parquet files have corresponding target files!")
        print("⚠️  This suggests targets haven't been created yet. Aborting cleanup.")
        return None
    
    # Safety check: don't delete more than 50% of files
    deletion_percentage = len(files_without_targets) / len(parquet_files) * 100
    if deletion_percentage > 50:
        print(f"⚠️  SAFETY CHECK WARNING: Would delete {deletion_percentage:.1f}% of files ({len(files_without_targets)}/{len(parquet_files)})")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cleanup aborted by user.")
            return None
    
    if len(files_without_targets) == 0:
        print("All parquet files have corresponding target files. No cleanup needed.")
        return {
            'total_files': len(parquet_files),
            'files_with_targets': len(files_with_targets),
            'files_without_targets': len(files_without_targets),
            'cleanup_needed': False
        }
    
    # Show summary
    total_size_mb = sum(f['size_mb'] for f in files_without_targets)
    print(f"Found {len(files_without_targets)} files without targets ({total_size_mb:.2f} MB)")
    if len(files_without_targets) <= 5:
        for file_info in files_without_targets:
            print(f"  - {file_info['parquet'].name}")
    else:
        for file_info in files_without_targets[:5]:
            print(f"  - {file_info['parquet'].name}")
        print(f"  ... and {len(files_without_targets) - 5} more")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        return {
            'total_files': len(parquet_files),
            'files_with_targets': len(files_with_targets),
            'files_without_targets': len(files_without_targets),
            'cleanup_needed': True,
            'total_size_mb': total_size_mb,
            'dry_run': True,
            'files_to_delete': [str(f['parquet']) for f in files_without_targets]
        }
    
    # Actual cleanup
    if backup_dir:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
    
    deleted_files = []
    moved_files = []
    errors = []
    
    for file_info in files_without_targets:
        parquet_file = file_info['parquet']
        
        try:
            if backup_dir:
                backup_file = backup_path / parquet_file.name
                parquet_file.rename(backup_file)
                moved_files.append(str(parquet_file))
            else:
                parquet_file.unlink()
                deleted_files.append(str(parquet_file))
                
        except Exception as e:
            error_msg = f"Error processing {parquet_file.name}: {str(e)}"
            errors.append(error_msg)
    
    # Summary
    action = "moved" if backup_dir else "deleted"
    successful_count = len(moved_files) if backup_dir else len(deleted_files)
    
    print(f"✅ {action.capitalize()} {successful_count} files ({total_size_mb:.2f} MB)")
    
    if errors:
        print(f"❌ {len(errors)} errors encountered")
        for error in errors[:3]:  # Show first 3 errors
            print(f"  - {error}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more errors")
    
    return {
        'total_files': len(parquet_files),
        'files_with_targets': len(files_with_targets),
        'files_without_targets': len(files_without_targets),
        'cleanup_needed': True,
        'total_size_mb': total_size_mb,
        'dry_run': False,
        'successful_deletions': successful_count,
        'errors': len(errors),
        'backup_used': backup_dir is not None,
        'backup_dir': str(backup_dir) if backup_dir else None,
        'deleted_files': deleted_files if not backup_dir else [],
        'moved_files': moved_files if backup_dir else [],
        'error_details': errors
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MOCAS Dataset Processing Pipeline for EEG Cognitive Workload Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (all steps)
  python load_mocas.py
  
  # Only extract raw EEG without processing (for LaBraM embeddings)
  python load_mocas.py --raw-only
  
  # Only create time-series datasets (skip feature extraction)
  python load_mocas.py --time-only
  
  # Only create feature datasets (skip time-series)
  python load_mocas.py --feature-only
  
  # Include legacy folder-based classification datasets
  python load_mocas.py --old-classification

Steps:
  1   - Extract and reformat raw EEG (with subband decomposition)
  1.5 - Create target files for raw EEG
  2   - Organize raw EEG by workload (legacy, optional)
  3   - Extract statistical features
  3.5 - Organize features by workload (legacy, optional)
  4   - Create target files for features
  5   - Cleanup files without targets
  5.5 - Create metadata-based classification datasets
  6   - Organize by TLX bins (legacy, optional)
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
    
    parser.add_argument(
        '--old-classification',
        action='store_true',
        help='Include legacy folder-based classification datasets'
    )
    
    args = parser.parse_args()
    
    # Override the global flag if --old-classification is passed
    if args.old_classification:
        CREATE_OLD_CLASSIFICATION_DATASETS = True
    
    if args.raw_only:
        # Only extract raw EEG without processing
        print(f"\n{'#'*60}")
        print(f"# MOCAS RAW EEG EXTRACTION (NO PROCESSING)")
        print(f"# For LaBraM embeddings")
        print(f"{'#'*60}\n")
        print("Note: TLX target files should already exist from previous full pipeline run.")
        print("If missing, run without --raw-only flag to create them.\n")
        reformat_raw_eeg(data_path, output_path, sampling_rate=128.0, raw_only=True)
        print(f"\n{'#'*60}")
        print(f"# RAW EXTRACTION COMPLETE!")
        print(f"{'#'*60}\n")
    elif args.time_only:
        # Time-series only mode - simplified pipeline
        print(f"\n{'#'*60}")
        print(f"# MOCAS TIME-SERIES ONLY PIPELINE")
        print(f"{'#'*60}\n")
        
        # STEP 1: Reformat raw EEG
        print("STEP 1: Reformatting and extracting raw EEG files...")
        print("="*60)
        file_info = reformat_raw_eeg(data_path, output_path)
        
        # STEP 1.5: Create target files
        print("\nSTEP 1.5: Creating target files...")
        print("="*60)
        scores_file = os.path.join(data_path, 'scores.csv')
        create_target_files_from_scores(scores_file, output_path, 'tlx_raw', version="raw")
        cleanup_parquet_files_without_targets(output_path, dry_run=False, backup_dir=None, version="raw")
        
        # STEP 5.5: Create metadata-based classification
        print("\nSTEP 5.5: Creating metadata-based classification datasets...")
        print("="*60)
        create_metadata_based_classification_datasets(output_path, features_output_path, workload_classes_path)
        
        print(f"\n{'#'*60}")
        print(f"# TIME-SERIES PIPELINE COMPLETE!")
        print(f"{'#'*60}\n")
    elif args.feature_only:
        # Feature-only mode
        print(f"\n{'#'*60}")
        print(f"# MOCAS FEATURE-ONLY PIPELINE")
        print(f"{'#'*60}\n")
        
        # STEP 1: Reformat raw EEG
        print("STEP 1: Reformatting and extracting raw EEG files...")
        print("="*60)
        file_info = reformat_raw_eeg(data_path, output_path)
        
        # STEP 3: Extract features
        print("\nSTEP 3: Extracting statistical features...")
        print("="*60)
        extract_features_from_mocas_files(output_path, features_output_path, data_path, sampling_rate=128.0)
        
        # STEP 4: Create target files for features
        print("\nSTEP 4: Creating target files for features...")
        print("="*60)
        scores_file = os.path.join(data_path, 'scores.csv')
        create_target_files_from_scores(scores_file, features_output_path, 'tlx_raw', version="feature")
        
        # STEP 5: Cleanup
        print("\nSTEP 5: Cleaning up files without targets...")
        print("="*60)
        cleanup_parquet_files_without_targets(features_output_path, dry_run=False, backup_dir=None, version="feature")
        
        # STEP 5.5: Create metadata-based classification
        print("\nSTEP 5.5: Creating metadata-based classification datasets...")
        print("="*60)
        create_metadata_based_classification_datasets(output_path, features_output_path, workload_classes_path)
        
        print(f"\n{'#'*60}")
        print(f"# FEATURE PIPELINE COMPLETE!")
        print(f"{'#'*60}\n")
    else:
        # Run full pipeline (original inline code)
        # STEP 1: Reformat and extract raw EEG files  
        print("STEP 1: Reformatting and extracting raw EEG files...")
        print("="*60)
        file_info = reformat_raw_eeg(data_path, output_path)
        
        # Save file info to JSON
        import json
        with open(os.path.join(output_path, 'file_info.json'), 'w') as f:
            json.dump(file_info, f, indent=4)
        
        print(f"\nFile info saved to {os.path.join(output_path, 'file_info.json')}")
        
        print("\n" + "="*60 + "\n")
        
        print("STEP 1.5: Creating target files from raw cognitive workload scores...")
        print("="*60)
        scores_file = os.path.join(data_path, 'scores.csv')
        
        target_results = create_target_files_from_scores(
            scores_file=scores_file,
            features_dir=output_path,
            target_column='tlx_raw',
            version="raw"
        )
        cleanup_parquet_files_without_targets(output_path, dry_run=False, backup_dir=None,version="raw")
        
        
        if target_results:
            print(f"\nTarget files created successfully!")
            print(f"Files created: {target_results['processing_stats']['created_files']}")
            print(f"Target range: {target_results['target_statistics']['min']:.2f} - {target_results['target_statistics']['max']:.2f}")
        else:
            print("\nFailed to create target files")
        
        # STEP 2: Organize raw EEG files by workload class (legacy folder-based)
        if CREATE_OLD_CLASSIFICATION_DATASETS:
            print("STEP 2: Organizing raw EEG files by workload class (legacy)...")
            print("="*60)
            eeg_organization_results = organize_raw_eeg_by_workload(output_path, data_path, workload_classes_path)
            
            if eeg_organization_results:
                print(f"EEG file organization completed!")
                print(f"Processed {eeg_organization_results['processed_files']} EEG files")
                print(f"Workload distribution: {eeg_organization_results['workload_distribution']}")
            else:
                print("EEG file organization failed")
        else:
            print("STEP 2: Skipped legacy workload-based organization (CREATE_OLD_CLASSIFICATION_DATASETS=False)")
            print("="*60)
        
        print("\n" + "="*60 + "\n") 
        
       
        
        # STEP 3: Extract features from EEG files
        print("STEP 3: Extracting statistical features from EEG files...")
        print("="*60)
        results = extract_features_from_mocas_files(output_path, features_output_path, data_path, sampling_rate=128.0)
        #print(results)
        
        print("\n" + "="*60 + "\n")
        
        # STEP 3.5: Organize extracted features by workload class (legacy folder-based)
        if CREATE_OLD_CLASSIFICATION_DATASETS:
            print("STEP 3.5: Organizing extracted features by workload class (legacy)...")
            print("="*60)
            feature_organization_results = organize_extracted_features_by_workload(
                features_output_path, data_path, workload_classes_path
            )
            
            if feature_organization_results:
                print(f"✅ Feature organization completed!")
                print(f"Processed {feature_organization_results['processed_files']} feature files")
                print(f"Workload distribution: {feature_organization_results['workload_distribution']}")
            else:
                print("❌ Feature organization failed")
        else:
            print("STEP 3.5: Skipped legacy workload-based feature organization (CREATE_OLD_CLASSIFICATION_DATASETS=False)")
            print("="*60)
        
        print("\n" + "="*60 + "\n")
        
        # STEP 4: Create target files from scores.csv using tlx_raw column
        print("STEP 4: Creating target files from fearture cognitive workload scores...")
        print("="*60)
        
        target_results = create_target_files_from_scores(
            scores_file=scores_file,
            features_dir=features_output_path,
            target_column='tlx_raw',
            version="feature"
        )
        
        
        if target_results:
            print(f"\nTarget files created successfully!")
            print(f"Files created: {target_results['processing_stats']['created_files']}")
            print(f"Target range: {target_results['target_statistics']['min']:.2f} - {target_results['target_statistics']['max']:.2f}")
        else:
            print("\nFailed to create target files")
            
            
       
        print("\n" + "="*60 + "\n")
            
        # STEP 5: Cleanup files without targets (this removes orphaned feature files)
        print("STEP 5: Cleaning up feature files without corresponding targets...")
        print("="*60)
        cleanup_parquet_files_without_targets(features_output_path, dry_run=False, backup_dir=None,version="feature")
        
        print("\n" + "="*60 + "\n")
        
        # STEP 5.5: Create metadata-based multi-subscale classification datasets
        print("STEP 5.5: Creating metadata-based multi-subscale classification datasets...")
        print("="*60)
        metadata_classification_results = create_metadata_based_classification_datasets(
            raw_eeg_dir=output_path,
            features_dir=features_output_path,
            output_base_dir=workload_classes_path
        )
        
        if metadata_classification_results:
            print(f"✅ Metadata-based classification datasets created!")
            print(f"Total samples: {metadata_classification_results['total_samples']}")
            print(f"Subscales: {', '.join(metadata_classification_results['subscales'])}")
        else:
            print("❌ Metadata-based classification creation failed")
        
        print("\n" + "="*60 + "\n")
        
        # STEP 6: Organize files by TLX bins (3-class classification: low/medium/high)
        # Note: This is the old folder-based method, kept for backward compatibility
        if CREATE_OLD_CLASSIFICATION_DATASETS:
            print("STEP 6: Organizing files by TLX bins (legacy folder-based classification)...")
            print("="*60)
            tlx_binning_results = organize_files_by_tlx_bins(
                scores_file=scores_file,
                raw_eeg_dir=output_path,
                features_dir=features_output_path,
                output_base_dir=workload_classes_path,
                low_threshold=30,
                high_threshold=50
            )
            
            if tlx_binning_results:
                print(f"✅ TLX binning organization completed!")
                print(f"Raw EEG files organized: {tlx_binning_results['statistics']['raw_processed']}")
                print(f"Feature files organized: {tlx_binning_results['statistics']['feature_processed']}")
                print(f"Bin distribution: {tlx_binning_results['statistics']['raw_bins']}")
            else:
                print("❌ TLX binning organization failed")
        else:
            print("STEP 6: Skipped legacy folder-based classification (CREATE_OLD_CLASSIFICATION_DATASETS=False)")
            print("="*60)
            print("✅ Using modern metadata-based classification only (created in STEP 5.5)")
        
        print("\n" + "="*60 + "\n")
        print("🎉 All processing steps completed!")






