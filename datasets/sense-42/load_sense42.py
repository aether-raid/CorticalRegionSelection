from pathlib import Path
import pandas as pd
import numpy as np
import json
import traceback
import shutil
from tqdm import tqdm
import mne
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directories to path for imports
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from channel_importance.eeg import EEG


"""
SENSE-42 Dataset Processing Pipeline for EEG Analysis
====================================================================

Complete preprocessing pipeline for the SENSE-42 dataset that transforms raw 
.bdf EEG data into processed features with NASA-TLX workload ratings.

DATASET SOURCE:
    SENSE-42 - HCI dataset from Synapse (syn68713182)
    Contains EEG recordings from 42 participants performing various cognitive tasks
    Recorded with BioSemi ActiveTwo system (32 EEG channels + auxiliary channels)
    Sampling rate: 1024 Hz
    Duration: ~130 minutes per participant
    Workload labels: NASA-TLX ratings collected ~25 times per session

EXPECTED INPUT STRUCTURE:
    data/SENSE-42/
        download_metadata.json           # Download information from Synapse
        001_explorer_2025-02-15_*.csv    # Behavioral data with TLX ratings
        002_explorer_*.csv               # One CSV per participant
        ...
        EEG/
            p001.bdf                     # Participant 1 EEG recording
            p002.bdf                     # Participant 2 EEG recording
            ...
            p042.bdf                     # Participant 42 EEG recording

BDF FILE STRUCTURE:
    48 channels total:
        - 32 EEG channels: Fp1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz,
                          PO3, O1, Oz, O2, PO4, P4, P8, CP6, CP2, C4, T8, FC6, FC2, 
                          F4, F8, AF4, Fp2, Fz, Cz
        - 1 reference: EarL (left ear reference)
        - 8 external channels: EXG2-EXG8 (EOG, EMG, etc.)
        - 6 auxiliary channels: GSR1, GSR2, Erg1, Erg2, Resp, Plet, Temp
        - 1 status channel: Status
    
    Sampling rate: 1024 Hz
    Duration: ~7800-8200 seconds (~130 minutes) per recording

WORKLOAD LABELS:
    5 NASA-TLX subscales (TLX-correlated dimensions):
        - Mental Demand: Cognitive workload
        - Temporal Demand: Time pressure
        - Performance: Success in task (inverted: higher performance = lower workload)
        - Effort: Physical and mental effort
        - Frustration: Stress and annoyance
    
    Rating encoding in EEG Status channel:
        - Event code formula: QUESTION_BASE (100) + QUESTION_LEAP (10) × INDEX + RATING
        - Mental demand: codes 110-119 (28 events - has 2 duplicates)
        - Temporal demand: codes 120-129 (25 events - used as reference)
        - Performance: codes 130-139 (25 events)
        - Effort: codes 140-149 (25 events)
        - Frustration: codes 150-159 (25 events)
        - Rating scale: 0-9 in event codes, rescaled to 0-100
        - Combined TLX: Average of 5 subscales
    
    Segmentation approach:
        - Uses temporal_demand events as segment boundaries (stable 25 events)
        - Event timestamps extracted from Status channel (exact timing)
        - Mental_demand duplicates handled via nearest-neighbor matching
        - Zero timing error (event-based, not trial-based estimation)
        - Each participant: ~25 segments aligned with TLX rating times

PROCESSING PIPELINE (5 STEPS):

    STEP 1: EEG File Extraction & Reformatting (with Subband Decomposition)
        • Loads .bdf files using MNE
        • Extracts ONLY 32 EEG channels (removes auxiliary, reference, status channels)
        • Applies subband decomposition (Overall, delta, theta, alpha, beta, gamma)
        • Creates MultiIndex DataFrame with (band, channel) structure
        • Preserves original 1024 Hz sampling rate (no resampling in step1)
        • Standardizes filenames: P{participant}_eeg_raw.parquet
        • Output: data/SENSE-42/sense42_raw_eeg_extracted/
    
    STEP 2: Statistical Feature Extraction (with Event-Based Segmentation)
        • Extracts temporal_demand events from Status channel (codes 120-129)
        • Uses event timestamps as exact segment boundaries (~25 per participant)
        • Loads ORIGINAL .bdf files for each segment (prevents double-processing bug)
        • Applies signal processing: bandpass filtering, frequency decomposition
        • Downsamples from 1024 Hz to 128 Hz during feature extraction
        • Extracts 400+ statistical features per segment using EEG class
        • Features: power bands, spectral entropy, hjorth parameters, etc.
        • Zero timing error (event-based, not uniform trial spacing)
        • Output: data/SENSE-42/sense42_features_extracted/P{X}_seg{YY}_features.parquet

    STEP 2.5: Create NASA-TLX Target Files (from EEG Event Encoding)
        • Extracts all 5 TLX dimensions from Status channel events (codes 110-159)
        • Decodes ratings directly from event codes (rating = code - base)
        • Uses temporal_demand events as reference boundaries (25 events)
        • Handles mental_demand duplicates via nearest-neighbor matching
        • Rescales 0-9 → 0-100, inverts performance (higher = lower workload)
        • Creates 6 target files per segment (combined + 5 subscales)
        • Zero timing error (same event source as segmentation)
        • Output: data/SENSE-42/sense42_features_extracted/P{X}_seg{YY}_features*.txt

    STEP 3: Create Time-Series Dataset
        • Segments raw EEG based on temporal_demand events (~25 segments per participant)
        • Applies subband decomposition (Overall, delta, theta, alpha, beta, gamma)
        • Preserves 1024 Hz sampling rate
        • Saves directly to sense42_time_classification_dataset/all/ (no intermediate copy)
        • Output: data/SENSE-42/sense42_time_classification_dataset/all/

    STEP 4: Create Feature Classification Dataset
        • Copies extracted feature files to feature classification dataset directory
        • Output: data/SENSE-42/sense42_feature_classification_dataset/
    
    STEP 4.1: Verify Time Classification Dataset
        • Verifies time-series segments exist (created directly by Step 3)
        • No file copying needed
        • Output: data/SENSE-42/sense42_time_classification_dataset/

    STEP 5: Create Regression Datasets
        • Copies target txt files to time classification all/ folder (time regression)
        • Copies feature files + target files to feature regression dataset
        • Time regression uses same all/ folder as classification (no file duplication)
        • Output: 
            - Time regression: sense42_time_classification_dataset/all/ (txt files added)
            - Feature regression: sense42_feature_regression_dataset/

FINAL OUTPUT STRUCTURE:
    data/SENSE-42/
        sense42_raw_eeg_extracted/              # Extracted with decomposition
            P001_eeg_raw.parquet                # Full-length EEG
            P002_eeg_raw.parquet
            ...
        
        sense42_features_extracted/             # Segmented features + targets
            P001_seg01_features.parquet         # Segment 1 features
            P001_seg01_features.txt             # Combined TLX
            P001_seg01_features_mental.txt      # Mental demand (0-100)
            P001_seg01_features_temporal.txt    # Temporal demand (0-100)
            P001_seg01_features_performance.txt # Performance inverted (0-100)
            P001_seg01_features_effort.txt      # Effort (0-100)
            P001_seg01_features_frustration.txt # Frustration (0-100)
            P001_seg02_features.parquet
            P001_seg02_features*.txt
            ...
            P001_seg25_features.parquet
            P001_seg25_features*.txt
        
        sense42_time_classification_dataset/     # Time classification & regression dataset (shared)
            all/
                P001_seg01_eeg_raw.parquet      # Segmented time-series (created by Step 3)
                P001_seg01_eeg_raw.txt          # Combined TLX target (for regression)
                P001_seg01_eeg_raw_mental.txt   # Mental demand (for regression)
                P001_seg01_eeg_raw_temporal.txt # Temporal demand (for regression)
                P001_seg01_eeg_raw_performance.txt # Performance (for regression)
                P001_seg01_eeg_raw_effort.txt   # Effort (for regression)
                P001_seg01_eeg_raw_frustration.txt # Frustration (for regression)
                ...
            classification_metadata.json        # For classification tasks
        
        sense42_feature_regression_dataset/     # Feature regression dataset
            P001_seg01_features.parquet
            P001_seg01_features*.txt            # All 6 target files
            ...

NOTES:
    • Only processes standard .bdf files (p001-p042)
    • Skips p005 02.bdf (secondary session) for consistency
    • Extracts only 32 EEG channels, excluding auxiliary sensors
    • Each participant contributes ~25 samples (segments)
    • Total dataset: ~1050 samples (42 participants × 25 segments)

INTEGRATION:
    The processed data is compatible with:
    • EEG class for loading and analysis
    • Regression models (Random Forest, SVM, Neural Networks)
    • Multi-output regression (5 TLX subscales)
    • Time-series models (RNNs, Transformers) with full-length data
    • Channel importance analysis

USAGE:
    python datasets/sense-42/load_sense42.py
    
    Or call individual functions:
    • step1_extract_raw_eeg()
    • step2_extract_features()
    • step2_5_create_targets()
    • step3_create_time_dataset()
    • step4_create_feature_dataset()
    • step4_1_create_time_classification_dataset()
    • step4_5_create_classification_metadata()
    • step5_create_regression_datasets()
"""

# =============================================================================
# Configuration
# =============================================================================

# Paths
SENSE42_ROOT = Path("data/SENSE-42")
SOURCE_EEG_DIR = SENSE42_ROOT / "EEG"
SOURCE_CSV_DIR = SENSE42_ROOT  # CSV files at root level
OUTPUT_RAW_EEG = SENSE42_ROOT / "sense42_raw_eeg_extracted"
OUTPUT_FEATURES = SENSE42_ROOT / "sense42_features_extracted"
# OUTPUT_TIME_DATASET is no longer needed - Step 3 saves directly to OUTPUT_TIME_CLASSIFICATION/all/
OUTPUT_FEATURE_CLASSIFICATION = SENSE42_ROOT / "sense42_feature_classification_dataset"
OUTPUT_TIME_CLASSIFICATION = SENSE42_ROOT / "sense42_time_classification_dataset"
# OUTPUT_TIME_REGRESSION is no longer needed - time regression uses OUTPUT_TIME_CLASSIFICATION/all/
OUTPUT_FEATURE_REGRESSION = SENSE42_ROOT / "sense42_feature_regression_dataset"

# EEG Configuration
SAMPLING_RATE = 1024.0  # Hz

# EEG channels to extract (32 standard EEG channels only)
EEG_CHANNELS = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 
    'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 
    'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 
    'Fz', 'Cz'
]

# Channels to exclude
EXCLUDE_CHANNELS = [
    'EarL',  # Reference
    'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8',  # External
    'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp',  # Auxiliary
    'Status'  # Status channel
]

# NASA-TLX subscale column mapping (DEPRECATED - now extract from EEG events)
# Kept for backward compatibility reference only
TLX_COLUMNS = {
    'mental': 'mental_demand:_how_mentally_demanding_was_the_task_slider.rating',
    'temporal': 'temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider.rating',
    'performance': 'performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider.rating',
    'effort': 'effort:_how_hard_did_you_have_to_work_to_accomplish_your_level_of_performance_slider.rating',
    'frustration': 'frustration:_how_insecure__discouraged__irritated__stressed__and_annoyed_were_you_slider.rating'
}


# =============================================================================
# STEP 1: Extract and Reformat Raw EEG from .bdf Files
# =============================================================================

def step1_extract_raw_eeg(raw_only=False):
    """
    Extract and reformat raw EEG files from SENSE-42 .bdf files.
    
    STEP 1: Processes .bdf files for each participant.
    
    Args:
        raw_only (bool): If True, skip EEG class processing (no filtering, no decomposition).
                        Just extract 32 EEG channels at 1024 Hz without any processing.
    
    This function:
    1. Loads .bdf files using MNE
    2. Extracts ONLY 32 EEG channels (removes auxiliary, reference, status)
    3. If raw_only=False: Applies bandpass filtering and subband decomposition via EEG class
       If raw_only=True: Saves raw channels without processing
    4. Preserves original 1024 Hz sampling rate (no resampling)
    5. Creates MultiIndex (band, channel) structure (unless raw_only=True)
    6. Saves with standardized naming: P{participant}_eeg_raw.parquet
    
    Note: Unlike step2, this step does NOT downsample to maintain full temporal resolution.
    
    Returns:
        List of dictionaries with processing details for each file
    """
    
    # Determine output directory based on raw_only flag
    if raw_only:
        output_dir = SENSE42_ROOT / "sense42_raw_eeg_extracted_rawonly"
    else:
        output_dir = OUTPUT_RAW_EEG
    
    if not SOURCE_EEG_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_EEG_DIR}")
        return []
        
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create output directory: {e}")
        return []

    if raw_only:
        print(f"\n{'='*80}")
        print(f"SENSE-42 RAW EEG EXTRACTION (NO PROCESSING)")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"SENSE-42 EEG EXTRACTION WITH SUBBAND DECOMPOSITION")
        print(f"{'='*80}")
    print(f"Source: {SOURCE_EEG_DIR}")
    print(f"Output: {output_dir}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"EEG channels: {len(EEG_CHANNELS)} (32 standard EEG)")
    print(f"Raw only (no processing): {raw_only}")
    print(f"{'='*80}\n")

    processed_files = 0
    missing_files = 0
    failed_files = 0
    failed_files_info = []
    file_info = []
    
    # Get all .bdf files (exclude p005 02.bdf - secondary session)
    bdf_files = sorted([f for f in SOURCE_EEG_DIR.glob("p*.bdf") 
                       if f.name != "p005 02.bdf"])
    
    print(f"Found {len(bdf_files)} .bdf files to process\n")
    
    for bdf_file in tqdm(bdf_files, desc="Extracting EEG files"):
        try:
            # Extract participant number from filename (p001.bdf -> P001)
            participant_name = bdf_file.stem  # p001
            participant_id = participant_name.upper()  # P001
            
            output_filename = f"{participant_id}_eeg_raw.parquet"
            output_file = output_dir / output_filename
            
            # Skip if already processed
            if raw_only:
                # Check if segment files exist for this participant
                seg_files = list(output_dir.glob(f"{participant_id}_seg*_eeg_raw.parquet"))
                if seg_files:
                    print(f"  ⚠ Skipping {participant_id} (segments already exist: {len(seg_files)} files)")
                    continue
            else:
                # Check if full file exists
                if output_file.exists():
                    print(f"  ⚠ Skipping {participant_id} (already exists)")
                    continue
            
            # Load .bdf file with MNE
            print(f"\n  Processing {participant_id}...")
            raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose=False)
            
            # Get original info
            original_channels = raw.ch_names
            original_sfreq = raw.info['sfreq']
            original_duration = raw.times[-1]
            original_n_samples = raw.n_times
            
            print(f"    Original: {len(original_channels)} channels, "
                  f"{original_sfreq} Hz, {original_duration:.1f}s, "
                  f"{original_n_samples:,} samples")
            
            # Extract events from Status channel BEFORE picking EEG channels
            # (needed for raw_only mode segmentation)
            events = None
            if raw_only and 'Status' in raw.ch_names:
                events = mne.find_events(raw, stim_channel='Status', shortest_event=1, verbose=False)
            
            # Extract only EEG channels
            # Check if all required channels are present
            missing_channels = [ch for ch in EEG_CHANNELS if ch not in raw.ch_names]
            if missing_channels:
                print(f"  ✗ Missing channels: {missing_channels}")
                failed_files += 1
                continue
            
            raw.pick_channels(EEG_CHANNELS, ordered=True)
            
            # Convert to DataFrame
            data = raw.get_data().T  # Shape: (n_samples, n_channels)
            if data.size == 0:
                print(f"  ✗ Empty data after extraction")
                failed_files += 1
                continue
            
            df = pd.DataFrame(data, columns=raw.ch_names)
            
            print(f"    Extracted: {len(df.columns)} EEG channels")
            
            # Validate data
            if len(df) == 0:
                print(f"  ✗ Empty dataframe")
                failed_files += 1
                continue
            
            if raw_only:
                # RAW ONLY MODE: Segment raw EEG without any processing
                print(f"    Extracting temporal_demand events for segmentation...")
                
                if events is None:
                    print(f"  ✗ No events available - skipping")
                    failed_files += 1
                    continue
                
                # Extract temporal_demand events (codes 120-129) for segmentation
                TEMPORAL_DEMAND_BASE = 120
                temporal_events = events[
                    (events[:, 2] >= TEMPORAL_DEMAND_BASE) & 
                    (events[:, 2] < TEMPORAL_DEMAND_BASE + 10)
                ]
                
                if len(temporal_events) == 0:
                    print(f"  ✗ No temporal_demand events found - skipping")
                    failed_files += 1
                    continue
                
                # Extract event timestamps and TLX ratings
                event_timestamps = temporal_events[:, 0] / original_sfreq
                event_ratings = temporal_events[:, 2] - TEMPORAL_DEMAND_BASE  # 0-9 scale
                num_segments = len(event_timestamps)
                
                print(f"    Found {num_segments} segments from temporal_demand events")
                print(f"    Segmenting and saving raw data...")
                
                # Define segment boundaries
                segment_boundaries = np.concatenate([[0], event_timestamps, [len(df) / original_sfreq]])
                
                # Extract all TLX dimensions for target files
                TLX_DIMENSIONS = {
                    'mental': 110,
                    'temporal': 120,
                    'performance': 130,
                    'effort': 140,
                    'frustration': 150
                }
                
                dimension_data = {}
                for dim_name, base_code in TLX_DIMENSIONS.items():
                    dim_events = events[
                        (events[:, 2] >= base_code) & 
                        (events[:, 2] < base_code + 10)
                    ]
                    timestamps = dim_events[:, 0] / original_sfreq
                    ratings = dim_events[:, 2] - base_code
                    dimension_data[dim_name] = {'timestamps': timestamps, 'ratings': ratings}
                
                # Process each segment
                segments_saved = 0
                for seg_idx in range(num_segments):
                    seg_start_time = segment_boundaries[seg_idx]
                    seg_end_time = segment_boundaries[seg_idx + 1]
                    seg_start_sample = int(seg_start_time * original_sfreq)
                    seg_end_sample = int(seg_end_time * original_sfreq)
                    
                    # Extract segment from dataframe
                    df_segment = df.iloc[seg_start_sample:seg_end_sample].copy()
                    
                    if len(df_segment) == 0:
                        continue
                    
                    # Save segment
                    seg_name = f"{participant_id}_seg{seg_idx+1:02d}"
                    seg_output_file = output_dir / f"{seg_name}_eeg_raw.parquet"
                    df_segment.to_parquet(seg_output_file)
                    
                    # Create TLX target files for this segment
                    segment_time = event_timestamps[seg_idx]
                    subscale_scores = {}
                    
                    for dim_name in TLX_DIMENSIONS.keys():
                        dim_times = dimension_data[dim_name]['timestamps']
                        dim_ratings = dimension_data[dim_name]['ratings']
                        
                        if len(dim_times) == 0:
                            subscale_scores[dim_name] = 0.0
                            continue
                        
                        # Find nearest event to this segment's reference time
                        nearest_idx = np.argmin(np.abs(dim_times - segment_time))
                        rating_0_9 = dim_ratings[nearest_idx]
                        rating_scaled = (rating_0_9 / 9) * 100
                        
                        # Invert performance
                        if dim_name == 'performance':
                            rating_scaled = 100 - rating_scaled
                        
                        subscale_scores[dim_name] = rating_scaled
                    
                    # Combined TLX score
                    combined_tlx = np.mean(list(subscale_scores.values()))
                    
                    # Save target files
                    target_file = output_dir / f"{seg_name}_eeg_raw.txt"
                    with open(target_file, 'w') as f:
                        f.write(f"{combined_tlx:.2f}")
                    
                    for subscale, score in subscale_scores.items():
                        subscale_file = output_dir / f"{seg_name}_eeg_raw_{subscale}.txt"
                        with open(subscale_file, 'w') as f:
                            f.write(f"{score:.2f}")
                    
                    segments_saved += 1
                
                print(f"    ✓ Saved {segments_saved} segments with TLX targets")
                
                file_info.append({
                    'participant': participant_id,
                    'input_file': str(bdf_file),
                    'output_dir': str(output_dir),
                    'original_channels': int(len(original_channels)),
                    'eeg_channels': int(len(EEG_CHANNELS)),
                    'original_sampling_rate': float(original_sfreq),
                    'sampling_rate': float(original_sfreq),
                    'duration_seconds': float(original_duration),
                    'original_n_samples': int(original_n_samples),
                    'num_segments': segments_saved,
                    'bands': 0,
                    'raw_only': True,
                    'status': 'success'
                })
                
                processed_files += 1
                print(f"    ✓ Saved: {participant_id} ({segments_saved} segments)")
                continue  # Skip normal processing
                
            else:
                # NORMAL MODE: Create EEG instance (applies filtering and decomposition, preserves sampling rate)
                print(f"    Creating EEG instance (filtering, decomposition)...")
                print(f"    Preserving original {original_sfreq} Hz sampling rate")
                
                # Create time array (relative seconds)
                timestamps = np.arange(len(df)) / original_sfreq
                sample_numbers = np.arange(len(df))
                
                # Prepare channel data dictionary
                channels_dict = {col: df[col].values for col in df.columns}
                
                # Validate data before processing
                if any(np.all(np.isnan(v)) for v in channels_dict.values()):
                    print(f"  ✗ All NaN values in one or more channels")
                    failed_files += 1
                    continue
                
                # extract_time=False preserves original 1024 Hz (like all other loaders in step1)
                eeg_instance = EEG(
                    s_n=sample_numbers,
                    t=timestamps,
                    channels=channels_dict,
                    frequency=SAMPLING_RATE,
                    extract_time=False  # Don't resample, just decompose (aligned with other loaders)
                )
                
                # Get decomposed data
                decomposed_df = eeg_instance.data
                
                # Validate decomposed data
                if decomposed_df is None or decomposed_df.empty:
                    print(f"  ✗ Empty decomposed data")
                    failed_files += 1
                    continue
                
                print(f"    Decomposed shape: {decomposed_df.shape}")
                print(f"    Sampling rate: {eeg_instance.frequency} Hz (preserved)")
                print(f"    Bands: {decomposed_df.columns.get_level_values(0).unique().tolist()}")
                
                output_df = decomposed_df
                output_sampling_rate = eeg_instance.frequency
                n_bands = 6
            
            # Save to parquet with error handling
            try:
                output_df.to_parquet(output_file)
                if not output_file.exists() or output_file.stat().st_size == 0:
                    print(f"  ✗ Failed to write file or file is empty")
                    failed_files += 1
                    continue
                output_size = output_file.stat().st_size / (1024 * 1024)
            except Exception as write_error:
                print(f"  ✗ Error writing file: {write_error}")
                failed_files += 1
                continue
            
            file_info.append({
                'participant': participant_id,
                'input_file': str(bdf_file),
                'output_file': str(output_file),
                'original_channels': int(len(original_channels)),
                'eeg_channels': int(len(EEG_CHANNELS)),
                'original_sampling_rate': float(original_sfreq),
                'sampling_rate': float(output_sampling_rate),
                'duration_seconds': float(original_duration),
                'original_n_samples': int(original_n_samples),
                'final_n_samples': int(output_df.shape[0]),
                'output_shape': [int(x) for x in output_df.shape],
                'output_size_mb': float(output_size),
                'bands': n_bands,
                'raw_only': raw_only,
                'status': 'success'
            })
            
            processed_files += 1
            print(f"    ✓ Saved: {output_filename}")
            
        except FileNotFoundError:
            print(f"  ✗ File not found: {bdf_file}")
            missing_files += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {bdf_file.name}: {e}")
            failed_files += 1
            tb = traceback.format_exc()
            file_info.append({
                'participant': bdf_file.stem.upper(),
                'input_file': str(bdf_file),
                'error': str(e),
                'status': 'failed'
            })
            failed_files_info.append({
                'participant': bdf_file.stem.upper(),
                'input_file': str(bdf_file),
                'error': str(e),
                'traceback': tb
            })
    
    # Summary
    total_files = processed_files + failed_files + missing_files
    print(f"\n{'='*80}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {processed_files} files")
    print(f"Failed: {failed_files} files")
    print(f"Missing: {missing_files} files")
    
    if processed_files > 0:
        total_samples = sum(f['original_n_samples'] for f in file_info if f['status'] == 'success')
        total_duration = sum(f['duration_seconds'] for f in file_info if f['status'] == 'success')
        avg_duration = total_duration / processed_files
        
        print(f"\nOutput directory: {output_dir}")
        if raw_only:
            total_segments = sum(f.get('num_segments', 0) for f in file_info if f['status'] == 'success')
            print(f"Files contain:")
            print(f"  - Raw channels (no processing)")
            print(f"  - {len(EEG_CHANNELS)} EEG channels")
            print(f"  - Original 1024 Hz sampling rate")
            print(f"  - Event-based segmentation (~25 segments per participant)")
            print(f"\nStatistics:")
            print(f"  Total participants: {processed_files}")
            print(f"  Total segments: {total_segments}")
            print(f"  Total samples: {total_samples:,}")
            print(f"  Total duration: {total_duration/60:.1f} minutes")
            print(f"  Average duration per participant: {avg_duration:.1f} seconds")
        else:
            total_size = sum(f['output_size_mb'] for f in file_info if f['status'] == 'success')
            print(f"Files contain:")
            print(f"  - MultiIndex columns: (band, channel)")
            print(f"  - Bands: Overall, delta, theta, alpha, beta, gamma")
            print(f"  - {len(EEG_CHANNELS)} EEG channels")
            print(f"\nStatistics:")
            print(f"  Total samples: {total_samples:,}")
            print(f"  Total duration: {total_duration/60:.1f} minutes")
            print(f"  Average duration per file: {avg_duration:.1f} seconds")
            print(f"  Total data size: {total_size:.1f} MB")
    
    # Save processing summary (convert numpy types to native Python types)
    summary_file = output_dir / 'extraction_summary.json'
    
    # Convert file_info numpy types to native Python types
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    with open(summary_file, 'w') as f:
        json.dump({
            'total_files': len(bdf_files),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'missing_files': missing_files,
            'sampling_rate': int(SAMPLING_RATE),
            'eeg_channels': EEG_CHANNELS,
            'raw_only': raw_only,
            'file_details': convert_to_native(file_info),
            'failed_files_info': failed_files_info
        }, f, indent=2)
    
    print(f"\nProcessing summary saved to: {summary_file}")
    
    if failed_files_info:
        print('\nFailed Files (first 5):')
        for f in failed_files_info[:5]:
            print(f"  - {Path(f['input_file']).name}: {f['error']}")
        if len(failed_files_info) > 5:
            print(f"  ... and {len(failed_files_info) - 5} more failed files")
    
    return file_info


# =============================================================================
# STEP 2: Extract Features from Raw EEG
# =============================================================================

def _process_single_participant(eeg_file, source_eeg_dir, output_features_dir, sampling_rate):
    """
    Worker function to process a single participant's EEG file.
    
    Extracts features from all segments for one participant.
    This function is designed to be called by multiprocessing.Pool.
    
    Args:
        eeg_file: Path to extracted EEG file (P###_eeg_raw.parquet)
        source_eeg_dir: Path to original .bdf files
        output_features_dir: Path to save feature files
        sampling_rate: Original sampling rate (1024 Hz)
        
    Returns:
        dict with 'participant', 'segments', 'file_info', 'status', 'error'
    """
    import time
    import random
    try:
        participant_id = eeg_file.stem.replace('_eeg_raw', '')
        
        # Small random delay to stagger workers and avoid simultaneous memory allocation
        # This prevents workers from hitting the load_data() call at the exact same time
        delay = random.uniform(0, 15.0)  # 0-15 second random delay
        time.sleep(delay)
        
        print(f"[{participant_id}] Starting processing...", flush=True)
        
        # Find corresponding .bdf file
        participant_num = participant_id[1:].lstrip('0') or '0'
        bdf_pattern = f"p{participant_num.zfill(3)}.bdf"
        original_file = source_eeg_dir / bdf_pattern
        
        if not original_file.exists():
            print(f"[{participant_id}] ERROR: .bdf file not found: {original_file}", flush=True)
            return {
                'participant': participant_id,
                'segments': 0,
                'file_info': [],
                'status': 'failed',
                'error': f'Original .bdf file not found: {original_file}'
            }
        
        # Load ORIGINAL .bdf file - load only EEG channels to save memory
        # Use preload=False to avoid all workers allocating 2GB simultaneously
        t_start = time.time()
        print(f"[{participant_id}] Loading .bdf file metadata...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Load without preload first to get events without allocating full data
            raw = mne.io.read_raw_bdf(
                original_file, 
                include=EEG_CHANNELS + ['Status'],  # Only load these channels
                preload=False,  # Don't allocate memory yet
                verbose=False
            )
        print(f"[{participant_id}] Metadata loaded in {time.time()-t_start:.1f}s", flush=True)
        
        original_sfreq = raw.info['sfreq']
        
        # Extract events from Status channel (this loads Status channel only temporarily)
        t_event_start = time.time()
        events = mne.find_events(raw, stim_channel='Status', shortest_event=1, verbose=False)
        print(f"[{participant_id}] Events extracted in {time.time()-t_event_start:.1f}s", flush=True)
        
        # Extract temporal_demand events (codes 120-129)
        TEMPORAL_DEMAND_BASE = 120
        temporal_events = events[
            (events[:, 2] >= TEMPORAL_DEMAND_BASE) & 
            (events[:, 2] < TEMPORAL_DEMAND_BASE + 10)
        ]
        
        if len(temporal_events) == 0:
            return {
                'participant': participant_id,
                'segments': 0,
                'file_info': [],
                'status': 'failed',
                'error': 'No temporal_demand events found'
            }
        
        # Extract timestamps
        event_timestamps = temporal_events[:, 0] / original_sfreq
        num_segments = len(event_timestamps)
        
        # Define segment boundaries
        segment_boundaries = np.concatenate([[0], event_timestamps, [raw.times[-1]]])
        
        # Drop Status channel now that we have events (keep only EEG channels)
        raw.pick_channels(EEG_CHANNELS, ordered=True)
        
        print(f"[{participant_id}] Processing {num_segments} segments...", flush=True)
        
        # Process each segment
        segment_info = []
        segments_processed = 0
        segment_errors = []
        
        for seg_idx in range(num_segments):
            try:
                seg_start_time = segment_boundaries[seg_idx]
                seg_end_time = segment_boundaries[seg_idx + 1]
                duration = seg_end_time - seg_start_time
                
                t_seg_start = time.time()
                print(f"[{participant_id}] Segment {seg_idx+1}/{num_segments}: {duration:.1f}s, loading data...", flush=True)
                
                # Crop segment - this loads ONLY this segment's data into memory
                # Using crop on non-preloaded raw loads only the requested time range
                raw_segment = raw.copy().crop(tmin=seg_start_time, tmax=seg_end_time)
                raw_segment.load_data()  # Explicit load for just this segment
                raw_segment.pick_channels(EEG_CHANNELS, ordered=True)  # Drop Status if still present
                
                # Convert to DataFrame
                data = raw_segment.get_data().T
                if data.size == 0:
                    print(f"[{participant_id}] Segment {seg_idx+1}: SKIPPED (empty data)", flush=True)
                    continue
                
                df_segment = pd.DataFrame(data, columns=raw_segment.ch_names)
                if len(df_segment) == 0:
                    print(f"[{participant_id}] Segment {seg_idx+1}: SKIPPED (empty df)", flush=True)
                    continue
                
                n_samples = len(df_segment)
                print(f"[{participant_id}] Segment {seg_idx+1}: {n_samples:,} samples, creating EEG instance...", flush=True)
                
                # Create time array
                timestamps = np.arange(len(df_segment)) / original_sfreq
                sample_numbers = np.arange(len(df_segment))
                channels_dict = {col: df_segment[col].values for col in df_segment.columns}
                
                # Create EEG instance with downsampling to 128 Hz for faster feature extraction
                t_eeg = time.time()
                eeg_instance = EEG(
                    s_n=sample_numbers,
                    t=timestamps,
                    channels=channels_dict,
                    frequency=original_sfreq,  # Original frequency (1024 Hz)
                    extract_time=True,  # Downsample to 128 Hz for faster processing
                    apply_notch=(50, 60)  # Apply notch filter
                )
                print(f"[{participant_id}] Segment {seg_idx+1}: EEG created in {time.time()-t_eeg:.1f}s, generating stats...", flush=True)
                
                # Extract features
                t_stats = time.time()
                eeg_instance.generate_stats()
                print(f"[{participant_id}] Segment {seg_idx+1}: Stats generated in {time.time()-t_stats:.1f}s", flush=True)
                features_df = eeg_instance.stats
                
                if features_df is None or features_df.empty:
                    print(f"[{participant_id}] Segment {seg_idx+1}: SKIPPED (empty features)", flush=True)
                    continue
                
                # Ensure MultiIndex columns are properly formatted with named levels
                if not isinstance(features_df.columns, pd.MultiIndex):
                    # If columns are strings like "('band', 'channel')", convert to proper MultiIndex
                    features_df.columns = pd.MultiIndex.from_tuples(features_df.columns)
                
                # Ensure MultiIndex has proper names (required by fastparquet)
                features_df.columns.names = ['band', 'channel']
                
                # Save features
                seg_name = f"{participant_id}_seg{seg_idx+1:02d}"
                output_file = output_features_dir / f"{seg_name}_features.parquet"
                
                features_df.to_parquet(output_file, engine="fastparquet", index=False)
                if not output_file.exists():
                    print(f"[{participant_id}] Segment {seg_idx+1}: FAILED to save!", flush=True)
                    continue
                
                print(f"[{participant_id}] Segment {seg_idx+1}: SAVED in {time.time()-t_seg_start:.1f}s total", flush=True)
                
                segment_info.append({
                    'participant': participant_id,
                    'segment': int(seg_idx + 1),
                    'seg_name': seg_name,
                    'time_range': f"{seg_start_time:.1f}-{seg_end_time:.1f}s",
                    'event_time': f"{event_timestamps[seg_idx]:.1f}s",
                    'n_samples': int(len(df_segment)),
                    'duration_sec': float(len(df_segment) / original_sfreq),
                    'n_features': int(len(features_df.columns)),
                    'status': 'success'
                })
                segments_processed += 1
                
            except Exception as seg_error:
                print(f"[{participant_id}] Segment {seg_idx+1}: ERROR - {seg_error}", flush=True)
                # Log segment error for debugging
                segment_errors.append({
                    'segment': seg_idx + 1,
                    'error': str(seg_error)
                })
                continue
        
        print(f"[{participant_id}] DONE: {segments_processed}/{num_segments} segments saved", flush=True)
        
        return {
            'participant': participant_id,
            'segments': segments_processed,
            'file_info': segment_info,
            'status': 'success',
            'error': None,
            'segment_errors': segment_errors if segment_errors else None
        }
        
    except Exception as e:
        print(f"[{eeg_file.stem.replace('_eeg_raw', '')}] FATAL ERROR: {e}", flush=True)
        return {
            'participant': eeg_file.stem.replace('_eeg_raw', ''),
            'segments': 0,
            'file_info': [],
            'status': 'failed',
            'error': str(e)
        }


def step2_extract_features(n_jobs=14):
    """
    Extract statistical features from SENSE-42 EEG using event-based segmentation.
    
    STEP 2: Segments each participant's EEG based on temporal_demand events from Status channel.
    
    Uses parallel processing to speed up feature extraction across participants.
    
    Args:
        n_jobs (int): Number of parallel workers. None = all CPUs, 1 = sequential, 14 = recommended
    
    For each participant:
    1. Extracts temporal_demand events (codes 120-129) from Status channel
    2. Uses event timestamps as exact segment boundaries (~25 per participant)
    3. Loads ORIGINAL .bdf file and crops to event-defined segments
    4. Extracts features from each segment independently (downsamples 1024→128 Hz)
    5. Saves features as P{ID}_seg{XX}_features.parquet
    6. Zero timing error (event-based, not uniform trial spacing)
    
    This approach uses actual event timestamps (not CSV indices) for perfect alignment.
    
    Returns:
        List of dictionaries with feature extraction details
    """
    
    if not OUTPUT_RAW_EEG.exists():
        print(f"ERROR: Extracted EEG directory not found: {OUTPUT_RAW_EEG}")
        print("Please run step1_extract_raw_eeg() first")
        return []
    
    if not SOURCE_EEG_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_EEG_DIR}")
        return []
    
    if not SOURCE_CSV_DIR.exists():
        print(f"ERROR: CSV directory not found: {SOURCE_CSV_DIR}")
        return []
        
    try:
        OUTPUT_FEATURES.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create output directory: {e}")
        return []

    # Determine number of workers
    if n_jobs is None:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    print(f"\n{'='*80}")
    print(f"SENSE-42 FEATURE EXTRACTION WITH EVENT-BASED SEGMENTATION (PARALLEL)")
    print(f"{'='*80}")
    print(f"Input: {OUTPUT_RAW_EEG}")
    print(f"Original data: {SOURCE_EEG_DIR}")
    print(f"Output: {OUTPUT_FEATURES}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz -> 128 Hz")
    print(f"Segmentation: temporal_demand events (codes 120-129)")
    print(f"Parallel workers: {n_workers}")
    print(f"{'='*80}\n")

    # Get all extracted EEG files (convert to list to ensure serialization)
    eeg_files = sorted([Path(f) for f in OUTPUT_RAW_EEG.glob("P*_eeg_raw.parquet")])
    
    if not eeg_files:
        print(f"ERROR: No extracted EEG files found in {OUTPUT_RAW_EEG}")
        return []
    
    print(f"Found {len(eeg_files)} participants to process\n")
    
    # Process files in parallel or sequentially
    if n_workers > 1:
        print(f"Processing participants in parallel using {n_workers} workers...")
        
        # Create partial function with fixed arguments
        process_func = partial(
            _process_single_participant,
            source_eeg_dir=SOURCE_EEG_DIR,
            output_features_dir=OUTPUT_FEATURES,
            sampling_rate=SAMPLING_RATE
        )
        
        # Use multiprocessing Pool with progress bar and error handling
        try:
            with Pool(processes=n_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_func, eeg_files),
                    total=len(eeg_files),
                    desc="Extracting features"
                ))
        except Exception as pool_error:
            print(f"\n❌ Error in parallel processing: {pool_error}")
            print(f"Falling back to sequential processing...")
            results = []
            for eeg_file in tqdm(eeg_files, desc="Extracting features (sequential fallback)"):
                result = _process_single_participant(
                    eeg_file,
                    source_eeg_dir=SOURCE_EEG_DIR,
                    output_features_dir=OUTPUT_FEATURES,
                    sampling_rate=SAMPLING_RATE
                )
                results.append(result)
    else:
        print(f"Processing participants sequentially...")
        results = []
        for eeg_file in tqdm(eeg_files, desc="Extracting features"):
            result = _process_single_participant(
                eeg_file,
                source_eeg_dir=SOURCE_EEG_DIR,
                output_features_dir=OUTPUT_FEATURES,
                sampling_rate=SAMPLING_RATE
            )
            results.append(result)
    
    # Aggregate results
    processed_files = sum(1 for r in results if r['status'] == 'success')
    failed_files = sum(1 for r in results if r['status'] == 'failed')
    total_segments = sum(r['segments'] for r in results)
    
    file_info = []
    failed_files_info = []
    
    for result in results:
        if result['status'] == 'success':
            file_info.extend(result['file_info'])
        else:
            failed_files_info.append({
                'participant': result['participant'],
                'input_file': str(result.get('participant', 'unknown')),
                'error': result.get('error', 'Unknown error'),
                'traceback': ''
            })
    
    # Summary
    print(f"\n{'='*80}")
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {processed_files} participants")
    print(f"Total segments: {total_segments}")
    print(f"Failed: {failed_files} files")
    
    if processed_files > 0:
        total_duration = sum(f['duration_sec'] for f in file_info if f['status'] == 'success')
        total_samples = sum(f['n_samples'] for f in file_info if f['status'] == 'success')
        
        print(f"\nOutput directory: {OUTPUT_FEATURES}")
        print(f"Segments per participant: ~{total_segments/processed_files:.0f}")
        
        print(f"\nStatistics:")
        print(f"  Total segments: {total_segments}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total duration: {total_duration/60:.1f} minutes")
    
    # Save processing summary (convert numpy types to native Python types)
    summary_file = OUTPUT_FEATURES / 'feature_extraction_summary.json'
    
    # Helper function to convert numpy types
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    with open(summary_file, 'w') as f:
        json.dump({
            'total_files': int(len(eeg_files)),
            'processed_files': int(processed_files),
            'total_segments': int(total_segments),
            'failed_files': int(failed_files),
            'sampling_rate': int(SAMPLING_RATE),
            'file_details': convert_to_native(file_info),
            'failed_files_info': failed_files_info
        }, f, indent=2)
    
    print(f"\nProcessing summary saved to: {summary_file}")
    
    if failed_files_info:
        print('\nFailed Files (first 5):')
        for f in failed_files_info[:5]:
            print(f"  - {Path(f['input_file']).name}: {f['error']}")
        if len(failed_files_info) > 5:
            print(f"  ... and {len(failed_files_info) - 5} more failed files")
    
    return file_info


# =============================================================================
# STEP 2.5: Create Target Files from EEG Event Encoding
# =============================================================================

def step2_5_create_targets():
    """
    Create target files for NASA-TLX subscales from EEG Status channel events.
    
    SENSE-42 encodes TLX ratings directly in the EEG Status channel using event codes:
        Event Code = QUESTION_BASE + QUESTION_LEAP * INDEX + RATING
        
    Where:
        QUESTION_BASE = 100
        QUESTION_LEAP = 10
        
    TLX dimensions and their base codes:
        - mental_demand: 110-119 (28 events - has duplicates)
        - temporal_demand: 120-129 (25 events - reference)
        - performance: 130-139 (25 events)
        - effort: 140-149 (25 events)
        - frustration: 150-159 (25 events)
    
    Ratings are encoded on 0-9 scale in event codes, rescaled to 0-100 for consistency.
    Performance is inverted (higher performance = lower workload).
    
    Uses temporal_demand events as segment boundaries (stable 25 events).
    Mental_demand duplicates handled via nearest-neighbor matching to temporal boundaries.
    """
    print("\n" + "="*80)
    print("STEP 2.5: Creating NASA-TLX Target Files from EEG Event Encoding")
    print("="*80)
    
    created_count = 0
    segment_count = 0
    
    # TLX dimension definitions (event code bases)
    TLX_DIMENSIONS = {
        'mental': 110,
        'temporal': 120,
        'performance': 130,
        'effort': 140,
        'frustration': 150
    }
    
    # Find all .bdf files
    bdf_files = list(SOURCE_EEG_DIR.glob("*.bdf"))
    
    if not bdf_files:
        print(f"\n⚠ No .bdf files found in {SOURCE_EEG_DIR}")
        return
    
    print(f"\nFound {len(bdf_files)} .bdf file(s) to process\n")
    
    for bdf_file in tqdm(bdf_files, desc="Creating target files from EEG events"):
        try:
            # Extract participant ID from filename (p001.bdf -> P001)
            stem = bdf_file.stem.upper()
            participant_id = stem if stem.startswith('P') else f"P{stem}"
            
            # Load EEG and extract events from Status channel
            raw = mne.io.read_raw_bdf(bdf_file, preload=False, verbose=False)
            events = mne.find_events(raw, stim_channel='Status', shortest_event=1, verbose=False)
            sfreq = raw.info['sfreq']
            
            # Extract all TLX dimension events
            dimension_data = {}
            
            for dim_name, base_code in TLX_DIMENSIONS.items():
                # Extract events for this dimension (codes: base to base+9)
                dim_events = events[
                    (events[:, 2] >= base_code) & 
                    (events[:, 2] < base_code + 10)
                ]
                
                if len(dim_events) == 0:
                    print(f"  ⚠ {participant_id}: No {dim_name} events found")
                    dimension_data[dim_name] = {'timestamps': [], 'ratings': []}
                    continue
                
                # Extract timestamps and ratings
                timestamps = dim_events[:, 0] / sfreq  # Convert to seconds
                ratings = dim_events[:, 2] - base_code  # Extract 0-9 rating
                
                dimension_data[dim_name] = {
                    'timestamps': timestamps,
                    'ratings': ratings,
                    'count': len(timestamps)
                }
            
            # Use temporal_demand as reference for segment boundaries (stable 25 events)
            if len(dimension_data['temporal']['timestamps']) == 0:
                print(f"  ⚠ {participant_id}: No temporal_demand events - skipping")
                continue
            
            temporal_times = dimension_data['temporal']['timestamps']
            num_segments = len(temporal_times)
            
            print(f"\n  Processing {participant_id}: {num_segments} segments (from temporal_demand events)")
            
            # Create target files for each segment
            for seg_idx in range(num_segments):
                seg_name = f"{participant_id}_seg{seg_idx+1:02d}"
                segment_time = temporal_times[seg_idx]
                
                # For each dimension, find nearest event to this segment's temporal_demand time
                subscale_scores = {}
                
                for dim_name in TLX_DIMENSIONS.keys():
                    dim_times = dimension_data[dim_name]['timestamps']
                    dim_ratings = dimension_data[dim_name]['ratings']
                    
                    if len(dim_times) == 0:
                        subscale_scores[dim_name] = 0.0  # Default if missing
                        continue
                    
                    # Find nearest event to this segment's reference time
                    nearest_idx = np.argmin(np.abs(dim_times - segment_time))
                    rating_0_9 = dim_ratings[nearest_idx]
                    
                    # Rescale from 0-9 to 0-100
                    rating_scaled = (rating_0_9 / 9) * 100
                    
                    # Invert performance (higher performance = lower workload)
                    if dim_name == 'performance':
                        rating_scaled = 100 - rating_scaled
                    
                    subscale_scores[dim_name] = rating_scaled
                
                # Calculate combined TLX score (average of 5 subscales)
                combined_tlx = np.mean(list(subscale_scores.values()))
                
                # Create target files for this segment
                # Combined TLX score
                target_file = OUTPUT_FEATURES / f"{seg_name}_features.txt"
                with open(target_file, 'w') as f:
                    f.write(f"{combined_tlx:.2f}")
                
                # Individual subscales
                for subscale, score in subscale_scores.items():
                    subscale_file = OUTPUT_FEATURES / f"{seg_name}_features_{subscale}.txt"
                    with open(subscale_file, 'w') as f:
                        f.write(f"{score:.2f}")
                
                segment_count += 1
            
            created_count += 1
            
        except Exception as e:
            print(f"\n  ❌ Error processing {bdf_file.name}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n✅ Step 2.5 complete: {segment_count} segments from {created_count} participant(s)")
    print(f"   Each segment has 6 target files (1 combined + 5 subscales)")
    print(f"   Ratings extracted from EEG Status channel events (zero timing error)")
    print(f"   Output: {OUTPUT_FEATURES}")


# =============================================================================
# STEP 3: Create Time-Series Dataset with Segmentation
# =============================================================================

def _process_single_participant_time(eeg_file, source_eeg_dir, output_time_dir, sampling_rate):
    """
    Worker function to segment time-series for a single participant.
    
    Segments raw EEG time-series at temporal_demand event boundaries.
    
    Args:
        eeg_file: Path to extracted EEG file (P###_eeg_raw.parquet)
        source_eeg_dir: Path to original .bdf files
        output_time_dir: Path to save segmented time-series files
        sampling_rate: Original sampling rate (1024 Hz)
        
    Returns:
        dict with 'participant', 'segments', 'file_info', 'status', 'error'
    """
    try:
        participant_id = eeg_file.stem.replace('_eeg_raw', '')
        
        # Find corresponding .bdf file
        participant_num = participant_id[1:].lstrip('0') or '0'
        bdf_pattern = f"p{participant_num.zfill(3)}.bdf"
        original_file = source_eeg_dir / bdf_pattern
        
        if not original_file.exists():
            return {
                'participant': participant_id,
                'segments': 0,
                'file_info': [],
                'status': 'failed',
                'error': f'Original .bdf file not found: {original_file}'
            }
        
        # Load ORIGINAL .bdf file once (suppress MNE warnings)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = mne.io.read_raw_bdf(original_file, preload=True, verbose=False)
        
        original_sfreq = raw.info['sfreq']
        
        # Validate EEG channels exist
        missing_channels = [ch for ch in EEG_CHANNELS if ch not in raw.ch_names]
        if missing_channels:
            return {
                'participant': participant_id,
                'segments': 0,
                'file_info': [],
                'status': 'failed',
                'error': f'Missing channels: {missing_channels}'
            }
        
        # Extract events from Status channel
        events = mne.find_events(raw, stim_channel='Status', shortest_event=1, verbose=False)
        
        # Extract temporal_demand events (codes 120-129)
        TEMPORAL_DEMAND_BASE = 120
        temporal_events = events[
            (events[:, 2] >= TEMPORAL_DEMAND_BASE) & 
            (events[:, 2] < TEMPORAL_DEMAND_BASE + 10)
        ]
        
        if len(temporal_events) == 0:
            return {
                'participant': participant_id,
                'segments': 0,
                'file_info': [],
                'status': 'failed',
                'error': 'No temporal_demand events found'
            }
        
        # Extract timestamps
        event_timestamps = temporal_events[:, 0] / original_sfreq
        num_segments = len(event_timestamps)
        
        # Define segment boundaries
        segment_boundaries = np.concatenate([[0], event_timestamps, [raw.times[-1]]])
        
        # Pick EEG channels
        raw.pick_channels(EEG_CHANNELS, ordered=True)
        
        # Process each segment
        segment_info = []
        segments_processed = 0
        segment_errors = []
        
        for seg_idx in range(num_segments):
            try:
                seg_start_time = segment_boundaries[seg_idx]
                seg_end_time = segment_boundaries[seg_idx + 1]
                
                # Crop segment
                raw_segment = raw.copy().crop(tmin=seg_start_time, tmax=seg_end_time)
                
                # Convert to DataFrame
                data = raw_segment.get_data().T
                if data.size == 0:
                    continue
                
                df_segment = pd.DataFrame(data, columns=raw_segment.ch_names)
                if len(df_segment) == 0:
                    continue
                
                # Apply subband decomposition using EEG class
                timestamps = np.arange(len(df_segment)) / original_sfreq
                sample_numbers = np.arange(len(df_segment))
                
                # Prepare channel data dictionary
                channels_dict = {col: df_segment[col].values for col in df_segment.columns}
                
                # Create EEG instance with subband decomposition (preserve sampling rate)
                eeg_instance = EEG(
                    s_n=sample_numbers,
                    t=timestamps,
                    channels=channels_dict,
                    frequency=original_sfreq,
                    extract_time=False  # Preserve original sampling rate
                )
                
                # Get decomposed data with MultiIndex (band, channel)
                decomposed_df = eeg_instance.data
                
                if decomposed_df is None or decomposed_df.empty:
                    continue
                
                # Save time-series segment with subband decomposition
                seg_name = f"{participant_id}_seg{seg_idx+1:02d}"
                output_file = output_time_dir / f"{seg_name}_eeg_raw.parquet"
                
                decomposed_df.to_parquet(output_file, index=True)
                if not output_file.exists():
                    continue
                
                segment_info.append({
                    'participant': participant_id,
                    'segment': int(seg_idx + 1),
                    'seg_name': seg_name,
                    'time_range': f"{seg_start_time:.1f}-{seg_end_time:.1f}s",
                    'event_time': f"{event_timestamps[seg_idx]:.1f}s",
                    'n_samples': int(len(decomposed_df)),
                    'duration_sec': float(len(decomposed_df) / original_sfreq),
                    'bands': decomposed_df.columns.get_level_values(0).unique().tolist(),
                    'status': 'success'
                })
                segments_processed += 1
                
            except Exception as seg_error:
                # Log segment error for debugging
                segment_errors.append({
                    'segment': seg_idx + 1,
                    'error': str(seg_error)
                })
                continue
        
        return {
            'participant': participant_id,
            'segments': segments_processed,
            'file_info': segment_info,
            'status': 'success',
            'error': None,
            'segment_errors': segment_errors if segment_errors else None
        }
        
    except Exception as e:
        return {
            'participant': eeg_file.stem.replace('_eeg_raw', ''),
            'segments': 0,
            'file_info': [],
            'status': 'failed',
            'error': str(e)
        }


def step3_create_time_dataset(n_jobs=14):
    """
    Create time-series dataset by segmenting raw EEG at event boundaries WITH subband decomposition.
    
    Segments each participant's EEG based on temporal_demand events (same as features).
    Applies subband decomposition (Overall, delta, theta, alpha, beta, gamma) to each segment.
    Uses parallel processing for efficiency.
    Saves directly to time classification dataset folder (no intermediate copy needed).
    
    Args:
        n_jobs (int): Number of parallel workers. None = all CPUs, 1 = sequential, 14 = recommended
    
    Output: sense42_time_classification_dataset/all/ (with MultiIndex: band × channel)
    """
    print("\n" + "="*80)
    print("STEP 3: Creating Time-Series Dataset (with segmentation + subband decomposition)")
    print("="*80)
    
    if not OUTPUT_RAW_EEG.exists():
        print(f"ERROR: Extracted EEG directory not found: {OUTPUT_RAW_EEG}")
        print("Please run step1_extract_raw_eeg() first")
        return
    
    if not SOURCE_EEG_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_EEG_DIR}")
        return
    
    # Create output directory with 'all' subdirectory - save directly to classification folder
    all_dir = OUTPUT_TIME_CLASSIFICATION / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if n_jobs is None:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())
    
    print(f"\nInput: {OUTPUT_RAW_EEG}")
    print(f"Original data: {SOURCE_EEG_DIR}")
    print(f"Output: {all_dir}")
    print(f"Segmentation: temporal_demand events (codes 120-129)")
    print(f"Parallel workers: {n_workers}\n")
    
    # Get all extracted EEG files
    eeg_files = sorted([Path(f) for f in OUTPUT_RAW_EEG.glob("P*_eeg_raw.parquet")])
    
    if not eeg_files:
        print(f"ERROR: No extracted EEG files found in {OUTPUT_RAW_EEG}")
        return
    
    print(f"Found {len(eeg_files)} participants to process\n")
    
    # Process files in parallel or sequentially
    if n_workers > 1:
        print(f"Processing participants in parallel using {n_workers} workers...")
        
        # Create partial function with fixed arguments
        process_func = partial(
            _process_single_participant_time,
            source_eeg_dir=SOURCE_EEG_DIR,
            output_time_dir=all_dir,
            sampling_rate=SAMPLING_RATE
        )
        
        # Use multiprocessing Pool with progress bar and error handling
        try:
            with Pool(processes=n_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_func, eeg_files),
                    total=len(eeg_files),
                    desc="Segmenting time-series"
                ))
        except Exception as pool_error:
            print(f"\n❌ Error in parallel processing: {pool_error}")
            print(f"Falling back to sequential processing...")
            results = []
            for eeg_file in tqdm(eeg_files, desc="Segmenting time-series (sequential fallback)"):
                result = _process_single_participant_time(
                    eeg_file,
                    source_eeg_dir=SOURCE_EEG_DIR,
                    output_time_dir=all_dir,
                    sampling_rate=SAMPLING_RATE
                )
                results.append(result)
    else:
        print(f"Processing participants sequentially...")
        results = []
        for eeg_file in tqdm(eeg_files, desc="Segmenting time-series"):
            result = _process_single_participant_time(
                eeg_file,
                source_eeg_dir=SOURCE_EEG_DIR,
                output_time_dir=all_dir,
                sampling_rate=SAMPLING_RATE
            )
            results.append(result)
    
    # Aggregate results
    processed_files = sum(1 for r in results if r['status'] == 'success')
    failed_files = sum(1 for r in results if r['status'] == 'failed')
    total_segments = sum(r['segments'] for r in results)
    
    file_info = []
    failed_files_info = []
    
    for result in results:
        if result['status'] == 'success':
            file_info.extend(result['file_info'])
        else:
            failed_files_info.append({
                'participant': result['participant'],
                'error': result.get('error', 'Unknown error')
            })
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TIME-SERIES SEGMENTATION SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {processed_files} participants")
    print(f"Total segments: {total_segments}")
    print(f"Failed: {failed_files} files")
    
    if processed_files > 0:
        total_duration = sum(f['duration_sec'] for f in file_info if f['status'] == 'success')
        total_samples = sum(f['n_samples'] for f in file_info if f['status'] == 'success')
        
        print(f"\nOutput directory: {all_dir}")
        print(f"Segments per participant: ~{total_segments/processed_files:.0f}")
        
        print(f"\nStatistics:")
        print(f"  Total segments: {total_segments}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total duration: {total_duration/60:.1f} minutes")
    
    if failed_files_info:
        print('\nFailed Files (first 5):')
        for f in failed_files_info[:5]:
            print(f"  - {f['participant']}: {f['error']}")
        if len(failed_files_info) > 5:
            print(f"  ... and {len(failed_files_info) - 5} more failed files")
    
    print(f"\n✅ Step 3 complete:")
    print(f"   Time-series segments: {total_segments}")
    print(f"   Output: {all_dir}")


# =============================================================================
# STEP 4: Create Feature Dataset
# =============================================================================

def step4_create_feature_dataset():
    """
    Create feature classification dataset by copying extracted feature files to 'all/' subdirectory.
    
    This creates the structure required for EEGRawDataset:
        sense42_feature_classification_dataset/
            all/
                P001_seg01_features.parquet
                P001_seg02_features.parquet
                ...
    
    Output: sense42_feature_classification_dataset/all/
    """
    print("\n" + "="*80)
    print("STEP 4: Creating Feature Classification Dataset")
    print("="*80)
    
    # Create output directory with 'all' subdirectory
    all_dir = OUTPUT_FEATURE_CLASSIFICATION / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy feature files to all/ subdirectory
    copied_count = 0
    for feature_file in tqdm(list(OUTPUT_FEATURES.glob("P*_seg*_features.parquet")), desc="Copying feature files"):
        output_file = all_dir / feature_file.name
        if not output_file.exists():
            shutil.copy2(feature_file, output_file)
            copied_count += 1
    
    print(f"\n✅ Step 4 complete:")
    print(f"   Feature files copied: {copied_count}")
    print(f"   Output: {all_dir}")


# =============================================================================
# STEP 4.1: Create Time Classification Dataset
# =============================================================================

def step4_1_create_time_classification_dataset():
    """
    Verify time classification dataset exists (files created directly in Step 3).
    
    Step 3 now saves directly to sense42_time_classification_dataset/all/,
    so this step just verifies the files exist.
    
    Output: sense42_time_classification_dataset/all/
    """
    print("\n" + "="*80)
    print("STEP 4.1: Verifying Time Classification Dataset")
    print("="*80)
    
    # Check if output directory exists
    all_dir = OUTPUT_TIME_CLASSIFICATION / "all"
    
    if not all_dir.exists():
        print(f"\n⚠ Warning: Time classification directory not found: {all_dir}")
        print(f"   Run step 3 first to create time-series segments")
        return
    
    # Count existing time-series files
    time_files = list(all_dir.glob("P*_seg*_eeg_raw.parquet"))
    
    if not time_files:
        print(f"\n⚠ Warning: No time-series files found in {all_dir}")
        print(f"   Run step 3 first to create time-series segments")
        return
    
    print(f"\n✅ Step 4.1 complete:")
    print(f"   Time-series files found: {len(time_files)}")
    print(f"   Location: {all_dir}")
    print(f"   (Files created directly by Step 3, no copy needed)")


# =============================================================================
# STEP 4.5: Create Classification Metadata
# =============================================================================

def step4_5_create_classification_metadata():
    """
    Create classification_metadata.json for multi-subscale classification.
    
    Converts continuous TLX scores (0-100) to categorical labels (low/medium/high)
    using tertile-based binning for each subscale independently.
    
    Output: sense42_feature_dataset/classification_metadata.json
    """
    print("\n" + "="*80)
    print("STEP 4.5: Creating Classification Metadata")
    print("="*80)
    
    import json
    
    # Get all feature files
    feature_files = sorted(OUTPUT_FEATURES.glob("P*_seg*_features.parquet"))
    
    if not feature_files:
        print("\n⚠ No feature files found. Run step 2 first.")
        return
    
    # Collect TLX scores for all subscales
    subscales = ['combined', 'mental', 'temporal', 'performance', 'effort', 'frustration']
    scores_by_subscale = {sub: [] for sub in subscales}
    file_scores = {}  # {filename: {subscale: score}}
    
    print("\nCollecting TLX scores...")
    for feature_file in tqdm(feature_files, desc="Reading targets"):
        base_name = feature_file.stem  # e.g., "P001_seg01_features"
        filename = feature_file.name
        
        file_scores[filename] = {}
        
        # Read combined TLX
        combined_file = OUTPUT_FEATURES / f"{base_name}.txt"
        if combined_file.exists():
            with open(combined_file, 'r') as f:
                score = float(f.read().strip())
                scores_by_subscale['combined'].append(score)
                file_scores[filename]['combined'] = score
        
        # Read subscale TLX scores
        for subscale in ['mental', 'temporal', 'performance', 'effort', 'frustration']:
            subscale_file = OUTPUT_FEATURES / f"{base_name}_{subscale}.txt"
            if subscale_file.exists():
                with open(subscale_file, 'r') as f:
                    score = float(f.read().strip())
                    scores_by_subscale[subscale].append(score)
                    file_scores[filename][subscale] = score
    
    # Compute tertile thresholds for each subscale
    print("\nComputing tertile thresholds...")
    thresholds = {}
    for subscale in subscales:
        if len(scores_by_subscale[subscale]) > 0:
            scores = np.array(scores_by_subscale[subscale])
            t33 = np.percentile(scores, 33.33)
            t66 = np.percentile(scores, 66.67)
            thresholds[subscale] = (t33, t66)
            print(f"  {subscale}: low < {t33:.1f} < medium < {t66:.1f} < high")
    
    # Convert scores to labels (for both feature and time filenames)
    feature_metadata = {}
    time_metadata = {}
    
    for filename, scores in file_scores.items():
        # Feature metadata (P001_seg01_features.parquet)
        feature_metadata[filename] = {}
        
        # Time metadata (P001_seg01_eeg_raw.parquet)
        time_filename = filename.replace('_features.parquet', '_eeg_raw.parquet')
        time_metadata[time_filename] = {}
        
        for subscale, score in scores.items():
            if subscale in thresholds:
                t33, t66 = thresholds[subscale]
                if score < t33:
                    label = 0  # low
                elif score < t66:
                    label = 1  # medium
                else:
                    label = 2  # high
                
                feature_metadata[filename][subscale] = label
                time_metadata[time_filename][subscale] = label
    
    # Save metadata to both feature and time classification dataset directories
    feature_metadata_file = OUTPUT_FEATURE_CLASSIFICATION / "classification_metadata.json"
    time_metadata_file = OUTPUT_TIME_CLASSIFICATION / "classification_metadata.json"
    
    # Ensure directories exist
    OUTPUT_FEATURE_CLASSIFICATION.mkdir(parents=True, exist_ok=True)
    OUTPUT_TIME_CLASSIFICATION.mkdir(parents=True, exist_ok=True)
    
    with open(feature_metadata_file, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    with open(time_metadata_file, 'w') as f:
        json.dump(time_metadata, f, indent=2)
    
    print(f"\n✅ Step 4.5 complete:")
    print(f"   Classification metadata:")
    print(f"     Feature files: {len(feature_metadata)}")
    print(f"     Time files: {len(time_metadata)}")
    print(f"   Subscales: {', '.join(subscales)}")
    print(f"   Label distribution per subscale:")
    
    for subscale in subscales:
        labels = [meta.get(subscale, -1) for meta in feature_metadata.values()]
        counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
        print(f"     {subscale}: low={counts[0]}, medium={counts[1]}, high={counts[2]}")
    
    print(f"   Output:")
    print(f"     Feature: {feature_metadata_file}")
    print(f"     Time: {time_metadata_file}")


# =============================================================================
# STEP 5: Create Regression Datasets
# =============================================================================

def step5_create_regression_datasets():
    """
    Create regression datasets by copying target files.
    
    Creates:
        - Copies txt target files to sense42_time_classification_dataset/all/ (time regression)
        - Copies feature + target files to sense42_feature_regression_dataset/ (feature regression)
        
    Note: Time regression shares the all/ folder with time classification to avoid file duplication.
    """
    print("\n" + "="*80)
    print("STEP 5: Creating Regression Datasets")
    print("="*80)
    
    # Create feature regression output directory
    OUTPUT_FEATURE_REGRESSION.mkdir(parents=True, exist_ok=True)
    
    # For time regression: copy txt files to time classification's all/ folder
    # The parquet files are already there from Step 3
    time_target_count = 0
    
    # Get all time-series segment files from time classification dataset
    time_classification_dir = OUTPUT_TIME_CLASSIFICATION / "all"
    if time_classification_dir.exists():
        time_files = list(time_classification_dir.glob("P*_seg*_eeg_raw.parquet"))
        
        print(f"\nCopying target txt files to time classification all/ folder...")
        for time_file in tqdm(time_files, desc="Time regression targets"):
            # Extract base name without extension (e.g., "P001_seg01_eeg_raw")
            base_name = time_file.stem
            # Convert to feature base name (e.g., "P001_seg01_features")
            feature_base = base_name.replace("_eeg_raw", "_features")
            
            # Check if all target files exist (combined + 5 subscales)
            target_base = OUTPUT_FEATURES / f"{feature_base}.txt"
            
            all_targets_exist = target_base.exists()
            for subscale in ['mental', 'temporal', 'performance', 'effort', 'frustration']:
                subscale_file = OUTPUT_FEATURES / f"{feature_base}_{subscale}.txt"
                all_targets_exist = all_targets_exist and subscale_file.exists()
            
            if not all_targets_exist:
                continue
            
            # Copy all target files to time classification all/ folder (rename from features to eeg_raw)
            for target_suffix in ['', '_mental', '_temporal', '_performance', '_effort', '_frustration']:
                source_target = OUTPUT_FEATURES / f"{feature_base}{target_suffix}.txt"
                target_filename = f"{base_name}{target_suffix}.txt"
                output_target = time_classification_dir / target_filename
                if source_target.exists() and not output_target.exists():
                    shutil.copy2(source_target, output_target)
                    time_target_count += 1
    else:
        print(f"\n⚠ Warning: Time classification dataset directory not found: {time_classification_dir}")
        print(f"   Run step 3 first to create time-series segments")
    
    # Copy feature regression (features + targets)
    feature_count = 0
    target_count = 0
    
    # Get all feature files (segmented)
    feature_files = list(OUTPUT_FEATURES.glob("P*_seg*_features.parquet"))
    
    print(f"\nCopying feature regression files...")
    for feature_file in tqdm(feature_files, desc="Feature regression"):
        base_name = feature_file.stem  # e.g., "P001_seg01_features"
        
        # Check if all target files exist (combined + 5 subscales)
        target_base = OUTPUT_FEATURES / f"{base_name}.txt"
        
        all_targets_exist = target_base.exists()
        for subscale in ['mental', 'temporal', 'performance', 'effort', 'frustration']:
            subscale_file = OUTPUT_FEATURES / f"{base_name}_{subscale}.txt"
            all_targets_exist = all_targets_exist and subscale_file.exists()
        
        if not all_targets_exist:
            continue
        
        # Copy feature file
        output_feature = OUTPUT_FEATURE_REGRESSION / feature_file.name
        if not output_feature.exists():
            shutil.copy2(feature_file, output_feature)
            feature_count += 1
        
        # Copy all target files
        for target_file in OUTPUT_FEATURES.glob(f"{base_name}*.txt"):
            output_target = OUTPUT_FEATURE_REGRESSION / target_file.name
            if not output_target.exists():
                shutil.copy2(target_file, output_target)
                target_count += 1
    
    print(f"\n✅ Step 5 complete:")
    print(f"   Time regression: {time_target_count} target txt files (parquet files in time classification all/)")
    print(f"   Feature regression: {feature_count} feature files, {target_count} target files")
    print(f"   Output:")
    print(f"     Time: {time_classification_dir} (shared with classification)")
    print(f"     Features: {OUTPUT_FEATURE_REGRESSION}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_all_steps(n_jobs=14):
    """
    Run complete processing pipeline with workload regression.
    
    Args:
        n_jobs (int): Number of parallel workers for step 2 (default: 14)
    """
    print("\n" + "="*80)
    print("   SENSE-42 DATASET PROCESSING PIPELINE")
    print("="*80)
    print("\nThis will process the SENSE-42 dataset into standardized datasets:")
    print("  1. sense42_time_classification_dataset (time-series for classification & regression)")
    print("  2. sense42_feature_classification_dataset (statistical features)")
    print("  3. sense42_feature_regression_dataset (features + TLX targets)")
    print("\nDataset characteristics:")
    print("  - 6 TLX subscales: combined, mental, temporal, performance, effort, frustration")
    print("  - ~25 segments per participant (continuous workload monitoring)")
    print("  - Ratings: 0-100 scale for regression, low/medium/high for classification")
    print("="*80)
    
    # Step 1: Extract raw EEG
    step1_extract_raw_eeg()
    
    # Step 2: Extract features (with segmentation)
    step2_extract_features(n_jobs=n_jobs)
    
    # Step 2.5: Create TLX target files
    step2_5_create_targets()
    
    # Step 3: Create time-series dataset
    step3_create_time_dataset(n_jobs=n_jobs)
    
    # Step 4: Create feature classification dataset
    step4_create_feature_dataset()
    
    # Step 4.1: Create time classification dataset
    step4_1_create_time_classification_dataset()
    
    # Step 4.5: Create classification metadata
    step4_5_create_classification_metadata()
    
    # Step 5: Create regression datasets
    step5_create_regression_datasets()
    
    print("\n" + "="*80)
    print("   PROCESSING COMPLETE!")
    print("="*80)
    print("\nDatasets created:")
    print(f"  1. {OUTPUT_TIME_CLASSIFICATION} (time-series created directly by Step 3)")
    print(f"  2. {OUTPUT_FEATURE_CLASSIFICATION}")
    print(f"  3. {OUTPUT_FEATURE_REGRESSION}")
    print(f"  (Note: Time regression uses {OUTPUT_TIME_CLASSIFICATION}/all/)")
    print("\nYou can now use these datasets for:")
    print("  - Workload regression (predicting TLX scores)")
    print("  - Time-series modeling (RNNs, Transformers)")
    print("  - Traditional ML (Random Forest, SVM, etc.)")
    print("  - Multi-output regression (5 TLX subscales)")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SENSE-42 Dataset Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (all steps)
  python load_sense42.py
  
  # Extract raw EEG without processing (for LaBraM)
  python load_sense42.py --raw-only
  
  # Only create feature datasets (skip time-series)
  python load_sense42.py --feature-only
  
  # Only create time-series datasets (skip features)
  python load_sense42.py --time-only
  
  # Run specific steps
  python load_sense42.py --steps 1 2 2.5 4 4.5 5
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--raw-only',
        action='store_true',
        help='Only extract raw EEG without signal processing (for LaBraM embeddings). Saves to _rawonly folder.'
    )
    mode_group.add_argument(
        '--time-only',
        action='store_true',
        help='Only create time-series datasets (steps 1, 3)'
    )
    mode_group.add_argument(
        '--feature-only',
        action='store_true',
        help='Only create feature-based datasets (steps 1, 2, 2.5, 4, 4.5, 5)'
    )
    mode_group.add_argument(
        '--steps',
        nargs='+',
        choices=['1', '2', '2.5', '3', '4', '4.1', '4.5', '5'],
        help='Run specific steps only (e.g., --steps 1 2 4 4.1 4.5 5)'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=14,
        help='Number of parallel workers for step 2 (default: 14)'
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if args.raw_only:
        # Only extract raw EEG without processing
        print("\n" + "="*80)
        print("   SENSE-42 RAW EEG EXTRACTION (NO PROCESSING)")
        print("   For LaBraM embeddings")
        print("="*80 + "\n")
        print("Note: NASA-TLX target files should already exist from previous full pipeline run.")
        print("If missing, run without --raw-only flag to create them.\n")
        
        step1_extract_raw_eeg(raw_only=True)
        
        print("\n" + "="*80)
        print("   RAW EXTRACTION COMPLETE!")
        print("="*80)
        print(f"   Output: {SENSE42_ROOT / 'sense42_raw_eeg_extracted_rawonly'}")
        print("="*80 + "\n")
        
    elif args.steps:
        # Run only specified steps
        print("\n" + "="*80)
        print(f"   SENSE-42 DATASET PROCESSING - CUSTOM STEPS: {', '.join(args.steps)}")
        print("="*80 + "\n")
        
        if '1' in args.steps:
            step1_extract_raw_eeg()
        if '2' in args.steps:
            step2_extract_features(n_jobs=args.n_jobs)
        if '2.5' in args.steps:
            step2_5_create_targets()
        if '3' in args.steps:
            step3_create_time_dataset(n_jobs=args.n_jobs)
        if '4' in args.steps:
            step4_create_feature_dataset()
        if '4.1' in args.steps:
            step4_1_create_time_classification_dataset()
        if '4.5' in args.steps:
            step4_5_create_classification_metadata()
        if '5' in args.steps:
            step5_create_regression_datasets()
            
    elif args.feature_only:
        # Feature-only mode
        print("\n" + "="*80)
        print("   SENSE-42 DATASET PROCESSING - FEATURE DATASETS ONLY")
        print("="*80)
        print("\nThis will create feature-based datasets:")
        print("  - sense42_feature_dataset (statistical features)")
        print("  - sense42_feature_regression_dataset (features + TLX targets)")
        print("="*80 + "\n")
        
        step1_extract_raw_eeg()
        step2_extract_features(n_jobs=args.n_jobs)
        step2_5_create_targets()
        step4_create_feature_dataset()
        step4_5_create_classification_metadata()
        step5_create_regression_datasets()
        
        print("\n" + "="*80)
        print("   FEATURE DATASETS COMPLETE!")
        print("="*80)
        print(f"\n  ✓ {OUTPUT_FEATURE_CLASSIFICATION}")
        print(f"  ✓ {OUTPUT_FEATURE_REGRESSION}")
        print("="*80 + "\n")
        
    elif args.time_only:
        # Time-series only mode
        print("\n" + "="*80)
        print("   SENSE-42 DATASET PROCESSING - TIME-SERIES DATASETS ONLY")
        print("="*80)
        print("\nThis will create time-series datasets:")
        print("  - sense42_time_classification_dataset (segmented time-series)")
        print("="*80 + "\n")
        
        step1_extract_raw_eeg()
        step3_create_time_dataset(n_jobs=args.n_jobs)
        step4_5_create_classification_metadata()
        
        print("\n" + "="*80)
        print("   TIME-SERIES DATASETS COMPLETE!")
        print("="*80)
        print(f"\n  ✓ {OUTPUT_TIME_CLASSIFICATION}")
        print("="*80 + "\n")
        
    else:
        # Run full pipeline
        run_all_steps(n_jobs=args.n_jobs)
