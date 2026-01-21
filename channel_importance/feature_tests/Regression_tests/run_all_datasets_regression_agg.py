#!/usr/bin/env python3
"""
Run Aggregated Channel Importance Analysis for All Datasets - REGRESSION
==========================================================================

This script runs the aggregated regression analysis for all 4 datasets:
- MOCAS
- Heat the Chair (HTC)
- N-Back
- WAUC

Results are saved to the current directory.

Usage:
    python run_all_datasets_agg.py           # Full run (all datasets)
    python run_all_datasets_agg.py --test    # Test run (quick validation)
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime

# Dataset configurations
DATASETS = [
    {'name': 'MOCAS', 'flag': '--mocas'},
    {'name': 'Heat the Chair (HTC)', 'flag': '--htc'},
    {'name': 'N-Back', 'flag': '--nback'},
    {'name': 'WAUC', 'flag': '--wauc'}
]

# Get absolute path to the script in the same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.join(SCRIPT_DIR, 'test_feature_regression_agg.py')


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def run_analysis(dataset_info, test_mode=False):
    """
    Run analysis for a single dataset.
    
    Args:
        dataset_info: Dictionary with 'name' and 'flag' keys
        test_mode: If True, run in test mode
        
    Returns:
        bool: True if successful, False otherwise
    """
    dataset_name = dataset_info['name']
    dataset_flag = dataset_info['flag']
    
    print_header(f"Running {dataset_name} Dataset...")
    
    # Verify script exists
    if not os.path.exists(SCRIPT_NAME):
        print(f"✗ ERROR: Script not found at: {SCRIPT_NAME}")
        return False
    
    try:
        # Build command with absolute path
        cmd = [sys.executable, SCRIPT_NAME, dataset_flag]
        if test_mode:
            cmd.append('--test')
        
        # Run the analysis script from the script's directory
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=SCRIPT_DIR  # Run from the script directory
        )
        
        print(f"\n✓ {dataset_name} completed successfully\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: {dataset_name} analysis failed with error code {e.returncode}\n")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {dataset_name} analysis failed: {str(e)}\n")
        return False


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run aggregated channel importance analysis for all datasets (REGRESSION)'
    )
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (quick validation, 3 channel groups only)')
    args = parser.parse_args()
    
    test_mode = args.test
    start_time = datetime.now()
    
    mode_text = "TEST MODE - Quick Validation" if test_mode else "FULL RUN"
    print_header(f"EEG Channel Importance - Aggregated Regression Ensemble - ALL DATASETS ({mode_text})")
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Script directory: {SCRIPT_DIR}")
    print(f"  Target script: {SCRIPT_NAME}")
    print(f"  Script exists: {os.path.exists(SCRIPT_NAME)}")
    
    if not os.path.exists(SCRIPT_NAME):
        print(f"\n❌ ERROR: Analysis script not found!")
        print(f"   Expected location: {SCRIPT_NAME}")
        print(f"   Please ensure the script is in the correct location.")
        return 1
    
    print(f"\nThis will run the analysis for {len(DATASETS)} datasets sequentially:")
    for i, dataset in enumerate(DATASETS, 1):
        print(f"  {i}. {dataset['name']}")
    
    if test_mode:
        print(f"\n🧪 TEST MODE: Only 3 channel groups per dataset (All, Frontal, Central)")
        print(f"Estimated time: 8-12 minutes total (~2-3 min per dataset)")
    else:
        print(f"\nEstimated time: 60-100 minutes total (~15-25 min per dataset)")
    
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Confirm execution
    try:
        response = input("Press Enter to continue or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        return 1
    
    # Track results
    results = []
    
    # Run analysis for each dataset
    for i, dataset in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}]", end=" ")
        success = run_analysis(dataset, test_mode=test_mode)
        results.append({'name': dataset['name'], 'success': success})
        
        if not success:
            print(f"\n⚠️ Warning: {dataset['name']} failed but continuing with remaining datasets...\n")
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("ANALYSIS SUMMARY")
    
    print(f"\nCompletion time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\nResults:")
    
    success_count = 0
    for result in results:
        status = "✓" if result['success'] else "✗"
        status_text = "SUCCESS" if result['success'] else "FAILED"
        print(f"  {status} {result['name']:<25} {status_text}")
        if result['success']:
            success_count += 1
    
    print(f"\nSuccess rate: {success_count}/{len(DATASETS)} datasets")
    
    if success_count == len(DATASETS):
        print("\n🎉 ALL DATASETS COMPLETED SUCCESSFULLY!")
        print(f"\nResults have been saved to: {SCRIPT_DIR}")
        if test_mode:
            print("\nOutput files (TEST MODE):")
            print("  • channel_importance_aggregated_mocas_regression_TEST_*.csv")
            print("  • channel_importance_aggregated_htc_regression_TEST_*.csv")
            print("  • channel_importance_aggregated_nback_regression_TEST_*.csv")
            print("  • channel_importance_aggregated_wauc_regression_TEST_*.csv")
            print("  • run_details_aggregated_*_regression_TEST_*.csv (detailed metrics)")
            print("\n💡 Test passed! Run without --test for full analysis")
        else:
            print("\nOutput files:")
            print("  • channel_importance_aggregated_mocas_regression_*.csv")
            print("  • channel_importance_aggregated_htc_regression_*.csv")
            print("  • channel_importance_aggregated_nback_regression_*.csv")
            print("  • channel_importance_aggregated_wauc_regression_*.csv")
            print("  • run_details_aggregated_*_regression_*.csv (detailed metrics)")
        return 0
    else:
        print(f"\n⚠️ WARNING: {len(DATASETS) - success_count} dataset(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
