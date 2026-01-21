"""
Automation Script for Running Aggregated LOSO Channel Importance Analysis
==========================================================================

This script runs the aggregated 20-model ensemble with LOSO cross-validation
channel importance analysis across all available datasets.

Model Ensemble (20 models):
  • Core Models (16): LogisticRegression, RandomForest, HistGradientBoosting,
    SVM, MLP, NaiveBayes, AdaBoost, DecisionTree, ExtraTrees, KNeighbors,
    RidgeClassifier, SGDClassifier, LinearSVC, LDA, QDA, Bagging
  • Optional Models (4): XGBoost, CatBoost, LightGBM
  
Hyperparameters optimized for statistical EEG features.
Channel rankings are aggregated across all models using average rank method.

Author: EEG Biosensing Team
Date: October 2025

Usage:
    python run_all_datasets_loso.py              # Run all datasets (full analysis, 20 models)
    python run_all_datasets_loso.py --test       # Run all datasets (test mode, 3 channel groups)
    python run_all_datasets_loso.py --mocas      # Run only MOCAS dataset (20 models)
    python run_all_datasets_loso.py --htc --test # Run only HTC dataset in test mode (20 models)
"""

import subprocess
import sys
import os
from datetime import datetime
import argparse

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.join(SCRIPT_DIR, 'test_feature_classification_agg_loso.py')

# Dataset configurations
DATASETS = {
    'mocas': {
        'name': 'MOCAS (Multi-Operator Cognitive Assessment)',
        'flag': '--mocas',
        'description': 'Air traffic controller workload dataset'
    },
    'htc': {
        'name': 'Heat the Chair',
        'flag': '--htc',
        'description': 'Driving task workload dataset'
    },
    'nback': {
        'name': 'N-Back',
        'flag': '--nback',
        'description': 'Working memory task dataset'
    },
    'wauc': {
        'name': 'WAUC',
        'flag': '--wauc',
        'description': 'Workload assessment dataset'
    }
}


def run_analysis(dataset_key, test_mode=False):
    """
    Run the aggregated LOSO channel importance analysis for a specific dataset.
    
    Args:
        dataset_key: Key of the dataset to run (e.g., 'mocas', 'htc')
        test_mode: If True, run in test mode (faster, 3 channel groups only)
    
    Returns:
        bool: True if successful, False otherwise
    """
    dataset_info = DATASETS[dataset_key]
    
    print(f"\n{'='*80}")
    print(f"Running Analysis: {dataset_info['name']}")
    print(f"Description: {dataset_info['description']}")
    print(f"Mode: {'TEST (3 channel groups)' if test_mode else 'FULL (11 channel groups)'}")
    print(f"{'='*80}\n")
    
    # Verify script exists
    if not os.path.exists(SCRIPT_NAME):
        print(f"❌ ERROR: Script not found: {SCRIPT_NAME}")
        return False
    
    # Build command
    cmd = [sys.executable, SCRIPT_NAME, dataset_info['flag']]
    if test_mode:
        cmd.append('--test')
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {SCRIPT_DIR}\n")
    
    # Run the analysis
    try:
        start_time = datetime.now()
        
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=False,
            text=True
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ SUCCESS: {dataset_info['name']} completed in {duration}")
            return True
        else:
            print(f"\n❌ FAILED: {dataset_info['name']} (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR running {dataset_info['name']}: {str(e)}")
        return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run aggregated 20-model ensemble LOSO channel importance analysis across all datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Model Ensemble (20 models):
  Core (16): LogisticRegression, RandomForest, HistGradientBoosting, SVM, MLP,
             NaiveBayes, AdaBoost, DecisionTree, ExtraTrees, KNeighbors,
             RidgeClassifier, SGDClassifier, LinearSVC, LDA, QDA, Bagging
  Optional (4): XGBoost, CatBoost, LightGBM (if installed)

Examples:
  python run_all_datasets_loso.py              # Run all datasets (full analysis, 20 models)
  python run_all_datasets_loso.py --test       # Run all datasets (test mode, 3 channel groups)
  python run_all_datasets_loso.py --mocas      # Run only MOCAS dataset (20 models)
  python run_all_datasets_loso.py --htc --test # Run only HTC dataset in test mode (20 models)
        '''
    )
    
    # Dataset selection (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument('--mocas', action='store_true', help='Run only MOCAS dataset')
    dataset_group.add_argument('--htc', action='store_true', help='Run only Heat the Chair dataset')
    dataset_group.add_argument('--nback', action='store_true', help='Run only N-Back dataset')
    dataset_group.add_argument('--wauc', action='store_true', help='Run only WAUC dataset')
    dataset_group.add_argument('--all', action='store_true', help='Run all datasets (default)')
    
    # Test mode
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode (3 channel groups only, faster)')
    
    args = parser.parse_args()
    
    # Determine which datasets to run
    datasets_to_run = []
    
    if args.mocas:
        datasets_to_run = ['mocas']
    elif args.htc:
        datasets_to_run = ['htc']
    elif args.nback:
        datasets_to_run = ['nback']
    elif args.wauc:
        datasets_to_run = ['wauc']
    else:
        # Default: run all datasets
        datasets_to_run = list(DATASETS.keys())
    
    # Display configuration
    print(f"\n{'='*80}")
    print(f"AGGREGATED LOSO CHANNEL IMPORTANCE ANALYSIS - BATCH RUNNER")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Mode: {'TEST (3 channel groups, ~15-30 min per dataset)' if args.test else 'FULL (11 channel groups, ~60-120 min per dataset)'}")
    print(f"  Datasets to run: {len(datasets_to_run)}")
    for dataset_key in datasets_to_run:
        print(f"    - {DATASETS[dataset_key]['name']}")
    
    # Estimate total time
    if args.test:
        time_per_dataset = "15-30 minutes"
        total_time = f"{15 * len(datasets_to_run)}-{30 * len(datasets_to_run)} minutes"
    else:
        time_per_dataset = "60-120 minutes"
        total_time = f"{60 * len(datasets_to_run)}-{120 * len(datasets_to_run)} minutes"
    
    print(f"\n  Estimated time:")
    print(f"    Per dataset: {time_per_dataset}")
    print(f"    Total: {total_time}")
    
    # Confirm execution
    print(f"\n{'='*80}")
    try:
        response = input("Press Enter to continue or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\n\n❌ Analysis cancelled by user")
        return
    
    # Run analyses
    overall_start_time = datetime.now()
    results = {}
    
    for i, dataset_key in enumerate(datasets_to_run, 1):
        print(f"\n\n{'#'*80}")
        print(f"# DATASET {i}/{len(datasets_to_run)}: {DATASETS[dataset_key]['name'].upper()}")
        print(f"{'#'*80}")
        
        success = run_analysis(dataset_key, test_mode=args.test)
        results[dataset_key] = success
    
    # Summary
    overall_end_time = datetime.now()
    total_duration = overall_end_time - overall_start_time
    
    print(f"\n\n{'='*80}")
    print(f"BATCH RUN SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal duration: {total_duration}")
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    
    print(f"\nResults:")
    success_count = sum(1 for success in results.values() if success)
    for dataset_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {DATASETS[dataset_key]['name']}")
    
    print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
    
    # Output location
    output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), '..', '..', 'channel_importance', 'tests')
    output_dir = os.path.abspath(output_dir)
    print(f"\nResults saved to: {output_dir}")
    print(f"  Look for files: channel_importance_aggregated_*_LOSO_*.csv")
    
    print(f"\n{'='*80}\n")
    
    # Exit with appropriate code
    if success_count == len(results):
        print("🎉 All analyses completed successfully!")
        sys.exit(0)
    else:
        print("⚠️ Some analyses failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
