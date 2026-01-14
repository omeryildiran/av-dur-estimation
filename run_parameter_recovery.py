#!/usr/bin/env python3
"""
Parameter Recovery Analysis with Multiprocessing Support

This script runs parameter recovery analysis for model validation.
For each participant:
  1. Load fitted parameters (ground truth)
  2. Simulate synthetic datasets
  3. Re-fit the model to each synthetic dataset
  4. Compare recovered parameters to ground truth

Usage:
    # Single process (testing)
    python run_parameter_recovery.py --participants as dt jw --n_recovery 20

    # Parallel processing (recommended)
    python run_parameter_recovery.py --participants all --n_recovery 50 --n_jobs 8
        python run_parameter_recovery.py --participants as dt oy mh hh ip ln2 ml qs sx mt --n_recovery 50 --n_jobs 8


    # Specific model type
    python run_parameter_recovery.py --participants all --model lognorm --n_recovery 50 --n_jobs -1
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

# Import your project modules
import loadData
import loadResults
import monteCarloClass


def run_single_participant_recovery(participantID, modelType, n_recovery, nSimul, nStarts, save_dir):
    """
    Run parameter recovery for a single participant.
    
    Returns:
        dict: Recovery results or None if failed
    """
    # Check if results already exist
    participant_dir = os.path.join(save_dir, participantID)
    result_path = os.path.join(participant_dir, f"{participantID}_{modelType}_recovery.json")
    
    if os.path.exists(result_path):
        print(f"✓ Loading existing results for {participantID}")
        with open(result_path, 'r') as f:
            return json.load(f)
    
    try:
        # Load original data and fitted parameters
        data, dataName = loadData.loadData(participantID + "_all.csv", verbose=False)
        
        mc_original = monteCarloClass.OmerMonteCarlo(data)
        mc_original.modelName = modelType
        mc_original.freeP_c = False
        mc_original.sharedLambda = False
        mc_original.dataName = dataName
        mc_original.nSimul = nSimul
        mc_original.nStart = nStarts
        
        res = loadResults.loadFitResults(mc_original, dataName, modelName=modelType)
        true_params = np.array(res['fittedParams'])
        
    except Exception as e:
        print(f"✗ Could not load {modelType} results for {participantID}: {e}")
        return None
    
    # Run recovery iterations
    recovered_params_list = []
    
    for iter_idx in range(n_recovery):
        try:
            # Simulate data from true parameters
            sim_data = mc_original.simulateMonteCarloData(true_params, data)
            
            # Create fitter for simulated data
            mc_recovery = monteCarloClass.OmerMonteCarlo(sim_data)
            mc_recovery.modelName = modelType
            mc_recovery.freeP_c = False
            mc_recovery.sharedLambda = False
            mc_recovery.nSimul = nSimul
            mc_recovery.nStart = nStarts
            mc_recovery.optimizationMethod = 'scipy'
            mc_recovery.dataName = f"{participantID}_recovery_{iter_idx}"
            
            # Fit model to recover parameters
            recovered_params = mc_recovery.fitCausalInferenceMonteCarlo(mc_recovery.groupedData)
            if recovered_params is not None:
                recovered_params_list.append(recovered_params)
                
        except Exception as e:
            print(f"  Recovery iteration {iter_idx} failed for {participantID}: {e}")
            continue
    
    if len(recovered_params_list) > 0:
        recovered_params_array = np.array(recovered_params_list)
        
        # Calculate statistics
        result = {
            'participantID': participantID,
            'modelType': modelType,
            'true_params': true_params.tolist(),
            'recovered_params_mean': np.mean(recovered_params_array, axis=0).tolist(),
            'recovered_params_std': np.std(recovered_params_array, axis=0).tolist(),
            'recovered_params_median': np.median(recovered_params_array, axis=0).tolist(),
            'n_successful': len(recovered_params_list),
            'n_attempted': n_recovery,
            'all_recovered_params': recovered_params_array.tolist()
        }
        
        # Save results
        os.makedirs(participant_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        bias = np.mean(recovered_params_array - true_params, axis=0)
        print(f"✓ {participantID}: {len(recovered_params_list)}/{n_recovery} successful")
        print(f"  Mean bias: {np.round(bias, 4)}")
        
        return result
    else:
        print(f"✗ {participantID}: No successful recoveries")
        return None


def get_all_participant_ids():
    """Get all available participant IDs from model_fits directory."""
    json_files = glob("model_fits/**/*.json", recursive=True)
    participant_ids = set()
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 1:
            participant_ids.add(parts[0])
    
    return sorted(participant_ids)


def main():
    parser = argparse.ArgumentParser(
        description='Run parameter recovery analysis with multiprocessing support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 2 participants, 20 iterations each
  python run_parameter_recovery.py --participants as dt --n_recovery 20

  # All participants with parallel processing (8 cores)
  python run_parameter_recovery.py --participants all --n_recovery 50 --n_jobs 8

  # Use all available cores
  python run_parameter_recovery.py --participants all --n_recovery 100 --n_jobs -1
        """
    )
    
    parser.add_argument(
        '--participants',
        nargs='+',
        default=['all'],
        help='Participant IDs to analyze (e.g., as dt jw) or "all" for all participants'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lognorm',
        help='Model type to test (default: lognorm)'
    )
    parser.add_argument(
        '--n_recovery',
        type=int,
        default=50,
        help='Number of recovery iterations per participant (default: 50, recommend 50-100 for publication)'
    )
    parser.add_argument(
        '--nSimul',
        type=int,
        default=500,
        help='Monte Carlo simulations for fitting (default: 500)'
    )
    parser.add_argument(
        '--nStarts',
        type=int,
        default=1,
        help='Number of optimization starting points (default: 1)'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs. Use -1 for all CPU cores (default: 1 for serial)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='parameter_recovery_results',
        help='Directory to save results (default: parameter_recovery_results)'
    )
    
    args = parser.parse_args()
    
    # Get participant IDs
    if args.participants == ['all']:
        participant_ids = get_all_participant_ids()
        print(f"Found {len(participant_ids)} participants: {participant_ids}")
    else:
        participant_ids = args.participants
    
    if not participant_ids:
        print("No participants found!")
        sys.exit(1)
    
    # Determine number of jobs
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs > cpu_count():
        print(f"Warning: Requested {n_jobs} jobs but only {cpu_count()} CPUs available")
        n_jobs = cpu_count()
    
    print("\n" + "="*70)
    print("PARAMETER RECOVERY ANALYSIS")
    print("="*70)
    print(f"Model type:        {args.model}")
    print(f"Participants:      {len(participant_ids)}")
    print(f"Recovery iters:    {args.n_recovery}")
    print(f"MC simulations:    {args.nSimul}")
    print(f"Parallel jobs:     {n_jobs}")
    print(f"Save directory:    {args.save_dir}")
    print("="*70 + "\n")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create partial function with fixed parameters
    recovery_func = partial(
        run_single_participant_recovery,
        modelType=args.model,
        n_recovery=args.n_recovery,
        nSimul=args.nSimul,
        nStarts=args.nStarts,
        save_dir=args.save_dir
    )
    
    # Run parameter recovery
    if n_jobs == 1:
        # Serial processing
        print("Running in serial mode...")
        results = []
        for pid in tqdm(participant_ids, desc="Participants"):
            result = recovery_func(pid)
            if result is not None:
                results.append(result)
    else:
        # Parallel processing
        print(f"Running in parallel mode with {n_jobs} processes...")
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(recovery_func, participant_ids),
                total=len(participant_ids),
                desc="Participants"
            ))
        # Filter out None results
        results = [r for r in results if r is not None]
    
    # Print summary
    print("\n" + "="*70)
    print("PARAMETER RECOVERY SUMMARY")
    print("="*70)
    print(f"Total participants processed: {len(participant_ids)}")
    print(f"Successful recoveries:        {len(results)}")
    print(f"Failed:                       {len(participant_ids) - len(results)}")
    
    if results:
        print("\nPer-participant statistics:")
        for result in results:
            pid = result['participantID']
            n_success = result['n_successful']
            n_total = result['n_attempted']
            success_rate = 100 * n_success / n_total
            print(f"  {pid}: {n_success}/{n_total} ({success_rate:.1f}%)")
        
        # Calculate overall bias
        print("\nOverall parameter recovery:")
        all_biases = []
        for result in results:
            true_p = np.array(result['true_params'])
            recovered_p = np.array(result['recovered_params_mean'])
            bias = recovered_p - true_p
            all_biases.append(bias)
        
        mean_bias = np.mean(all_biases, axis=0)
        std_bias = np.std(all_biases, axis=0)
        print(f"  Mean bias (across all participants):")
        print(f"    {np.round(mean_bias, 4)}")
        print(f"  Std of bias:")
        print(f"    {np.round(std_bias, 4)}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {args.save_dir}/")
    print("="*70)
    
    print("\nTo visualize results in your notebook, run:")
    print(f"  recovery_results = load_recovery_results(modelType='{args.model}')")
    print(f"  plot_parameter_recovery(recovery_results)")


if __name__ == '__main__':
    main()
