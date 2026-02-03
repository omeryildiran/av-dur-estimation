#!/usr/bin/env python3
"""
Model Recovery Analysis Script (Group-Level Parameter Sampling)
===============================================================
Run this script to perform comprehensive model recovery analysis.

This version samples parameters from distributions based on group-level statistics
(mean/SD of fitted parameters across participants). This provides a realistic test 
of model discriminability across the empirically-observed parameter space.

No participant loop is needed - we simply run N iterations per generating model,
sampling parameters from group distributions each time.

Usage:
    python run_model_recovery.py [--n_recovery N] [--n_jobs N]
    
Example:
    python run_model_recovery.py --n_recovery 100 --n_jobs 8
    python run_model_recovery.py --n_recovery 50  # Uses default CPU count
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from glob import glob
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Import project modules
import loadData
import monteCarloClass


def load_group_parameter_statistics(model_name, participants=None):
    """
    Load fitted parameters for all participants and compute group statistics.
    
    Args:
        model_name: Name of the model (e.g., 'lognorm', 'fusionOnlyLogNorm')
        participants: Optional list of participant IDs. If None, loads all available.
    
    Returns:
        dict with 'mean', 'std', 'n_participants', 'all_params' for each parameter
    """
    # Model name to filename mapping
    # LapseFix = sharedLambda=True (one shared lambda across conflict conditions)
    model_suffix = f"{model_name}_LapseFix_sharedPrior"
    
    # Find all participant directories
    model_fits_dir = "model_fits"
    if participants is None:
        participant_dirs = [d for d in os.listdir(model_fits_dir) 
                          if os.path.isdir(os.path.join(model_fits_dir, d)) 
                          and d not in ['.DS_Store', 'all']]
    else:
        participant_dirs = participants
    
    # Load parameters from each participant
    all_params = []
    loaded_participants = []
    
    for pid in participant_dirs:
        filepath = os.path.join(model_fits_dir, pid, f"{pid}_{model_suffix}_fit.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                all_params.append(results['fittedParams'])
                loaded_participants.append(pid)
            except Exception as e:
                print(f"  Warning: Could not load {filepath}: {e}")
    
    if len(all_params) == 0:
        raise ValueError(f"No fitted parameters found for model '{model_name}'")
    
    all_params = np.array(all_params)
    
    return {
        'mean': np.mean(all_params, axis=0),
        'std': np.std(all_params, axis=0),
        'n_participants': len(loaded_participants),
        'participants': loaded_participants,
        'all_params': all_params
    }


def sample_parameters_from_group(group_stats, seed=None, clip_to_bounds=True):
    """
    Sample parameters from Normal distributions based on group statistics.
    
    Args:
        group_stats: Dict with 'mean' and 'std' arrays from load_group_parameter_statistics
        seed: Optional random seed
        clip_to_bounds: Whether to clip parameters to valid ranges
    
    Returns:
        np.array: Sampled parameter vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    mean = group_stats['mean']
    std = group_stats['std']
    
    # Sample from normal distributions
    # Use at least a small std to avoid degenerate sampling
    std_safe = np.maximum(std, 0.01)
    params = np.random.normal(mean, std_safe)
    
    if clip_to_bounds:
        # Get number of params to determine model type
        # With sharedLambda=True (default):
        # - fusionOnlyLogNorm: 4 params [λ, σa1, σv, σa2]
        # - causal inference (lognorm, etc.): 5 params [λ, σa1, σv, pc, σa2]
        # - switchingFree: 6 params [λ, σa1, σv, p_switch1, σa2, p_switch2]
        n_params = len(params)
        
        if n_params == 4:
            # fusionOnlyLogNorm with sharedLambda=True: [λ, σa1, σv, σa2]
            params[0] = np.clip(params[0], 0.01, 0.4)   # λ
            params[1] = np.clip(params[1], 0.01, 2)   # σa1
            params[2] = np.clip(params[2], 0.01, 2)   # σv
            params[3] = np.clip(params[3], 0.01, 2)   # σa2
            
        elif n_params == 5:
            # lognorm/gaussian/switching/etc with sharedLambda=True: [λ, σa1, σv, pc, σa2]
            params[0] = np.clip(params[0], 0.01, 0.4)   # λ
            params[1] = np.clip(params[1], 0.01, 2)   # σa1
            params[2] = np.clip(params[2], 0.01, 2)   # σv
            params[3] = np.clip(params[3], 0.01, 0.99)  # pc (probability)
            params[4] = np.clip(params[4], 0.01, 2)   # σa2
            
        elif n_params == 6:
            # switchingFree with sharedLambda=True: [λ, σa1, σv, p_switch1, σa2, p_switch2]
            params[0] = np.clip(params[0], 0.01, 0.4)   # λ
            params[1] = np.clip(params[1], 0.01, 2)   # σa1
            params[2] = np.clip(params[2], 0.01, 2)   # σv
            params[3] = np.clip(params[3], 0.01, 0.99)  # p_switch1
            params[4] = np.clip(params[4], 0.01, 2)   # σa2
            params[5] = np.clip(params[5], 0.01, 0.99)  # p_switch2
    
    return params


def run_single_recovery_iteration(args):
    """
    Run a single model recovery iteration (worker function for parallel execution).
    
    Args:
        args: tuple of (iteration_idx, generating_model, group_stats, models_to_test, 
                       template_data, nSimul, nStarts)
    
    Returns:
        dict with iteration results or None if failed
    """
    (iteration_idx, generating_model, group_stats, models_to_test, 
     template_data, nSimul, nStarts) = args
    
    # Sample parameters from group distribution
    sampled_params = sample_parameters_from_group(group_stats, seed=None)
    
    # Set up the Monte Carlo fitter for the generating model
    mc_gen = monteCarloClass.OmerMonteCarlo(template_data)
    mc_gen.modelName = generating_model
    mc_gen.freeP_c = False
    mc_gen.sharedLambda = True  # Use shared lambda across conflict conditions
    mc_gen.nSimul = nSimul
    mc_gen.nStart = nStarts
    mc_gen.optimizationMethod = 'bads'
    
    # Simulate data from generating model with sampled parameters
    try:
        sim_data = mc_gen.simulateMonteCarloData(sampled_params, template_data)
    except Exception as e:
        return None
    
    # Fit all competing models to simulated data
    model_fits = {}
    
    for fit_model in models_to_test:
        mc_fit = monteCarloClass.OmerMonteCarlo(sim_data)
        mc_fit.modelName = fit_model
        mc_fit.freeP_c = False
        mc_fit.sharedLambda = True  # Use shared lambda across conflict conditions
        mc_fit.nSimul = nSimul
        mc_fit.nStart = nStarts
        mc_fit.optimizationMethod = 'bads'
        
        try:
            fitted_params = mc_fit.fitCausalInferenceMonteCarlo(mc_fit.groupedData)
            if fitted_params is not None:
                # Calculate log-likelihood and AIC
                nLL = mc_fit.nLLMonteCarloCausal(fitted_params, mc_fit.groupedData)
                LL = -nLL
                n_params = len(fitted_params)
                AIC = 2 * n_params - 2 * LL
                BIC = n_params * np.log(len(sim_data)) - 2 * LL
                
                model_fits[fit_model] = {
                    'fittedParams': fitted_params.tolist(),
                    'logLikelihood': float(LL),
                    'AIC': float(AIC),
                    'BIC': float(BIC),
                    'nParams': n_params
                }
        except Exception as e:
            continue
    
    if len(model_fits) > 0:
        # Find best model by AIC/BIC
        best_model_aic = min(model_fits.keys(), key=lambda m: model_fits[m]['AIC'])
        best_model_bic = min(model_fits.keys(), key=lambda m: model_fits[m]['BIC'])
        
        return {
            'iteration': iteration_idx,
            'sampled_params': sampled_params.tolist(),
            'model_fits': model_fits,
            'best_model_aic': best_model_aic,
            'best_model_bic': best_model_bic
        }
    
    return None


def run_model_recovery_for_model(generating_model, models_to_test, group_stats_dict,
                                  template_data, n_recovery=50, nSimul=500, nStarts=1,
                                  save_dir="model_recovery_results", n_jobs=1):
    """
    Run model recovery for a single generating model.
    
    Args:
        generating_model: Name of the model that generates the data
        models_to_test: List of model names to fit to the simulated data
        group_stats_dict: Dict mapping model names to their group statistics
        template_data: DataFrame with experimental design (used as template for simulation)
        n_recovery: Number of recovery iterations
        nSimul: Number of Monte Carlo simulations for fitting
        nStarts: Number of optimization starting points
        save_dir: Directory to save results
        n_jobs: Number of parallel workers
    
    Returns:
        dict with recovery results
    """
    result_path = os.path.join(save_dir, f"group_{generating_model}_model_recovery.json")
    
    # Check if already exists
    if os.path.exists(result_path):
        print(f"  Results already exist for {generating_model}, loading...")
        with open(result_path, 'r') as f:
            return json.load(f)
    
    group_stats = group_stats_dict[generating_model]
    
    print(f"\n  Running {n_recovery} iterations for {generating_model}...")
    print(f"  Group stats based on {group_stats['n_participants']} participants")
    print(f"  Parameter means: {np.round(group_stats['mean'], 3)}")
    print(f"  Parameter SDs:   {np.round(group_stats['std'], 3)}")
    
    # Prepare arguments for parallel execution
    iteration_args = [
        (i, generating_model, group_stats, models_to_test, 
         template_data, nSimul, nStarts)
        for i in range(n_recovery)
    ]
    
    # Run iterations
    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(run_single_recovery_iteration, iteration_args),
                total=n_recovery,
                desc=f"  {generating_model}",
                leave=False
            ))
    else:
        results = []
        for args in tqdm(iteration_args, desc=f"  {generating_model}", leave=False):
            results.append(run_single_recovery_iteration(args))
    
    # Filter out None results
    recovery_iterations = [r for r in results if r is not None]
    
    if len(recovery_iterations) > 0:
        result = {
            'generating_model': generating_model,
            'group_stats': {
                'mean': group_stats['mean'].tolist(),
                'std': group_stats['std'].tolist(),
                'n_participants': group_stats['n_participants'],
                'participants': group_stats['participants']
            },
            'n_iterations': len(recovery_iterations),
            'iterations': recovery_iterations,
            'best_model_counts_aic': {},
            'best_model_counts_bic': {}
        }
        
        # Count best model selections
        for m in models_to_test:
            result['best_model_counts_aic'][m] = sum(
                1 for it in recovery_iterations if it['best_model_aic'] == m
            )
            result['best_model_counts_bic'][m] = sum(
                1 for it in recovery_iterations if it['best_model_bic'] == m
            )
        
        # Save result
        os.makedirs(save_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved results to {result_path}")
        
        return result
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Run model recovery analysis with group-level parameter sampling')
    parser.add_argument('--models', nargs='+', 
                        default=['lognorm', 'fusionOnlyLogNorm', 'switchingFree', 'probabilityMatchingLogNorm', 'selection'],
                        help='Models to test (both as generating and fitting models)')
    parser.add_argument('--n_recovery', type=int, default=50,
                        help='Number of recovery iterations per generating model')
    parser.add_argument('--nSimul', type=int, default=500,
                        help='Monte Carlo simulations for fitting')
    parser.add_argument('--nStarts', type=int, default=1,
                        help='Optimization starting points')
    parser.add_argument('--save_dir', type=str, default='model_recovery_results_group',
                        help='Directory to save results')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs (default: number of CPUs - 1)')
    parser.add_argument('--template_participant', type=str, default='as',
                        help='Participant ID to use as template for experimental design')
    
    args = parser.parse_args()
    
    # Determine number of parallel workers
    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)
    
    print("="*70)
    print("MODEL RECOVERY ANALYSIS (Group-Level Parameter Sampling)")
    print("="*70)
    print(f"Models to test: {args.models}")
    print(f"Recovery iterations per model: {args.n_recovery}")
    print(f"Monte Carlo simulations: {args.nSimul}")
    print(f"Optimization starts: {args.nStarts}")
    print(f"Parallel jobs: {n_jobs} (CPUs available: {cpu_count()})")
    print(f"Template participant: {args.template_participant}")
    print("="*70)
    
    start_time = time.time()
    
    # Load template data (experimental design)
    print(f"\nLoading template data from {args.template_participant}...")
    try:
        template_data, _ = loadData.loadData(f"{args.template_participant}_all.csv", verbose=False)
        print(f"  Template data: {len(template_data)} trials")
    except Exception as e:
        print(f"Error loading template data: {e}")
        return
    
    # Load group statistics for each model
    print("\nLoading group parameter statistics...")
    group_stats_dict = {}
    for model in args.models:
        try:
            stats = load_group_parameter_statistics(model)
            group_stats_dict[model] = stats
            print(f"  {model}: {stats['n_participants']} participants, {len(stats['mean'])} params")
        except Exception as e:
            print(f"  Warning: Could not load stats for {model}: {e}")
    
    if len(group_stats_dict) == 0:
        print("Error: No group statistics loaded. Check that model_fits directory exists.")
        return
    
    # Run model recovery for each generating model
    print("\n" + "-"*70)
    print("Running Model Recovery")
    print("-"*70)
    
    all_results = []
    for gen_model in args.models:
        if gen_model not in group_stats_dict:
            print(f"\nSkipping {gen_model} (no group statistics available)")
            continue
            
        result = run_model_recovery_for_model(
            generating_model=gen_model,
            models_to_test=args.models,
            group_stats_dict=group_stats_dict,
            template_data=template_data,
            n_recovery=args.n_recovery,
            nSimul=args.nSimul,
            nStarts=args.nStarts,
            save_dir=args.save_dir,
            n_jobs=n_jobs
        )
        if result is not None:
            all_results.append(result)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("MODEL RECOVERY COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {args.save_dir}/")
    print(f"Successful recoveries: {len(all_results)} models")
    
    # Print confusion matrix summary
    if all_results:
        print("\n" + "-"*70)
        print("Recovery Confusion Matrix (AIC)")
        print("-"*70)
        
        # Header
        header = "Generated \\ Recovered"
        model_abbrevs = {
            'lognorm': 'CI',
            'fusionOnlyLogNorm': 'Fus',
            'switchingFree': 'SwF',
            'probabilityMatchingLogNorm': 'PM',
            'selection': 'Sel'
        }
        
        print(f"{'Generating':<25}", end="")
        for m in args.models:
            abbrev = model_abbrevs.get(m, m[:4])
            print(f"{abbrev:>8}", end="")
        print(f"{'Correct':>10}")
        print("-"*70)
        
        for result in all_results:
            gen_model = result['generating_model']
            total = result['n_iterations']
            print(f"{gen_model:<25}", end="")
            
            correct_count = 0
            for fit_model in args.models:
                count = result['best_model_counts_aic'].get(fit_model, 0)
                pct = count / total * 100 if total > 0 else 0
                if fit_model == gen_model:
                    correct_count = count
                    print(f"{pct:>7.0f}%", end="")
                else:
                    print(f"{pct:>7.0f}%", end="")
            
            correct_pct = correct_count / total * 100 if total > 0 else 0
            print(f"{correct_pct:>9.1f}%")
        
        # Overall accuracy
        print("-"*70)
        total_correct = sum(r['best_model_counts_aic'].get(r['generating_model'], 0) for r in all_results)
        total_iterations = sum(r['n_iterations'] for r in all_results)
        overall_pct = total_correct / total_iterations * 100 if total_iterations > 0 else 0
        print(f"{'Overall Accuracy:':<25}{' '*len(args.models)*8}{overall_pct:>9.1f}%")


if __name__ == "__main__":
    main()
