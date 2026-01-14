#!/usr/bin/env python3
"""
Model Recovery Analysis Script (Random Parameter Sampling)
==========================================================
Run this script overnight to perform comprehensive model recovery analysis.

This version samples RANDOM parameters from prior distributions for each iteration,
rather than using fitted parameters. This provides a more rigorous test of model
discriminability across the full parameter space.

Usage:
    python run_model_recovery.py [--participants PARTICIPANT_IDS] [--n_recovery N]
    
Example:
    python run_model_recovery.py --participants as dt hh ip --n_recovery 50
    python run_model_recovery.py --participants as --n_recovery 100 --n_jobs 8
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
import loadResults
import monteCarloClass


def sample_random_parameters(model_name, seed=None):
    """
    Sample random parameters from prior distributions for a given model.
    
    Each model has different parameters with biologically/psychophysically plausible ranges.
    
    Args:
        model_name: Name of the model (e.g., 'lognorm', 'fusionOnlyLogNorm', 'switching')
        seed: Optional random seed for reproducibility
    
    Returns:
        np.array: Sampled parameter vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define parameter priors for each model
    # Format: (min, max) for uniform sampling, or (mean, std) for normal
    
    if model_name == 'lognorm':
        # Parameters: [lambda1, sigma_a, sigma_v, p_common, sigma_a2, lambda2, lambda3]
        # Full causal inference model with log-normal prior
        params = np.array([
            np.random.uniform(0.3, 1.5),      # lambda1: audio weight (0.3-1.5)
            np.random.uniform(0.05, 0.4),     # sigma_a: auditory noise (0.05-0.4)
            np.random.uniform(0.05, 0.4),     # sigma_v: visual noise (0.05-0.4)
            np.random.uniform(0.2, 0.9),      # p_common: prior probability of common cause (0.2-0.9)
            np.random.uniform(0.1, 0.5),      # sigma_a2: prior width (0.1-0.5)
            np.random.uniform(0.3, 1.5),      # lambda2: visual weight (0.3-1.5)
            np.random.uniform(0.3, 1.5),      # lambda3: bimodal weight (0.3-1.5)
        ])
        
    elif model_name == 'fusionOnlyLogNorm':
        # Fusion-only model: always assumes common cause (p_common=1 effectively)
        # Parameters: [lambda1, sigma_a, sigma_v, sigma_prior, lambda2, lambda3]
        params = np.array([
            np.random.uniform(0.3, 1.5),      # lambda1
            np.random.uniform(0.05, 0.4),     # sigma_a
            np.random.uniform(0.05, 0.4),     # sigma_v
            np.random.uniform(0.1, 0.5),      # sigma_prior
            np.random.uniform(0.3, 1.5),      # lambda2
            np.random.uniform(0.3, 1.5),      # lambda3
        ])
        
    elif model_name == 'switching' or model_name == 'switchingFree':
        # Switching model: probabilistically switches between modalities
        # Parameters: [lambda1, sigma_a, sigma_v, p_audio, lambda2, lambda3]
        params = np.array([
            np.random.uniform(0.3, 1.5),      # lambda1
            np.random.uniform(0.05, 0.4),     # sigma_a
            np.random.uniform(0.05, 0.4),     # sigma_v
            np.random.uniform(0.1, 0.9),      # p_audio: probability of using audio
            np.random.uniform(0.3, 1.5),      # lambda2
            np.random.uniform(0.3, 1.5),      # lambda3
        ])
        
    elif model_name == 'probabilityMatchingLogNorm':
        # Probability matching: responds proportionally to posterior
        # Parameters: [lambda1, sigma_a, sigma_v, p_common, sigma_prior, lambda2, lambda3]
        params = np.array([
            np.random.uniform(0.3, 1.5),      # lambda1
            np.random.uniform(0.05, 0.4),     # sigma_a
            np.random.uniform(0.05, 0.4),     # sigma_v
            np.random.uniform(0.2, 0.9),      # p_common
            np.random.uniform(0.1, 0.5),      # sigma_prior
            np.random.uniform(0.3, 1.5),      # lambda2
            np.random.uniform(0.3, 1.5),      # lambda3
        ])
        
    elif model_name == 'selection':
        # Selection model: always picks one modality
        # Parameters: [lambda1, sigma_a, sigma_v, lambda2, lambda3]
        params = np.array([
            np.random.uniform(0.3, 1.5),      # lambda1
            np.random.uniform(0.05, 0.4),     # sigma_a
            np.random.uniform(0.05, 0.4),     # sigma_v
            np.random.uniform(0.3, 1.5),      # lambda2
            np.random.uniform(0.3, 1.5),      # lambda3
        ])
        
    else:
        # Generic fallback: sample 7 parameters with reasonable defaults
        print(f"  Warning: Unknown model '{model_name}', using generic parameter sampling")
        params = np.array([
            np.random.uniform(0.3, 1.5),      # lambda1
            np.random.uniform(0.05, 0.4),     # sigma_a
            np.random.uniform(0.05, 0.4),     # sigma_v
            np.random.uniform(0.2, 0.9),      # p_common or similar
            np.random.uniform(0.1, 0.5),      # sigma_prior or similar
            np.random.uniform(0.3, 1.5),      # lambda2
            np.random.uniform(0.3, 1.5),      # lambda3
        ])
    
    return params


def run_model_recovery_single(participantID, generating_model, models_to_test, 
                               n_recovery=10, nSimul=500, nStarts=1, 
                               save_dir="model_recovery_results"):
    """
    Run model recovery for a single participant-generating_model combination.
    
    For each iteration:
      1. Sample RANDOM parameters from prior distributions
      2. Simulate data using those parameters
      3. Fit all competing models
      4. Record which model wins by AIC/BIC
    """
    result_path = os.path.join(save_dir, f"{participantID}_{generating_model}_model_recovery.json")
    
    # Check if already exists
    if os.path.exists(result_path):
        print(f"  Results already exist for {participantID} - {generating_model}, skipping...")
        return None
    
    # Load original data structure (we need the experimental design/conditions)
    try:
        data, dataName = loadData.loadData(participantID + "_all.csv", verbose=False)
    except Exception as e:
        print(f"  Could not load data for {participantID}: {e}")
        return None
    
    # Set up the Monte Carlo fitter for the generating model
    mc_gen = monteCarloClass.OmerMonteCarlo(data)
    mc_gen.modelName = generating_model
    mc_gen.freeP_c = False
    mc_gen.sharedLambda = False
    mc_gen.dataName = dataName
    mc_gen.nSimul = nSimul
    mc_gen.nStart = nStarts
    
    print(f"  Running {n_recovery} recovery iterations with RANDOM parameters...")
    recovery_iterations = []
    all_sampled_params = []
    
    for iter_idx in tqdm(range(n_recovery), desc=f"  {participantID}-{generating_model}", leave=False):
        # Sample RANDOM parameters for this iteration
        sampled_params = sample_random_parameters(generating_model, seed=None)
        all_sampled_params.append(sampled_params.tolist())
        
        # Simulate data from generating model with sampled parameters
        try:
            sim_data = mc_gen.simulateMonteCarloData(sampled_params, data)
        except Exception as e:
            print(f"    Simulation failed for iteration {iter_idx}: {e}")
            continue
        
        # Fit all competing models to simulated data
        model_fits = {}
        
        for fit_model in models_to_test:
            mc_fit = monteCarloClass.OmerMonteCarlo(sim_data)
            mc_fit.modelName = fit_model
            mc_fit.freeP_c = False
            mc_fit.sharedLambda = False
            mc_fit.nSimul = nSimul
            mc_fit.nStart = nStarts
            mc_fit.optimizationMethod = 'scipy'
            mc_fit.dataName = f"{participantID}_recovery"
            
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
                print(f"    Fit failed for {fit_model}: {e}")
                continue
        
        if len(model_fits) > 0:
            # Find best model by AIC
            best_model_aic = min(model_fits.keys(), key=lambda m: model_fits[m]['AIC'])
            best_model_bic = min(model_fits.keys(), key=lambda m: model_fits[m]['BIC'])
            
            recovery_iterations.append({
                'iteration': iter_idx,
                'sampled_params': sampled_params.tolist(),  # Store the params used for this iteration
                'model_fits': model_fits,
                'best_model_aic': best_model_aic,
                'best_model_bic': best_model_bic
            })
    
    if len(recovery_iterations) > 0:
        result = {
            'participantID': participantID,
            'generating_model': generating_model,
            'sampled_params': all_sampled_params,  # Store all sampled parameter sets
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
    parser = argparse.ArgumentParser(description='Run model recovery analysis')
    parser.add_argument('--participants', nargs='+', default=['as', 'dt', 'hh', 'ip', 'oy','ln2','mh','ml','mt','qs', 'sx' ],
                        help='Participant IDs to process')
    parser.add_argument('--models', nargs='+', 
                        default=['lognorm', 'fusionOnlyLogNorm', 'switching', 'probabilityMatchingLogNorm'],
                        help='Models to test')
    parser.add_argument('--n_recovery', type=int, default=10,
                        help='Number of recovery iterations per participant-model')
    parser.add_argument('--nSimul', type=int, default=500,
                        help='Monte Carlo simulations for fitting')
    parser.add_argument('--nStarts', type=int, default=1,
                        help='Optimization starting points')
    parser.add_argument('--save_dir', type=str, default='model_recovery_results',
                        help='Directory to save results')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs (default: number of CPUs - 1)')
    
    args = parser.parse_args()
    
    # Determine number of parallel workers
    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)
    
    print("="*70)
    print("MODEL RECOVERY ANALYSIS")
    print("="*70)
    print(f"Participants: {args.participants}")
    print(f"Models to test: {args.models}")
    print(f"Recovery iterations: {args.n_recovery}")
    print(f"Monte Carlo simulations: {args.nSimul}")
    print(f"Optimization starts: {args.nStarts}")
    print(f"Parallel jobs: {n_jobs} (CPUs available: {cpu_count()})")
    print("="*70)
    
    start_time = time.time()
    
    # Create all participant-model combinations
    combinations = [
        (pid, gen_model) 
        for gen_model in args.models 
        for pid in args.participants
    ]
    
    print(f"\nTotal combinations to process: {len(combinations)}")
    
    # Create partial function with fixed arguments
    worker_func = partial(
        run_model_recovery_single,
        models_to_test=args.models,
        n_recovery=args.n_recovery,
        nSimul=args.nSimul,
        nStarts=args.nStarts,
        save_dir=args.save_dir
    )
    
    # Run in parallel
    results = []
    if n_jobs > 1:
        print(f"\nRunning in parallel with {n_jobs} workers...")
        with Pool(processes=n_jobs) as pool:
            # Use starmap for multiple arguments
            results = list(tqdm(
                pool.starmap(worker_func, combinations),
                total=len(combinations),
                desc="Model Recovery"
            ))
    else:
        print("\nRunning sequentially...")
        for pid, gen_model in tqdm(combinations, desc="Model Recovery"):
            result = worker_func(pid, gen_model)
            results.append(result)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("MODEL RECOVERY COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {args.save_dir}/")
    print(f"Successful recoveries: {len(results)}")
    
    # Print summary
    if results:
        print("\nRecovery Summary (by AIC):")
        print("-"*50)
        
        for gen_model in args.models:
            gen_results = [r for r in results if r['generating_model'] == gen_model]
            if gen_results:
                correct = sum(r['best_model_counts_aic'].get(gen_model, 0) for r in gen_results)
                total = sum(r['n_iterations'] for r in gen_results)
                pct = correct / total * 100 if total > 0 else 0
                print(f"  {gen_model}: {correct}/{total} ({pct:.1f}%) correctly recovered")


if __name__ == "__main__":
    main()
