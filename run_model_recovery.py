#!/usr/bin/env python3
"""
Model Recovery Analysis Script
==============================
Run this script overnight to perform comprehensive model recovery analysis.

Usage:
    python run_model_recovery.py [--participants PARTICIPANT_IDS] [--n_recovery N]
    
Example:
    python run_model_recovery.py --participants as dt hh ip --n_recovery 10
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from glob import glob
from tqdm import tqdm
import time

# Import project modules
import loadData
import loadResults
import monteCarloClass


def run_model_recovery_single(participantID, generating_model, models_to_test, 
                               n_recovery=10, nSimul=500, nStarts=1, 
                               save_dir="model_recovery_results"):
    """
    Run model recovery for a single participant-generating_model combination.
    """
    result_path = os.path.join(save_dir, f"{participantID}_{generating_model}_model_recovery.json")
    
    # Check if already exists
    if os.path.exists(result_path):
        print(f"  Results already exist for {participantID} - {generating_model}, skipping...")
        return None
    
    # Load original data structure
    try:
        data, dataName = loadData.loadData(participantID + "_all.csv", verbose=False)
    except Exception as e:
        print(f"  Could not load data for {participantID}: {e}")
        return None
    
    # Load fitted parameters for the generating model
    mc_gen = monteCarloClass.OmerMonteCarlo(data)
    mc_gen.modelName = generating_model
    mc_gen.freeP_c = False
    mc_gen.sharedLambda = False
    mc_gen.dataName = dataName
    mc_gen.nSimul = nSimul
    mc_gen.nStart = nStarts
    
    try:
        res_gen = loadResults.loadFitResults(mc_gen, dataName, modelName=generating_model)
        true_params = np.array(res_gen['fittedParams'])
    except Exception as e:
        print(f"  Could not load {generating_model} fit for {participantID}: {e}")
        return None
    
    print(f"  Running {n_recovery} recovery iterations...")
    recovery_iterations = []
    
    for iter_idx in tqdm(range(n_recovery), desc=f"  {participantID}-{generating_model}", leave=False):
        # Simulate data from generating model
        sim_data = mc_gen.simulateMonteCarloData(true_params, data)
        
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
                'model_fits': model_fits,
                'best_model_aic': best_model_aic,
                'best_model_bic': best_model_bic
            })
    
    if len(recovery_iterations) > 0:
        result = {
            'participantID': participantID,
            'generating_model': generating_model,
            'true_params': true_params.tolist(),
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
    parser.add_argument('--participants', nargs='+', default=['as', 'dt', 'hh', 'ip', 'ln1'],
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
    
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL RECOVERY ANALYSIS")
    print("="*70)
    print(f"Participants: {args.participants}")
    print(f"Models to test: {args.models}")
    print(f"Recovery iterations: {args.n_recovery}")
    print(f"Monte Carlo simulations: {args.nSimul}")
    print(f"Optimization starts: {args.nStarts}")
    print("="*70)
    
    start_time = time.time()
    results = []
    
    total_combinations = len(args.participants) * len(args.models)
    current = 0
    
    for generating_model in args.models:
        print(f"\n{'='*50}")
        print(f"GENERATING MODEL: {generating_model}")
        print(f"{'='*50}")
        
        for participantID in args.participants:
            current += 1
            print(f"\n[{current}/{total_combinations}] Participant: {participantID}")
            
            result = run_model_recovery_single(
                participantID=participantID,
                generating_model=generating_model,
                models_to_test=args.models,
                n_recovery=args.n_recovery,
                nSimul=args.nSimul,
                nStarts=args.nStarts,
                save_dir=args.save_dir
            )
            
            if result is not None:
                results.append(result)
    
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
