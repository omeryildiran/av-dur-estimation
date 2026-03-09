#!/usr/bin/env python3
"""
Model Recovery Analysis Script (Free / Wide-Range Parameter Sampling)
=====================================================================
Run this script to perform model recovery using parameters drawn from
wide, plausible uniform distributions — NOT from the distribution of
fitted participant parameters.

Motivation:
    If p_c recovery fails when parameters are centred on data-fitted values
    but succeeds here, we can conclude that the experimental range (not the
    model) is the limiting factor.

Key differences from run_model_recovery.py:
    - Parameters are sampled from wide uniform priors (no participant data needed).
    - Boundary-clipping diagnostics are reported (how many fitted values sit at the
      boundary, which would indicate the optimizer is stuck).
    - Dedicated parameter-recovery scatter analysis for p_c (and all params).

Usage:
    python run_model_recovery_freeParams.py [--n_recovery N] [--n_jobs N]

Example:
    python run_model_recovery_freeParams.py --n_recovery 100 --n_jobs 8
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

# Import project modules
import loadData
import monteCarloClass


# ---------------------------------------------------------------------------
# Simplified (unique) parameter ranges for model recovery.
# Since this is model recovery (not fitting real data with two SNR conditions),
# we use a single σa (duplicated into σa1=σa2) and a single p_switch
# (duplicated into p_switch1=p_switch2).  This reduces the effective
# dimensionality and focuses the test on identifiability of the key params.
# ---------------------------------------------------------------------------

# Unique parameters that we actually sample  (label → uniform range)
PARAM_RANGES_UNIQUE = {
    # fusionOnlyLogNorm: [λ, σa, σv]   →  expanded to [λ, σa, σv, σa]
    'fusionOnlyLogNorm': [
        (0.02, 0.35),   # λ  – lapse rate
        (0.01, 1.5),    # σa – auditory noise (single)
        (0.01, 1.5),    # σv – visual noise
    ],
    # CI models: [λ, σa, σv, p_c]  →  expanded to [λ, σa, σv, p_c, σa]
    'lognorm': [
        (0.02, 0.35),   # λ
        (0.01, 1.5),    # σa
        (0.01, 1.5),    # σv
        (0.1, 0.9),     # p_c  – wide range to test recovery
    ],
    'probabilityMatchingLogNorm': [
        (0.02, 0.35),
        (0.01, 1.5),
        (0.01, 1.5),
        (0.1, 0.9),
    ],
    'selection': [
        (0.02, 0.35),
        (0.01, 1.5),
        (0.01, 1.5),
        (0.1, 0.9),
    ],
    # switchingFree: [λ, σa, σv, p_switch]  →  expanded to [λ, σa, σv, p_sw, σa, p_sw]
    'switchingFree': [
        (0.02, 0.35),   # λ
        (0.05, 1.5),    # σa
        (0.05, 1.5),    # σv
        (0.1, 0.9),     # p_switch (single)
    ],
}

# Human-readable names for the *unique* parameters we sample
PARAM_NAMES_UNIQUE = {
    'fusionOnlyLogNorm': ['λ', 'σa', 'σv'],
    'lognorm':           ['λ', 'σa', 'σv', 'p_c'],
    'probabilityMatchingLogNorm': ['λ', 'σa', 'σv', 'p_c'],
    'selection':         ['λ', 'σa', 'σv', 'p_c'],
    'switchingFree':     ['λ', 'σa', 'σv', 'p_sw'],
}

# Names for the *full* (expanded) parameter vector that monteCarloClass expects
PARAM_NAMES = {
    'fusionOnlyLogNorm': ['λ', 'σa1', 'σv', 'σa2'],
    'lognorm':           ['λ', 'σa1', 'σv', 'p_c', 'σa2'],
    'probabilityMatchingLogNorm': ['λ', 'σa1', 'σv', 'p_c', 'σa2'],
    'selection':         ['λ', 'σa1', 'σv', 'p_c', 'σa2'],
    'switchingFree':     ['λ', 'σa1', 'σv', 'p_sw1', 'σa2', 'p_sw2'],
}

# Fitting bounds used by the optimizer (must match monteCarloClass)
# These are the hard boundaries used for clipping diagnostics.
FITTING_BOUNDS = {
    'fusionOnlyLogNorm': [
        (0.001, 0.4),   # λ
        (0.01, 2.0),    # σa1
        (0.01, 2.0),    # σv
        (0.01, 2.0),    # σa2
    ],
    'lognorm': [
        (0.001, 0.4),   # λ
        (0.0001, 2.0),  # σa1
        (0.0001, 2.0),  # σv
        (0.0, 1.0),     # p_c
        (0.0001, 2.0),  # σa2
    ],
    'probabilityMatchingLogNorm': [
        (0.001, 0.4),
        (0.0001, 2.0),
        (0.0001, 2.0),
        (0.0, 1.0),
        (0.0001, 2.0),
    ],
    'selection': [
        (0.001, 0.4),
        (0.0001, 2.0),
        (0.0001, 2.0),
        (0.0, 1.0),
        (0.0001, 2.0),
    ],
    'switchingFree': [
        (0.001, 0.4),
        (0.0001, 2.0),
        (0.0001, 2.0),
        (0.0, 1.0),
        (0.0001, 2.0),
        (0.0, 1.0),
    ],
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_free_parameters(model_name, rng=None):
    """
    Draw one parameter vector from wide uniform distributions.

    We sample a *reduced* set of unique parameters and then *expand* the
    vector to the full layout that monteCarloClass expects by duplicating
    σa → (σa1, σa2) and p_switch → (p_switch1, p_switch2).

    Returns:
        unique_params : np.ndarray – the unique values actually sampled
        full_params   : np.ndarray – expanded vector for monteCarloClass
    """
    if rng is None:
        rng = np.random.default_rng()

    ranges = PARAM_RANGES_UNIQUE[model_name]
    unique = np.array([rng.uniform(lo, hi) for lo, hi in ranges])

    # Expand to the full parameter vector expected by monteCarloClass
    if model_name == 'fusionOnlyLogNorm':
        # unique: [λ, σa, σv]  →  full: [λ, σa, σv, σa]
        lam, sa, sv = unique
        full = np.array([lam, sa, sv, sa])
    elif model_name == 'switchingFree':
        # unique: [λ, σa, σv, p_sw]  →  full: [λ, σa, σv, p_sw, σa, p_sw]
        lam, sa, sv, psw = unique
        full = np.array([lam, sa, sv, psw, sa, psw])
    else:
        # CI models: unique: [λ, σa, σv, p_c]  →  full: [λ, σa, σv, p_c, σa]
        lam, sa, sv, pc = unique
        full = np.array([lam, sa, sv, pc, sa])

    return unique, full


# ---------------------------------------------------------------------------
# Boundary-clipping diagnostics
# ---------------------------------------------------------------------------

def count_boundary_clips(fitted_params, model_name, tol=0.01):
    """
    Count how many fitted param values sit within *tol* of the optimiser
    hard boundary, which suggests the optimiser got stuck.

    Returns:
        dict with keys 'n_lower', 'n_upper', 'details' (list of param-name
        strings that were clipped)
    """
    bounds = FITTING_BOUNDS[model_name]
    names = PARAM_NAMES[model_name]
    n_lower = 0
    n_upper = 0
    details = []
    for val, (lo, hi), name in zip(fitted_params, bounds, names):
        if abs(val - lo) <= tol * (hi - lo):
            n_lower += 1
            details.append(f"{name}@lower")
        if abs(val - hi) <= tol * (hi - lo):
            n_upper += 1
            details.append(f"{name}@upper")
    return {'n_lower': n_lower, 'n_upper': n_upper, 'details': details}


# ---------------------------------------------------------------------------
# Single-iteration worker
# ---------------------------------------------------------------------------

def run_single_recovery_iteration(args):
    """
    Worker function executed in parallel (or sequentially).

    Returns dict with iteration results, or None on failure.
    """
    (iteration_idx, generating_model, models_to_test,
     template_data, nSimul, nStarts) = args

    rng = np.random.default_rng()

    # 1. Sample ground-truth parameters from wide uniform priors
    sampled_unique, sampled_params = sample_free_parameters(generating_model, rng=rng)

    # 2. Simulate data from the generating model
    mc_gen = monteCarloClass.OmerMonteCarlo(template_data)
    mc_gen.modelName = generating_model
    mc_gen.freeP_c = False
    mc_gen.sharedLambda = True
    mc_gen.nSimul = nSimul
    mc_gen.nStart = nStarts
    mc_gen.optimizationMethod = 'bads'

    try:
        sim_data = mc_gen.simulateMonteCarloData(sampled_params, template_data)
    except Exception:
        return None

    # 3. Fit every competing model to the simulated data
    model_fits = {}
    for fit_model in models_to_test:
        mc_fit = monteCarloClass.OmerMonteCarlo(sim_data)
        mc_fit.modelName = fit_model
        mc_fit.freeP_c = False
        mc_fit.sharedLambda = True
        mc_fit.nSimul = nSimul
        mc_fit.nStart = nStarts
        mc_fit.optimizationMethod = 'bads'

        try:
            fitted_params = mc_fit.fitCausalInferenceMonteCarlo(mc_fit.groupedData)
            if fitted_params is not None:
                nLL = mc_fit.nLLMonteCarloCausal(fitted_params, mc_fit.groupedData)
                LL = -nLL
                n_params = len(fitted_params)
                AIC = 2 * n_params - 2 * LL
                BIC = n_params * np.log(len(sim_data)) - 2 * LL

                clip_info = count_boundary_clips(fitted_params, fit_model)

                model_fits[fit_model] = {
                    'fittedParams': fitted_params.tolist(),
                    'logLikelihood': float(LL),
                    'AIC': float(AIC),
                    'BIC': float(BIC),
                    'nParams': n_params,
                    'boundary_clips': clip_info,
                }
        except Exception:
            continue

    if len(model_fits) == 0:
        return None

    best_model_aic = min(model_fits, key=lambda m: model_fits[m]['AIC'])
    best_model_bic = min(model_fits, key=lambda m: model_fits[m]['BIC'])

    return {
        'iteration': iteration_idx,
        'sampled_params': sampled_params.tolist(),
        'sampled_unique': sampled_unique.tolist(),
        'model_fits': model_fits,
        'best_model_aic': best_model_aic,
        'best_model_bic': best_model_bic,
    }


# ---------------------------------------------------------------------------
# Per-model recovery driver
# ---------------------------------------------------------------------------

def run_model_recovery_for_model(generating_model, models_to_test,
                                  template_data, n_recovery=50,
                                  nSimul=500, nStarts=1,
                                  save_dir="model_recovery_results_freeParams",
                                  n_jobs=1):
    result_path = os.path.join(save_dir, f"freeParam_{generating_model}_model_recovery.json")

    if os.path.exists(result_path):
        print(f"  Results already exist for {generating_model}, loading …")
        with open(result_path, 'r') as f:
            return json.load(f)

    ranges = PARAM_RANGES_UNIQUE[generating_model]
    names_unique = PARAM_NAMES_UNIQUE[generating_model]
    names_full = PARAM_NAMES[generating_model]
    print(f"\n  Running {n_recovery} iterations for {generating_model} …")
    print(f"  Unique free parameters sampled (duplicated σa / p_switch for full vector):")
    for nm, (lo, hi) in zip(names_unique, ranges):
        print(f"    {nm}: U({lo}, {hi})")

    iteration_args = [
        (i, generating_model, models_to_test, template_data, nSimul, nStarts)
        for i in range(n_recovery)
    ]

    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(run_single_recovery_iteration, iteration_args),
                total=n_recovery,
                desc=f"  {generating_model}",
                leave=False,
            ))
    else:
        results = []
        for a in tqdm(iteration_args, desc=f"  {generating_model}", leave=False):
            results.append(run_single_recovery_iteration(a))

    recovery_iterations = [r for r in results if r is not None]

    if len(recovery_iterations) == 0:
        return None

    # ---- Aggregate results ----
    result = {
        'generating_model': generating_model,
        'param_ranges_unique': {nm: list(r) for nm, r in zip(names_unique, ranges)},
        'n_iterations': len(recovery_iterations),
        'iterations': recovery_iterations,
        'best_model_counts_aic': {},
        'best_model_counts_bic': {},
    }

    for m in models_to_test:
        result['best_model_counts_aic'][m] = sum(
            1 for it in recovery_iterations if it['best_model_aic'] == m
        )
        result['best_model_counts_bic'][m] = sum(
            1 for it in recovery_iterations if it['best_model_bic'] == m
        )

    # ---- Parameter recovery diagnostics (same-model fits only) ----
    # We compare the *unique* generating values against the corresponding
    # slots in the recovered (full) parameter vector.
    # Mapping: unique-param-index → full-param-index
    if generating_model == 'fusionOnlyLogNorm':
        unique_to_full = [0, 1, 2]          # λ, σa→σa1, σv
    elif generating_model == 'switchingFree':
        unique_to_full = [0, 1, 2, 3]       # λ, σa→σa1, σv, p_sw→p_sw1
    else:
        unique_to_full = [0, 1, 2, 3]       # λ, σa→σa1, σv, p_c

    same_model_fits = [
        it for it in recovery_iterations
        if generating_model in it['model_fits']
    ]
    if same_model_fits:
        gen_arr = np.array([it['sampled_unique'] for it in same_model_fits])
        rec_full = np.array([
            it['model_fits'][generating_model]['fittedParams']
            for it in same_model_fits
        ])

        param_recovery = {}
        for uid, nm in enumerate(names_unique):
            fid = unique_to_full[uid]
            gen_vals = gen_arr[:, uid]
            rec_vals = rec_full[:, fid]
            corr = float(np.corrcoef(gen_vals, rec_vals)[0, 1])
            rmse = float(np.sqrt(np.mean((gen_vals - rec_vals) ** 2)))
            bias = float(np.mean(rec_vals - gen_vals))
            param_recovery[nm] = {
                'correlation': corr,
                'rmse': rmse,
                'bias': bias,
            }
        result['param_recovery'] = param_recovery

        # ---- Boundary-clipping summary (same-model fit) ----
        total_clips_lower = 0
        total_clips_upper = 0
        clipped_params_counter = {}
        for it in same_model_fits:
            clip = it['model_fits'][generating_model].get('boundary_clips', {})
            total_clips_lower += clip.get('n_lower', 0)
            total_clips_upper += clip.get('n_upper', 0)
            for d in clip.get('details', []):
                clipped_params_counter[d] = clipped_params_counter.get(d, 0) + 1

        result['boundary_clip_summary'] = {
            'total_lower': total_clips_lower,
            'total_upper': total_clips_upper,
            'total_iterations': len(same_model_fits),
            'clipped_params': clipped_params_counter,
        }

    # ---- Save ----
    os.makedirs(save_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved results to {result_path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Model recovery with wide free-parameter sampling')
    parser.add_argument('--models', nargs='+',
                        default=['lognorm', 'fusionOnlyLogNorm', 'switchingFree',
                                 'probabilityMatchingLogNorm', 'selection'],
                        help='Models to test')
    parser.add_argument('--n_recovery', type=int, default=50,
                        help='Recovery iterations per generating model')
    parser.add_argument('--nSimul', type=int, default=500,
                        help='Monte Carlo simulations for fitting')
    parser.add_argument('--nStarts', type=int, default=1,
                        help='Optimisation starting points')
    parser.add_argument('--save_dir', type=str,
                        default='model_recovery_results_freeParams',
                        help='Directory to save results')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Parallel jobs (default: CPUs - 1)')
    parser.add_argument('--template_participant', type=str, default='as',
                        help='Participant ID for experimental design template')
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)

    print("=" * 70)
    print("MODEL RECOVERY – FREE / WIDE-RANGE PARAMETER SAMPLING")
    print("=" * 70)
    print(f"Models to test:     {args.models}")
    print(f"Iterations / model: {args.n_recovery}")
    print(f"MC simulations:     {args.nSimul}")
    print(f"Optim starts:       {args.nStarts}")
    print(f"Parallel jobs:      {n_jobs} (CPUs: {cpu_count()})")
    print(f"Template participant: {args.template_participant}")
    print("=" * 70)

    start_time = time.time()

    # ---- Load template data ----
    print(f"\nLoading template data from {args.template_participant} …")
    try:
        template_data, _ = loadData.loadData(
            f"{args.template_participant}_all.csv", verbose=False)
        print(f"  Template data: {len(template_data)} trials")
    except Exception as e:
        print(f"Error loading template data: {e}")
        return

    # ---- Run recovery per generating model ----
    print("\n" + "-" * 70)
    print("Running Model Recovery (free params)")
    print("-" * 70)

    all_results = []
    for gen_model in args.models:
        if gen_model not in PARAM_RANGES_UNIQUE:
            print(f"\nSkipping {gen_model} (no PARAM_RANGES_UNIQUE defined)")
            continue
        result = run_model_recovery_for_model(
            generating_model=gen_model,
            models_to_test=args.models,
            template_data=template_data,
            n_recovery=args.n_recovery,
            nSimul=args.nSimul,
            nStarts=args.nStarts,
            save_dir=args.save_dir,
            n_jobs=n_jobs,
        )
        if result is not None:
            all_results.append(result)

    elapsed = time.time() - start_time

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("MODEL RECOVERY COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {args.save_dir}/")
    print(f"Successful recoveries: {len(all_results)} models")

    # ---- Confusion matrix (AIC) ----
    if all_results:
        model_abbrevs = {
            'lognorm': 'CI',
            'fusionOnlyLogNorm': 'Fus',
            'switchingFree': 'SwF',
            'probabilityMatchingLogNorm': 'PM',
            'selection': 'Sel',
        }

        print("\n" + "-" * 70)
        print("Recovery Confusion Matrix (AIC)")
        print("-" * 70)
        print(f"{'Generating':<25}", end="")
        for m in args.models:
            print(f"{model_abbrevs.get(m, m[:4]):>8}", end="")
        print(f"{'Correct':>10}")
        print("-" * 70)

        for result in all_results:
            gen = result['generating_model']
            total = result['n_iterations']
            print(f"{gen:<25}", end="")
            correct = 0
            for fit_m in args.models:
                cnt = result['best_model_counts_aic'].get(fit_m, 0)
                pct = cnt / total * 100 if total > 0 else 0
                if fit_m == gen:
                    correct = cnt
                print(f"{pct:>7.0f}%", end="")
            c_pct = correct / total * 100 if total > 0 else 0
            print(f"{c_pct:>9.1f}%")

        print("-" * 70)
        tot_correct = sum(
            r['best_model_counts_aic'].get(r['generating_model'], 0) for r in all_results)
        tot_iters = sum(r['n_iterations'] for r in all_results)
        overall = tot_correct / tot_iters * 100 if tot_iters > 0 else 0
        print(f"{'Overall Accuracy:':<25}{' ' * len(args.models) * 8}{overall:>9.1f}%")

        # ---- Parameter recovery table ----
        print("\n" + "-" * 70)
        print("Parameter Recovery (same-model fits)")
        print("-" * 70)
        print(f"{'Model':<25} {'Param':<8} {'Corr':>7} {'RMSE':>8} {'Bias':>8}")
        print("-" * 70)
        for result in all_results:
            gen = result['generating_model']
            pr = result.get('param_recovery', {})
            for pname, stats in pr.items():
                print(f"{gen:<25} {pname:<8} {stats['correlation']:>7.3f} "
                      f"{stats['rmse']:>8.4f} {stats['bias']:>8.4f}")

        # ---- Boundary clip summary ----
        print("\n" + "-" * 70)
        print("Boundary Clipping Summary (same-model fits)")
        print("-" * 70)
        print(f"{'Model':<25} {'Iters':>6} {'Lower':>7} {'Upper':>7} {'Details'}")
        print("-" * 70)
        for result in all_results:
            gen = result['generating_model']
            bc = result.get('boundary_clip_summary', {})
            n_it = bc.get('total_iterations', 0)
            lo = bc.get('total_lower', 0)
            hi = bc.get('total_upper', 0)
            details_str = ", ".join(
                f"{k}:{v}" for k, v in sorted(bc.get('clipped_params', {}).items(),
                                               key=lambda x: -x[1])
            ) or "none"
            print(f"{gen:<25} {n_it:>6} {lo:>7} {hi:>7} {details_str}")


if __name__ == "__main__":
    main()
