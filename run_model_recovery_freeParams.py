#!/usr/bin/env python3
"""
Model Recovery Analysis Script (Data-Informed Wide-Range Parameter Sampling)
===========================================================================
Run this script to perform model recovery using parameters drawn from
wide uniform distributions centred on the group-level fitted means,
with range = mean +/- n_sigma * std (default 3), clipped to valid
optimiser bounds.

Motivation:
    If p_c recovery fails when parameters are centred on data-fitted values
    (group-level recovery with Normal sampling) but succeeds under this wider
    uniform sampling, the experimental range — not the model — is the
    limiting factor.

Key differences from run_model_recovery.py:
    - Parameters are sampled from U(mean - 3*std, mean + 3*std) instead of
      N(mean, std).  This covers a much wider range while still being
      anchored to real data.
    - Boundary-clipping diagnostics are reported.
    - Dedicated parameter-recovery scatter analysis for p_c (and all params).

Usage:
    python run_model_recovery_freeParams.py [--n_recovery N] [--n_jobs N]

Example:
    python run_model_recovery_freeParams.py --n_recovery 100 --n_jobs 8 --n_sigma 3
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

# The actual sampling ranges are computed at runtime from participant data
# via build_param_ranges_from_data().  PARAM_RANGES_UNIQUE will be populated
# in main() after loading the fits.
PARAM_RANGES_UNIQUE = {}          # filled by build_param_ranges_from_data()

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
# Loading fitted data & building parameter ranges
# ---------------------------------------------------------------------------

# Mapping: for each model, which full-vector indices to average into each
# unique parameter.  Each entry is (unique_name, [full_indices_to_average]).
UNIQUE_FROM_FULL = {
    # fusionOnlyLogNorm full: [λ(0), σa1(1), σv(2), σa2(3)]
    'fusionOnlyLogNorm': [('lambda', [0]), ('sigma_a', [1, 3]), ('sigma_v', [2])],
    # CI models full: [λ(0), σa1(1), σv(2), p_c(3), σa2(4)]
    'lognorm':                      [('lambda', [0]), ('sigma_a', [1, 4]), ('sigma_v', [2]), ('p_c', [3])],
    'probabilityMatchingLogNorm':   [('lambda', [0]), ('sigma_a', [1, 4]), ('sigma_v', [2]), ('p_c', [3])],
    'selection':                    [('lambda', [0]), ('sigma_a', [1, 4]), ('sigma_v', [2]), ('p_c', [3])],
    # switchingFree full: [λ(0), σa1(1), σv(2), p_sw1(3), σa2(4), p_sw2(5)]
    'switchingFree': [('lambda', [0]), ('sigma_a', [1, 4]), ('sigma_v', [2]), ('p_sw', [3, 5])],
}


def load_group_parameter_statistics(model_name, fits_dir='model_fits'):
    """
    Load fitted parameters for all participants and return the full array.
    """
    suffix = f"{model_name}_LapseFix_sharedPrior"
    pids = [d for d in os.listdir(fits_dir)
            if os.path.isdir(os.path.join(fits_dir, d))
            and d not in ['.DS_Store', 'all']]

    all_params = []
    for pid in pids:
        fp = os.path.join(fits_dir, pid, f"{pid}_{suffix}_fit.json")
        if os.path.exists(fp):
            with open(fp) as f:
                r = json.load(f)
            all_params.append(r['fittedParams'])

    if not all_params:
        raise ValueError(f"No fitted params found for '{model_name}' in {fits_dir}")
    return np.array(all_params)


def build_param_ranges_from_data(models, n_sigma=3, fits_dir='model_fits'):
    """
    For each model, load participant fits, compute unique-parameter
    mean and std, then set range = [mean - n_sigma*std, mean + n_sigma*std]
    clipped to the optimizer hard bounds.

    Populates the global PARAM_RANGES_UNIQUE dict and returns a summary
    dict for saving into the results JSON.
    """
    global PARAM_RANGES_UNIQUE
    summary = {}

    for model in models:
        if model not in UNIQUE_FROM_FULL:
            continue
        full_params = load_group_parameter_statistics(model, fits_dir)
        mapping = UNIQUE_FROM_FULL[model]
        names = PARAM_NAMES_UNIQUE[model]
        bounds_full = FITTING_BOUNDS[model]

        ranges = []
        info = []
        for (uname, fidxs), pname in zip(mapping, names):
            # Average the full-vector columns that map to this unique param
            vals = np.mean(full_params[:, fidxs], axis=1)
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sigma = max(sigma, 0.01)  # avoid degenerate zero-width range

            lo = mu - n_sigma * sigma
            hi = mu + n_sigma * sigma

            # Clip to fitting bounds (take the tightest among mapped indices)
            bound_lo = max(bounds_full[idx][0] for idx in fidxs)
            bound_hi = min(bounds_full[idx][1] for idx in fidxs)
            lo = max(lo, bound_lo)
            hi = min(hi, bound_hi)

            ranges.append((lo, hi))
            info.append({'name': pname, 'mean': round(mu, 4),
                         'std': round(sigma, 4), 'range': [round(lo, 4), round(hi, 4)]})

        PARAM_RANGES_UNIQUE[model] = ranges
        summary[model] = info

    return summary



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
        with open(result_path, 'r') as f:
            return json.load(f)

    ranges = PARAM_RANGES_UNIQUE[generating_model]
    names_unique = PARAM_NAMES_UNIQUE[generating_model]
    names_full = PARAM_NAMES[generating_model]

    iteration_args = [
        (i, generating_model, models_to_test, template_data, nSimul, nStarts)
        for i in range(n_recovery)
    ]

    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(run_single_recovery_iteration, iteration_args),
                total=n_recovery,
                desc=f"  {generating_model:<30}",
                bar_format='{l_bar}{bar:30}{r_bar}',
                leave=True,
            ))
    else:
        results = []
        for a in tqdm(iteration_args, desc=f"  {generating_model:<30}",
                      bar_format='{l_bar}{bar:30}{r_bar}', leave=True):
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
    parser.add_argument('--nStarts', type=int, default=5,
                        help='Optimisation starting points')
    parser.add_argument('--n_sigma', type=float, default=3.0,
                        help='Range width in std devs around group mean (default: 3)')
    parser.add_argument('--save_dir', type=str,
                        default='model_recovery_results_freeParams',
                        help='Directory to save results')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Parallel jobs (default: CPUs - 1)')
    parser.add_argument('--template_participant', type=str, default='as',
                        help='Participant ID for experimental design template')
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)

    print(f"Free-param model recovery | {len(args.models)} models x "
          f"{args.n_recovery} iters | nSimul={args.nSimul} nStarts={args.nStarts} "
          f"n_sigma={args.n_sigma} jobs={n_jobs}")

    start_time = time.time()

    # ---- Build sampling ranges from fitted participant data ----
    range_summary = build_param_ranges_from_data(
        args.models, n_sigma=args.n_sigma)
    for model, info in range_summary.items():
        rng_str = "  ".join(f"{p['name']}:[{p['range'][0]:.3f},{p['range'][1]:.3f}]" for p in info)
        print(f"  {model}: {rng_str}")

    # ---- Load template data ----
    try:
        template_data, _ = loadData.loadData(
            f"{args.template_participant}_all.csv", verbose=False)
    except Exception as e:
        print(f"Error loading template data: {e}")
        return

    # ---- Run recovery per generating model ----
    all_results = []
    for gen_model in args.models:
        if gen_model not in PARAM_RANGES_UNIQUE:
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
    # SUMMARY (compact)
    # ==================================================================
    print(f"\nDone in {elapsed / 60:.1f} min — saved to {args.save_dir}/")

    if all_results:
        abbr = {'lognorm': 'CI', 'fusionOnlyLogNorm': 'Fus',
                'switchingFree': 'SwF', 'probabilityMatchingLogNorm': 'PM',
                'selection': 'Sel'}

        # Confusion matrix
        header = f"{'Gen':<12}" + "".join(f"{abbr.get(m, m[:4]):>6}" for m in args.models) + f"{'%OK':>7}"
        print(f"\nConfusion (AIC):\n{header}")
        for result in all_results:
            gen = result['generating_model']
            total = result['n_iterations']
            row = f"{abbr.get(gen, gen[:4]):<12}"
            correct = 0
            for fit_m in args.models:
                cnt = result['best_model_counts_aic'].get(fit_m, 0)
                if fit_m == gen:
                    correct = cnt
                row += f"{cnt / total * 100 if total else 0:>5.0f}%"
            row += f"{correct / total * 100 if total else 0:>6.0f}%"
            print(row)

        # Parameter recovery
        print(f"\n{'Model':<12} {'Param':<6} {'r':>6} {'RMSE':>7} {'Bias':>7}")
        for result in all_results:
            gen = result['generating_model']
            for pname, s in result.get('param_recovery', {}).items():
                print(f"{abbr.get(gen, gen[:4]):<12} {pname:<6} "
                      f"{s['correlation']:>6.3f} {s['rmse']:>7.4f} {s['bias']:>7.4f}")

        # Boundary clips (one-liner per model)
        print(f"\nBoundary clips:")
        for result in all_results:
            bc = result.get('boundary_clip_summary', {})
            lo, hi = bc.get('total_lower', 0), bc.get('total_upper', 0)
            top = sorted(bc.get('clipped_params', {}).items(), key=lambda x: -x[1])[:3]
            detail = ", ".join(f"{k}:{v}" for k, v in top) or "none"
            print(f"  {abbr.get(result['generating_model'], '?'):<5} lo={lo} hi={hi}  {detail}")


if __name__ == "__main__":
    main()
