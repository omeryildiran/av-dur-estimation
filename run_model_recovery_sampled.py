#!/usr/bin/env python3
"""
Model recovery — sampled-parameter regime
==========================================
Unlike run_model_recovery_favorable.py (which pins σ and λ per sweep cell),
this script samples σ, λ, and p_c freely from specified ranges on every
iteration.  Run it twice with different sigma ranges to produce the two-case
comparison needed for the claim:

  "In a low-noise regime models are reliably distinguishable. When visual
   noise reaches empirically observed levels, model recovery collapses even
   under a wide conflict range.  The bottleneck is sensory noise, not
   experimental design."

Empirical measurement-σ reference (task_σ = measurement_σ × √2):
  Auditory High Rel  : mean = 0.21,  SD = 0.04
  Visual             : mean = 0.48,  SD = 0.23   ← drives the collapse
  Auditory Low Rel   : mean = 1.43,  SD = 1.17

Suggested usage
---------------
# Case 1 — low noise (well below empirical levels):
python run_model_recovery_sampled.py \\
    --sigma_min 0.05 --sigma_max 0.30 \\
    --save_dir results_low_noise

# Case 2 — high noise (covers empirical visual range):
python run_model_recovery_sampled.py \\
    --sigma_min 0.30 --sigma_max 0.70 \\
    --save_dir results_high_noise

# Quick pilot (5 iters, no parallelism):
python run_model_recovery_sampled.py --pilot

Outputs (in save_dir/)
----------------------
  recovery_raw.json        — all iteration-level results (gen + rec params)
  recovery_results.json    — confusion matrices + per-param r/RMSE/bias
  recovery_params.csv      — tidy table of gen vs rec values (for notebook)
"""

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

import run_param_recovery_favorable as favo


MODELS_DEFAULT = ['lognorm', 'fusionOnlyLogNorm', 'switchingFree']

# Unique-param names for each model
PARAM_NAMES_UNIQUE = {
    'fusionOnlyLogNorm':          ['lambda', 'sigma_a', 'sigma_v'],
    'lognorm':                    ['lambda', 'sigma_a', 'sigma_v', 'p_c'],
    'probabilityMatchingLogNorm': ['lambda', 'sigma_a', 'sigma_v', 'p_c'],
    'selection':                  ['lambda', 'sigma_a', 'sigma_v', 'p_c'],
    'switchingFree':              ['lambda', 'sigma_a', 'sigma_v', 'p_sw'],
}

# Maps unique-param index → fitted-param index (full vector from monteCarloClass)
# Full vectors: fusionOnly=[λ,σa,σv,σa]  CI=[λ,σa,σv,p_c,σa]  switching=[λ,σa,σv,p_sw,σa,p_sw]
U2F = {
    'fusionOnlyLogNorm':          [0, 1, 2],
    'lognorm':                    [0, 1, 2, 3],
    'probabilityMatchingLogNorm': [0, 1, 2, 3],
    'selection':                  [0, 1, 2, 3],
    'switchingFree':              [0, 1, 2, 3],
}

ABBR = {
    'lognorm':                    'CI ',
    'fusionOnlyLogNorm':          'Fus',
    'switchingFree':              'SwF',
    'probabilityMatchingLogNorm': 'PM ',
    'selection':                  'Sel',
}


# ---------------------------------------------------------------------------
# Range builder
# ---------------------------------------------------------------------------

def build_ranges(models, sigma_min, sigma_max, lambda_min, lambda_max,
                 pc_min, pc_max):
    """Return {model: [(lo, hi), ...]} with λ, σ, and p_c all freely sampled."""
    sigma_range  = (sigma_min,  sigma_max)
    lambda_range = (lambda_min, lambda_max)
    pc_range     = (pc_min,     pc_max)

    ranges = {}
    for m in models:
        if m == 'fusionOnlyLogNorm':
            ranges[m] = [lambda_range, sigma_range, sigma_range]
        elif m == 'switchingFree':
            ranges[m] = [lambda_range, sigma_range, sigma_range, pc_range]
        else:  # CI models: lognorm, probabilityMatchingLogNorm, selection
            ranges[m] = [lambda_range, sigma_range, sigma_range, pc_range]
    return ranges


# ---------------------------------------------------------------------------
# Run recovery
# ---------------------------------------------------------------------------

def run_recovery(models, ranges, template, n_recovery, nSimul, nStarts,
                 n_jobs, save_dir, force=False):
    """
    Run n_recovery iterations for each generating model.
    Returns {gen_model: [iteration_result, ...]} (None entries removed).
    Caches raw results to recovery_raw.json.
    """
    cache_path = os.path.join(save_dir, 'recovery_raw.json')
    if (not force) and os.path.exists(cache_path):
        print(f"  [loaded from cache: {cache_path}]")
        with open(cache_path) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    all_iters = {}

    for gen_model in models:
        print(f"\n  Generating from {gen_model} …")
        sys.stdout.flush()
        iargs = [
            (i, gen_model, list(models), template, nSimul, nStarts, ranges)
            for i in range(n_recovery)
        ]
        if n_jobs > 1:
            with Pool(processes=n_jobs) as pool:
                raw = list(tqdm(
                    pool.imap(favo.run_single_recovery, iargs),
                    total=n_recovery,
                    desc=f"    {gen_model:<36}",
                    bar_format='{l_bar}{bar:28}{r_bar}',
                ))
        else:
            raw = [favo.run_single_recovery(a) for a in tqdm(
                iargs,
                desc=f"    {gen_model:<36}",
                bar_format='{l_bar}{bar:28}{r_bar}',
            )]

        completed = [r for r in raw if r is not None]
        print(f"    completed: {len(completed)}/{n_recovery}")
        all_iters[gen_model] = completed

    with open(cache_path, 'w') as f:
        json.dump(all_iters, f, indent=2)
    return all_iters


# ---------------------------------------------------------------------------
# Compute results
# ---------------------------------------------------------------------------

def compute_results(all_iters, models, save_dir):
    """
    Build confusion matrices, parameter recovery statistics, and a tidy CSV.
    Saves recovery_results.json and recovery_params.csv to save_dir.
    """
    confusion_aic = {m: {fm: 0 for fm in models} for m in models}
    confusion_bic = {m: {fm: 0 for fm in models} for m in models}
    param_recovery = {}
    csv_rows = []

    for gen_model in models:
        iters = all_iters.get(gen_model, [])
        if not iters:
            continue

        for it in iters:
            confusion_aic[gen_model][it['best_model_aic']] += 1
            confusion_bic[gen_model][it['best_model_bic']] += 1

        # Parameter recovery (same-model fits only)
        same = [it for it in iters if gen_model in it['model_fits']]
        names = PARAM_NAMES_UNIQUE.get(gen_model, [])
        u2f   = U2F.get(gen_model, list(range(len(names))))

        if same and names:
            gen_arr = np.array([it['sampled_unique'] for it in same])
            rec_arr = np.array([
                it['model_fits'][gen_model]['fittedParams'] for it in same
            ])

            pr = {}
            for uid, nm in enumerate(names):
                fid = u2f[uid]
                gv = gen_arr[:, uid]
                rv = rec_arr[:, fid]
                corr = (float(np.corrcoef(gv, rv)[0, 1])
                        if len(gv) > 1 else float('nan'))
                rmse = float(np.sqrt(np.mean((gv - rv) ** 2)))
                bias = float(np.mean(rv - gv))
                pr[nm] = {
                    'correlation': corr,
                    'rmse':        rmse,
                    'bias':        bias,
                    'gen_values':  gv.tolist(),
                    'rec_values':  rv.tolist(),
                }

                # Rows for CSV export
                for i, (gval, rval) in enumerate(zip(gv, rv)):
                    it = same[i]
                    csv_rows.append({
                        'gen_model':   gen_model,
                        'param':       nm,
                        'gen_value':   gval,
                        'rec_value':   rval,
                        'best_aic':    it['best_model_aic'],
                        'best_bic':    it['best_model_bic'],
                        'correct_aic': int(it['best_model_aic'] == gen_model),
                        'correct_bic': int(it['best_model_bic'] == gen_model),
                        # Also store all unique gen params for context
                        **{f'gen_{nm2}': it['sampled_unique'][uid2]
                           for uid2, nm2 in enumerate(names)},
                    })

            param_recovery[gen_model] = pr

    # Diagonal recovery rates
    diag_rates = [
        confusion_aic[m][m] / max(sum(confusion_aic[m].values()), 1)
        for m in models
    ]
    mean_diag = float(np.mean(diag_rates))

    per_model_diag_aic = {
        m: confusion_aic[m][m] / max(sum(confusion_aic[m].values()), 1)
        for m in models
    }

    results = {
        'confusion_aic':           confusion_aic,
        'confusion_bic':           confusion_bic,
        'mean_diag_recovery_aic':  mean_diag,
        'per_model_diag_aic':      per_model_diag_aic,
        'param_recovery':          param_recovery,
    }

    os.makedirs(save_dir, exist_ok=True)

    results_path = os.path.join(save_dir, 'recovery_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if csv_rows:
        csv_path = os.path.join(save_dir, 'recovery_params.csv')
        # Union of all keys across rows (models have different gen_* columns)
        fieldnames = list(dict.fromkeys(k for row in csv_rows for k in row))
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore',
                                    restval=float('nan'))
            writer.writeheader()
            writer.writerows(csv_rows)

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_confusion(confusion, models):
    header = (f"  {'gen':>5}  "
              + "  ".join(f"{ABBR.get(m, m[:3]):>4}" for m in models)
              + "   diag%")
    print(header)
    for gen in models:
        row   = confusion[gen]
        total = sum(row.values())
        diag  = row[gen] / total if total else 0.0
        vals  = "  ".join(f"{row[m]:>4}" for m in models)
        print(f"  {ABBR.get(gen, gen[:3]):>5}  {vals}   {diag*100:5.1f}%")


def print_param_recovery(param_recovery):
    print(f"\n{'Model':<10} {'Param':<10} {'r':>7} {'RMSE':>8} {'Bias':>8}")
    for gen_model, pr in param_recovery.items():
        for pname, s in pr.items():
            flag = ''
            if pname == 'p_c':
                flag = '  *** p_c ***' if s['correlation'] > 0.5 else '  (low r)'
            print(f"{ABBR.get(gen_model, gen_model[:4]):<10} {pname:<10} "
                  f"{s['correlation']:>7.3f} {s['rmse']:>8.4f} "
                  f"{s['bias']:>8.4f}{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Model recovery with freely sampled σ, λ, p_c')
    parser.add_argument('--models', nargs='+', default=MODELS_DEFAULT,
                        help='Models to generate from and fit')
    # Sigma range — the key experimental variable
    parser.add_argument('--sigma_min', type=float, default=0.05,
                        help='Lower bound of σa/σv sampling range (default 0.05)')
    parser.add_argument('--sigma_max', type=float, default=0.30,
                        help='Upper bound of σa/σv sampling range (default 0.30)')
    # Lapse rate — freely sampled
    parser.add_argument('--lambda_min', type=float, default=0.001,
                        help='Lower bound of λ (lapse rate) range (default 0.001)')
    parser.add_argument('--lambda_max', type=float, default=0.40,
                        help='Upper bound of λ (lapse rate) range (default 0.40)')
    # Causal prior
    parser.add_argument('--pc_min', type=float, default=0.001,
                        help='Lower bound of p_c / p_sw range (default 0.001)')
    parser.add_argument('--pc_max', type=float, default=0.999,
                        help='Upper bound of p_c / p_sw range (default 0.999)')
    # Trial structure
    parser.add_argument('--conflict_max',      type=float, default=0.45,
                        help='AV conflict half-range in seconds (default 0.45)')
    parser.add_argument('--n_conflict_steps',  type=int,   default=9)
    parser.add_argument('--n_trials_per_cell', type=int,   default=20)
    parser.add_argument('--delta_max_pct',     type=float, default=0.80)
    # Optimisation
    parser.add_argument('--n_recovery', type=int, default=100,
                        help='Recovery iterations per generating model (default 100)')
    parser.add_argument('--nSimul',     type=int, default=300,
                        help='Monte Carlo draws per nLL evaluation (default 300)')
    parser.add_argument('--nStarts',    type=int, default=1,
                        help='Optimiser restarts (default 3)')
    parser.add_argument('--n_jobs',     type=int, default=None)
    parser.add_argument('--save_dir',   type=str, default='model_recovery_sampled',
                        help='Output directory (default: model_recovery_sampled)')
    parser.add_argument('--force',  action='store_true',
                        help='Re-run even if cached results exist')
    parser.add_argument('--pilot', action='store_true',
                        help='Tiny run: 5 iters, nSimul=100, nStarts=1, single job')
    args = parser.parse_args()

    if args.pilot:
        args.n_recovery = 5
        args.nSimul     = 100
        args.nStarts    = 1
        args.n_jobs     = 1
        if args.save_dir == 'model_recovery_sampled':
            args.save_dir = 'model_recovery_sampled_pilot'

    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)

    print("=" * 72)
    print("Model recovery — sampled-parameter regime")
    print("=" * 72)
    print(f"  σ range        : [{args.sigma_min:.3f}, {args.sigma_max:.3f}]"
          f"   [empirical visual measurement-σ ≈ 0.48]")
    print(f"  λ range        : [{args.lambda_min:.3f}, {args.lambda_max:.3f}]")
    print(f"  p_c / p_sw     : [{args.pc_min:.3f}, {args.pc_max:.3f}]")
    print(f"  conflict_max   : ±{args.conflict_max:.2f} s  "
          f"({args.n_conflict_steps} levels)   [empirical ±0.25 s]")
    print(f"  n_recovery     : {args.n_recovery} per model")
    print(f"  nSimul={args.nSimul}  nStarts={args.nStarts}  jobs={n_jobs}")
    print(f"  models         : {args.models}")
    print(f"  save_dir       : {args.save_dir}")
    print()

    ranges = build_ranges(
        args.models,
        args.sigma_min,  args.sigma_max,
        args.lambda_min, args.lambda_max,
        args.pc_min,     args.pc_max,
    )

    template = favo.build_synthetic_template(
        conflict_max=args.conflict_max,
        n_conflict_steps=args.n_conflict_steps,
        n_trials_per_cell=args.n_trials_per_cell,
        delta_max_pct=args.delta_max_pct,
    )
    print(f"  Template: {len(template)} rows")
    print(f"  Conflict levels: {sorted(template['conflictDur'].unique().tolist())}")
    print()

    start = time.time()

    all_iters = run_recovery(
        models=args.models,
        ranges=ranges,
        template=template,
        n_recovery=args.n_recovery,
        nSimul=args.nSimul,
        nStarts=args.nStarts,
        n_jobs=n_jobs,
        save_dir=args.save_dir,
        force=args.force,
    )

    results = compute_results(all_iters, args.models, args.save_dir)
    elapsed = time.time() - start

    print("\n" + "=" * 72)
    print("Confusion matrix (AIC):")
    print_confusion(results['confusion_aic'], args.models)
    print(f"\n  Mean diagonal recovery: {results['mean_diag_recovery_aic']*100:.1f}%")
    per = results['per_model_diag_aic']
    for m in args.models:
        print(f"    {ABBR.get(m, m[:4])}: {per.get(m, 0)*100:.1f}%")

    print_param_recovery(results['param_recovery'])

    print(f"\nDone in {elapsed / 60:.1f} min")
    print(f"Results → {args.save_dir}/")
    print(f"  recovery_raw.json       — full iteration data")
    print(f"  recovery_results.json   — confusion + param recovery stats")
    print(f"  recovery_params.csv     — tidy gen vs rec table for notebook")


if __name__ == '__main__':
    main()
