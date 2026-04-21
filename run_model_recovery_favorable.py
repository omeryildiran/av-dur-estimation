#!/usr/bin/env python3
"""
Model recovery sweep — favorable regime
========================================
Sweeps sensory noise σ ∈ [sigma_min, sigma_max] at a fixed wide conflict
range (cmax=0.45 s by default) to answer: "at what noise level do the
three CI-family models become distinguishable?"

Key differences from run_identifiability_sweep.py:
  - Designed for the favorable (low-σ, wide-conflict) half of the space
  - Higher default n_iter (50) and nSimul (300) for tighter confidence intervals
  - nStarts=3 for better optimization (avoids local minima)
  - Prints per-cell confusion matrices immediately so you can watch progress
  - --also_empirical flag adds cmax=0.25 cells for direct comparison

Usage
-----
    python run_model_recovery_favorable.py [options]

Quick pilot (one cell, ~2 min):
    python run_model_recovery_favorable.py --pilot

Full sweep (6 sigma × 1 conflict = 6 cells, ~2-4 h on 8 cores):
    python run_model_recovery_favorable.py

Also compare against empirical conflict range:
    python run_model_recovery_favorable.py --also_empirical
"""

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

import run_param_recovery_favorable as favo


MODELS_DEFAULT = ['lognorm', 'fusionOnlyLogNorm', 'switchingFree']


def run_single_cell(sigma, conflict_max,
                    lambda_fixed,
                    pc_min, pc_max,
                    models, n_iter,
                    n_conflict_steps, n_trials_per_cell,
                    nSimul, nStarts, n_jobs,
                    save_dir, force=False, delta_max_pct=0.80):
    """Run one (sigma, conflict_max) cell; skip if cached unless force=True.

    p_c is sampled uniformly from [pc_min, pc_max] each iteration so that
    recovery is tested across the plausible range of causal priors, not just
    at the single most-discriminative value (p_c=0.5).
    """
    cell_tag = (f"fav_sa{sigma:.2f}_sv{sigma:.2f}_cmax{conflict_max:.2f}"
                f"_pc{pc_min:.2f}-{pc_max:.2f}_lam{lambda_fixed:.3f}"
                f"_ni{n_iter}_ns{nSimul}")
    cell_path = os.path.join(save_dir, f"{cell_tag}.json")

    if (not force) and os.path.exists(cell_path):
        with open(cell_path) as f:
            data = json.load(f)
        if data.get('mean_diag_recovery_aic', 0) > 0:
            return data, True  # (result, was_cached)

    pin_lambda  = (lambda_fixed, lambda_fixed)
    pin_sigma_a = (sigma, sigma)
    pin_sigma_v = (sigma, sigma)
    pc_range    = (pc_min, pc_max)   # sampled each iteration — not pinned

    ranges = {}
    for m in models:
        if m == 'fusionOnlyLogNorm':
            ranges[m] = [pin_lambda, pin_sigma_a, pin_sigma_v]
        elif m == 'switchingFree':
            ranges[m] = [pin_lambda, pin_sigma_a, pin_sigma_v, pc_range]
        else:
            ranges[m] = [pin_lambda, pin_sigma_a, pin_sigma_v, pc_range]

    template = favo.build_synthetic_template(
        conflict_max=conflict_max,
        n_conflict_steps=n_conflict_steps,
        n_trials_per_cell=n_trials_per_cell,
        delta_max_pct=delta_max_pct,
    )

    cell = {
        'sigma':         sigma,
        'conflict_max':  conflict_max,
        'pc_range':      [pc_min, pc_max],
        'lambda_fixed':  lambda_fixed,
        'n_iter':        n_iter,
        'nSimul':        nSimul,
        'nStarts':       nStarts,
        'models':        list(models),
        'per_model':     {},
        'confusion_aic': {m: {fm: 0 for fm in models} for m in models},
        'confusion_bic': {m: {fm: 0 for fm in models} for m in models},
    }

    for gen_model in models:
        iargs = [
            (i, gen_model, list(models), template, nSimul, nStarts, ranges)
            for i in range(n_iter)
        ]
        if n_jobs > 1:
            with Pool(processes=n_jobs) as pool:
                raw = pool.map(favo.run_single_recovery, iargs)
        else:
            raw = [favo.run_single_recovery(a) for a in iargs]
        iters = [r for r in raw if r is not None]

        for it in iters:
            cell['confusion_aic'][gen_model][it['best_model_aic']] += 1
            cell['confusion_bic'][gen_model][it['best_model_bic']] += 1

        same = [it for it in iters if gen_model in it['model_fits']]
        if gen_model == 'fusionOnlyLogNorm':
            names = ['lambda', 'sigma_a', 'sigma_v']
            u2f   = [0, 1, 2]
        elif gen_model == 'switchingFree':
            names = ['lambda', 'sigma_a', 'sigma_v', 'p_sw']
            u2f   = [0, 1, 2, 3]
        else:
            names = ['lambda', 'sigma_a', 'sigma_v', 'p_c']
            u2f   = [0, 1, 2, 3]

        param_recov = {}
        if same:
            gen_arr = np.array([it['sampled_unique'] for it in same])
            rec_arr = np.array([it['model_fits'][gen_model]['fittedParams']
                                for it in same])
            for uid, nm in enumerate(names):
                gv = gen_arr[:, uid]
                rv = rec_arr[:, u2f[uid]]
                param_recov[nm] = {
                    'mean_gen': float(np.mean(gv)),
                    'mean_rec': float(np.mean(rv)),
                    'std_rec':  float(np.std(rv)),
                    'rmse':     float(np.sqrt(np.mean((rv - gv) ** 2))),
                    'bias':     float(np.mean(rv - gv)),
                }

        cell['per_model'][gen_model] = {
            'n_completed':       len(iters),
            'n_same_model_fit':  sum(1 for it in iters
                                     if it['best_model_aic'] == gen_model),
            'param_recovery':    param_recov,
        }

    diag_rates = []
    for gen in models:
        row   = cell['confusion_aic'][gen]
        total = sum(row.values())
        if total > 0:
            diag_rates.append(row[gen] / total)
    cell['mean_diag_recovery_aic'] = float(np.mean(diag_rates)) if diag_rates else 0.0

    os.makedirs(save_dir, exist_ok=True)
    with open(cell_path, 'w') as f:
        json.dump(cell, f, indent=2)
    return cell, False


def print_confusion(cell, models):
    abbr = {'lognorm': 'CI ', 'fusionOnlyLogNorm': 'Fus', 'switchingFree': 'SwF',
            'probabilityMatchingLogNorm': 'PM ', 'selection': 'Sel'}
    header = f"  {'gen':>4}  " + "  ".join(f"{abbr.get(m, m[:3]):>4}" for m in models) + "   diag"
    print(header)
    for gen in models:
        row   = cell['confusion_aic'][gen]
        total = sum(row.values())
        diag  = row[gen] / total if total else 0
        vals  = "  ".join(f"{row[m]:>4}" for m in models)
        print(f"  {abbr.get(gen, gen[:3]):>4}  {vals}   {diag*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Model recovery sweep — favorable (low-σ, wide-conflict) regime')
    parser.add_argument('--models', nargs='+', default=MODELS_DEFAULT)
    parser.add_argument('--sigma_levels', nargs='+', type=float,
                        default=[0.10, 0.20, 0.30, 0.40, 0.55, 0.60],
                        help='σ values to sweep (default: 0.10 0.20 0.30 0.40 0.55 0.60)')
    parser.add_argument('--conflict_max', type=float, default=0.45,
                        help='AV conflict half-range in seconds (default 0.45)')
    parser.add_argument('--also_empirical', action='store_true',
                        help='Also run cmax=0.25 cells for comparison')
    parser.add_argument('--pc_min',             type=float, default=0.20,
                        help='Lower bound of p_c sampling range (default 0.20)')
    parser.add_argument('--pc_max',             type=float, default=0.80,
                        help='Upper bound of p_c sampling range (default 0.80)')
    parser.add_argument('--lambda_fixed',       type=float, default=0.05)
    parser.add_argument('--n_iter',             type=int,   default=50,
                        help='Recovery iterations per generating model per cell (default 50)')
    parser.add_argument('--n_conflict_steps',   type=int,   default=9)
    parser.add_argument('--n_trials_per_cell',  type=int,   default=20)
    parser.add_argument('--nSimul',             type=int,   default=300)
    parser.add_argument('--nStarts',            type=int,   default=3)
    parser.add_argument('--n_jobs',             type=int,   default=None)
    parser.add_argument('--delta_max_pct',      type=float, default=0.80)
    parser.add_argument('--save_dir',           type=str,
                        default='model_recovery_favorable_results')
    parser.add_argument('--force',   action='store_true',
                        help='Re-run cells even if cached JSON exists')
    parser.add_argument('--pilot',   action='store_true',
                        help='Run one tiny cell (σ=0.20, cmax=0.45, n_iter=5) and exit')
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)

    conflict_levels = [args.conflict_max]
    if args.also_empirical:
        conflict_levels = sorted(set(conflict_levels + [0.25]))

    if args.pilot:
        print(">> PILOT: σ=0.20  cmax=0.45  p_c~U(0.2,0.8)  n_iter=5  nSimul=200  nStarts=1")
        cell, cached = run_single_cell(
            sigma=0.20, conflict_max=0.45,
            lambda_fixed=0.05, pc_min=0.01, pc_max=0.999,
            models=['lognorm', 'fusionOnlyLogNorm', 'switchingFree'],
            n_iter=5, n_conflict_steps=7, n_trials_per_cell=15,
            nSimul=200, nStarts=1, n_jobs=n_jobs,
            save_dir=args.save_dir, force=args.force,
        )
        print(f"  mean_diag = {cell['mean_diag_recovery_aic']*100:.1f}%")
        print_confusion(cell, ['lognorm', 'fusionOnlyLogNorm', 'switchingFree'])
        return

    grid = list(itertools.product(args.sigma_levels, conflict_levels))
    n_cells   = len(grid)
    n_models  = len(args.models)
    n_iters_c = n_models * args.n_iter

    print("=" * 72)
    print("Model recovery sweep — favorable regime")
    print("=" * 72)
    print(f"  sigma levels : {args.sigma_levels}")
    print(f"  conflict     : {conflict_levels} s")
    print(f"  p_c range    : [{args.pc_min:.2f}, {args.pc_max:.2f}]  (sampled each iter — NOT pinned)")
    print(f"  grid         : {n_cells} cells")
    print(f"  models       : {args.models}")
    print(f"  per cell     : {args.n_iter} iters × {n_models} gen models "
          f"= {n_iters_c} sims, each fit by {n_models} models")
    print(f"  nSimul={args.nSimul}  nStarts={args.nStarts}  "
          f"delta_max=±{args.delta_max_pct*100:.0f}%  jobs={n_jobs}")
    print(f"  save_dir     : {args.save_dir}")
    print()

    start   = time.time()
    summary = {'args': vars(args), 'grid': []}

    for k, (sigma, cmax) in enumerate(grid, start=1):
        print(f"[{k}/{n_cells}] σ={sigma:.2f}  cmax={cmax:.2f}")
        cell, cached = run_single_cell(
            sigma=sigma, conflict_max=cmax,
            lambda_fixed=args.lambda_fixed,
            pc_min=args.pc_min, pc_max=args.pc_max,
            models=list(args.models), n_iter=args.n_iter,
            n_conflict_steps=args.n_conflict_steps,
            n_trials_per_cell=args.n_trials_per_cell,
            nSimul=args.nSimul, nStarts=args.nStarts, n_jobs=n_jobs,
            save_dir=args.save_dir, force=args.force,
            delta_max_pct=args.delta_max_pct,
        )

        diag = cell['mean_diag_recovery_aic']
        per_model_diag = {
            m: (cell['confusion_aic'][m][m] /
                max(sum(cell['confusion_aic'][m].values()), 1))
            for m in args.models
        }
        tag = " [cached]" if cached else ""
        print(f"  mean_diag = {diag*100:.1f}%{tag}")
        print_confusion(cell, args.models)
        print()
        sys.stdout.flush()

        summary['grid'].append({
            'sigma':                    sigma,
            'conflict_max':             cmax,
            'mean_diag_recovery_aic':   diag,
            'per_model_diag_aic':       per_model_diag,
            'confusion_aic':            cell['confusion_aic'],
        })

    elapsed = time.time() - start

    # Final summary table
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    abbr = {'lognorm': 'CI', 'fusionOnlyLogNorm': 'Fus', 'switchingFree': 'SwF'}
    header = f"  {'σ':>5}  {'cmax':>5}  {'mean':>6}  " + \
             "  ".join(f"{abbr.get(m, m[:3]):>5}" for m in args.models)
    print(header)
    for entry in summary['grid']:
        pm = entry['per_model_diag_aic']
        vals = "  ".join(f"{pm.get(m, 0)*100:5.1f}%" for m in args.models)
        print(f"  {entry['sigma']:5.2f}  {entry['conflict_max']:5.2f}  "
              f"{entry['mean_diag_recovery_aic']*100:5.1f}%  {vals}")

    print(f"\nDone in {elapsed/60:.1f} min")

    out = os.path.join(args.save_dir, 'favorable_sweep_summary.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {out}")


if __name__ == '__main__':
    main()
