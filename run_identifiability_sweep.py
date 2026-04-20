#!/usr/bin/env python3
"""
Identifiability sweep: 2D grid over (sensory noise) x (conflict range)
======================================================================
For each grid cell, run a small model-recovery experiment with fixed
generating parameters and record:
  - confusion matrix (which model wins by AIC)
  - diagonal recovery rate (% of iterations where the true model wins)
  - p_c parameter recovery correlation (only meaningful for CI models)

The result is a 2D landscape that directly answers the eLife reviewers'
question: "under which experimental conditions can the models be
disentangled?"

Reuses the synthetic-template builder and the per-iteration worker from
run_param_recovery_favorable.py.

Usage
-----
    python run_identifiability_sweep.py [options]

Recommended quick pilot (one cell, fast):
    python run_identifiability_sweep.py --pilot

Default sweep (4 sigma levels x 3 conflict ranges, 15 iters/cell, 5 models)
takes ~1-2h with 8 jobs and nSimul=300.
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


# ---------------------------------------------------------------------------
# Single-cell driver (one (sigma_a, sigma_v, conflict_max) point)
# ---------------------------------------------------------------------------

def run_single_cell(sigma_a, sigma_v, conflict_max,
                    p_c_fixed, lambda_fixed,
                    models, n_iter, n_conflict_steps,
                    n_trials_per_cell, nSimul, nStarts,
                    n_jobs, save_dir, force=False, delta_max_pct=0.80):
    """
    Run a small model-recovery experiment at one grid cell with fixed
    generating parameters (sigma_a, sigma_v, p_c, lambda), then aggregate.

    Returns a dict with the cell's confusion matrix and diagnostic stats.
    """
    cell_tag = (f"sa{sigma_a:.2f}_sv{sigma_v:.2f}_cmax{conflict_max:.2f}"
                f"_pc{p_c_fixed:.2f}_lam{lambda_fixed:.3f}")
    cell_path = os.path.join(save_dir, f"cell_{cell_tag}.json")

    if (not force) and os.path.exists(cell_path):
        with open(cell_path) as f:
            return json.load(f)

    # Pin all parameters to single values via degenerate uniform ranges
    # (favo.sample_params samples uniform(lo, hi); lo==hi -> deterministic)
    pin_lambda = (lambda_fixed, lambda_fixed)
    pin_sigma_a = (sigma_a, sigma_a)
    pin_sigma_v = (sigma_v, sigma_v)
    pin_pc = (p_c_fixed, p_c_fixed)
    pin_psw = (p_c_fixed, p_c_fixed)  # treat p_switch like p_c for switchingFree

    ranges = {}
    for m in models:
        if m == 'fusionOnlyLogNorm':
            ranges[m] = [pin_lambda, pin_sigma_a, pin_sigma_v]
        elif m == 'switchingFree':
            ranges[m] = [pin_lambda, pin_sigma_a, pin_sigma_v, pin_psw]
        else:
            ranges[m] = [pin_lambda, pin_sigma_a, pin_sigma_v, pin_pc]

    # Build template at this conflict range
    template = favo.build_synthetic_template(
        conflict_max=conflict_max,
        n_conflict_steps=n_conflict_steps,
        n_trials_per_cell=n_trials_per_cell,
        delta_max_pct=delta_max_pct,
    )

    cell = {
        'sigma_a': sigma_a,
        'sigma_v': sigma_v,
        'conflict_max': conflict_max,
        'p_c_fixed': p_c_fixed,
        'lambda_fixed': lambda_fixed,
        'n_conflict_steps': n_conflict_steps,
        'n_trials_per_cell': n_trials_per_cell,
        'n_iter_per_model': n_iter,
        'models': list(models),
        'per_model': {},
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

        # Tally confusion row
        for it in iters:
            cell['confusion_aic'][gen_model][it['best_model_aic']] += 1
            cell['confusion_bic'][gen_model][it['best_model_bic']] += 1

        # Same-model param recovery (correlation of recovered vs generating)
        same = [it for it in iters if gen_model in it['model_fits']]
        param_recov = {}
        if same:
            gen_arr = np.array([it['sampled_unique'] for it in same])
            # Map unique -> full index (same as favorable script)
            if gen_model == 'fusionOnlyLogNorm':
                u2f = [0, 1, 2]
                names = ['lambda', 'sigma_a', 'sigma_v']
            elif gen_model == 'switchingFree':
                u2f = [0, 1, 2, 3]
                names = ['lambda', 'sigma_a', 'sigma_v', 'p_sw']
            else:
                u2f = [0, 1, 2, 3]
                names = ['lambda', 'sigma_a', 'sigma_v', 'p_c']

            rec_arr = np.array([it['model_fits'][gen_model]['fittedParams']
                                for it in same])

            for uid, nm in enumerate(names):
                gv = gen_arr[:, uid]
                rv = rec_arr[:, u2f[uid]]
                # With pinned generators, gv has zero variance -> corr undefined.
                # Report mean recovered value, std, bias, rmse instead.
                param_recov[nm] = {
                    'mean_gen': float(np.mean(gv)),
                    'mean_rec': float(np.mean(rv)),
                    'std_rec':  float(np.std(rv)),
                    'rmse':     float(np.sqrt(np.mean((rv - gv) ** 2))),
                    'bias':     float(np.mean(rv - gv)),
                }

        cell['per_model'][gen_model] = {
            'n_completed': len(iters),
            'n_same_model_fit': len(same),
            'param_recovery_pinned': param_recov,
        }

    # Diagonal recovery rate (averaged over generating models)
    diag_rates = []
    for gen in models:
        row = cell['confusion_aic'][gen]
        total = sum(row.values())
        if total > 0:
            diag_rates.append(row[gen] / total)
    cell['mean_diag_recovery_aic'] = float(np.mean(diag_rates)) if diag_rates else 0.0

    os.makedirs(save_dir, exist_ok=True)
    with open(cell_path, 'w') as f:
        json.dump(cell, f, indent=2)
    return cell


# ---------------------------------------------------------------------------
# Main: iterate over the grid
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='2D identifiability sweep: sensory noise x conflict range')
    parser.add_argument('--models', nargs='+',
                        default=['lognorm', 'fusionOnlyLogNorm',
                                 'switchingFree',
                                 'probabilityMatchingLogNorm', 'selection'])
    parser.add_argument('--sigma_levels', nargs='+', type=float,
                        default=[0.10, 0.20, 0.30, 0.50],
                        help='Values of sigma_a (=sigma_v) to sweep')
    parser.add_argument('--conflict_levels', nargs='+', type=float,
                        default=[0.125, 0.25, 0.45],
                        help='Values of conflict_max (s) to sweep')
    parser.add_argument('--p_c_fixed', type=float, default=0.5,
                        help='Fixed generating p_c (also used as p_switch)')
    parser.add_argument('--lambda_fixed', type=float, default=0.05,
                        help='Fixed generating lapse rate')
    parser.add_argument('--n_iter', type=int, default=15,
                        help='Recovery iterations per generating model per cell')
    parser.add_argument('--n_conflict_steps', type=int, default=9)
    parser.add_argument('--n_trials_per_cell', type=int, default=20)
    parser.add_argument('--nSimul', type=int, default=300)
    parser.add_argument('--nStarts', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=None)
    parser.add_argument('--save_dir', type=str,
                        default='identifiability_sweep_results')
    parser.add_argument('--pilot', action='store_true',
                        help='Run a tiny single-cell pilot and exit')
    parser.add_argument('--force', action='store_true',
                        help='Re-run cells even if cached')
    parser.add_argument('--delta_max_pct', type=float, default=0.80,
                        help='Delta range as fraction of standard_dur (default 0.80 = ±80%%)')
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)

    if args.pilot:
        print(">> PILOT: 1 cell, 3 iters per gen model, 3 models, fast settings")
        models = ['lognorm', 'fusionOnlyLogNorm', 'switchingFree']
        cell = run_single_cell(
            sigma_a=0.15, sigma_v=0.15, conflict_max=0.45,
            p_c_fixed=0.5, lambda_fixed=0.05,
            models=models, n_iter=3,
            n_conflict_steps=7, n_trials_per_cell=15,
            nSimul=200, nStarts=1, n_jobs=n_jobs,
            save_dir=args.save_dir, force=args.force,
        )
        print(json.dumps(cell['confusion_aic'], indent=2))
        print(f"diag_recovery_aic = {cell['mean_diag_recovery_aic']:.2f}")
        return

    grid = list(itertools.product(args.sigma_levels, args.conflict_levels))
    n_cells = len(grid)
    n_models = len(args.models)
    n_iters_per_cell = n_models * args.n_iter
    print("=" * 72)
    print(f"Identifiability sweep")
    print("=" * 72)
    print(f"  grid:        {len(args.sigma_levels)} sigma x "
          f"{len(args.conflict_levels)} conflict = {n_cells} cells")
    print(f"  sigma:       {args.sigma_levels}")
    print(f"  conflict:    {args.conflict_levels}")
    print(f"  fixed p_c:   {args.p_c_fixed}    fixed lambda: {args.lambda_fixed}")
    print(f"  models:      {args.models}")
    print(f"  per-cell:    {args.n_iter} iters x {n_models} gen models = "
          f"{n_iters_per_cell} sims, each fitted by {n_models} models")
    print(f"  Monte Carlo: nSimul={args.nSimul}  nStarts={args.nStarts}  "
          f"jobs={n_jobs}")
    print(f"  delta_max:   ±{args.delta_max_pct*100:.0f}% of standard_dur")
    print(f"  save_dir:    {args.save_dir}")
    print()

    start = time.time()
    summary = {
        'args': vars(args),
        'grid': [],
    }

    for k, (sigma, cmax) in enumerate(grid, start=1):
        print(f"[{k}/{n_cells}] sigma={sigma:.2f}  conflict_max={cmax:.2f}")
        cell = run_single_cell(
            sigma_a=sigma, sigma_v=sigma, conflict_max=cmax,
            p_c_fixed=args.p_c_fixed, lambda_fixed=args.lambda_fixed,
            models=list(args.models), n_iter=args.n_iter,
            n_conflict_steps=args.n_conflict_steps,
            n_trials_per_cell=args.n_trials_per_cell,
            nSimul=args.nSimul, nStarts=args.nStarts,
            n_jobs=n_jobs, save_dir=args.save_dir, force=args.force,
            delta_max_pct=args.delta_max_pct,
        )
        diag = cell['mean_diag_recovery_aic']
        per_model_diag = {
            m: (cell['confusion_aic'][m][m] /
                max(sum(cell['confusion_aic'][m].values()), 1))
            for m in args.models
        }
        summary['grid'].append({
            'sigma_a': sigma,
            'sigma_v': sigma,
            'conflict_max': cmax,
            'mean_diag_recovery_aic': diag,
            'per_model_diag_aic': per_model_diag,
            'confusion_aic': cell['confusion_aic'],
        })
        diag_str = " ".join(f"{m[:3]}:{per_model_diag[m]*100:3.0f}%"
                            for m in args.models)
        print(f"    -> mean diag = {diag*100:5.1f}%   {diag_str}")
        sys.stdout.flush()

    elapsed = time.time() - start
    summary['elapsed_minutes'] = elapsed / 60.0

    out = os.path.join(args.save_dir, 'identifiability_sweep_summary.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {elapsed/60:.1f} min  ->  {out}")


if __name__ == '__main__':
    main()
