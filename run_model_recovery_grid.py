#!/usr/bin/env python3
"""
Model recovery grid — σ-range × conflict_max
=============================================
4 sigma-level × 2 conflict_max = 8-cell grid.  Each cell runs full model
recovery for all 5 CI-family models with freely sampled parameters.

Sigma levels (ranges, NOT fixed values):
  a  σ_a = σ_v ~ U[0.01, 0.20]   very low noise
  b  σ_a = σ_v ~ U[0.20, 0.40]   moderate noise
  c  σ_a = σ_v ~ U[0.30, 0.70]   high noise
  d  σ_a ~ U[0.08, 0.47]         empirical noise (p5-p95 of fitted σ_a)
     σ_v ~ U[0.12, 1.30]         empirical noise (p5-p95 of fitted σ_v)

conflict_max: [0.25, 0.45] s

All cells use:
  λ    ~ U[0.001, 0.40]
  p_c  ~ U[0.001, 0.999]
  p_sw ~ U[0.001, 0.999]

Usage
-----
    python run_model_recovery_grid.py [options]

Pilot (single cell, ~5 min):
    python run_model_recovery_grid.py --pilot

Full 8-cell grid:
    python run_model_recovery_grid.py
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
# Sigma level definitions
# ---------------------------------------------------------------------------

SIGMA_LEVELS = {
    'a': {
        'label':   'a [0.01–0.20]',
        'sigma_a': (0.01, 0.20),
        'sigma_v': (0.01, 0.20),
    },
    'b': {
        'label':   'b [0.20–0.40]',
        'sigma_a': (0.20, 0.40),
        'sigma_v': (0.20, 0.40),
    },
    'c': {
        'label':   'c [0.30–0.70]',
        'sigma_a': (0.30, 0.70),
        'sigma_v': (0.30, 0.70),
    },
    'd': {
        'label':   'd empirical [σa:0.08–0.47, σv:0.12–1.30]',
        'sigma_a': (0.08, 0.47),
        'sigma_v': (0.12, 1.30),
    },
}

MODELS_DEFAULT = [
    'lognorm',
    'fusionOnlyLogNorm',
    'switchingFree',
    'probabilityMatchingLogNorm',
    'selection',
]

LAMBDA_RANGE = (0.001, 0.40)
PC_RANGE     = (0.001, 0.999)

ABBR = {
    'lognorm':                    'CI ',
    'fusionOnlyLogNorm':          'Fus',
    'switchingFree':              'SwF',
    'probabilityMatchingLogNorm': 'PM ',
    'selection':                  'Sel',
}


# ---------------------------------------------------------------------------
# Ranges builder (per cell — supports asymmetric σ_a / σ_v)
# ---------------------------------------------------------------------------

def build_ranges_for_cell(models, sigma_a_range, sigma_v_range,
                           lambda_range=LAMBDA_RANGE, pc_range=PC_RANGE):
    """Return {model: [(lo, hi), ...]} for unique params in the order used by
    run_param_recovery_favorable.sample_params:
        fusionOnly  → [λ, σa, σv]
        CI models   → [λ, σa, σv, p_c]
        switching   → [λ, σa, σv, p_sw]
    """
    ranges = {}
    for m in models:
        if m == 'fusionOnlyLogNorm':
            ranges[m] = [lambda_range, sigma_a_range, sigma_v_range]
        elif m == 'switchingFree':
            ranges[m] = [lambda_range, sigma_a_range, sigma_v_range, pc_range]
        else:  # lognorm, probabilityMatchingLogNorm, selection
            ranges[m] = [lambda_range, sigma_a_range, sigma_v_range, pc_range]
    return ranges


# ---------------------------------------------------------------------------
# Single-cell driver
# ---------------------------------------------------------------------------

def _param_names_u2f(gen_model):
    if gen_model == 'fusionOnlyLogNorm':
        return ['lambda', 'sigma_a', 'sigma_v'], [0, 1, 2]
    if gen_model == 'switchingFree':
        return ['lambda', 'sigma_a', 'sigma_v', 'p_sw'], [0, 1, 2, 3]
    return ['lambda', 'sigma_a', 'sigma_v', 'p_c'], [0, 1, 2, 3]


def _recompute_summaries(cell, models):
    """Rebuild per_model param_recovery + mean_diag from the raw iteration
    list stored under cell['raw_iters']. Called after every iteration so the
    on-disk JSON is always consistent if the run is interrupted."""
    cell['confusion_aic'] = {m: {fm: 0 for fm in models} for m in models}
    cell['confusion_bic'] = {m: {fm: 0 for fm in models} for m in models}

    for gen_model in models:
        iters = cell['raw_iters'].get(gen_model, [])
        for it in iters:
            cell['confusion_aic'][gen_model][it['best_model_aic']] += 1
            cell['confusion_bic'][gen_model][it['best_model_bic']] += 1

        names, u2f = _param_names_u2f(gen_model)
        same = [it for it in iters if gen_model in it.get('model_fits', {})]
        param_recov = {}
        if same:
            gen_arr = np.array([it['sampled_unique'] for it in same])
            rec_arr = np.array([it['model_fits'][gen_model]['fittedParams']
                                for it in same])
            for uid, nm in enumerate(names):
                gv = gen_arr[:, uid]
                rv = rec_arr[:, u2f[uid]]
                corr = float(np.corrcoef(gv, rv)[0, 1]) if len(gv) > 1 else float('nan')
                param_recov[nm] = {
                    'correlation': corr,
                    'rmse':        float(np.sqrt(np.mean((rv - gv) ** 2))),
                    'bias':        float(np.mean(rv - gv)),
                    'gen_values':  gv.tolist(),
                    'rec_values':  rv.tolist(),
                }

        cell['per_model'][gen_model] = {
            'n_completed':      len(iters),
            'n_correct_aic':    sum(1 for it in iters
                                    if it['best_model_aic'] == gen_model),
            'param_recovery':   param_recov,
        }

    diag_rates = []
    for gen in models:
        row   = cell['confusion_aic'][gen]
        total = sum(row.values())
        if total > 0:
            diag_rates.append(row[gen] / total)
    cell['mean_diag_recovery_aic'] = float(np.mean(diag_rates)) if diag_rates else 0.0


def _atomic_write_json(path, payload):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def run_single_cell(sigma_key, conflict_max,
                    models, n_iter,
                    n_conflict_steps, n_trials_per_cell,
                    nSimul, nStarts, n_jobs,
                    save_dir, force=False, delta_max_pct=0.80,
                    save_every=1):
    """
    Run (or resume) one (sigma_level, conflict_max) cell.

    Resumable: every completed iteration is appended to ``cell['raw_iters']``
    and the JSON is rewritten atomically every ``save_every`` iterations.  An
    interrupted run can be resumed simply by re-invoking with the same args.

    Returns (cell_dict, was_cached_complete).
    """
    slevel  = SIGMA_LEVELS[sigma_key]
    sa_lo, sa_hi = slevel['sigma_a']
    sv_lo, sv_hi = slevel['sigma_v']

    cell_tag = (f"grid_sl{sigma_key}"
                f"_sa{sa_lo:.2f}-{sa_hi:.2f}"
                f"_sv{sv_lo:.2f}-{sv_hi:.2f}"
                f"_cmax{conflict_max:.2f}"
                f"_ni{n_iter}_ns{nSimul}")
    cell_path = os.path.join(save_dir, f"{cell_tag}.json")

    os.makedirs(save_dir, exist_ok=True)

    # Try to resume from existing JSON unless --force
    cell = None
    if (not force) and os.path.exists(cell_path):
        with open(cell_path) as f:
            cell = json.load(f)
        # Fully-complete cell?  Skip.
        completed = all(len(cell.get('raw_iters', {}).get(m, [])) >= n_iter
                        for m in models)
        if completed and cell.get('mean_diag_recovery_aic', 0) > 0:
            return cell, True

    if cell is None or force:
        cell = {
            'sigma_key':    sigma_key,
            'sigma_label':  slevel['label'],
            'sigma_a_range': list(slevel['sigma_a']),
            'sigma_v_range': list(slevel['sigma_v']),
            'conflict_max': conflict_max,
            'lambda_range': list(LAMBDA_RANGE),
            'pc_range':     list(PC_RANGE),
            'n_iter':       n_iter,
            'nSimul':       nSimul,
            'nStarts':      nStarts,
            'models':       list(models),
            'per_model':    {},
            'confusion_aic': {m: {fm: 0 for fm in models} for m in models},
            'confusion_bic': {m: {fm: 0 for fm in models} for m in models},
            'raw_iters':    {m: [] for m in models},
        }
    else:
        cell.setdefault('raw_iters', {})
        for m in models:
            cell['raw_iters'].setdefault(m, [])

    ranges = build_ranges_for_cell(
        models,
        sigma_a_range=slevel['sigma_a'],
        sigma_v_range=slevel['sigma_v'],
    )

    template = favo.build_synthetic_template(
        conflict_max=conflict_max,
        n_conflict_steps=n_conflict_steps,
        n_trials_per_cell=n_trials_per_cell,
        delta_max_pct=delta_max_pct,
    )

    for gen_model in models:
        already = len(cell['raw_iters'][gen_model])
        if already >= n_iter:
            print(f"    {ABBR.get(gen_model, gen_model[:3])}: cached "
                  f"({already}/{n_iter})")
            continue
        if already:
            print(f"    {ABBR.get(gen_model, gen_model[:3])}: resuming at "
                  f"{already}/{n_iter}")

        remaining_idx = list(range(already, n_iter))
        iargs = [
            (i, gen_model, list(models), template, nSimul, nStarts, ranges)
            for i in remaining_idx
        ]

        if n_jobs > 1:
            # imap_unordered → results trickle in as workers finish.  We
            # checkpoint every `save_every` iterations.
            with Pool(processes=n_jobs) as pool:
                done = 0
                for r in pool.imap_unordered(favo.run_single_recovery, iargs):
                    done += 1
                    if r is not None:
                        cell['raw_iters'][gen_model].append(r)
                    if done % save_every == 0 or done == len(iargs):
                        _recompute_summaries(cell, models)
                        _atomic_write_json(cell_path, cell)
        else:
            for k, a in enumerate(iargs, start=1):
                r = favo.run_single_recovery(a)
                if r is not None:
                    cell['raw_iters'][gen_model].append(r)
                if k % save_every == 0 or k == len(iargs):
                    _recompute_summaries(cell, models)
                    _atomic_write_json(cell_path, cell)

    _recompute_summaries(cell, models)
    _atomic_write_json(cell_path, cell)
    return cell, False


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_confusion(cell, models):
    header = (f"  {'gen':>4}  " +
              "  ".join(f"{ABBR.get(m, m[:3]):>4}" for m in models) +
              "   diag")
    print(header)
    for gen in models:
        row   = cell['confusion_aic'][gen]
        total = sum(row.values())
        diag  = row[gen] / total if total else 0
        vals  = "  ".join(f"{row[m]:>4}" for m in models)
        print(f"  {ABBR.get(gen, gen[:3]):>4}  {vals}   {diag*100:.0f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Model recovery grid — σ-range × conflict_max (8 cells)')
    parser.add_argument('--sigma_levels', nargs='+', default=list(SIGMA_LEVELS.keys()),
                        choices=list(SIGMA_LEVELS.keys()),
                        help='Sigma levels to run (default: a b c d)')
    parser.add_argument('--conflict_levels', nargs='+', type=float, default=[0.25, 0.45],
                        help='conflict_max values in seconds (default: 0.25 0.45)')
    parser.add_argument('--models', nargs='+', default=MODELS_DEFAULT)
    parser.add_argument('--n_iter',            type=int,   default=100,
                        help='Recovery iterations per generating model per cell (default 100)')
    parser.add_argument('--n_conflict_steps',  type=int,   default=9)
    parser.add_argument('--n_trials_per_cell', type=int,   default=20)
    parser.add_argument('--nSimul',            type=int,   default=200,
                        help='MC draws per nLL eval (default 200; was 500). '
                             'Recovery quality plateaus around 200 for these models.')
    parser.add_argument('--nStarts',           type=int,   default=3,
                        help='Optimiser restarts (default 3; was 5).')
    parser.add_argument('--delta_max_pct',     type=float, default=0.90)
    parser.add_argument('--n_jobs',            type=int,   default=None)
    parser.add_argument('--save_dir',          type=str,
                        default='model_recovery_grid_results')
    parser.add_argument('--save_every',        type=int,   default=1,
                        help='Checkpoint to JSON every N completed iterations '
                             '(default 1 = save after every iteration). '
                             'Lower = safer if interrupted, slightly more I/O.')
    parser.add_argument('--force',  action='store_true',
                        help='Re-run cells even if cached JSON exists '
                             '(otherwise resume from where it left off)')
    parser.add_argument('--pilot',  action='store_true',
                        help='Run one tiny cell (level a, cmax=0.45, n_iter=3) and exit')
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)

    if args.pilot:
        print(">> PILOT: σ-level=a  cmax=0.45  n_iter=3  nSimul=100  nStarts=1")
        pilot_models = ['lognorm', 'fusionOnlyLogNorm', 'switchingFree']
        cell, cached = run_single_cell(
            sigma_key='a', conflict_max=0.45,
            models=pilot_models,
            n_iter=3, n_conflict_steps=7, n_trials_per_cell=10,
            nSimul=100, nStarts=1, n_jobs=n_jobs,
            save_dir=args.save_dir, force=args.force,
            save_every=args.save_every,
        )
        print(f"  mean_diag = {cell['mean_diag_recovery_aic']*100:.1f}%")
        print_confusion(cell, pilot_models)
        return

    grid = list(itertools.product(args.sigma_levels, args.conflict_levels))
    n_cells  = len(grid)
    n_models = len(args.models)

    print("=" * 72)
    print("Model recovery grid — σ-range × conflict_max")
    print("=" * 72)
    for key in args.sigma_levels:
        sl = SIGMA_LEVELS[key]
        print(f"  level {key}: σa ~ U{sl['sigma_a']}  σv ~ U{sl['sigma_v']}")
    print(f"  conflict     : {args.conflict_levels} s")
    print(f"  λ range      : {LAMBDA_RANGE}  (freely sampled)")
    print(f"  p_c/p_sw     : {PC_RANGE}  (freely sampled)")
    print(f"  grid         : {n_cells} cells  ({len(args.sigma_levels)} σ-levels × "
          f"{len(args.conflict_levels)} conflict)")
    print(f"  models       : {args.models}")
    print(f"  per cell     : {args.n_iter} iters × {n_models} gen models "
          f"= {n_models * args.n_iter} sims")
    print(f"  nSimul={args.nSimul}  nStarts={args.nStarts}  "
          f"delta_max=±{args.delta_max_pct*100:.0f}%  jobs={n_jobs}")
    print(f"  save_dir     : {args.save_dir}")
    print()

    start   = time.time()
    summary = {'args': vars(args), 'sigma_levels': SIGMA_LEVELS, 'grid': []}

    for k, (sigma_key, cmax) in enumerate(grid, start=1):
        sl = SIGMA_LEVELS[sigma_key]
        print(f"[{k}/{n_cells}] level={sigma_key} ({sl['label']})  cmax={cmax:.2f}")
        cell, cached = run_single_cell(
            sigma_key=sigma_key, conflict_max=cmax,
            models=list(args.models),
            n_iter=args.n_iter,
            n_conflict_steps=args.n_conflict_steps,
            n_trials_per_cell=args.n_trials_per_cell,
            nSimul=args.nSimul, nStarts=args.nStarts, n_jobs=n_jobs,
            save_dir=args.save_dir, force=args.force,
            delta_max_pct=args.delta_max_pct,
            save_every=args.save_every,
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
            'sigma_key':              sigma_key,
            'sigma_label':            sl['label'],
            'sigma_a_range':          list(sl['sigma_a']),
            'sigma_v_range':          list(sl['sigma_v']),
            'conflict_max':           cmax,
            'mean_diag_recovery_aic': diag,
            'per_model_diag_aic':     per_model_diag,
            'confusion_aic':          cell['confusion_aic'],
        })

    elapsed = time.time() - start

    # --- Final summary table ---
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    header = (f"  {'lvl':>4}  {'cmax':>5}  {'mean':>6}  " +
              "  ".join(f"{ABBR.get(m, m[:3]):>5}" for m in args.models))
    print(header)
    for entry in summary['grid']:
        pm   = entry['per_model_diag_aic']
        vals = "  ".join(f"{pm.get(m, 0)*100:5.1f}%" for m in args.models)
        print(f"  {entry['sigma_key']:>4}  {entry['conflict_max']:5.2f}  "
              f"{entry['mean_diag_recovery_aic']*100:5.1f}%  {vals}")

    print(f"\nDone in {elapsed/60:.1f} min")

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, 'grid_sweep_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {out}")


if __name__ == '__main__':
    main()
