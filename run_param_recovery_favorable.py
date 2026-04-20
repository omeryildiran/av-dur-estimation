#!/usr/bin/env python3
"""
Favorable-regime p_c parameter recovery
========================================
Tests whether p_c can be recovered when task conditions are designed to
maximise causal-inference dissociation:

  1. Smaller sensory noise  (σa, σv sampled from [0.05, 0.30])
  2. Wider AV conflict range  (±0.50 s instead of the empirical ±0.25 s)

Motivation
----------
In the empirical data the likelihood ratio already strongly favours one
causal structure over most of the tested conflict range, so different p_c
values produce nearly identical predicted behaviour — p_c is not
identifiable.  Smaller σ sharpens the LR function, moving the transition
zone into the tested conflict range; a wider conflict range ensures the
experiment samples both sides of that transition.  Together these conditions
make p_c influential and therefore recoverable.

If p_c recovery improves markedly here but is poor with the empirical
template, the conclusion is that the experimental design — not the model —
limits p_c identifiability.

Usage
-----
    python run_param_recovery_favorable.py [options]

    --n_recovery N      recovery iterations per model (default 100)
    --n_jobs N          parallel workers (default: n_cpus - 1)
    --nSimul N          Monte Carlo draws per nLL evaluation (default 500)
    --nStarts N         optimiser restarts (default 5)
    --save_dir DIR      output directory (default: model_recovery_results_favorable)
    --models M [M ...]  models to test (default: lognorm probabilityMatchingLogNorm selection)
    --conflict_max C    max absolute conflict in seconds (default: 0.45)
    --sigma_max S       upper bound of σ sampling range (default: 0.30)
    --sigma_min S       lower bound of σ sampling range (default: 0.05)
    --pc_min P          lower bound of p_c sampling range (default: 0.10)
    --pc_max P          upper bound of p_c sampling range (default: 0.90)
"""

import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

import monteCarloClass


# ---------------------------------------------------------------------------
# Favorable parameter ranges (explicit, NOT data-derived)
# ---------------------------------------------------------------------------

# These are the unique (reduced) parameters actually sampled.
# Expanded to the full vector expected by monteCarloClass inside the worker.
FAVORABLE_PARAM_RANGES = {
    # CI models: [λ, σa, σv, p_c]
    'lognorm': {
        'lambda':  (0.001, 0.15),
        'sigma_a': None,          # filled from CLI args
        'sigma_v': None,
        'p_c':     None,
    },
    'probabilityMatchingLogNorm': {
        'lambda':  (0.001, 0.15),
        'sigma_a': None,
        'sigma_v': None,
        'p_c':     None,
    },
    'selection': {
        'lambda':  (0.001, 0.15),
        'sigma_a': None,
        'sigma_v': None,
        'p_c':     None,
    },
    # Fusion / switching kept for optional confusion matrix
    'fusionOnlyLogNorm': {
        'lambda':  (0.001, 0.15),
        'sigma_a': None,
        'sigma_v': None,
    },
    'switchingFree': {
        'lambda':  (0.001, 0.15),
        'sigma_a': None,
        'sigma_v': None,
        'p_sw':    (0.10, 0.90),
    },
}

PARAM_NAMES_UNIQUE = {
    'fusionOnlyLogNorm':         ['λ', 'σa', 'σv'],
    'lognorm':                   ['λ', 'σa', 'σv', 'p_c'],
    'probabilityMatchingLogNorm':['λ', 'σa', 'σv', 'p_c'],
    'selection':                 ['λ', 'σa', 'σv', 'p_c'],
    'switchingFree':             ['λ', 'σa', 'σv', 'p_sw'],
}

FITTING_BOUNDS = {
    'fusionOnlyLogNorm': [(0.001, 0.4), (0.01, 2.0), (0.01, 2.0), (0.01, 2.0)],
    'lognorm':           [(0.001, 0.4), (0.0001, 2.0), (0.0001, 2.0), (0.0, 1.0), (0.0001, 2.0)],
    'probabilityMatchingLogNorm': [(0.001, 0.4), (0.0001, 2.0), (0.0001, 2.0), (0.0, 1.0), (0.0001, 2.0)],
    'selection':         [(0.001, 0.4), (0.0001, 2.0), (0.0001, 2.0), (0.0, 1.0), (0.0001, 2.0)],
    'switchingFree':     [(0.001, 0.4), (0.0001, 2.0), (0.0001, 2.0), (0.0, 1.0), (0.0001, 2.0), (0.0, 1.0)],
}


# ---------------------------------------------------------------------------
# Synthetic template builder
# ---------------------------------------------------------------------------

def build_synthetic_template(conflict_max=0.50,
                              n_conflict_steps=9,
                              n_delta_steps=9,
                              standard_dur=0.50,
                              noise_levels=(0.1, 1.2),
                              n_trials_per_cell=20,
                              delta_max_pct=0.80):
    """
    Build a synthetic pandas DataFrame that mimics the structure of a real
    participant's data file but with:
      - A wider, symmetric conflict range  [-conflict_max, +conflict_max]
      - Two auditory noise levels (matching the empirical design)
      - Balanced ΔDur levels spanning ±delta_max_pct of the standard duration
        (default 0.80 = ±80%, matching empirical staircase range of ~±90%)

    Returns
    -------
    pd.DataFrame  — ready to be passed directly to OmerMonteCarlo()
    """
    delta_durs = np.linspace(-delta_max_pct * standard_dur,
                              delta_max_pct * standard_dur,
                              n_delta_steps)
    conflicts  = np.linspace(-conflict_max, conflict_max, n_conflict_steps)

    rows = []
    for noise in noise_levels:
        for conflict in conflicts:
            s_v_s = standard_dur + conflict
            if s_v_s <= 0:
                continue                # skip physically impossible visual durations
            for delta in delta_durs:
                test_dur = standard_dur + delta
                if test_dur <= 0:
                    continue
                s_v_t = test_dur        # visual test = audio test (no extra conflict on test)
                for _ in range(n_trials_per_cell):
                    # Placeholder binary choice — actual behaviour is regenerated
                    # by simulateMonteCarloData, so only the trial *structure*
                    # matters here.
                    chose = int(np.random.rand() > 0.5)
                    rows.append({
                        'standardDur':               standard_dur,
                        'testDurS':                  round(test_dur, 5),
                        'deltaDurS':                 round(delta, 5),
                        'logDeltaDur':               np.log(test_dur) - np.log(standard_dur),
                        'logDeltaDurMs':             np.log(test_dur * 1000) - np.log(standard_dur * 1000),
                        'audNoise':                  noise,
                        'conflictDur':               round(conflict, 4),
                        'unbiasedVisualStandardDur': round(s_v_s, 5),
                        'unbiasedVisualTestDur':      round(s_v_t, 5),
                        'chose_test':                chose,
                        'chose_standard':            1 - chose,
                        'responses':                 1,
                        # dummy columns that loadData / monteCarloClass may access
                        'recordedDurVisualStandard': round(s_v_s * 1000, 2),
                        'recordedDurVisualTest':     round(s_v_t * 1000, 2),
                        'VisualPSE':                 0.0,
                        'visualPSEBias':             0.0,
                        'visualPSEBiasTest':         0.0,
                        'riseDur':                   1,
                        'delta_dur_percents':        round(delta / standard_dur * 100, 2),
                        'standard_dur':              standard_dur,
                        'logStandardDur':            np.log(standard_dur),
                        'logTestDur':                np.log(test_dur),
                        'logConflictDur':            np.log(abs(conflict)) if conflict != 0 else 0.0,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def build_ranges(models, sigma_min, sigma_max, pc_min, pc_max):
    """Return a dict {model: [(lo, hi), ...]} of per-model unique-param ranges."""
    sigma_range = (sigma_min, sigma_max)
    pc_range    = (pc_min,    pc_max)
    ranges = {}
    for m in models:
        if m not in FAVORABLE_PARAM_RANGES:
            continue
        base = FAVORABLE_PARAM_RANGES[m]
        r = []
        for key in ['lambda', 'sigma_a', 'sigma_v', 'p_c', 'p_sw']:
            if key not in base:
                continue
            if key in ('sigma_a', 'sigma_v'):
                r.append(sigma_range)
            elif key == 'p_c':
                r.append(pc_range)
            elif key == 'p_sw':
                r.append(base.get('p_sw', pc_range))
            else:
                r.append(base['lambda'])
        ranges[m] = r
    return ranges


def sample_params(model_name, ranges, rng):
    """
    Sample unique params and expand to the full vector for monteCarloClass.

    Full-vector layouts (sharedLambda=True):
      fusionOnlyLogNorm : [λ, σa, σv, σa]
      CI models         : [λ, σa, σv, p_c, σa]
      switchingFree     : [λ, σa, σv, p_sw, σa, p_sw]
    """
    r = ranges[model_name]
    unique = np.array([rng.uniform(lo, hi) for lo, hi in r])

    if model_name == 'fusionOnlyLogNorm':
        lam, sa, sv = unique
        full = np.array([lam, sa, sv, sa])
    elif model_name == 'switchingFree':
        lam, sa, sv, psw = unique
        full = np.array([lam, sa, sv, psw, sa, psw])
    else:
        lam, sa, sv, pc = unique
        full = np.array([lam, sa, sv, pc, sa])

    return unique, full


# ---------------------------------------------------------------------------
# Boundary-clip diagnostics
# ---------------------------------------------------------------------------

def count_boundary_clips(fitted_params, model_name, tol=0.01):
    bounds = FITTING_BOUNDS[model_name]
    param_name_map = {
        'fusionOnlyLogNorm':         ['λ', 'σa1', 'σv', 'σa2'],
        'lognorm':                   ['λ', 'σa1', 'σv', 'p_c', 'σa2'],
        'probabilityMatchingLogNorm':['λ', 'σa1', 'σv', 'p_c', 'σa2'],
        'selection':                 ['λ', 'σa1', 'σv', 'p_c', 'σa2'],
        'switchingFree':             ['λ', 'σa1', 'σv', 'p_sw1', 'σa2', 'p_sw2'],
    }
    names = param_name_map.get(model_name, [f'p{i}' for i in range(len(fitted_params))])
    n_lower, n_upper, details = 0, 0, []
    for val, (lo, hi), name in zip(fitted_params, bounds, names):
        if abs(val - lo) <= tol * (hi - lo):
            n_lower += 1
            details.append(f"{name}@lower")
        if abs(val - hi) <= tol * (hi - lo):
            n_upper += 1
            details.append(f"{name}@upper")
    return {'n_lower': n_lower, 'n_upper': n_upper, 'details': details}


# ---------------------------------------------------------------------------
# Single recovery worker
# ---------------------------------------------------------------------------

def run_single_recovery(args):
    """
    One recovery iteration: sample → simulate → fit all models → return results.
    Designed to run in a multiprocessing Pool.
    """
    (iteration_idx, generating_model, models_to_test,
     template_data, nSimul, nStarts, ranges) = args

    rng = np.random.default_rng()

    # 1. Sample ground-truth parameters
    sampled_unique, sampled_full = sample_params(generating_model, ranges, rng)

    # 2. Simulate data
    mc_gen = monteCarloClass.OmerMonteCarlo(template_data)
    mc_gen.modelName    = generating_model
    mc_gen.freeP_c      = False
    mc_gen.sharedLambda = True
    mc_gen.nSimul       = nSimul
    mc_gen.nStart       = nStarts
    mc_gen.optimizationMethod = 'bads'

    try:
        sim_data = mc_gen.simulateMonteCarloData(sampled_full, template_data)
    except Exception:
        return None

    # 3. Fit all competing models to the simulated data
    model_fits = {}
    for fit_model in models_to_test:
        mc_fit = monteCarloClass.OmerMonteCarlo(sim_data)
        mc_fit.modelName    = fit_model
        mc_fit.freeP_c      = False
        mc_fit.sharedLambda = True
        mc_fit.nSimul       = nSimul
        mc_fit.nStart       = nStarts
        mc_fit.optimizationMethod = 'bads'

        try:
            fp = mc_fit.fitCausalInferenceMonteCarlo(mc_fit.groupedData)
            if fp is not None:
                nLL = mc_fit.nLLMonteCarloCausal(fp, mc_fit.groupedData)
                LL  = -nLL
                k   = len(fp)
                AIC = 2 * k - 2 * LL
                BIC = k * np.log(len(sim_data)) - 2 * LL
                clips = count_boundary_clips(fp, fit_model)
                model_fits[fit_model] = {
                    'fittedParams':   fp.tolist(),
                    'logLikelihood':  float(LL),
                    'AIC':            float(AIC),
                    'BIC':            float(BIC),
                    'nParams':        k,
                    'boundary_clips': clips,
                }
        except Exception:
            continue

    if not model_fits:
        return None

    best_aic = min(model_fits, key=lambda m: model_fits[m]['AIC'])
    best_bic = min(model_fits, key=lambda m: model_fits[m]['BIC'])

    return {
        'iteration':       iteration_idx,
        'sampled_unique':  sampled_unique.tolist(),
        'sampled_full':    sampled_full.tolist(),
        'model_fits':      model_fits,
        'best_model_aic':  best_aic,
        'best_model_bic':  best_bic,
    }


# ---------------------------------------------------------------------------
# Per-generating-model driver
# ---------------------------------------------------------------------------

def run_recovery_for_model(generating_model, models_to_test,
                            template_data, ranges,
                            n_recovery=100, nSimul=500, nStarts=5,
                            save_dir='model_recovery_results_favorable',
                            n_jobs=1):
    result_path = os.path.join(save_dir, f"favorable_{generating_model}_recovery.json")
    if os.path.exists(result_path):
        print(f"  [cached] {generating_model}")
        with open(result_path) as f:
            return json.load(f)

    iargs = [
        (i, generating_model, models_to_test,
         template_data, nSimul, nStarts, ranges)
        for i in range(n_recovery)
    ]

    if n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            raw = list(tqdm(
                pool.imap(run_single_recovery, iargs),
                total=n_recovery,
                desc=f"  {generating_model:<34}",
                bar_format='{l_bar}{bar:30}{r_bar}',
            ))
    else:
        raw = [run_single_recovery(a)
               for a in tqdm(iargs, desc=f"  {generating_model:<34}",
                              bar_format='{l_bar}{bar:30}{r_bar}')]

    iters = [r for r in raw if r is not None]
    if not iters:
        return None

    names_unique = PARAM_NAMES_UNIQUE[generating_model]

    # Map unique-param index → full-vector index for same-model recovery
    if generating_model == 'fusionOnlyLogNorm':
        u2f = [0, 1, 2]
    elif generating_model == 'switchingFree':
        u2f = [0, 1, 2, 3]
    else:
        u2f = [0, 1, 2, 3]   # λ, σa1, σv, p_c

    result = {
        'generating_model': generating_model,
        'n_iterations':     len(iters),
        'param_ranges':     {nm: list(ranges[generating_model][uid])
                             for uid, nm in enumerate(names_unique)},
        'iterations':       iters,
        'best_model_counts_aic': {},
        'best_model_counts_bic': {},
    }

    for m in models_to_test:
        result['best_model_counts_aic'][m] = sum(
            1 for it in iters if it['best_model_aic'] == m)
        result['best_model_counts_bic'][m] = sum(
            1 for it in iters if it['best_model_bic'] == m)

    # --- Parameter recovery statistics (same-model fits) ---
    same = [it for it in iters if generating_model in it['model_fits']]
    if same:
        gen_arr = np.array([it['sampled_unique'] for it in same])
        rec_arr = np.array([it['model_fits'][generating_model]['fittedParams']
                            for it in same])

        param_recovery = {}
        for uid, nm in enumerate(names_unique):
            fid = u2f[uid]
            gv = gen_arr[:, uid]
            rv = rec_arr[:, fid]
            corr = float(np.corrcoef(gv, rv)[0, 1]) if len(gv) > 1 else float('nan')
            rmse = float(np.sqrt(np.mean((gv - rv) ** 2)))
            bias = float(np.mean(rv - gv))
            param_recovery[nm] = {'correlation': corr, 'rmse': rmse, 'bias': bias,
                                   'gen_values': gv.tolist(), 'rec_values': rv.tolist()}
        result['param_recovery'] = param_recovery

        # Boundary-clip summary
        lo_tot = hi_tot = 0
        clip_counter = {}
        for it in same:
            clip = it['model_fits'][generating_model].get('boundary_clips', {})
            lo_tot += clip.get('n_lower', 0)
            hi_tot += clip.get('n_upper', 0)
            for d in clip.get('details', []):
                clip_counter[d] = clip_counter.get(d, 0) + 1
        result['boundary_clip_summary'] = {
            'total_lower': lo_tot, 'total_upper': hi_tot,
            'total_iterations': len(same), 'clipped_params': clip_counter,
        }

    os.makedirs(save_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='p_c parameter recovery under favorable (low-σ, wide-conflict) conditions')
    parser.add_argument('--models', nargs='+',
                        default=['lognorm', 'probabilityMatchingLogNorm', 'selection',
                                 'fusionOnlyLogNorm', 'switchingFree'],
                        help='Models to generate and recover')
    parser.add_argument('--n_recovery', type=int, default=100)
    parser.add_argument('--nSimul',     type=int, default=500)
    parser.add_argument('--nStarts',    type=int, default=5)
    parser.add_argument('--n_jobs',     type=int, default=None)
    parser.add_argument('--save_dir',   type=str,
                        default='model_recovery_results_favorable')
    # Favorable condition knobs
    parser.add_argument('--conflict_max', type=float, default=0.45,
                        help='Half-range of AV conflict in seconds (default 0.45). '
                             'Empirical range is ~0.25 s. Note: levels where '
                             'standard + conflict ≤ 0 are automatically excluded.')
    parser.add_argument('--n_conflict_steps', type=int, default=9,
                        help='Number of conflict levels (default 9)')
    parser.add_argument('--sigma_min', type=float, default=0.05,
                        help='Lower bound of σa/σv sampling range (default 0.05)')
    parser.add_argument('--sigma_max', type=float, default=0.30,
                        help='Upper bound of σa/σv sampling range (default 0.30)')
    parser.add_argument('--pc_min', type=float, default=0.10,
                        help='Lower bound of p_c sampling range (default 0.10)')
    parser.add_argument('--pc_max', type=float, default=0.90,
                        help='Upper bound of p_c sampling range (default 0.90)')
    parser.add_argument('--n_trials_per_cell', type=int, default=20,
                        help='Simulated trials per (conflict × noise × Δdur) cell (default 20)')
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)

    print("=" * 70)
    print("Favorable-regime p_c parameter recovery")
    print("=" * 70)
    print(f"  conflict range : ±{args.conflict_max:.2f} s  "
          f"({args.n_conflict_steps} levels)   [empirical: ±0.25 s, 7 levels]")
    print(f"  σ range        : [{args.sigma_min:.2f}, {args.sigma_max:.2f}]"
          f"                     [empirical σ mean ≈ 0.53]")
    print(f"  p_c range      : [{args.pc_min:.2f}, {args.pc_max:.2f}]")
    print(f"  n_recovery     : {args.n_recovery}  |  "
          f"nSimul={args.nSimul}  nStarts={args.nStarts}  jobs={n_jobs}")
    print()

    # 1. Build ranges
    ranges = build_ranges(args.models, args.sigma_min, args.sigma_max,
                          args.pc_min, args.pc_max)

    # 2. Build synthetic template
    template = build_synthetic_template(
        conflict_max=args.conflict_max,
        n_conflict_steps=args.n_conflict_steps,
        n_trials_per_cell=args.n_trials_per_cell,
    )
    print(f"  Synthetic template: {len(template)} rows  |  "
          f"conflict levels: {sorted(template['conflictDur'].unique())}")
    print(f"  noise levels: {sorted(template['audNoise'].unique())}")
    print()

    start = time.time()
    all_results = []
    for gen_model in args.models:
        if gen_model not in ranges:
            print(f"  Skipping {gen_model} (not in ranges)")
            continue
        res = run_recovery_for_model(
            generating_model=gen_model,
            models_to_test=args.models,
            template_data=template,
            ranges=ranges,
            n_recovery=args.n_recovery,
            nSimul=args.nSimul,
            nStarts=args.nStarts,
            save_dir=args.save_dir,
            n_jobs=n_jobs,
        )
        if res is not None:
            all_results.append(res)

    elapsed = time.time() - start

    # ================================================================
    # Summary
    # ================================================================
    print(f"\nDone in {elapsed / 60:.1f} min — results in '{args.save_dir}/'")

    abbr = {'lognorm': 'CI', 'fusionOnlyLogNorm': 'Fus',
            'switchingFree': 'SwF', 'probabilityMatchingLogNorm': 'PM',
            'selection': 'Sel'}

    if all_results:
        # Confusion matrix (AIC)
        header = (f"{'Gen':<12}" +
                  "".join(f"{abbr.get(m, m[:4]):>6}" for m in args.models) +
                  f"{'%OK':>7}")
        print(f"\nConfusion (AIC):\n{header}")
        for res in all_results:
            gen   = res['generating_model']
            total = res['n_iterations']
            row   = f"{abbr.get(gen, gen[:4]):<12}"
            correct = 0
            for m in args.models:
                cnt = res['best_model_counts_aic'].get(m, 0)
                if m == gen:
                    correct = cnt
                row += f"{cnt / total * 100 if total else 0:>5.0f}%"
            row += f"{correct / total * 100 if total else 0:>6.0f}%"
            print(row)

        # Parameter recovery table
        print(f"\n{'Model':<12} {'Param':<6} {'r':>6} {'RMSE':>7} {'Bias':>7}")
        for res in all_results:
            gen = res['generating_model']
            for pname, s in res.get('param_recovery', {}).items():
                flag = ''
                if pname == 'p_c':
                    r_val = s['correlation']
                    flag  = '  *** p_c recovery ***' if r_val > 0.5 else '  (low r)'
                print(f"{abbr.get(gen, gen[:4]):<12} {pname:<6} "
                      f"{s['correlation']:>6.3f} {s['rmse']:>7.4f} "
                      f"{s['bias']:>7.4f}{flag}")

        # Boundary-clip summary
        print(f"\nBoundary clips:")
        for res in all_results:
            bc  = res.get('boundary_clip_summary', {})
            lo  = bc.get('total_lower', 0)
            hi  = bc.get('total_upper', 0)
            top = sorted(bc.get('clipped_params', {}).items(),
                         key=lambda x: -x[1])[:3]
            detail = ", ".join(f"{k}:{v}" for k, v in top) or "none"
            print(f"  {abbr.get(res['generating_model'], '?'):<5}  "
                  f"lo={lo}  hi={hi}  {detail}")

        # Save consolidated summary
        summary = {
            'args': vars(args),
            'conflict_levels': sorted(template['conflictDur'].unique().tolist()),
            'noise_levels':    sorted(template['audNoise'].unique().tolist()),
            'param_ranges':    ranges,
            'results_by_model': {
                r['generating_model']: {
                    'n_iterations': r['n_iterations'],
                    'param_recovery': r.get('param_recovery', {}),
                    'best_model_counts_aic': r['best_model_counts_aic'],
                }
                for r in all_results
            },
        }
        os.makedirs(args.save_dir, exist_ok=True)
        summary_path = os.path.join(args.save_dir, 'favorable_recovery_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved → {summary_path}")


if __name__ == '__main__':
    main()
