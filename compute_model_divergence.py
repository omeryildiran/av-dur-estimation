#!/usr/bin/env python3
"""
Model-prediction divergence diagnostic (no fitting)
===================================================
For each (sigma_a, sigma_v, conflict_max) cell, compute:

    div(A,B) = mean over (delta, conflict) of
                | p_test_longer^A(stim; theta) - p_test_longer^B(stim; theta) |

across all model pairs A != B with shared parameters.  Higher divergence
means the models predict more different psychometric curves -> more
identifiable in principle.

This is much cheaper than full model recovery (no optimisation, no
binomial sampling), so we can densely sweep the parameter space and
get a smooth identifiability landscape.

Usage:
    python compute_model_divergence.py
    python compute_model_divergence.py --sigma_levels 0.05 0.1 0.2 0.3 0.5 0.8
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import monteCarloClass


# Models we evaluate divergences between
DEFAULT_MODELS = [
    'lognorm',                       # full causal inference
    'fusionOnlyLogNorm',             # forced fusion (p_c=1)
    'switchingFree',                 # free random switching
    'probabilityMatchingLogNorm',    # probability matching
    'selection',                     # MAP selection
]


def build_param_vector(model_name, sigma_a, sigma_v, p_c, lambda_):
    """Construct the full parameter vector expected by monteCarloClass."""
    if model_name == 'fusionOnlyLogNorm':
        return np.array([lambda_, sigma_a, sigma_v, sigma_a])
    if model_name == 'switchingFree':
        return np.array([lambda_, sigma_a, sigma_v, p_c, sigma_a, p_c])
    return np.array([lambda_, sigma_a, sigma_v, p_c, sigma_a])


def build_template(conflict_max, n_conflict_steps=9, n_delta_steps=11,
                   standard_dur=0.5, n_trials_per_cell=10, delta_max_pct=0.90):
    """Synthetic balanced design at the requested conflict range."""
    deltas = np.linspace(-delta_max_pct * standard_dur, delta_max_pct * standard_dur, n_delta_steps)
    conflicts = np.linspace(-conflict_max, conflict_max, n_conflict_steps)
    rows = []
    for noise in [0.1, 1.2]:                      # two SNR levels (matched in sim)
        for c in conflicts:
            sv_s = standard_dur + c
            if sv_s <= 0:
                continue
            for d in deltas:
                t = standard_dur + d
                if t <= 0:
                    continue
                for _ in range(n_trials_per_cell):
                    rows.append({
                        'standardDur': standard_dur,
                        'testDurS': round(t, 5),
                        'deltaDurS': round(d, 5),
                        'logDeltaDur': np.log(t) - np.log(standard_dur),
                        'logDeltaDurMs': np.log(t * 1000) - np.log(standard_dur * 1000),
                        'audNoise': noise,
                        'conflictDur': round(c, 4),
                        'unbiasedVisualStandardDur': round(sv_s, 5),
                        'unbiasedVisualTestDur': round(t, 5),
                        'chose_test': 0, 'chose_standard': 1, 'responses': 1,
                        'recordedDurVisualStandard': round(sv_s * 1000, 2),
                        'recordedDurVisualTest': round(t * 1000, 2),
                        'VisualPSE': 0.0, 'visualPSEBias': 0.0,
                        'visualPSEBiasTest': 0.0, 'riseDur': 1,
                        'delta_dur_percents': round(d / standard_dur * 100, 2),
                        'standard_dur': standard_dur,
                        'logStandardDur': np.log(standard_dur),
                        'logTestDur': np.log(t),
                        'logConflictDur': np.log(abs(c)) if c != 0 else 0.0,
                    })
    return pd.DataFrame(rows)


def predicted_p_longer(model_name, params, template):
    """
    For each unique (delta, conflict) cell, return the model-predicted
    p(test_longer) using monteCarloClass's vectorised predictor.
    """
    mc = monteCarloClass.OmerMonteCarlo(template)
    mc.modelName = model_name
    mc.freeP_c = False
    mc.sharedLambda = True
    mc.nSimul = 2000

    # Use one SNR slice (collapse across noise levels for the diagnostic)
    rows = mc.groupedData
    preds = []
    for _, trial in rows.iterrows():
        delta = trial['deltaDurS']
        snr = trial['audNoise']
        conflict = trial['conflictDur']
        pres = mc.getParamsCausal(params, snr, conflict)
        if model_name == 'switchingFree':
            lam, sa, sv, psw, tmin, tmax = pres
            p_c = psw
        elif model_name in ['fusionOnly', 'fusionOnlyLogNorm']:
            lam, sa, sv, p_c, tmin, tmax = pres
        else:
            lam, sa, sv, p_c, tmin, tmax = pres

        S_a_s = 0.5
        S_v_s = S_a_s + conflict
        S_a_t = S_a_s + delta
        S_v_t = S_a_t
        trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
        p = mc.probTestLonger_vectorized_mc(trueStims, sa, sv, p_c, lam, tmin, tmax)
        preds.append({
            'deltaDurS': delta, 'audNoise': snr, 'conflictDur': conflict,
            'p_longer': float(p),
        })
    return pd.DataFrame(preds)


def divergence_matrix(models, params_by_model, template):
    """Return a DataFrame with mean |Delta p| across stim conds for every model pair."""
    preds = {m: predicted_p_longer(m, params_by_model[m], template) for m in models}
    keys = ['deltaDurS', 'audNoise', 'conflictDur']
    div = {}
    for a, b in itertools.combinations(models, 2):
        merged = preds[a].merge(preds[b], on=keys, suffixes=('_a', '_b'))
        d = float(np.mean(np.abs(merged['p_longer_a'] - merged['p_longer_b'])))
        div[f"{a}__vs__{b}"] = d
    div['mean_pairwise'] = float(np.mean(list(div.values())))
    return div, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma_levels', nargs='+', type=float,
                        default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7])
    parser.add_argument('--conflict_levels', nargs='+', type=float,
                        default=[0.083, 0.167, 0.25, 0.35, 0.45])
    parser.add_argument('--p_c_levels', nargs='+', type=float,
                        default=[0.3, 0.5, 0.7])
    parser.add_argument('--lambda_fixed', type=float, default=0.05)
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--save_dir', type=str, default='model_divergence_results')
    parser.add_argument('--n_conflict_steps', type=int, default=9)
    parser.add_argument('--n_delta_steps', type=int, default=11)
    parser.add_argument('--n_trials_per_cell', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    grid = list(itertools.product(args.sigma_levels, args.conflict_levels,
                                   args.p_c_levels))
    print(f"Computing divergences for {len(grid)} cells "
          f"({len(args.sigma_levels)} sigma x {len(args.conflict_levels)} "
          f"conflict x {len(args.p_c_levels)} p_c)")

    rows = []
    start = time.time()
    for k, (sigma, cmax, pc) in enumerate(grid, 1):
        params_by_model = {
            m: build_param_vector(m, sigma_a=sigma, sigma_v=sigma,
                                  p_c=pc, lambda_=args.lambda_fixed)
            for m in args.models
        }
        template = build_template(conflict_max=cmax,
                                   n_conflict_steps=args.n_conflict_steps,
                                   n_delta_steps=args.n_delta_steps,
                                   n_trials_per_cell=args.n_trials_per_cell)
        div, _ = divergence_matrix(args.models, params_by_model, template)

        row = {'sigma_a': sigma, 'sigma_v': sigma,
               'conflict_max': cmax, 'p_c': pc,
               'lambda': args.lambda_fixed, **div}
        rows.append(row)
        elapsed = time.time() - start
        print(f"[{k:3d}/{len(grid)}] sigma={sigma:.2f} cmax={cmax:.2f} "
              f"pc={pc:.2f}  mean_div={div['mean_pairwise']:.3f}  "
              f"({elapsed:.1f}s)")

    out = pd.DataFrame(rows)
    csv_path = os.path.join(args.save_dir, 'divergence_grid.csv')
    out.to_csv(csv_path, index=False)
    json_path = os.path.join(args.save_dir, 'divergence_grid.json')
    out.to_json(json_path, orient='records', indent=2)
    print(f"\nSaved {csv_path}")


if __name__ == '__main__':
    main()
