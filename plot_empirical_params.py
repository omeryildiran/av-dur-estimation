#!/usr/bin/env python3
"""Quick script: plot per-participant empirical parameters (σ_a1, σ_a2, σ_v, p_c).

Place this at the repo root and run: `python plot_empirical_params.py`.
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FITS_DIR = 'model_fits'
MODELS = ['lognorm', 'fusionOnlyLogNorm', 'switchingFree',
          'probabilityMatchingLogNorm', 'selection']
PARAM_IDX = {
    'lambda': {m: 0 for m in MODELS},
    'sigma_a1': {m: 1 for m in MODELS},
    'sigma_v':  {m: 2 for m in MODELS},
    'sigma_a2': {'lognorm': 4, 'fusionOnlyLogNorm': 3,
                 'switchingFree': 4, 'probabilityMatchingLogNorm': 4,
                 'selection': 4},
    'p_c':      {'lognorm': 3, 'probabilityMatchingLogNorm': 3, 'selection': 3},
    'p_sw1':    {'switchingFree': 3},
}


def load_fits():
    if not os.path.isdir(FITS_DIR):
        return pd.DataFrame()
    pids = sorted([d for d in os.listdir(FITS_DIR)
                   if os.path.isdir(os.path.join(FITS_DIR, d))
                   and d not in ['.DS_Store', 'all']])
    rows = []
    for pid in pids:
        for model in MODELS:
            suffix = f'{model}_LapseFix_sharedPrior'
            fp = os.path.join(FITS_DIR, pid, f'{pid}_{suffix}_fit.json')
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                r = json.load(f)
            p = r.get('fittedParams', [])
            if not p:
                continue
            row = dict(pid=pid, model=model,
                       AIC=r.get('AIC', np.nan), BIC=r.get('BIC', np.nan), LL=r.get('logLikelihood', np.nan),
                       nParams=len(p),
                       lam=p[0], sigma_a1=p[1], sigma_v=p[2],
                       sigma_a2=p[PARAM_IDX['sigma_a2'][model]] if model in PARAM_IDX['sigma_a2'] else np.nan,
                       sigma_a_mean=np.mean([p[1], p[PARAM_IDX['sigma_a2'][model]]]) if model in PARAM_IDX['sigma_a2'] else p[1],
                       p_c=p[3] if model in ['lognorm','probabilityMatchingLogNorm','selection'] else np.nan,
                       p_sw=p[3] if model == 'switchingFree' else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_empirical_params():
    df = load_fits()
    if df.empty:
        print(f'No fit files found in {FITS_DIR}/ — ensure fits exist.')
        return

    # Use CI (`lognorm`) as reference for σ estimates
    ci_df = df[df['model'] == 'lognorm'].copy().sort_values('pid')
    pids = ci_df['pid'].values
    x = np.arange(len(ci_df))

    # Single composite plot: each participant as a point per parameter and group mean
    params = ['sigma_a1', 'sigma_a2', 'sigma_v', 'p_c']
    x = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(8, 5))

    # per-participant points (from lognorm/CI fits)
    for pid in pids:
        row = ci_df[ci_df['pid'] == pid]
        if row.empty:
            continue
        yvals = [
            row['sigma_a1'].values[0] if not np.isnan(row['sigma_a1'].values[0]) else np.nan,
            row['sigma_a2'].values[0] if not np.isnan(row['sigma_a2'].values[0]) else np.nan,
            row['sigma_v'].values[0] if not np.isnan(row['sigma_v'].values[0]) else np.nan,
            row['p_c'].values[0]    if not np.isnan(row['p_c'].values[0])    else np.nan,
        ]
        jitter = (np.random.rand(len(x)) - 0.5) * 0.12
        ax.scatter(x + jitter, yvals, color='#377eb8', alpha=0.8, s=28)

    # group means (ignore NaNs)
    means = [ci_df['sigma_a1'].mean(), ci_df['sigma_a2'].mean(), ci_df['sigma_v'].mean(), ci_df['p_c'].mean()]
    ax.scatter(x, means, color='#e41a1c', s=120, marker='D', label='group mean')

    ax.set_xticks(x)
    ax.set_xticklabels(['σ_a1', 'σ_a2', 'σ_v', 'p_c'])
    ax.set_ylabel('Parameter value')
    ax.set_title('Per-participant parameter estimates (points) and group mean')
    ax.legend()

    plt.tight_layout()
    os.makedirs('identifiability_figures', exist_ok=True)
    out = os.path.join('identifiability_figures', 'empirical_params.png')
    plt.savefig(out, bbox_inches='tight', dpi=200)
    print(f'Saved plot to {out}')
    plt.show()


if __name__ == '__main__':
    plot_empirical_params()
