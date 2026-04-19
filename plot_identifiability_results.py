#!/usr/bin/env python3
"""
Plot identifiability sweep + divergence diagnostic results
==========================================================
Generates publication-quality figures for the manuscript revision.

Inputs:
  - identifiability_sweep_results/identifiability_sweep_summary.json
  - identifiability_sweep_results/cell_*.json (per-cell confusion matrices)
  - model_divergence_results/divergence_grid.csv (cheap diagnostic)

Outputs (saved next to the inputs):
  - fig_divergence_heatmap.{png,pdf}
  - fig_diagonal_recovery_heatmap.{png,pdf}
  - fig_confusion_matrices.{png,pdf}
  - fig_pse_curves_by_regime.{png,pdf}
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


MODEL_ABBR = {
    'lognorm': 'CI',
    'fusionOnlyLogNorm': 'Fus',
    'switchingFree': 'SwF',
    'probabilityMatchingLogNorm': 'PM',
    'selection': 'Sel',
}


# ---------------------------------------------------------------------------
# Divergence heatmap
# ---------------------------------------------------------------------------

def _heatmap_panel(ax, pivot, title, cb_label, vmax=None,
                    emp_sigma=0.28, emp_conflict=0.25, mark_empirical=True):
    if vmax is None:
        vmax = max(0.15, float(np.nanmax(pivot.values)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis',
                    vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel(r'$\sigma_a = \sigma_v$ (log-space SD)')
    ax.set_ylabel(r'Max conflict $|c|_{max}$ (s)')
    ax.set_title(title, fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 * vmax else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    color=color, fontsize=7)
    if mark_empirical:
        # Snap empirical regime to nearest grid cell
        sig_arr = np.array(pivot.columns, dtype=float)
        cf_arr = np.array(pivot.index, dtype=float)
        col_idx = int(np.argmin(np.abs(sig_arr - emp_sigma)))
        row_idx = int(np.argmin(np.abs(cf_arr - emp_conflict)))
        ax.add_patch(plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                     fill=False, edgecolor='red', lw=2.5))
    return im


def plot_divergence_heatmap(div_csv, out_dir, p_c=0.5):
    df = pd.read_csv(div_csv)
    sub = df[np.isclose(df['p_c'], p_c)].copy()
    if sub.empty:
        print(f"No rows with p_c={p_c}")
        return

    piv_mean = sub.pivot_table(index='conflict_max', columns='sigma_a',
                                values='mean_pairwise',
                                aggfunc='mean').sort_index(ascending=False)
    piv_ci_fus = sub.pivot_table(index='conflict_max', columns='sigma_a',
                                  values='lognorm__vs__fusionOnlyLogNorm',
                                  aggfunc='mean').sort_index(ascending=False)
    piv_ci_pm = sub.pivot_table(index='conflict_max', columns='sigma_a',
                                 values='lognorm__vs__probabilityMatchingLogNorm',
                                 aggfunc='mean').sort_index(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    vmax = max(float(piv_mean.values.max()), float(piv_ci_fus.values.max()),
                float(piv_ci_pm.values.max()))
    im0 = _heatmap_panel(axes[0], piv_mean,
                          f'Mean across all model pairs', None, vmax=vmax)
    _heatmap_panel(axes[1], piv_ci_fus,
                    'CI vs Forced Fusion', None, vmax=vmax)
    _heatmap_panel(axes[2], piv_ci_pm,
                    'CI vs Probability Matching', None, vmax=vmax)

    cb = fig.colorbar(im0, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(r'mean $|p^A - p^B|$ across stim conds')
    fig.suptitle(f'Predicted-behaviour divergence between models  (p_c={p_c}, λ=0.05)\n'
                  f'Red box = empirical regime (σ≈0.5, |c|≤0.25). '
                  f'Higher = more identifiable.',
                  fontsize=11)
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_divergence_heatmap.{ext}')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagonal-recovery heatmap from sweep results
# ---------------------------------------------------------------------------

def plot_diagonal_recovery_heatmap(sweep_dir, out_dir):
    summary_path = os.path.join(sweep_dir, 'identifiability_sweep_summary.json')
    if not os.path.exists(summary_path):
        print(f"  [skip] no sweep summary at {summary_path}")
        return
    with open(summary_path) as f:
        s = json.load(f)
    grid = s['grid']
    if not grid:
        print("  [skip] empty grid")
        return

    df = pd.DataFrame(grid)
    sigmas = sorted(df['sigma_a'].unique())
    cmaxes = sorted(df['conflict_max'].unique(), reverse=True)
    Z = np.zeros((len(cmaxes), len(sigmas)))
    for i, cm in enumerate(cmaxes):
        for j, sg in enumerate(sigmas):
            row = df[(df['sigma_a'] == sg) & (df['conflict_max'] == cm)]
            Z[i, j] = row['mean_diag_recovery_aic'].iloc[0] if len(row) else np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Z * 100, aspect='auto', cmap='RdYlGn',
                    vmin=20, vmax=100)
    ax.set_xticks(np.arange(len(sigmas)))
    ax.set_xticklabels([f"{v:.2f}" for v in sigmas])
    ax.set_yticks(np.arange(len(cmaxes)))
    ax.set_yticklabels([f"{v:.2f}" for v in cmaxes])
    ax.set_xlabel(r'$\sigma_a = \sigma_v$ (log-space SD)')
    ax.set_ylabel('Max conflict (s)')
    ax.set_title('Diagonal model-recovery rate (% true model wins by AIC)')
    for i in range(len(cmaxes)):
        for j in range(len(sigmas)):
            ax.text(j, i, f"{Z[i, j]*100:.0f}%", ha='center', va='center',
                    color='black', fontsize=9, fontweight='bold')
    fig.colorbar(im, ax=ax, label='% diagonal recovery')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_diagonal_recovery_heatmap.{ext}')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-cell confusion matrices grid
# ---------------------------------------------------------------------------

def plot_confusion_grid(sweep_dir, out_dir):
    cells = glob.glob(os.path.join(sweep_dir, 'cell_*.json'))
    if not cells:
        print("  [skip] no cell_*.json files")
        return
    data = []
    for fp in cells:
        with open(fp) as f:
            data.append(json.load(f))

    sigmas = sorted({c['sigma_a'] for c in data})
    cmaxes = sorted({c['conflict_max'] for c in data}, reverse=True)
    nrows, ncols = len(cmaxes), len(sigmas)
    if nrows == 0 or ncols == 0:
        return

    # Use models from the first cell
    models = data[0]['models']
    abbr = [MODEL_ABBR.get(m, m[:3]) for m in models]

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows),
                              squeeze=False)

    for i, cm in enumerate(cmaxes):
        for j, sg in enumerate(sigmas):
            ax = axes[i, j]
            cell = next((c for c in data
                          if np.isclose(c['sigma_a'], sg)
                          and np.isclose(c['conflict_max'], cm)), None)
            if cell is None:
                ax.set_visible(False)
                continue
            cm_aic = cell['confusion_aic']
            mat = np.zeros((len(models), len(models)))
            for ii, gen in enumerate(models):
                row = cm_aic.get(gen, {})
                tot = sum(row.values()) or 1
                for jj, fit in enumerate(models):
                    mat[ii, jj] = row.get(fit, 0) / tot
            im = ax.imshow(mat, vmin=0, vmax=1, cmap='Blues')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(abbr, fontsize=7)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(abbr, fontsize=7)
            for ii in range(len(models)):
                for jj in range(len(models)):
                    if mat[ii, jj] > 0.05:
                        ax.text(jj, ii, f"{mat[ii, jj]*100:.0f}",
                                ha='center', va='center', fontsize=6,
                                color='white' if mat[ii, jj] > 0.5 else 'black')
            ax.set_title(f'σ={sg:.2f}  c={cm:.2f}', fontsize=8)
            if j == 0:
                ax.set_ylabel('Generated by', fontsize=8)
            if i == nrows - 1:
                ax.set_xlabel('Recovered as', fontsize=8)

    fig.suptitle('Confusion matrices across (σ × conflict_max) regimes',
                  fontsize=11, y=1.0)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_confusion_matrices.{ext}')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PSE-vs-conflict curves at three regimes (illustrative)
# ---------------------------------------------------------------------------

def plot_pse_curves_by_regime(out_dir):
    """
    Show predicted PSE-vs-conflict curves for each model in three regimes:
    (i) empirical (sigma=0.5, cmax=0.25), (ii) low-noise (sigma=0.1,
    cmax=0.25), (iii) low-noise + wide conflict (sigma=0.1, cmax=0.6).
    """
    import compute_model_divergence as cmd

    regimes = [
        ('Empirical\n(σ≈0.5, |c|≤0.25)', 0.5, 0.25),
        ('Low noise\n(σ=0.1, |c|≤0.25)', 0.10, 0.25),
        ('Low noise + wide conflict\n(σ=0.1, |c|≤0.6)', 0.10, 0.60),
    ]
    models = cmd.DEFAULT_MODELS
    colors = {'lognorm': 'C0', 'fusionOnlyLogNorm': 'C1',
              'switchingFree': 'C2', 'probabilityMatchingLogNorm': 'C3',
              'selection': 'C4'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (label, sigma, cmax) in zip(axes, regimes):
        template = cmd.build_template(conflict_max=cmax, n_conflict_steps=11,
                                       n_delta_steps=15, n_trials_per_cell=4)
        for m in models:
            params = cmd.build_param_vector(m, sigma_a=sigma, sigma_v=sigma,
                                             p_c=0.5, lambda_=0.05)
            preds = cmd.predicted_p_longer(m, params, template)
            # Compute PSE per conflict using one SNR slice
            preds_snr = preds[np.isclose(preds['audNoise'], 1.2)]
            pse_per_conflict = []
            for c in sorted(preds_snr['conflictDur'].unique()):
                sub = preds_snr[preds_snr['conflictDur'] == c].sort_values('deltaDurS')
                ps = sub['p_longer'].values
                ds = sub['deltaDurS'].values * 1000  # ms
                if ps.min() <= 0.5 <= ps.max():
                    pse = float(np.interp(0.5, ps, ds))
                else:
                    pse = ds[np.argmin(np.abs(ps - 0.5))]
                pse_per_conflict.append((c * 1000, pse))
            arr = np.array(pse_per_conflict)
            ax.plot(arr[:, 0], arr[:, 1], 'o-', color=colors[m],
                    label=MODEL_ABBR.get(m, m[:3]), lw=2, ms=5)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Conflict (ms)')
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('PSE shift (ms)')
    axes[0].legend(fontsize=8, loc='best')

    fig.suptitle('Predicted PSE curves diverge only in low-noise + wide-conflict regimes',
                  fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_pse_curves_by_regime.{ext}')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', default='identifiability_sweep_results')
    parser.add_argument('--div_csv', default='model_divergence_results/divergence_grid.csv')
    parser.add_argument('--out_dir', default='identifiability_figures')
    parser.add_argument('--p_c_for_div', type=float, default=0.5)
    parser.add_argument('--skip_pse', action='store_true',
                        help='Skip the (slower) PSE-curves figure')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.div_csv):
        print("\n[1] Divergence heatmap")
        plot_divergence_heatmap(args.div_csv, args.out_dir, p_c=args.p_c_for_div)
    else:
        print(f"  [skip] no divergence csv at {args.div_csv}")

    print("\n[2] Diagonal recovery heatmap")
    plot_diagonal_recovery_heatmap(args.sweep_dir, args.out_dir)

    print("\n[3] Confusion matrices grid")
    plot_confusion_grid(args.sweep_dir, args.out_dir)

    if not args.skip_pse:
        print("\n[4] PSE-vs-conflict curves by regime")
        plot_pse_curves_by_regime(args.out_dir)


if __name__ == '__main__':
    main()
