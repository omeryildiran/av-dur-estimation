"""Fitted switching probabilities p_v (toward vision) per auditory-noise condition.

Mechanistic figure for the cue-switching account: each participant's fitted
probability of relying on vision rises with auditory noise. Source: boxcar
switchingFree fits (model_fits/boxcarFits/, LapseFix sharedPrior). Saves to the
repo root and ms_latex/assets/figures/.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

PIDS = ['as', 'dt', 'hh', 'ip', 'ln2', 'mh', 'ml', 'mt', 'oy', 'qs', 'sx']
MODEL_FITS_DIR = Path('model_fits/boxcarFits')
PARTICIPANT_COLORS = {
    pid: plt.get_cmap('tab20', len(PIDS))(i) for i, pid in enumerate(PIDS)
}

# raw switchingFree param order: [lambda, sigma_a_l, sigma_v, p_v_l, sigma_a_h, p_v_h]
# (display reorder [0,1,4,2,3,5] -> [lambda, sa_l, sa_h, sv, p_v_l, p_v_h]); p_v_l=raw[3], p_v_h=raw[5]
rows = []
for pid in PIDS:
    fit_path = MODEL_FITS_DIR / pid / f'{pid}_switchingFree_LapseFix_sharedPrior_fit.json'
    with fit_path.open() as fh:
        d = json.load(fh)
    p = d['fittedParams']
    rows.append((pid, float(p[3]), float(p[5])))   # p_v_low, p_v_high

pids = [r[0] for r in rows]
pvl = np.array([r[1] for r in rows]); pvh = np.array([r[2] for r in rows])
n = len(rows)
stat, pval = wilcoxon(pvh, pvl)

fig, ax = plt.subplots(figsize=(4.6, 5.0))
x0, x1 = 0, 1
rng = np.random.default_rng(3)
jit = rng.uniform(-0.04, 0.04, size=n)

# per-participant connecting lines
for i, pid in enumerate(pids):
    color = PARTICIPANT_COLORS[pid]
    ax.plot([x0 + jit[i], x1 + jit[i]], [pvl[i], pvh[i]], '-',
            color=color, lw=1.3, alpha=0.75, zorder=1)
    ax.scatter([x0 + jit[i]], [pvl[i]], s=58, color=color,
               edgecolor='black', linewidth=0.7, zorder=3)
    ax.scatter([x1 + jit[i]], [pvh[i]], s=58, color=color,
               edgecolor='black', linewidth=0.7, zorder=3)

# group means +/- SEM
for idx, (xx, vals) in enumerate([(x0, pvl), (x1, pvh)]):
    m = vals.mean(); sem = vals.std(ddof=1) / np.sqrt(n)
    ax.errorbar(xx, m, yerr=sem, fmt='D', color='black', markersize=9,
                capsize=6, elinewidth=2, zorder=4,
                label=r'Group mean $\pm$ SEM' if idx == 0 else '_nolegend_')
ax.plot([x0, x1], [pvl.mean(), pvh.mean()], '-', color='black', lw=2.2,
        zorder=4)

ax.set_xticks([x0, x1])
ax.set_xticklabels(['Low', 'High'], fontsize=12)
ax.set_xlabel('Auditory condition', fontsize=12)
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(-0.03, 1.0)
ax.set_ylabel(r'Switching probability toward vision $p_v$', fontsize=12)
ax.tick_params(axis='y', labelsize=11)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', frameon=False, fontsize=10, handlelength=2.0)
plt.tight_layout()

outs = ['switching_probability_by_noise.pdf', 'switching_probability_by_noise.png',
        'ms_latex/assets/figures/switching_probability_by_noise.pdf',
        'ms_latex/assets/figures/switching_probability_by_noise.png']
for o in outs:
    plt.savefig(o, dpi=300, bbox_inches='tight')
print('saved:', ', '.join(outs))
print(f'n={n}  mean p_v,l={pvl.mean():.3f}  p_v,h={pvh.mean():.3f}  Wilcoxon stat={stat} p={pval:.4f}')
