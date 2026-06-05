"""Fitted switching probabilities p_v (toward vision) per auditory-noise condition.

Mechanistic figure for the cue-switching account: each participant's fitted
probability of relying on vision rises with auditory noise. Source: free-sigma
switchingFree fits (model_fits/, LapseFix sharedPrior). Saves to repo root and
ms_latex/assets/figures/.
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

SWITCH_COLOR = '#F58518'   # matches 'Probabilistic cue switching' in the model-comparison figures

# raw switchingFree param order: [lambda, sigma_a_l, sigma_v, p_v_l, sigma_a_h, p_v_h]
# (display reorder [0,1,4,2,3,5] -> [lambda, sa_l, sa_h, sv, p_v_l, p_v_h]); p_v_l=raw[3], p_v_h=raw[5]
rows = []
for f in sorted(glob.glob('model_fits/**/*.json', recursive=True)):
    if 'switchingFree_LapseFix_sharedPrior' not in f or '/all/' in f:
        continue
    d = json.load(open(f)); pid = str(d['participantID']).lower()
    if pid in ('ln1', 'all'):
        continue
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
for i in range(n):
    ax.plot([x0 + jit[i], x1 + jit[i]], [pvl[i], pvh[i]], '-', color='0.65', lw=1.1, alpha=0.8, zorder=1)
    ax.scatter([x0 + jit[i]], [pvl[i]], s=55, color=SWITCH_COLOR, edgecolor='black', linewidth=0.7, zorder=3)
    ax.scatter([x1 + jit[i]], [pvh[i]], s=55, color=SWITCH_COLOR, edgecolor='black', linewidth=0.7, zorder=3)

# group means +/- SEM
for xx, vals in [(x0, pvl), (x1, pvh)]:
    m = vals.mean(); sem = vals.std(ddof=1) / np.sqrt(n)
    ax.errorbar(xx, m, yerr=sem, fmt='D', color='black', markersize=9,
                capsize=6, elinewidth=2, zorder=4)
ax.plot([x0, x1], [pvl.mean(), pvh.mean()], '-', color='black', lw=2.2, zorder=4)

ax.set_xticks([x0, x1])
ax.set_xticklabels(['Low\nauditory noise', 'High\nauditory noise'], fontsize=12)
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(-0.03, 1.0)
ax.set_ylabel(r'Switching probability toward vision  $p_v$', fontsize=12)
ax.tick_params(axis='y', labelsize=11)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
p_txt = 'p = %.3f' % pval if pval >= 0.001 else 'p < 0.001'
ax.set_title(f'Vision reliance increases with auditory noise\n'
             f'(Wilcoxon {p_txt}, {int((pvh>pvl).sum())}/{n} participants)',
             fontsize=11, pad=10)
ax.text(0.02, 0.97, f'mean $p_{{v,low}}$ = {pvl.mean():.2f}\nmean $p_{{v,high}}$ = {pvh.mean():.2f}',
        transform=ax.transAxes, va='top', ha='left', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))
plt.tight_layout()

outs = ['switching_probability_by_noise.pdf', 'switching_probability_by_noise.png',
        'ms_latex/assets/figures/switching_probability_by_noise.pdf',
        'ms_latex/assets/figures/switching_probability_by_noise.png']
for o in outs:
    plt.savefig(o, dpi=300, bbox_inches='tight')
print('saved:', ', '.join(outs))
print(f'n={n}  mean p_v,l={pvl.mean():.3f}  p_v,h={pvh.mean():.3f}  Wilcoxon stat={stat} p={pval:.4f}')
