#!/usr/bin/env python3
"""
SIMPLE explainer figure for the PI: why CI(lognorm) recovery drops at LOW
sensory noise but is perfect at HIGH noise (cmax=0.45).

Key correction over the first attempt: recovery does NOT compare CI-at-truth
vs Fusion-at-truth. It RE-FITS fusion (with its own free σ_a, σ_v, lapse) to
the CI-generated data. At low σ a CI observer segregates almost every trial
(reports audio, ignores vision) — which a fusion fit reproduces by making
σ_v ≫ σ_a (median σv/σa = 34 on the failures vs 7.5 on the successes, straight
from the grid JSON). Fusion then fits as well as CI, and AIC's +2 penalty for
CI's extra p_c flips the choice to the simpler fusion model.

Four plain panels:
  1. THE FACT   — CI recovery % across the three σ regimes.
  2. THE CAUSE  — recovery vs sampled σ: failures only at the tiniest σ.
  3. THE PROOF  — on a real misclassified case, the best-fit FUSION curve lands
                  on top of the CI-generated data (they are indistinguishable).
  4. THE KNOB   — fusion's recovered σv/σa is huge on failures = "ignore vision".
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

import monteCarloClass
from run_param_recovery_favorable import build_synthetic_template

np.random.seed(0)
GRID_DIR = 'model_recovery_grid_results_boxcar_ns1_nsim10000'
STANDARD = 0.50
BLUE, ORANGE, GREEN, RED = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'


def load_ci(sigma_key):
    fp = glob.glob(f"{GRID_DIR}/grid_sl{sigma_key}_*cmax0.45*.json")[0]
    d = json.load(open(fp))
    row = d['confusion_aic']['lognorm']
    pct = row['lognorm'] / sum(row.values()) * 100
    recs = []
    for it in d['raw_iters']['lognorm']:
        lam, sa, sv, pc = it['sampled_unique']
        mf = it.get('model_fits', {})
        r = {'sa': sa, 'sv': sv, 'pc': pc, 'lam': lam, 'min_sigma': min(sa, sv),
             'correct': it['best_model_aic'] == 'lognorm',
             'best': it['best_model_aic']}
        if 'fusionOnlyLogNorm' in mf:
            r['fus'] = mf['fusionOnlyLogNorm']['fittedParams']  # [λ, σa1, σv, σa2]
        recs.append(r)
    return pct, recs


keys = ['a', 'b', 'c']
labels = ['σ-a\n[0.01–0.20]\nlow noise', 'σ-b\n[0.20–0.40]', 'σ-c\n[0.30–0.70]\nhigh noise']
pcts, recs_by_key = zip(*[load_ci(k) for k in keys])
recs_a = recs_by_key[0]

# ── prediction machinery ───────────────────────────────────────────────────
template = build_synthetic_template(conflict_max=0.45, n_conflict_steps=9,
                                    standard_dur=STANDARD, noise_levels=(0.1, 1.2))
mc = monteCarloClass.OmerMonteCarlo(template)
mc.nSimul = 30000
t_min, t_max = mc.data_t_min, mc.data_t_max
conflicts = np.linspace(-0.45, 0.45, 13)
deltas = np.linspace(-0.40, 0.40, 61)


def psychometric(model_name, sigma_a, sigma_v, p_c, conflict, lam=0.0):
    """P(report test longer) vs test-minus-standard, at one conflict."""
    mc.modelName = model_name
    S_a_s, S_v_s = STANDARD, STANDARD + conflict
    P = np.empty_like(deltas)
    for i, d in enumerate(deltas):
        S_a_t = S_a_s + d
        pc_use = 1.0 if model_name == 'fusionOnlyLogNorm' else p_c
        P[i] = mc.probTestLonger_vectorized_mc(
            (S_a_s, S_a_t, S_v_s, S_a_t), sigma_a, sigma_v, pc_use, lam,
            t_min, t_max)
    return P


# pick one real misclassified low-σ case that has a fusion fit
case = next(r for r in recs_a
            if not r['correct'] and 'fus' in r and r['min_sigma'] < 0.10)

# ── σv/σa ratios for panel 4 ───────────────────────────────────────────────
def ratios(recs, want_correct):
    out = []
    for r in recs:
        if 'fus' not in r or r['correct'] != want_correct:
            continue
        sa1, sv = r['fus'][1], r['fus'][2]
        if sa1 > 0:
            out.append(sv / sa1)
    return np.array(out)

r_fail = ratios(recs_a, want_correct=False)
r_ok = ratios(recs_a, want_correct=True)

# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))

# Panel 1 — the fact
ax = axes[0]
bars = ax.bar(labels, pcts, color=[RED, '#f0ad4e', GREEN], edgecolor='k', lw=0.6)
for b, p in zip(bars, pcts):
    ax.text(b.get_x() + b.get_width() / 2, p + 1.5, f'{p:.0f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, 108)
ax.set_ylabel('CI correctly recovered  (%)')
ax.set_title('1.  The puzzle:\nCI recovery is WORSE at low noise', fontsize=11)

# Panel 2 — the cause
ax = axes[1]
ok = [r['min_sigma'] for r in recs_a if r['correct']]
bad = [r['min_sigma'] for r in recs_a if not r['correct']]
ax.scatter(ok, [1] * len(ok), s=55, color=GREEN, zorder=3,
           label=f'recovered  (n={len(ok)})')
ax.scatter(bad, [0] * len(bad), s=75, color=RED, marker='X', zorder=3,
           label=f'confused w/ fusion  (n={len(bad)})')
ax.axvspan(0, 0.10, color=RED, alpha=0.08)
ax.text(0.05, 0.5, 'danger\nzone', color=RED, ha='center', va='center', fontsize=10)
ax.set_yticks([0, 1]); ax.set_yticklabels(['WRONG', 'RIGHT'])
ax.set_xlabel('sensory noise σ (generating)')
ax.set_title('2.  The cause:\nfailures only at the TINIEST σ', fontsize=11)
ax.legend(fontsize=9, frameon=False, loc='center right')
ax.set_xlim(-0.005, 0.21)

# Panel 3 — the proof: fusion RE-FIT reproduces the CI psychometric curve
ax = axes[2]
CONF = 0.45                                        # largest conflict = hardest test
f = case['fus']                                    # [λ, σa1, σv, σa2]
P_ci = psychometric('lognorm', case['sa'], case['sv'], case['pc'], CONF, case['lam'])
P_fus = psychometric('fusionOnlyLogNorm', f[1], f[2], 1.0, CONF, f[0])
ax.plot(deltas, P_ci, '-', color=BLUE, lw=2.5,
        label=f'CI data\n(σa={case["sa"]:.2f}, σv={case["sv"]:.2f}, p_c={case["pc"]:.2f})')
ax.plot(deltas, P_fus, '--', color=ORANGE, lw=2.5,
        label=f'best-fit FUSION\n(σv/σa={f[2]/f[1]:.0f})')
ax.axhline(0.5, color='gray', lw=0.5)
ax.set_xlabel('test − standard duration (s)')
ax.set_ylabel('P(report test longer)')
ax.set_title('3.  Why they are confused (conflict = 0.45 s):\n'
             're-fit fusion reproduces the CI curve', fontsize=11)
ax.legend(fontsize=8, frameon=False, loc='upper left')

# Panel 4 — the knob fusion uses
ax = axes[3]
bp = ax.boxplot([r_ok, r_fail], labels=['recovered\ncorrectly', 'confused\nw/ fusion'],
                patch_artist=True, widths=0.5, showfliers=False)
for patch, col in zip(bp['boxes'], [GREEN, RED]):
    patch.set_facecolor(col); patch.set_alpha(0.35)
ax.scatter(np.random.normal(1, 0.05, len(r_ok)), r_ok, s=22, color=GREEN, zorder=3)
ax.scatter(np.random.normal(2, 0.05, len(r_fail)), r_fail, s=26, color=RED, zorder=3)
ax.axhline(1, color='gray', lw=0.6, ls=':')
ax.set_yscale('log')
ax.set_ylabel('fusion fit  σv / σa   (log scale)')
ax.set_title('4.  The trick fusion uses:\nblow up σv → "ignore vision" = CI segregation', fontsize=11)

fig.suptitle('Causal-Inference model recovery vs sensory noise  (large conflict = 0.45 s)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = 'identifiability_figures/lowsigma_ci_recovery_SIMPLE.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('saved', out)
print(f'median σv/σa  correct={np.median(r_ok):.1f}  failures={np.median(r_fail):.1f}')
