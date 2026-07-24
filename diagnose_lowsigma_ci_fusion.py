#!/usr/bin/env python3
"""
Diagnostic: why does Causal-Inference (lognorm) recovery DROP at LOW sensory
noise (σ-a) but hit 100% at HIGH noise (σ-c) when conflict_max = 0.45?

Grid observation (model_recovery_grid_results_boxcar_ns1_nsim10000, cmax=0.45):
    σ-a  [0.01–0.20] : CI self-recovery = 70%   (24% -> forced fusion)
    σ-b  [0.20–0.40] : CI self-recovery = 96%
    σ-c  [0.30–0.70] : CI self-recovery = 100%

WHAT THE RAW ITERATIONS SHOW
----------------------------
The misclassified CI runs in σ-a are NOT spread over the range — they cluster
at the BOTTOM of it:  mean min(σa,σv)=0.038 for the failures vs 0.082 for the
successes (corr(min_σ, correct)=+0.47). Every run with min_σ ≥ 0.15 recovers.

WHY: when σ → 0 the observer is nearly noiseless, so:
  • The psychometric functions become near-STEP functions.
  • On conflict trials the standardized discrepancy |m_a−m_v|/√(σa²+σv²) blows
    up, so posterior(common cause) saturates → segregation is all-or-nothing.
  • The resulting near-deterministic behaviour is absorbed by a simple FUSION
    model + lapse; the extra p_c parameter of CI buys almost no extra
    log-likelihood, so ΔAIC(fusion−CI) ≈ 0 and the +2 AIC penalty flips the
    decision to the simpler model.  → p_c becomes INESTIMABLE at low σ.
This is an estimation / identifiability effect, not a behavioural-equivalence
one: the *behavioural* CI−Fusion gap is actually large at low σ, but it is
carried by a handful of near-deterministic trials that AIC cannot reward.

This script reproduces both the estimation story (from the saved grid JSON) and
the behavioural story (fresh predicted psychometrics).
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
CONFLICT_MAX = 0.45
N_CONF = 9
NSIMUL = 20000

# ─────────────────────────────────────────────────────────────────────────
# PART A — estimation story straight from the saved grid iterations
# ─────────────────────────────────────────────────────────────────────────
def load_ci_iters(sigma_key):
    fp = glob.glob(f"{GRID_DIR}/grid_sl{sigma_key}_*cmax0.45*.json")[0]
    d = json.load(open(fp))
    out = []
    for it in d['raw_iters']['lognorm']:
        lam, sa, sv, pc = it['sampled_unique']
        mf = it.get('model_fits', {})
        rec = {
            'sa': sa, 'sv': sv, 'pc_true': pc, 'lam': lam,
            'min_sigma': min(sa, sv),
            'best': it['best_model_aic'],
            'correct': it['best_model_aic'] == 'lognorm',
        }
        if 'lognorm' in mf and 'fusionOnlyLogNorm' in mf:
            rec['aic_ci'] = mf['lognorm']['AIC']
            rec['aic_fus'] = mf['fusionOnlyLogNorm']['AIC']
            rec['dAIC'] = mf['fusionOnlyLogNorm']['AIC'] - mf['lognorm']['AIC']
            rec['pc_fit'] = mf['lognorm']['fittedParams'][3]
        out.append(rec)
    return out

ci_a = load_ci_iters('a')
ci_c = load_ci_iters('c')

# ─────────────────────────────────────────────────────────────────────────
# PART B — behavioural predictions (fresh) at very-low vs moderate σ
# ─────────────────────────────────────────────────────────────────────────
template = build_synthetic_template(conflict_max=CONFLICT_MAX,
                                    n_conflict_steps=N_CONF,
                                    standard_dur=STANDARD,
                                    noise_levels=(0.1, 1.2))
mc = monteCarloClass.OmerMonteCarlo(template)
mc.nSimul = NSIMUL
t_min, t_max = mc.data_t_min, mc.data_t_max
conflicts = np.linspace(-CONFLICT_MAX, CONFLICT_MAX, N_CONF)
deltas = np.linspace(-0.40, 0.40, 41)


def predicted_P(model_name, sigma, p_c):
    mc.modelName = model_name
    P = np.zeros((len(conflicts), len(deltas)))
    for ci_, conflict in enumerate(conflicts):
        S_a_s, S_v_s = STANDARD, STANDARD + conflict
        for di, d in enumerate(deltas):
            S_a_t = S_a_s + d
            trueStims = (S_a_s, S_a_t, S_v_s, S_a_t)
            pc_use = 1.0 if model_name == 'fusionOnlyLogNorm' else p_c
            P[ci_, di] = mc.probTestLonger_vectorized_mc(
                trueStims, sigma, sigma, pc_use, 0.0, t_min, t_max)
    return P


def pse(deltas, p_row):
    if p_row[0] > 0.5 or p_row[-1] < 0.5:
        return np.nan
    idx = np.searchsorted(p_row, 0.5)
    x0, x1 = deltas[idx - 1], deltas[idx]
    y0, y1 = p_row[idx - 1], p_row[idx]
    return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)


# ═════════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

# --- A1: recovery outcome vs sampled min-σ (the empirical driver) ----------
axA = fig.add_subplot(gs[0, 0])
for iters, xoff, lab, col in [(ci_a, 0, 'σ-a run', '#1f77b4')]:
    ok = np.array([r['min_sigma'] for r in iters if r['correct']])
    bad = np.array([r['min_sigma'] for r in iters if not r['correct']])
    axA.scatter(ok, np.random.uniform(0.9, 1.1, len(ok)), s=28,
                color='#2ca02c', label=f'recovered as CI (n={len(ok)})')
    axA.scatter(bad, np.random.uniform(-0.1, 0.1, len(bad)), s=40,
                color='#d62728', marker='x', label=f'misclassified (n={len(bad)})')
axA.axvline(0.15, ls=':', color='gray')
axA.text(0.155, 0.5, 'min σ = 0.15', color='gray', fontsize=8)
axA.set_yticks([0, 1]); axA.set_yticklabels(['WRONG', 'CI'])
axA.set_xlabel('sampled  min(σa, σv)')
axA.set_title('σ-a cmax=0.45:  failures cluster at the LOWEST σ')
axA.legend(fontsize=8, frameon=False, loc='center right')

# --- A2: ΔAIC(fusion−CI) vs min-σ  ----------------------------------------
axB = fig.add_subplot(gs[0, 1])
for iters, mk, lab in [(ci_a, 'o', 'σ-a'), (ci_c, '^', 'σ-c')]:
    x = [r['min_sigma'] for r in iters if 'dAIC' in r]
    y = [np.clip(r['dAIC'], -10, 60) for r in iters if 'dAIC' in r]
    c = ['#2ca02c' if r['correct'] else '#d62728'
         for r in iters if 'dAIC' in r]
    axB.scatter(x, y, c=c, marker=mk, s=30, label=lab, edgecolor='k', linewidth=0.2)
axB.axhline(0, color='k', lw=0.8)
axB.axhspan(-10, 0, color='#d62728', alpha=0.06)
axB.set_xlabel('sampled  min(σa, σv)')
axB.set_ylabel('ΔAIC = AIC(fusion) − AIC(CI)\n(>0 favours CI; clipped)')
axB.set_title('At low σ, ΔAIC ≈ 0 → p_c buys no LL,\nAIC penalty flips to fusion')
axB.legend(fontsize=8, frameon=False, title='green=correct, red=wrong')

# --- A3: p_c NOT identifiable at low σ  -----------------------------------
axC = fig.add_subplot(gs[0, 2])
for iters, mk, lab in [(ci_a, 'o', 'σ-a'), (ci_c, '^', 'σ-c')]:
    small = [r for r in iters if 'pc_fit' in r and r['min_sigma'] < 0.15]
    big = [r for r in iters if 'pc_fit' in r and r['min_sigma'] >= 0.15]
    if small:
        axC.scatter([r['pc_true'] for r in small], [r['pc_fit'] for r in small],
                    marker=mk, s=45, facecolor='none', edgecolor='#d62728',
                    label=f'{lab} min σ<0.15')
    if big:
        axC.scatter([r['pc_true'] for r in big], [r['pc_fit'] for r in big],
                    marker=mk, s=28, color='#2ca02c', alpha=0.7,
                    label=f'{lab} min σ≥0.15')
axC.plot([0, 1], [0, 1], 'k--', lw=0.8)
axC.set_xlabel('true p_c'); axC.set_ylabel('fitted p_c')
axC.set_title('p_c recovery: scattered when σ small,\ntight when σ adequate')
axC.legend(fontsize=7, frameon=False)

# --- B1/B2: predicted PSE-vs-conflict at very-low vs moderate σ -----------
for col, (sigma, tag) in enumerate([(0.03, 'very low  σ=0.03  (σ-a floor)'),
                                     (0.50, 'high  σ=0.50  (σ-c)')]):
    ax = fig.add_subplot(gs[1, col])
    for pc, cstyle in [(0.30, '-'), (0.70, '--')]:
        P_ci = predicted_P('lognorm', sigma, pc)
        pse_ci = [pse(deltas, P_ci[i]) for i in range(len(conflicts))]
        ax.plot(conflicts, pse_ci, cstyle, color='#1f77b4',
                label=f'CI  p_c={pc}')
    P_fus = predicted_P('fusionOnlyLogNorm', sigma, 1.0)
    pse_fus = [pse(deltas, P_fus[i]) for i in range(len(conflicts))]
    ax.plot(conflicts, pse_fus, ':', color='#ff7f0e', lw=2.5, label='Fusion')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'PSE vs conflict — {tag}')
    ax.set_xlabel('AV conflict (s)'); ax.set_ylabel('PSE shift (s)')
    ax.legend(fontsize=7, frameon=False)

# --- B3: posterior(common cause) saturation vs σ --------------------------
ax = fig.add_subplot(gs[1, 2])
for sigma, col in [(0.03, '#d62728'), (0.15, '#ff7f0e'), (0.50, '#2ca02c')]:
    post = []
    for conflict in conflicts:
        S_a_s, S_v_s = STANDARD, max(STANDARD + conflict, 1e-3)
        m_a = np.random.normal(np.log(S_a_s), sigma, NSIMUL)
        m_v = np.random.normal(np.log(S_v_s), sigma, NSIMUL)
        pc1 = mc.posterior_C1(m_a, m_v, sigma, sigma, 0.5,
                              np.log(t_min), np.log(t_max))
        post.append(np.mean(pc1))
    ax.plot(conflicts, post, 'o-', color=col, label=f'σ={sigma}')
ax.set_title('mean posterior(common cause)\nsaturates hard when σ small')
ax.set_xlabel('AV conflict (s)'); ax.set_ylabel('P(C=1 | m)')
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8, frameon=False)

fig.suptitle(
    'Why CI(lognorm) recovery DROPS at low σ but is 100% at high σ (cmax=0.45)\n'
    'Top: estimation story from grid JSON — failures sit at min σ<0.15 where ΔAIC≈0 and p_c is inestimable.\n'
    'Bottom: behavioural predictions — at σ→0 the CI PSE curve collapses toward fusion + saturated posterior.',
    fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = 'identifiability_figures/diagnose_lowsigma_ci_vs_fusion.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('saved', out)

# ── console summary ───────────────────────────────────────────────────────
for tag, iters in [('σ-a', ci_a), ('σ-c', ci_c)]:
    mins = np.array([r['min_sigma'] for r in iters])
    ok = np.array([r['correct'] for r in iters], float)
    print(f'\n{tag}: n={len(iters)}  recovered={int(ok.sum())}/{len(iters)}  '
          f'corr(min_σ,correct)={np.corrcoef(mins, ok)[0,1]:+.2f}')
    if (ok == 0).any():
        print(f'     mean min_σ  correct={mins[ok==1].mean():.3f}  '
              f'wrong={mins[ok==0].mean():.3f}')
    daic_lowsig = [r['dAIC'] for r in iters if 'dAIC' in r and r['min_sigma'] < 0.15]
    daic_hisig = [r['dAIC'] for r in iters if 'dAIC' in r and r['min_sigma'] >= 0.15]
    if daic_lowsig:
        print(f'     median ΔAIC(fus−CI): min σ<0.15 = {np.median(daic_lowsig):+.1f}  '
              f'(range {min(daic_lowsig):+.1f}..{max(daic_lowsig):+.1f})')
    if daic_hisig:
        print(f'                          min σ≥0.15 = {np.median(daic_hisig):+.1f}')
