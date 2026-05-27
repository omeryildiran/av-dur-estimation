"""
Forward-simulation slope diagnostic.

Question: At matched parameters, do the three causal-inference variants
(model averaging / selection / probability matching) produce visibly
different psychometric-function slopes across the conflict range?
If yes, the 2AFC likelihood already contains the information needed to
separate them, and our identifiability problem is a fitting/parameter-
range issue, not a task-design ceiling. If the slopes overlap, then 2AFC
is the bottleneck and the estimation-task simulation is the right next
move.

Outputs:
  slope_diagnostic.pdf / .png       — PSE & PF-sigma vs conflict, per regime
  slope_diagnostic_PFs.pdf / .png   — Full PFs per conflict, models overlaid
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

from monteCarloClass import OmerMonteCarlo
from loadData import loadData


# ---------------- Config ----------------

MODELS = [
    "lognorm",                       # CI w/ model averaging   (log space)
    "selection",                     # CI w/ model selection
    "probabilityMatchingLogNorm",    # CI w/ probability matching
    "fusionOnlyLogNorm",             # no-CI fusion baseline (reference)
]

LABELS = {
    "lognorm":                    "averaging",
    "selection":                  "selection",
    "probabilityMatchingLogNorm": "prob matching",
    "fusionOnlyLogNorm":          "fusion (no CI)",
}

COLORS = {
    "lognorm":                    "#1f77b4",
    "selection":                  "#d62728",
    "probabilityMatchingLogNorm": "#2ca02c",
    "fusionOnlyLogNorm":          "#7f7f7f",
}

# Empirical noise regimes — both experimental conditions, group means
# from the fitted parameters across participants (σ_v is shared across
# audio-noise conditions, σ_a varies).
NOISE_REGIMES = {
    "low_aud_noise":  dict(sigma_a=0.28, sigma_v=0.57),  # audNoise=0.1
    "high_aud_noise": dict(sigma_a=0.80, sigma_v=0.57),  # audNoise=1.2
}

# Conflicts: empirical levels + a couple of extended ones to test whether
# wider conflict helps separation.
CONFLICTS = np.array([-0.45, -0.40, -0.35, -0.30, -0.25, -0.17, -0.08, 0.0,
                      0.08, 0.17, 0.25, 0.30, 0.35, 0.40, 0.45])

# Fine Δ grid for clean PF fits.
DELTAS = np.linspace(-0.35, 0.35, 21)

P_C    = 0.7    # matched prior P(common cause)
LAMBDA = 0.02   # matched lapse
S_A_STD = 0.5   # auditory standard duration (matches experiment)

N_SIMUL = 5000  # MC samples per (Δ, conflict) — high to suppress MC noise

SEED = 0


# ---------------- Model evaluator ----------------

def make_mc():
    """Instantiate OmerMonteCarlo using the real dataset so t_min/t_max
    are set from data exactly as during fitting."""
    data, _ = loadData("mt_all.csv")
    mc = OmerMonteCarlo(data)
    mc.nSimul = N_SIMUL
    return mc


def predict_p_long(mc, model_name, sigma_a, sigma_v, p_c, lam, deltas, conflict):
    mc.modelName = model_name
    t_min = mc.data_t_min
    t_max = mc.data_t_max
    out = np.empty_like(deltas, dtype=float)
    for i, d in enumerate(deltas):
        S_a_s = S_A_STD
        S_v_s = S_a_s + conflict
        S_a_t = S_a_s + d
        S_v_t = S_a_t
        # NOTE: log-space models log t_min/t_max internally; pass linear.
        out[i] = mc.probTestLonger_vectorized_mc(
            (S_a_s, S_a_t, S_v_s, S_v_t),
            sigma_a, sigma_v, p_c, lam, t_min, t_max,
        )
    return out


def cum_gauss(x, mu, sigma, lam):
    return lam / 2.0 + (1.0 - lam) * norm.cdf(x, loc=mu, scale=sigma)


def _nll_pf(params, deltas, p_target):
    """Cross-entropy NLL between target probability curve and a cum-Gaussian
    PF parameterised by (μ, σ, λ). The right objective for binomial / choice-
    probability targets — weights each Δ by its information content rather
    than treating equal-variance residuals (which curve_fit would assume)."""
    mu, sigma, lam = params
    if sigma <= 0 or lam < 0 or lam > 0.4:
        return 1e10
    p_model = cum_gauss(deltas, mu, sigma, lam)
    eps = 1e-9
    p_model = np.clip(p_model, eps, 1.0 - eps)
    return -float(np.sum(p_target * np.log(p_model)
                         + (1.0 - p_target) * np.log(1.0 - p_model)))


def fit_pf(deltas, p_long):
    """Maximum-likelihood (μ, σ, λ) for a cum-Gaussian PF fit to a target
    probability curve. Tries multiple inits and returns the best fit."""
    bounds = [(-0.5, 0.5), (1e-3, 1.0), (0.0, 0.4)]
    inits = [
        [0.0,  0.10, 0.02],
        [0.0,  0.05, 0.02],
        [0.0,  0.20, 0.02],
        [0.05, 0.10, 0.05],
    ]
    best = None
    for x0 in inits:
        try:
            res = minimize(_nll_pf, x0, args=(deltas, p_long),
                           bounds=bounds, method="L-BFGS-B")
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            continue
    if best is None:
        return np.array([np.nan, np.nan, np.nan])
    return best.x  # (mu, sigma, lapse)


# ---------------- Sweep ----------------

def run():
    np.random.seed(SEED)
    mc = make_mc()

    results = []
    for regime_name, regime in NOISE_REGIMES.items():
        for model in MODELS:
            for c in CONFLICTS:
                p_long = predict_p_long(
                    mc, model, regime["sigma_a"], regime["sigma_v"],
                    P_C, LAMBDA, DELTAS, c,
                )
                mu, sigma, lam = fit_pf(DELTAS, p_long)
                results.append(dict(
                    regime=regime_name, model=model, conflict=c,
                    mu=mu, sigma=sigma, lam=lam,
                    deltas=DELTAS.copy(), p_long=p_long,
                ))
                print(f"[{regime_name:10s} | {LABELS[model]:14s} | "
                      f"conf={int(c*1000):+4d} ms]  "
                      f"μ={mu*1000:+6.1f} ms, σ={sigma*1000:5.1f} ms")
    return results


# ---------------- Plots ----------------

def _conflict_subset(max_abs_conflict):
    """Return CONFLICTS truncated to |c| <= max_abs_conflict."""
    if max_abs_conflict is None:
        return CONFLICTS
    return np.array([c for c in CONFLICTS if abs(c) <= max_abs_conflict + 1e-9])


def plot_pse_slope(results, max_abs_conflict=0.30,
                   outstem="report_pse_slope"):
    """PSE-vs-conflict and σ-vs-conflict, both noise regimes stacked
    in rows, two columns (PSE, σ). One figure per call."""
    regimes = list(NOISE_REGIMES.keys())
    conflicts = _conflict_subset(max_abs_conflict)

    fig, axes = plt.subplots(len(regimes), 2,
                             figsize=(12, 4.4 * len(regimes)),
                             constrained_layout=True)
    if len(regimes) == 1:
        axes = axes[None, :]

    for ri, regime_name in enumerate(regimes):
        sigma_a = NOISE_REGIMES[regime_name]["sigma_a"]
        sigma_v = NOISE_REGIMES[regime_name]["sigma_v"]
        for model in MODELS:
            rows = sorted([r for r in results
                           if r["regime"] == regime_name
                           and r["model"] == model
                           and any(np.isclose(r["conflict"], c) for c in conflicts)],
                          key=lambda r: r["conflict"])
            c   = np.array([r["conflict"] for r in rows]) * 1000
            mu  = np.array([r["mu"]       for r in rows]) * 1000
            sig = np.array([r["sigma"]    for r in rows]) * 1000

            axes[ri, 0].plot(c, mu,  "-o", color=COLORS[model],
                             label=LABELS[model], lw=2, ms=6)
            axes[ri, 1].plot(c, sig, "-o", color=COLORS[model],
                             label=LABELS[model], lw=2, ms=6)

        axes[ri, 0].set_title(f"{regime_name} (σ_a={sigma_a}, σ_v={sigma_v}): "
                              f"PSE μ vs conflict")
        axes[ri, 1].set_title(f"{regime_name} (σ_a={sigma_a}, σ_v={sigma_v}): "
                              f"PF σ vs conflict   (lower = steeper PF)")
        for ax in axes[ri]:
            ax.set_xlabel("conflict (ms)")
            ax.axvline(0, color="gray", ls="--", lw=0.6)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9, loc="best")
        axes[ri, 0].set_ylabel("PSE μ (ms)")
        axes[ri, 0].axhline(0, color="gray", ls="--", lw=0.6)
        axes[ri, 1].set_ylabel("PF σ (ms)")

    title_range = f"±{int(max_abs_conflict*1000)} ms" if max_abs_conflict \
                  else f"±{int(max(np.abs(CONFLICTS))*1000)} ms"
    fig.suptitle(
        f"Slope diagnostic — conflict range {title_range}, "
        f"matched p_c={P_C}, λ={LAMBDA}, nSimul={N_SIMUL}",
        fontsize=12,
    )
    fig.savefig(f"{outstem}.pdf", bbox_inches="tight")
    fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
    print(f"saved {outstem}.pdf / .png")


def plot_pfs(results, max_abs_conflict=0.30, outstem="report_pfs"):
    """Per-conflict PF curves, models overlaid. Rows = regimes,
    cols = conflicts."""
    regimes = list(NOISE_REGIMES.keys())
    conflicts = _conflict_subset(max_abs_conflict)

    fig, axes = plt.subplots(
        len(regimes), len(conflicts),
        figsize=(1.95 * len(conflicts), 2.7 * len(regimes)),
        sharey=True, constrained_layout=True,
    )
    if len(regimes) == 1:
        axes = axes[None, :]

    for ri, regime_name in enumerate(regimes):
        for ci, c in enumerate(conflicts):
            ax = axes[ri, ci]
            for model in MODELS:
                r = next(r for r in results
                         if r["regime"] == regime_name
                         and r["model"] == model
                         and np.isclose(r["conflict"], c))
                ax.plot(r["deltas"] * 1000, r["p_long"],
                        color=COLORS[model], lw=1.6,
                        label=LABELS[model])
            ax.set_title(f"conf {int(c*1000):+d} ms", fontsize=8)
            ax.axhline(0.5, color="gray", ls="--", lw=0.5)
            ax.axvline(0,   color="gray", ls="--", lw=0.5)
            ax.set_ylim(-0.02, 1.02)
            if ci == 0:
                ax.set_ylabel(f"{regime_name}\nP(test > std)", fontsize=9)
            if ri == len(regimes) - 1:
                ax.set_xlabel("Δ (ms)")
            if ri == 0 and ci == len(conflicts) - 1:
                ax.legend(fontsize=6, loc="lower right")
            ax.grid(alpha=0.25)

    title_range = f"±{int(max_abs_conflict*1000)} ms"
    fig.suptitle(f"Predicted PFs per conflict — both regimes, {title_range}",
                 fontsize=11)
    fig.savefig(f"{outstem}.pdf", bbox_inches="tight")
    fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
    print(f"saved {outstem}.pdf / .png")


def print_summary(results):
    print("\n=== SLOPE DIAGNOSTIC: σ-range across conflicts per model ===")
    for regime_name in NOISE_REGIMES:
        print(f"\n[{regime_name}]")
        sigs_by_model = {}
        for model in MODELS:
            rows = sorted([r for r in results
                           if r["regime"] == regime_name and r["model"] == model],
                          key=lambda r: r["conflict"])
            sigs = np.array([r["sigma"] for r in rows]) * 1000
            sigs_by_model[model] = sigs
            print(f"  {LABELS[model]:<14} "
                  f"σ(ms): min={np.nanmin(sigs):5.1f}  max={np.nanmax(sigs):5.1f}  "
                  f"range={np.nanmax(sigs)-np.nanmin(sigs):5.1f}")

        # pairwise max |Δσ| across conflicts between CI variants — the
        # quantity that determines whether 2AFC can separate them.
        ci_models = ["lognorm", "selection", "probabilityMatchingLogNorm"]
        print("  pairwise max |Δσ(c)| between CI variants (ms):")
        for i in range(len(ci_models)):
            for j in range(i + 1, len(ci_models)):
                a, b = ci_models[i], ci_models[j]
                d = np.nanmax(np.abs(sigs_by_model[a] - sigs_by_model[b]))
                print(f"    {LABELS[a]:>14} vs {LABELS[b]:<14}: {d:5.1f}")


def expected_dBIC(results, n_trials_per_regime, conflict_subset, label):
    """Information-theoretic upper bound on recoverability.

    For each pair of CI models, compute the *expected* log-likelihood
    gap if data is generated from model A and fit with model B:
        E[ΔLL] = Σ_{c,Δ}  N(c,Δ) · KL( Bern(p_A) ‖ Bern(p_B) )
    Assumes uniform Δ trials across the simulation grid (≈ true for
    staircased experiments at the slope-relevant range). Models have
    matching parameter counts so ΔBIC ≈ 2·ΔLL.

    BIC interpretation: |ΔBIC| < 2 ambiguous, 2-6 weak, 6-10 moderate,
    >10 decisive evidence.
    """
    ci_models = ["lognorm", "selection", "probabilityMatchingLogNorm"]
    eps = 1e-9
    n_conflicts = len(conflict_subset)
    n_deltas = len(DELTAS)
    n_per_cell = n_trials_per_regime / (n_conflicts * n_deltas)

    print(f"\n==== Expected ΔBIC — {label} ====")
    print(f"  conflicts (ms): {[int(c*1000) for c in conflict_subset]}")
    print(f"  N_trials_per_regime={n_trials_per_regime}  "
          f"→ ≈ {n_per_cell:.1f} trials per (Δ, conflict) cell")

    for regime_name in NOISE_REGIMES:
        print(f"\n  [regime: {regime_name}]")
        p_by_model = {}
        for m in ci_models:
            rows = sorted(
                [r for r in results
                 if r["regime"] == regime_name and r["model"] == m
                 and any(np.isclose(r["conflict"], c) for c in conflict_subset)],
                key=lambda r: r["conflict"],
            )
            p_by_model[m] = np.clip(np.concatenate([r["p_long"] for r in rows]),
                                    eps, 1 - eps)

        for i in range(len(ci_models)):
            for j in range(i + 1, len(ci_models)):
                pA = p_by_model[ci_models[i]]
                pB = p_by_model[ci_models[j]]
                kl_AB = float((pA * np.log(pA / pB)
                               + (1 - pA) * np.log((1 - pA) / (1 - pB))).sum())
                kl_BA = float((pB * np.log(pB / pA)
                               + (1 - pB) * np.log((1 - pB) / (1 - pA))).sum())
                edBIC_A = 2 * n_per_cell * kl_AB
                edBIC_B = 2 * n_per_cell * kl_BA
                avg = 0.5 * (edBIC_A + edBIC_B)
                verdict = ("decisive"  if avg > 10 else
                           "moderate"  if avg > 6  else
                           "weak"      if avg > 2  else
                           "ambiguous")
                print(f"    {LABELS[ci_models[i]]:>14} vs "
                      f"{LABELS[ci_models[j]]:<14}: "
                      f"E[ΔBIC|A]={edBIC_A:6.1f}  "
                      f"E[ΔBIC|B]={edBIC_B:6.1f}  "
                      f"avg={avg:6.1f}  → {verdict}")


def plot_dBIC(results, n_trials_per_regime=1078, outstem="report_dBIC"):
    """Bar chart of expected ΔBIC for each CI-model pair across conflict
    ranges. Two panels side by side, one per noise regime. Shared
    horizontal reference lines mark BIC interpretation thresholds."""
    ci_models = ["lognorm", "selection", "probabilityMatchingLogNorm"]
    pair_labels = {
        ("lognorm", "selection"):                   "averaging  vs  selection",
        ("lognorm", "probabilityMatchingLogNorm"):  "averaging  vs  prob matching",
        ("selection", "probabilityMatchingLogNorm"):"selection  vs  prob matching",
    }
    subsets = [
        ("±0.25", np.array([c for c in CONFLICTS if abs(c) <= 0.25 + 1e-9])),
        ("±0.30", np.array([c for c in CONFLICTS if abs(c) <= 0.30 + 1e-9])),
        ("±0.45", np.array([c for c in CONFLICTS if abs(c) <= 0.45 + 1e-9])),
    ]
    pairs = [(ci_models[i], ci_models[j])
             for i in range(len(ci_models))
             for j in range(i + 1, len(ci_models))]
    pair_colors = ["#1f77b4", "#2ca02c", "#d62728"]
    eps = 1e-9

    regimes = list(NOISE_REGIMES.keys())
    fig, axes = plt.subplots(1, len(regimes),
                             figsize=(6.5 * len(regimes), 4.6),
                             sharey=True, constrained_layout=True)
    if len(regimes) == 1:
        axes = [axes]

    # Compute all values first to know shared y-limit.
    all_vals = []
    pair_values_by_regime = {}
    for regime_name in regimes:
        pair_values = {p: [] for p in pairs}
        for _, conflicts in subsets:
            n_per_cell = n_trials_per_regime / (len(conflicts) * len(DELTAS))
            p_by_model = {}
            for m in ci_models:
                rows = sorted([r for r in results
                               if r["regime"] == regime_name and r["model"] == m
                               and any(np.isclose(r["conflict"], c) for c in conflicts)],
                              key=lambda r: r["conflict"])
                p_by_model[m] = np.clip(np.concatenate([r["p_long"] for r in rows]),
                                        eps, 1 - eps)
            for a, b in pairs:
                pA, pB = p_by_model[a], p_by_model[b]
                kl_AB = float((pA * np.log(pA / pB)
                               + (1 - pA) * np.log((1 - pA) / (1 - pB))).sum())
                kl_BA = float((pB * np.log(pB / pA)
                               + (1 - pB) * np.log((1 - pB) / (1 - pA))).sum())
                val = n_per_cell * (kl_AB + kl_BA)  # ΔBIC ≈ 2·avg·ΔLL
                pair_values[(a, b)].append(val)
                all_vals.append(val)
        pair_values_by_regime[regime_name] = pair_values

    y_top = max(max(all_vals) * 1.15, 12.0)

    for ri, regime_name in enumerate(regimes):
        ax = axes[ri]
        pair_values = pair_values_by_regime[regime_name]
        x = np.arange(len(subsets))
        width = 0.26
        for k, p in enumerate(pairs):
            ax.bar(x + (k - 1) * width, pair_values[p], width=width,
                   color=pair_colors[k], label=pair_labels[p])
        ax.axhline(2,  color="gray", ls=":", lw=0.8)
        ax.axhline(6,  color="gray", ls=":", lw=0.8)
        ax.axhline(10, color="gray", ls=":", lw=0.8)
        # Threshold annotations on rightmost panel only.
        if ri == len(regimes) - 1:
            ax.text(len(subsets) - 0.45, 2.1,  "weak",     fontsize=8, color="gray")
            ax.text(len(subsets) - 0.45, 6.1,  "moderate", fontsize=8, color="gray")
            ax.text(len(subsets) - 0.45, 10.1, "decisive", fontsize=8, color="gray")
        ax.set_xticks(x); ax.set_xticklabels([s[0] for s in subsets])
        ax.set_xlabel("Conflict range used")
        if ri == 0:
            ax.set_ylabel("Expected ΔBIC")
        sigma_a = NOISE_REGIMES[regime_name]["sigma_a"]
        sigma_v = NOISE_REGIMES[regime_name]["sigma_v"]
        ax.set_title(f"{regime_name}\n(σ_a={sigma_a}, σ_v={sigma_v})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, y_top)

    fig.suptitle("Information available to distinguish CI variants — "
                 f"N≈{n_trials_per_regime}/regime", fontsize=12)
    fig.savefig(f"{outstem}.pdf", bbox_inches="tight")
    fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
    print(f"saved {outstem}.pdf / .png")


if __name__ == "__main__":
    results = run()
    print_summary(results)

    # PSE / σ vs conflict — both regimes, two conflict-range variants.
    plot_pse_slope(results, max_abs_conflict=0.30,
                   outstem="report_pse_slope_pm30")
    plot_pse_slope(results, max_abs_conflict=0.45,
                   outstem="report_pse_slope_pm45")
    # Per-conflict PFs at ±0.30 (±0.45 would be too wide for a report figure).
    plot_pfs(results, max_abs_conflict=0.30, outstem="report_pfs_pm30")
    plot_pfs(results, max_abs_conflict=0.45, outstem="report_pfs_pm45")
    # ΔBIC bar chart — both regimes side by side, three conflict ranges.
    plot_dBIC(results, n_trials_per_regime=1078, outstem="report_dBIC")

    # Detailed ΔBIC tables across conflict subsets (printed only).
    subsets = [
        ("±0.25 (empirical)", np.array([c for c in CONFLICTS if abs(c) <= 0.25 + 1e-9])),
        ("±0.30",             np.array([c for c in CONFLICTS if abs(c) <= 0.30 + 1e-9])),
        ("±0.35",             np.array([c for c in CONFLICTS if abs(c) <= 0.35 + 1e-9])),
        ("±0.40",             np.array([c for c in CONFLICTS if abs(c) <= 0.40 + 1e-9])),
        ("±0.45",             np.array([c for c in CONFLICTS if abs(c) <= 0.45 + 1e-9])),
    ]
    for label, conflicts in subsets:
        expected_dBIC(results, n_trials_per_regime=1078,
                      conflict_subset=conflicts,
                      label=f"conflict range {label}, N≈1078/regime")
