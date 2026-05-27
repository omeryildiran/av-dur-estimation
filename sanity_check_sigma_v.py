"""
Sanity check on σ_v = 0.57.

Compares empirical PSE-vs-conflict slope (fit from the real data) to
the model-predicted PSE-vs-conflict slope using σ_v = 0.57. If the
empirical slope is substantially steeper than every model's prediction,
σ_v in the fit is too high — the data say the visual signal pulls
decisions more than σ_v = 0.57 allows. If empirical and predicted
slopes agree, σ_v is consistent with the data and the "CI variants are
informationally unreachable" conclusion stands.

All PFs fit with custom binomial / cross-entropy NLL + scipy.minimize.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

from monteCarloClass import OmerMonteCarlo
from loadData import loadData


# Empirical noise per audNoise condition, from group-mean fitted params
NOISE_REGIMES = {
    "low_aud_noise":  dict(audNoise=0.1, sigma_a=0.28, sigma_v=0.57),
    "high_aud_noise": dict(audNoise=1.2, sigma_a=0.80, sigma_v=0.57),
}

MODELS = ["lognorm", "selection", "probabilityMatchingLogNorm", "fusionOnlyLogNorm"]
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

P_C     = 0.75
LAMBDA  = 0.15
S_A_STD = 0.5
N_SIMUL = 5000
DELTAS  = np.linspace(-0.35, 0.35, 21)
EMP_CONFLICTS = np.array([-0.25, -0.17, -0.08, 0.0, 0.08, 0.17, 0.25])


# ---------------- PF fitters (NLL + minimize) ----------------

def cum_gauss(x, mu, sigma, lam):
    return lam / 2.0 + (1.0 - lam) * norm.cdf(x, loc=mu, scale=sigma)


def _binom_nll(params, deltas, n_chose, n_total):
    mu, sigma, lam = params
    if sigma <= 0 or lam < 0 or lam > 0.4:
        return 1e10
    p = cum_gauss(deltas, mu, sigma, lam)
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return -float(np.sum(n_chose * np.log(p)
                         + (n_total - n_chose) * np.log(1.0 - p)))


def _xent_nll(params, deltas, p_target):
    mu, sigma, lam = params
    if sigma <= 0 or lam < 0 or lam > 0.4:
        return 1e10
    p = cum_gauss(deltas, mu, sigma, lam)
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return -float(np.sum(p_target * np.log(p)
                         + (1.0 - p_target) * np.log(1.0 - p)))


def _multistart_minimize(nll_fn, args, bounds):
    inits = [[0.00, 0.10, 0.02],
             [0.00, 0.05, 0.02],
             [0.05, 0.15, 0.05],
             [-0.05, 0.20, 0.02]]
    best = None
    for x0 in inits:
        try:
            res = minimize(nll_fn, x0, args=args, bounds=bounds,
                           method="L-BFGS-B")
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            continue
    return best.x if best is not None else np.array([np.nan, np.nan, np.nan])


def fit_pf_binomial(deltas, n_chose, n_total):
    return _multistart_minimize(_binom_nll, (deltas, n_chose, n_total),
                                bounds=[(-0.5, 0.5), (1e-3, 1.0), (0.0, 0.4)])


def fit_pf_xent(deltas, p_target):
    return _multistart_minimize(_xent_nll, (deltas, p_target),
                                bounds=[(-0.5, 0.5), (1e-3, 1.0), (0.0, 0.4)])


# ---------------- Empirical & model PSEs ----------------

def empirical_pse_per_condition(mc, audNoise):
    """Fit PF per (conflict) at the given audNoise. Returns DataFrame
    with conflict, μ, σ, λ, N."""
    g = mc.groupedData
    sub = g[np.isclose(g["audNoise"], audNoise)]
    out = []
    for c in sorted(sub["conflictDur"].unique()):
        s = sub[np.isclose(sub["conflictDur"], c)]
        if len(s) < 3:
            continue
        deltas  = s["deltaDurS"].values.astype(float)
        n_chose = s["num_of_chose_test"].values.astype(float)
        n_total = s["total_responses"].values.astype(float)
        mu, sigma, lam = fit_pf_binomial(deltas, n_chose, n_total)
        out.append(dict(conflict=float(c), mu=mu, sigma=sigma, lam=lam,
                        N=int(n_total.sum())))
    return pd.DataFrame(out)


def model_predicted_pse(mc, model_name, sigma_a, sigma_v,
                        conflicts=EMP_CONFLICTS):
    mc.modelName = model_name
    t_min, t_max = mc.data_t_min, mc.data_t_max
    out = []
    for c in conflicts:
        p_long = np.empty_like(DELTAS)
        for i, d in enumerate(DELTAS):
            S_a_s = S_A_STD
            S_v_s = S_a_s + c
            S_a_t = S_a_s + d
            S_v_t = S_a_t
            p_long[i] = mc.probTestLonger_vectorized_mc(
                (S_a_s, S_a_t, S_v_s, S_v_t),
                sigma_a, sigma_v, P_C, LAMBDA, t_min, t_max,
            )
        mu, sigma, lam = fit_pf_xent(DELTAS, p_long)
        out.append(dict(conflict=float(c), mu=mu, sigma=sigma, lam=lam))
    return pd.DataFrame(out)


# ---------------- Main ----------------

def main():
    np.random.seed(0)
    data, _ = loadData("mt_all.csv")
    mc = OmerMonteCarlo(data)
    mc.nSimul = N_SIMUL

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    summary_rows = []

    for j, (regime_name, regime) in enumerate(NOISE_REGIMES.items()):
        ax = axes[j]

        emp = empirical_pse_per_condition(mc, regime["audNoise"])
        emp_slope, emp_int = np.polyfit(emp["conflict"], emp["mu"], 1)

        print(f"\n[{regime_name}]  (σ_a={regime['sigma_a']}, "
              f"σ_v={regime['sigma_v']}, audNoise={regime['audNoise']})")
        print(f"  empirical PSE slope (μ/Δconflict) = {emp_slope:.3f}   "
              f"intercept = {emp_int*1000:+.1f} ms")
        for _, row in emp.iterrows():
            print(f"    conflict {int(row['conflict']*1000):+4d} ms: "
                  f"μ={row['mu']*1000:+6.1f}  σ={row['sigma']*1000:5.1f}  "
                  f"λ={row['lam']:.3f}  N={int(row['N'])}")

        ax.plot(emp["conflict"] * 1000, emp["mu"] * 1000,
                "ko-", ms=8, lw=2.5, label="data (PF fit)")

        for model in MODELS:
            pred = model_predicted_pse(
                mc, model, regime["sigma_a"], regime["sigma_v"],
            )
            slope, intercept = np.polyfit(pred["conflict"], pred["mu"], 1)
            print(f"  {LABELS[model]:>14}  predicted slope = {slope:.3f}   "
                  f"intercept = {intercept*1000:+.1f} ms   "
                  f"(empirical/pred = {emp_slope/slope:.2f})")
            ax.plot(pred["conflict"] * 1000, pred["mu"] * 1000,
                    "-o", color=COLORS[model], label=LABELS[model],
                    lw=1.8, ms=5, alpha=0.85)
            summary_rows.append(dict(
                regime=regime_name, model=LABELS[model],
                emp_slope=emp_slope, pred_slope=slope,
                ratio=emp_slope / slope,
            ))

        ax.set_title(f"{regime_name}  (σ_a={regime['sigma_a']}, "
                     f"σ_v={regime['sigma_v']})")
        ax.set_xlabel("conflict (ms)")
        if j == 0:
            ax.set_ylabel("PSE μ (ms)")
        ax.axhline(0, color="gray", ls="--", lw=0.6)
        ax.axvline(0, color="gray", ls="--", lw=0.6)
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("Sanity check — empirical vs predicted PSE-vs-conflict "
                 f"at σ_v = {NOISE_REGIMES['low_aud_noise']['sigma_v']}",
                 fontsize=12)
    fig.savefig("sanity_check_sigma_v.pdf", bbox_inches="tight")
    fig.savefig("sanity_check_sigma_v.png", dpi=150, bbox_inches="tight")
    print("\nsaved sanity_check_sigma_v.pdf / .png")

    # Compact summary
    print("\n=== SUMMARY: empirical slope / predicted slope ===")
    print("(ratio >> 1 means data has a steeper visual-pull than σ_v=0.57 predicts)")
    sm = pd.DataFrame(summary_rows)
    for regime_name, grp in sm.groupby("regime"):
        print(f"\n  [{regime_name}]   emp slope = {grp['emp_slope'].iloc[0]:.3f}")
        for _, r in grp.iterrows():
            print(f"    {r['model']:<14}  pred slope = {r['pred_slope']:.3f}   "
                  f"ratio = {r['ratio']:.2f}")


if __name__ == "__main__":
    main()
