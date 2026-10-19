"""Psychometric-function fitting used by the bias correction and the analysis notebooks.

lognormal:  P(test longer) = lambda/2 + (1 - lambda) * Phi((log(test/standard) - mu) / sigma)
linear_v1:  same with x = test/standard - 1 (the v1 cross-modal notebook)
Conditions get their own mu and sigma; the lapse rate is shared unless per_condition_lapse.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def stimulus_axis(df, scale):
    ratio = df["testDurS"].to_numpy(float) / df["standardDur"].to_numpy(float)
    return np.log(ratio) if scale == "lognormal" else ratio - 1


def aggregate(df, condition_col, scale):
    d = pd.DataFrame({"cond": df[condition_col].to_numpy(), "x": np.round(stimulus_axis(df, scale), 10),
                      "y": df["chose_test"].astype(float).to_numpy()})
    g = d.groupby(["cond", "x"]).y.agg(["sum", "count"]).reset_index()
    return g.rename(columns={"sum": "k", "count": "n"})


def fit(df, condition_col, scale="lognormal", lapse_max=0.1, per_condition_lapse=False, mu_fixed=None):
    """Maximum-likelihood fit. Returns a dict with per-condition mu/sigma/lapse and diagnostics."""
    g = aggregate(df, condition_col, scale)
    conds = sorted(g.cond.unique(), key=lambda c: (str(type(c)), c))
    idx = g.cond.map({c: i for i, c in enumerate(conds)}).to_numpy()
    x, k, n = g.x.to_numpy(), g.k.to_numpy(), g.n.to_numpy()
    nc = len(conds)
    n_lapse = nc if per_condition_lapse else 1
    free_mu = mu_fixed is None
    sigma_lo, sigma_hi = (0.01, 3.0) if scale == "lognormal" else (0.01, 2.0)
    mu_bound = 1.5 if scale == "lognormal" else 0.95

    def unpack(p):
        lam = p[:n_lapse]
        mu = p[n_lapse:n_lapse + nc] if free_mu else np.full(nc, mu_fixed)
        sig = p[n_lapse + (nc if free_mu else 0):]
        return lam, mu, sig

    def nll(p):
        lam, mu, sig = unpack(p)
        l_i = lam[idx] if per_condition_lapse else lam[0]
        pr = l_i / 2 + (1 - l_i) * norm.cdf((x - mu[idx]) / sig[idx])
        pr = np.clip(pr, 1e-9, 1 - 1e-9)
        return -np.sum(k * np.log(pr) + (n - k) * np.log(1 - pr))

    bounds = [(0.0, lapse_max)] * n_lapse + ([(-mu_bound, mu_bound)] * nc if free_mu else []) + [(sigma_lo, sigma_hi)] * nc
    best = None
    for lam0, mu0, sig0 in itertools.product((0.01, lapse_max / 2), (-0.3, 0.0, 0.3), (0.1, 0.3, 0.8)):
        x0 = [lam0] * n_lapse + ([mu0] * nc if free_mu else []) + [sig0] * nc
        r = minimize(nll, x0, bounds=bounds, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    lam, mu, sig = unpack(best.x)
    at_bound = []                       # a lapse of 0 is normal; any other bound hit is suspicious
    for name, v, (lo, hi) in zip(_names(n_lapse, nc, free_mu), best.x, bounds):
        if abs(v - hi) < 1e-4 or (abs(v - lo) < 1e-4 and not name.startswith("lapse")):
            at_bound.append(name)
    rows = []
    for i, c in enumerate(conds):
        rows.append(dict(condition=c, mu=mu[i], sigma=sig[i], lapse=lam[i] if per_condition_lapse else lam[0],
                         n=int(n[idx == i].sum())))
    return dict(params=pd.DataFrame(rows), nll=float(best.fun), at_bound=at_bound, scale=scale,
                success=bool(best.success), x=best.x, unpack=unpack, data=g, conditions=conds)


def _names(n_lapse, nc, free_mu):
    return ([f"lapse{i}" for i in range(n_lapse)] + ([f"mu{i}" for i in range(nc)] if free_mu else [])
            + [f"sigma{i}" for i in range(nc)])


def pse_ratio(mu, scale):
    """Test/standard ratio at the PSE."""
    return float(np.exp(mu)) if scale == "lognormal" else float(1 + mu)


def predict(x, mu, sigma, lapse):
    return lapse / 2 + (1 - lapse) * norm.cdf((np.asarray(x) - mu) / sigma)


def bootstrap(df, condition_col, scale, n_boot, rng, **fit_kwargs):
    """Parametric bootstrap of mu per condition: returns an (n_boot, n_conditions) array."""
    base = fit(df, condition_col, scale, **fit_kwargs)
    g = base["data"]
    params = base["params"].set_index("condition")
    p = predict(g.x, params.loc[g.cond, "mu"].to_numpy(), params.loc[g.cond, "sigma"].to_numpy(),
                params.loc[g.cond, "lapse"].to_numpy())
    out = []
    for _ in range(n_boot):
        k = rng.binomial(g.n.to_numpy(), p)
        sim = pd.DataFrame({condition_col: np.repeat(g.cond.to_numpy(), g.n), "x": np.repeat(g.x.to_numpy(), g.n)})
        sim["chose_test"] = np.concatenate([np.r_[np.ones(kk), np.zeros(nn - kk)] for kk, nn in zip(k, g.n)])
        if scale == "lognormal":
            sim["testDurS"], sim["standardDur"] = np.exp(sim.x), 1.0
        else:
            sim["testDurS"], sim["standardDur"] = sim.x + 1, 1.0
        b = fit(sim, condition_col, scale, **fit_kwargs)["params"].set_index("condition")
        out.append(b.loc[base["conditions"], "mu"].to_numpy())
    return np.array(out)
