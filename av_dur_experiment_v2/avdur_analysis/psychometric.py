"""Lognormal psychometric functions per condition, their parametric bootstrap, and the saved results.

    P(test longer) = lapse/2 + (1 - lapse) * Phi((x - mu) / sigma),   x = log(test / standard)

Each condition has its own mu and sigma; conditions in one lapse group share the lapse rate (unimodal: one
per block, so A_low and A_high share one and V has its own; main: one per noise level, as
analysis/main_bias_plots.py). One lapse over all three unimodal conditions sat at the 0.1 bound for MZ,
BQ and OY (16 Nov 2026): their errors on the easy trials are all at high noise (80-90% correct, 100% at
low noise and visual), so a shared lapse also inflated the visual one. lapse_at_bound flags a lapse at 0.1.

sigma here is the psychometric (comparison) sigma of the 2IFC task. Both intervals are measured with noise,
so the noise of a single interval is sigma_single = sigma / sqrt(2); the models use sigma_single.
Derived (all from the fitted curve, so they hold for any sigma):
    PSE (ms)       = 1000 * standard * exp(mu)
    shift (ms)     = PSE - 500                          (v1's "PSE shift")
    JND (ms)       = PSE * (exp(z75 * sigma) - 1)       test increment from 50% to 75% "longer" (no lapse)
    Weber fraction = JND / PSE = exp(z75 * sigma) - 1
`beyond_range` marks a condition whose 75% point lies above the longest staircase test, so its JND is
extrapolated (a reviewer's point on v1: the high-noise threshold lay outside the tested range).

Bootstrap: parametric. Answers are drawn from the fitted curves at the tested levels (same number of
trials per level) and refitted, starting from the fit; percentile 95% intervals.
Presentation order (reviewer 1 on v1): the analyses "unimodal_order" and "main_order" fit every condition
twice, once on the test-first trials (order 1) and once on the test-second trials (order 2), with the same
lapse groups as the pooled fits (the lapse is shared by both orders). Their conditions are labelled
"<condition>|test_first" / "|test_second"; the summary has the columns base_condition and order.
Sessions: "main_part" fits every condition once per main-experiment part (part1-3, separate days), lapse
per noise level and part; the summary column part names it.
Results: results/psychometric/<complete|partial>/<pid>_<analysis>{.csv,_boot.npz,_fit.json}; a saved
fit is reused while the data files, trial count and settings are unchanged.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from . import data as D

Z75 = float(norm.ppf(0.75))
LAPSE_MAX = 0.1
MU_BOUNDS = (-2.0, 2.0)
SIGMA_BOUNDS = (0.01, 5.0)          # wider than the screening fit's 3.0, so hitting it is rare and flagged
N_BOOT = 1000
FIT_SETTINGS = dict(lapse_max=LAPSE_MAX, mu_bounds=MU_BOUNDS, sigma_bounds=SIGMA_BOUNDS, min_rt_s=D.MIN_RT_S)


@dataclass
class Fit:
    conditions: list
    lapse_group: np.ndarray          # per condition
    mu: np.ndarray
    sigma: np.ndarray
    lapse: np.ndarray                # per condition (shared within a lapse group)
    nll: float
    at_bound: list
    agg: pd.DataFrame = field(repr=False)


def aggregate(d, cond_col="condition"):
    g = (d.assign(xr=d["x"].round(10)).groupby([cond_col, "xr"])["chose_test"]
         .agg(k="sum", n="count").reset_index().rename(columns={cond_col: "condition", "xr": "x"}))
    return g


def predict(x, mu, sigma, lapse):
    return lapse / 2 + (1 - lapse) * norm.cdf((np.asarray(x) - mu) / sigma)


class _Problem:
    def __init__(self, agg, conditions, lapse_group):
        self.ci = agg["condition"].map({c: i for i, c in enumerate(conditions)}).to_numpy()
        self.x, self.k, self.n = agg["x"].to_numpy(float), agg["k"].to_numpy(float), agg["n"].to_numpy(float)
        self.lapse_group = np.asarray(lapse_group)
        self.gi = self.lapse_group[self.ci]
        self.nc, self.ng = len(conditions), int(self.lapse_group.max()) + 1
        self.bounds = ([(0.0, LAPSE_MAX)] * self.ng + [MU_BOUNDS] * self.nc + [SIGMA_BOUNDS] * self.nc)

    def unpack(self, th):
        return th[:self.ng], th[self.ng:self.ng + self.nc], th[self.ng + self.nc:]

    def nll_grad(self, th, k=None):
        k = self.k if k is None else k
        lam, mu, sig = self.unpack(th)
        lr, s = lam[self.gi], sig[self.ci]
        z = (self.x - mu[self.ci]) / s
        cdf, pdf = norm.cdf(z), norm.pdf(z)
        p = np.clip(lr / 2 + (1 - lr) * cdf, 1e-12, 1 - 1e-12)
        nll = -np.sum(k * np.log(p) + (self.n - k) * np.log1p(-p))
        dp = -k / p + (self.n - k) / (1 - p)
        g_lam = np.bincount(self.gi, dp * (0.5 - cdf), minlength=self.ng)
        g_mu = np.bincount(self.ci, dp * (-(1 - lr) * pdf / s), minlength=self.nc)
        g_sig = np.bincount(self.ci, dp * (-(1 - lr) * pdf * z / s), minlength=self.nc)
        return nll, np.concatenate([g_lam, g_mu, g_sig])

    def start(self, k=None):
        """Per-condition grid search (lapse 0.02) for mu and sigma."""
        k = self.k if k is None else k
        mus, sigs = np.linspace(-1.5, 1.5, 61), np.geomspace(0.03, 4.0, 40)
        M, Sg = np.meshgrid(mus, sigs, indexing="ij")
        M, Sg = M.ravel(), Sg.ravel()
        p = np.clip(0.01 + 0.98 * norm.cdf((self.x[:, None] - M[None]) / Sg[None]), 1e-12, 1 - 1e-12)
        ll = k[:, None] * np.log(p) + (self.n - k)[:, None] * np.log1p(-p)
        tot = np.zeros((self.nc, len(M)))
        np.add.at(tot, self.ci, ll)
        best = tot.argmax(1)
        return M[best], Sg[best]

    def solve(self, x0, k=None):
        return minimize(self.nll_grad, x0, args=(k,), jac=True, method="L-BFGS-B", bounds=self.bounds)


def _flags(prob, th):
    names = ([f"lapse[{g}]" for g in range(prob.ng)] + [f"mu[{c}]" for c in range(prob.nc)]
             + [f"sigma[{c}]" for c in range(prob.nc)])
    out = []
    for name, v, (lo, hi) in zip(names, th, prob.bounds):
        if abs(v - hi) < 1e-4 or (abs(v - lo) < 1e-4 and not name.startswith("lapse")):
            out.append(name)
    return out


def fit(d, lapse_by=None, cond_col="condition"):
    """Maximum-likelihood fit of every condition in `d`. lapse_by: column whose values form the lapse groups
    (None: one lapse for all conditions)."""
    agg = aggregate(d, cond_col)
    key = None if lapse_by is None else d.groupby(cond_col)[lapse_by].first().astype(str)
    return fit_cells(agg, key)


def fit_cells(agg, lapse_key=None):
    """Fit aggregated cells (condition, x, k, n); k may be fractional (expected counts of a model).
    lapse_key: Series condition -> lapse-group label (None: one lapse)."""
    conditions = sorted(agg["condition"].unique(), key=_cond_key)
    if lapse_key is None:
        lapse_group = np.zeros(len(conditions), int)
    else:
        levels = sorted(set(lapse_key[c] for c in conditions))
        lapse_group = np.array([levels.index(lapse_key[c]) for c in conditions])
    prob = _Problem(agg, conditions, lapse_group)
    mu0, sig0 = prob.start()
    best = None
    for lam0 in (0.02, 0.08):
        r = prob.solve(np.r_[np.full(prob.ng, lam0), mu0, sig0])
        if best is None or r.fun < best.fun:
            best = r
    lam, mu, sig = prob.unpack(best.x)
    return Fit(conditions, lapse_group, mu, sig, lam[lapse_group], float(best.fun), _flags(prob, best.x), agg)


def bootstrap(f, n_boot=N_BOOT, seed=0):
    """Parametric bootstrap: arrays (n_boot, n_conditions) of mu, sigma and lapse."""
    rng = np.random.default_rng(seed)
    prob = _Problem(f.agg, f.conditions, f.lapse_group)
    lam_g = np.array([f.lapse[np.argmax(f.lapse_group == g)] for g in range(prob.ng)])
    th0 = np.r_[lam_g, f.mu, f.sigma]
    lam, mu, sig = prob.unpack(th0)
    p = predict(prob.x, mu[prob.ci], sig[prob.ci], lam[prob.gi])
    out = np.empty((n_boot, 3, prob.nc))
    for b in range(n_boot):
        k = rng.binomial(prob.n.astype(int), p).astype(float)
        r = prob.solve(th0, k)
        lb, mb, sb = prob.unpack(r.x)
        out[b] = mb, sb, lb[f.lapse_group]
    return dict(mu=out[:, 0], sigma=out[:, 1], lapse=out[:, 2])


def derived(mu, sigma):
    """PSE, shift, JND (ms) and Weber fraction from mu and sigma (arrays of any shape)."""
    pse = 1000 * D.STANDARD_S * np.exp(mu)
    wf = np.expm1(Z75 * sigma)
    return dict(pse_ms=pse, shift_ms=pse - 1000 * D.STANDARD_S, jnd_ms=pse * wf, weber=wf,
                sigma_single=sigma / np.sqrt(2))


def summary(f, boot, d, cond_col="condition"):
    """One row per condition: estimates, 95% bootstrap intervals, flags."""
    est = dict(mu=f.mu, sigma=f.sigma, lapse=f.lapse, **derived(f.mu, f.sigma))
    bd = dict(mu=boot["mu"], sigma=boot["sigma"], lapse=boot["lapse"], **derived(boot["mu"], boot["sigma"]))
    first = d.groupby(cond_col).first()
    stair_max = d[d["kind"] == "staircase"].groupby(cond_col)["testDurS"].max()
    rows = []
    for i, c in enumerate(f.conditions):
        base, _, split = str(c).partition("|")
        row = dict(condition=c, base_condition=base, order=split if split in ORDER_NAMES.values() else "both",
                   part=split if split.startswith("part") else "all", task=first.loc[c, "task"], noise=first.loc[c, "noise"],
                   conflict_ms=int(first.loc[c, "conflict_ms"]), conflict_s=float(first.loc[c, "conflict"]),
                   n_trials=int((d[cond_col] == c).sum()))
        for name, v in est.items():
            row[name] = float(v[i])
            lo, hi = np.percentile(bd[name][:, i], [2.5, 97.5])
            row[f"{name}_lo"], row[f"{name}_hi"] = float(lo), float(hi)
        row["at_bound"] = ";".join(n for n in f.at_bound if n.endswith(f"[{i}]") and not n.startswith("lapse"))
        row["lapse_at_bound"] = bool(f.lapse[i] > LAPSE_MAX - 1e-4)
        row["beyond_range"] = bool(row["pse_ms"] + row["jnd_ms"] > 1000 * stair_max.get(c, np.inf))
        rows.append(row)
    return pd.DataFrame(rows)


# ── analyses, saving and loading ──────────────────────────────────────────────

ORDER_NAMES = {1: "test_first", 2: "test_second"}


def by_order(d):
    """Trials with the presentation order appended to the condition label (see the module docstring)."""
    if d.empty:
        return d
    d = d[d["order"].isin(list(ORDER_NAMES))].copy()
    d["condition"] = d["condition"].astype(str) + "|" + d["order"].astype(int).map(ORDER_NAMES)
    return d


def by_part(d):
    """Main-experiment trials with the part (session) appended to the condition label; lapse groups are
    noise level x part (column noise_part)."""
    if d.empty:
        return d
    d = d.copy()
    d["condition"] = d["condition"].astype(str) + "|" + d["block"].astype(str)
    d["noise_part"] = d["noise"].astype(str) + "|" + d["block"].astype(str)
    return d


ANALYSES = {
    # name: (loader, lapse grouping column)
    "unimodal": (lambda pid: D.load_unimodal(pid)[0], "task"),
    "main": (lambda pid: D.load(pid, "main")[0], "noise"),
    "unimodal_order": (lambda pid: by_order(D.load_unimodal(pid)[0]), "task"),
    "main_order": (lambda pid: by_order(D.load(pid, "main")[0]), "noise"),
    "main_part": (lambda pid: by_part(D.load(pid, "main")[0]), "noise_part"),
}


def _cond_key(c):
    c, _, order = str(c).partition("|")
    return _base_key(c) + (order,)


def _base_key(c):
    parts = str(c).split("_")
    order = {"A": 0, "V": 1, "AV": 2}
    noise = {"low": 0, "high": 1}
    conflict = int(parts[2]) if len(parts) > 2 and parts[2].lstrip("+-").isdigit() else 0
    return (order.get(parts[0], 9), noise.get(parts[1], 9) if len(parts) > 1 else 0, conflict, str(c))


def _fingerprint(d, n_boot, seed, lapse_by):
    files = sorted(d["session_file"].unique()) if "session_file" in d else []
    blob = json.dumps(dict(files=files, n=len(d), k=int(d["chose_test"].sum()), settings=FIT_SETTINGS,
                           n_boot=n_boot, seed=seed, lapse_by=lapse_by), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12], files


def result_paths(pid, analysis, state):
    folder = D.RESULTS / "psychometric" / state
    stem = folder / f"{pid}_{analysis}"
    return dict(csv=stem.with_suffix(".csv"), boot=folder / f"{pid}_{analysis}_boot.npz",
                meta=folder / f"{pid}_{analysis}_fit.json")


def run(pid, analysis="main", n_boot=N_BOOT, seed=0, refit=False, verbose=True):
    """Fit (or load the saved fit of) one participant's `analysis`. Returns the summary table, or None
    without data. Saved in results/psychometric/<complete|partial>/."""
    loader, lapse_by = ANALYSES[analysis]
    d = loader(pid)
    if d.empty:
        return None
    state = "complete" if D.status(pid)["complete"] else "partial"
    paths = result_paths(pid, analysis, state)
    key, files = _fingerprint(d, n_boot, seed, lapse_by)
    if not refit and paths["meta"].exists() and json.loads(paths["meta"].read_text()).get("fingerprint") == key:
        return pd.read_csv(paths["csv"])
    if verbose:
        print(f"[psychometric] fitting {pid} {analysis} ({len(d)} trials, {n_boot} bootstrap samples)", flush=True)
    f = fit(d, lapse_by)
    boot = bootstrap(f, n_boot, seed)
    table = summary(f, boot, d).assign(participant=pid, status=state, analysis=analysis)
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    for other in ("complete", "partial"):              # a participant who became complete: drop the old copy
        if other != state:
            for p in result_paths(pid, analysis, other).values():
                p.unlink(missing_ok=True)
    table.to_csv(paths["csv"], index=False)
    np.savez_compressed(paths["boot"], conditions=np.array(f.conditions), **boot)
    paths["meta"].write_text(json.dumps(dict(
        participant=pid, analysis=analysis, status=state, fingerprint=key, data_files=files, n_trials=len(d),
        nll=f.nll, at_bound=f.at_bound, lapse_groups=lapse_by, n_boot=n_boot, seed=seed, settings=FIT_SETTINGS,
        fitted=dt.datetime.now().isoformat(timespec="seconds")), indent=2))
    return table


def load_boot(pid, analysis):
    """Saved bootstrap arrays of one participant (dict with conditions, mu, sigma, lapse) or None."""
    for state in ("complete", "partial"):
        p = result_paths(pid, analysis, state)["boot"]
        if p.exists():
            z = np.load(p, allow_pickle=False)
            return {k: z[k] for k in z.files}
    return None


def collect(analysis="main", pids=None, **kw):
    """Summary tables of every participant with data for `analysis`, stacked."""
    tables = [run(pid, analysis, **kw) for pid in (pids or D.participants())]
    tables = [t for t in tables if t is not None]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
