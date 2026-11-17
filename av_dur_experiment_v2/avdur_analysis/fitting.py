"""Joint fits: the unimodal blocks and the main experiment in one likelihood, so the noise parameters are
constrained by both (v1 fitted the unimodal blocks first and fixed sigma, or freed it in the bimodal fit).

Parameters (single-interval sigmas, log units; see models.py):
    all models   lapse, sigma_a_low, sigma_a_high, sigma_v
    causal       + p_c                         (one prior for both noise levels; p_c="per_noise" for two)
    switching    + p_switch_low, p_switch_high (probability of using vision, per noise level)
    fusion       nothing more
Likelihood: binomial over every (block, noise, conflict, test duration) cell of the cleaned trials
(data.py): unimodal auditory P = lapse/2 + (1 - lapse) Phi(log(t/0.5) / (sqrt 2 sigma_a)), unimodal visual
likewise with sigma_v, main experiment the model's P with the same lapse. lapse="per_block" gives the
unimodal auditory, unimodal visual and main blocks a lapse each.

Optimiser: L-BFGS-B from N_STARTS starts (the first from the unimodal psychometric fits, the rest spread
over the bounds by a fixed-seed Sobol sequence). The likelihood has no Monte Carlo noise, so the best start
is the fit; `n_near_best` (starts within 0.5 of the best -log L) shows how reliably it is found.
optimizer="bads" (CLI --optimizer bads) uses BADS (pybads; Acerbi & Ma 2017) from the same starts instead,
as in v1 and the lab; its fits are saved with the suffix _bads so the two can be compared. v1 needed BADS
because its likelihoods were Monte Carlo (noisy); here they are exact and smooth, so the gradient method is
the faster default, and BADS is the cross-check. Checked 17 Nov 2026 (MZ, KJ, BQ, EK, all three models):
the same best -log L within 0.001 and the same parameters within 0.001; BADS took 15-60 s per fusion or
switching fit (L-BFGS-B under 7 s) and about the same time for causal inference.
Fits are saved to results/model_fits/<complete|partial>/<pid>_<model>[_options].json with every start,
the settings and the data files, and reused until the data or settings change.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from . import data as D
from . import models as M

N_STARTS = 6
OPTIMIZERS = ("lbfgsb", "bads")
BOUNDS = dict(lapse=(0.0, 0.2), sigma=(0.01, 4.0), prob=(0.0, 1.0))
PLAUSIBLE = dict(lapse=(0.001, 0.1), sigma=(0.03, 2.0), prob=(0.01, 0.99))     # BADS only: where the starts lie
FIT_DIR = D.RESULTS / "model_fits"
EPS = 1e-12


@dataclass
class Design:
    pid: str
    uni_a: pd.DataFrame        # noise, log_t, k, n
    uni_v: pd.DataFrame        # log_t, k, n
    main: pd.DataFrame         # noise, conflict, log_t, log_sv, k, n
    files: list
    n_trials: int


def design(pid):
    """Binomial cells of one participant's cleaned trials."""
    uni, _ = D.load_unimodal(pid)
    main, _ = D.load(pid, "main")
    cells = lambda d, by: (d.assign(log_t=np.log(d["testDurS"].astype(float)).round(10))
                           .groupby(by + ["log_t"])["chose_test"].agg(k="sum", n="count").reset_index())
    ua = cells(uni[uni.task == "unimodal_auditory"], ["noise"]) if not uni.empty else pd.DataFrame()
    uv = cells(uni[uni.task == "unimodal_visual"], []) if not uni.empty else pd.DataFrame()
    mm = cells(main, ["noise", "conflict"]) if not main.empty else pd.DataFrame()
    if not mm.empty:
        mm["log_sv"] = np.log(D.STANDARD_S + mm["conflict"])
    files = sorted(set(uni.get("session_file", pd.Series(dtype=str))) | set(main.get("session_file", pd.Series(dtype=str))))
    return Design(pid, ua, uv, mm, files, int(len(uni) + len(main)))


def param_names(model, lapse="shared", p_c="shared"):
    names = ["lapse"] if lapse == "shared" else ["lapse_uni_a", "lapse_uni_v", "lapse_main"]
    names += ["sigma_a_low", "sigma_a_high", "sigma_v"]
    if model == "causal":
        names += ["p_c"] if p_c == "shared" else ["p_c_low", "p_c_high"]
    elif model == "switching":
        names += ["p_switch_low", "p_switch_high"]
    return names


def _bounds(names):
    return [BOUNDS["lapse"] if n.startswith("lapse") else BOUNDS["sigma"] if n.startswith("sigma")
            else BOUNDS["prob"] for n in names]


def _binom_ll(p, k, n):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(k * np.log(p) + (n - k) * np.log1p(-p)))


def _noise_key(noise):
    return "low" if np.isclose(noise, D.NOISE_LOW) else "high"


def predict_main(model, prm, noise, log_t, log_sv):
    """Model P(test longer) without lapse for main-experiment cells of one noise level."""
    sa, sv = prm[f"sigma_a_{_noise_key(noise)}"], prm["sigma_v"]
    if model == "fusion":
        return M.p_fusion(log_t, log_sv, sa, sv)
    if model == "switching":
        return M.p_switching(log_t, log_sv, sa, sv, prm[f"p_switch_{_noise_key(noise)}"])
    pc = prm.get("p_c", prm.get(f"p_c_{_noise_key(noise)}"))
    return M.CausalNoiseLevel(sa, sv, pc).p(log_t, log_sv)


def neg_log_lik(theta, model, des, names, parts=False):
    prm = dict(zip(names, theta))
    lam = lambda block: prm.get("lapse", prm.get(f"lapse_{block}"))
    ll = {}
    if not des.uni_a.empty:
        sig = np.where(np.isclose(des.uni_a.noise, D.NOISE_LOW), prm["sigma_a_low"], prm["sigma_a_high"])
        p = lam("uni_a") / 2 + (1 - lam("uni_a")) * M.p_unimodal(des.uni_a.log_t.to_numpy(), sig)
        ll["uni_a"] = _binom_ll(p, des.uni_a.k.to_numpy(), des.uni_a.n.to_numpy())
    if not des.uni_v.empty:
        p = lam("uni_v") / 2 + (1 - lam("uni_v")) * M.p_unimodal(des.uni_v.log_t.to_numpy(), prm["sigma_v"])
        ll["uni_v"] = _binom_ll(p, des.uni_v.k.to_numpy(), des.uni_v.n.to_numpy())
    if not des.main.empty:
        ll["main"] = 0.0
        for noise, g in des.main.groupby("noise"):
            p = predict_main(model, prm, noise, g.log_t.to_numpy(), g.log_sv.to_numpy())
            p = lam("main") / 2 + (1 - lam("main")) * p
            ll["main"] += _binom_ll(p, g.k.to_numpy(), g.n.to_numpy())
    return ll if parts else -sum(ll.values())


def _start_from_data(des, names):
    """First start: sigma from the unimodal cells (probit slope by a quick 1-D search), lapse 0.02, p 0.5."""
    out = []
    for n in names:
        if n.startswith("lapse"):
            out.append(0.02)
        elif n.startswith("sigma"):
            if n == "sigma_v":
                cells = des.uni_v
            else:
                cells = des.uni_a[np.isclose(des.uni_a.noise, D.NOISE_LOW if n.endswith("low") else D.NOISE_HIGH)] \
                    if not des.uni_a.empty else pd.DataFrame()
            if cells is None or cells.empty:
                out.append(0.4)
                continue
            grid = np.geomspace(0.03, 3.0, 60)
            lls = [_binom_ll(0.01 + 0.98 * M.p_unimodal(cells.log_t.to_numpy(), s), cells.k.to_numpy(), cells.n.to_numpy())
                   for s in grid]
            out.append(float(grid[int(np.argmax(lls))]))
        else:
            out.append(0.5)
    return np.array(out)


def _starts(des, names, n_starts, seed):
    bounds = np.array(_bounds(names))
    first = _start_from_data(des, names)
    if n_starts <= 1:
        return [first]
    m = int(np.ceil(np.log2(max(2, n_starts - 1))))
    u = qmc.Sobol(len(names), scramble=True, seed=seed).random_base2(m)[: n_starts - 1]
    rest = []
    for row in u:
        x = []
        for ui, n, (lo, hi) in zip(row, names, bounds):
            if n.startswith("sigma"):
                x.append(float(np.exp(np.log(0.05) + ui * (np.log(2.0) - np.log(0.05)))))
            elif n.startswith("lapse"):
                x.append(float(0.1 * ui))
            else:
                x.append(float(0.05 + 0.9 * ui))
        rest.append(np.array(x))
    return [first] + rest


def _minimize(x0, model, des, names, bounds, optimizer="lbfgsb", seed=0):
    """One start: dict(x, nll, converged, n_eval)."""
    if optimizer == "lbfgsb":
        r = minimize(neg_log_lik, x0, args=(model, des, names), method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=500))
        return dict(x=[float(v) for v in r.x], nll=float(r.fun), converged=bool(r.success), n_eval=int(r.nfev))
    if optimizer != "bads":
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    from pybads import BADS
    kind = lambda n: "lapse" if n.startswith("lapse") else "sigma" if n.startswith("sigma") else "prob"
    lb, ub = np.array(bounds, float).T
    plb = np.array([PLAUSIBLE[kind(n)][0] for n in names])
    pub = np.array([PLAUSIBLE[kind(n)][1] for n in names])
    x0 = np.clip(np.asarray(x0, float), plb, pub)                  # BADS needs the start inside the plausible box
    r = BADS(lambda x: neg_log_lik(np.asarray(x), model, des, names), x0, lb, ub, plb, pub,
             options=dict(display="off", uncertainty_handling=False, random_seed=seed)).optimize()
    return dict(x=[float(v) for v in r["x"]], nll=float(r["fval"]), converged=bool(r["success"]),
                n_eval=int(r["func_count"]))


def _key(des, model, names, n_starts, seed, options):
    blob = json.dumps(dict(files=des.files, n=des.n_trials, model=model, names=names, n_starts=n_starts, seed=seed,
                           bounds=BOUNDS, box=[M.T_MIN, M.T_MAX], quad=[M.N_GH, M.N_GRID, M.N_E],
                           min_rt=D.MIN_RT_S, options=options), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def fit_path(pid, model, state, lapse="shared", p_c="shared", optimizer="lbfgsb"):
    suffix = ("" if lapse == "shared" else f"_lapse-{lapse}") + ("" if p_c == "shared" or model != "causal" else f"_pc-{p_c}")
    suffix += "" if optimizer == "lbfgsb" else f"_{optimizer}"
    return FIT_DIR / state / f"{pid}_{model}{suffix}.json"


def fit(pid, model, lapse="shared", p_c="shared", n_starts=N_STARTS, seed=0, refit=False, verbose=True,
        optimizer="lbfgsb"):
    """Joint fit of one participant (needs main-experiment data). Returns the saved record (dict)."""
    des = design(pid)
    if des.main.empty:
        return None
    state = "complete" if D.status(pid)["complete"] else "partial"
    names = param_names(model, lapse, p_c)
    options = dict(lapse=lapse, p_c=p_c, **({} if optimizer == "lbfgsb" else dict(optimizer=optimizer)))
    key = _key(des, model, names, n_starts, seed, options)
    path = fit_path(pid, model, state, lapse, p_c, optimizer)
    if not refit and path.exists():
        rec = json.loads(path.read_text())
        if rec.get("fingerprint") == key:
            return rec
    t0 = time.time()
    bounds = _bounds(names)
    starts = []
    for x0 in _starts(des, names, n_starts, seed):
        r = _minimize(x0, model, des, names, bounds, optimizer, seed + len(starts))
        starts.append(dict(start=[float(v) for v in x0], **r))
        if verbose:
            print(f"[fit] {pid} {model} ({optimizer}): start {len(starts)}/{n_starts} -log L {r['nll']:.2f}", flush=True)
    best = min(starts, key=lambda s: s["nll"])
    k, n = len(names), des.n_trials
    parts = neg_log_lik(np.array(best["x"]), model, des, names, parts=True)
    at_bound = [nm for nm, v, (lo, hi) in zip(names, best["x"], bounds)
                if (abs(v - hi) < 1e-4 or abs(v - lo) < 1e-4) and not (nm.startswith("lapse") and v < 1e-4)]
    rec = dict(participant=pid, status=state, model=model, lapse=lapse, p_c=p_c, optimizer=optimizer, fingerprint=key,
               params=dict(zip(names, best["x"])), nll=best["nll"], n_params=k, n_trials=n,
               aic=2 * k + 2 * best["nll"], bic=k * np.log(n) + 2 * best["nll"],
               loglik_parts={kk: float(v) for kk, v in parts.items()}, at_bound=at_bound,
               n_near_best=int(sum(s["nll"] < best["nll"] + 0.5 for s in starts)), starts=starts,
               data_files=des.files, settings=dict(bounds=BOUNDS, box_s=[M.T_MIN, M.T_MAX],
                                                   quadrature=dict(n_gh=M.N_GH, n_grid=M.N_GRID, n_e=M.N_E),
                                                   min_rt_s=D.MIN_RT_S, n_starts=n_starts, seed=seed),
               fitted=dt.datetime.now().isoformat(timespec="seconds"), seconds=round(time.time() - t0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    for other in ("complete", "partial"):
        if other != state:
            fit_path(pid, model, other, lapse, p_c, optimizer).unlink(missing_ok=True)
    path.write_text(json.dumps(rec, indent=2))
    return rec


def _fit_job(args):
    pid, model, kw = args
    return fit(pid, model, **kw)


def fit_all(pids=None, models=M.MODELS, workers=1, **kw):
    """Fit every participant with main-experiment data under every model; one row per fit. workers > 1 runs
    the fits in parallel processes (saved fits are loaded either way)."""
    jobs = [(pid, model, kw) for pid in (pids or D.participants()) for model in models]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            recs = list(ex.map(_fit_job, jobs))
    else:
        recs = [_fit_job(j) for j in jobs]
    return pd.DataFrame([record_row(r) for r in recs if r is not None])


def cell_probabilities(model, prm, des):
    """P(test longer) with lapse for every cell of the design: dict uni_a / uni_v / main -> array."""
    lam = lambda block: prm.get("lapse", prm.get(f"lapse_{block}"))
    out = {}
    if not des.uni_a.empty:
        sig = np.where(np.isclose(des.uni_a.noise, D.NOISE_LOW), prm["sigma_a_low"], prm["sigma_a_high"])
        out["uni_a"] = lam("uni_a") / 2 + (1 - lam("uni_a")) * M.p_unimodal(des.uni_a.log_t.to_numpy(), sig)
    if not des.uni_v.empty:
        out["uni_v"] = lam("uni_v") / 2 + (1 - lam("uni_v")) * M.p_unimodal(des.uni_v.log_t.to_numpy(), prm["sigma_v"])
    if not des.main.empty:
        p = np.empty(len(des.main))
        for noise, g in des.main.groupby("noise"):
            p[des.main.index.get_indexer(g.index)] = predict_main(model, prm, noise, g.log_t.to_numpy(), g.log_sv.to_numpy())
        out["main"] = lam("main") / 2 + (1 - lam("main")) * p
    return out


def simulate_design(des, model, prm, rng):
    """The same cells (stimuli and trial counts) with answers drawn from the model."""
    probs = cell_probabilities(model, prm, des)
    new = {}
    for part in ("uni_a", "uni_v", "main"):
        frame = getattr(des, part)
        if part in probs:
            frame = frame.copy()
            frame["k"] = rng.binomial(frame["n"].to_numpy(int), probs[part])
        new[part] = frame
    return Design(des.pid, new["uni_a"], new["uni_v"], new["main"], des.files, des.n_trials)


def fit_design(des, model, names=None, n_starts=N_STARTS, seed=0):
    """Fit a Design (e.g. simulated) without saving: (params dict, -log L, all start results)."""
    names = names or param_names(model)
    bounds = _bounds(names)
    res = [minimize(neg_log_lik, x0, args=(model, des, names), method="L-BFGS-B", bounds=bounds,
                    options=dict(maxiter=500)) for x0 in _starts(des, names, n_starts, seed)]
    best = min(res, key=lambda r: r.fun)
    return dict(zip(names, best.x)), float(best.fun), [float(r.fun) for r in res]


def record_row(rec):
    return dict(participant=rec["participant"], status=rec["status"], model=rec["model"], nll=rec["nll"],
                n_params=rec["n_params"], n_trials=rec["n_trials"], aic=rec["aic"], bic=rec["bic"],
                n_near_best=rec["n_near_best"], at_bound=";".join(rec["at_bound"]), seconds=rec["seconds"],
                **rec["params"])


def main(argv=None):
    """python -m avdur_analysis.fitting [--pids MZ BQ] [--models causal] [--workers 7] [--refit] [--optimizer bads]"""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pids", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=list(M.MODELS), choices=list(M.MODELS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--optimizer", default="lbfgsb", choices=OPTIMIZERS)
    args = ap.parse_args(argv)
    t = fit_all(args.pids, tuple(args.models), workers=args.workers, refit=args.refit, optimizer=args.optimizer)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
