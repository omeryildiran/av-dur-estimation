"""Order-aware observer models (exploratory, 19 Nov 2026): the three models of models.py plus memory
contraction of the first interval, so presentation order is part of the likelihood (reviewer 1).

Why: with the test second, psychometric sigma is 35-80% lower in nearly every condition, and the PSE
moves in opposite directions for the two orders (raw fig5_order). A noisier first interval cannot do
that (the comparison noise is the same whichever interval holds the test).

Internal-reference model (Dyjas, Bausenhart & Ulrich 2012, "internal reference model" of 2IFC): the
estimate of the FIRST interval is pulled toward a remembered reference duration before the comparison,
    e1' = g * e1 + (1 - g) * ref         (g = weight of the first interval's own estimate, ref in log s)
and the second interval's estimate is used as it is. Each model's own estimate e (fusion, switching,
causal inference, or the single cue in the unimodal blocks) is unchanged. So:
    test first:  the test is contracted, curve flatter by 1/g, PSE shifted by (1 - g)(ref - log 0.5) / g
    test second: the standard is contracted, slope unchanged, a conflict moves the PSE g times as far
g = 1 gives back the models of models.py exactly (checked in check()). g and ref are shared by every block
(unimodal auditory, unimodal visual, main) of a participant: 2 parameters more than models.py.

Likelihood: the CDF of one interval's estimate on a grid over the prior box (single cue and fusion: the
inverse of tn_mean, exact; switching: the mixture of the two single-cue CDFs; causal inference:
models.CausalNoiseLevel.cdf), then P(test longer) = integral of (1 - F_test(phi(y))) dF_standard(y), with
phi(y) = (y - c) / g for test first and g y + c for test second (c = (1 - g) ref). No Monte Carlo.
Cells are (block, noise, conflict, test level, order). Fits: results/model_fits_order/<state>/, apart from
the main fits.
"""

from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

from . import data as D
from . import fitting as F
from . import models as M

FIT_DIR = D.RESULTS / "model_fits_order"
E = np.linspace(M.BOX[0], M.BOX[1], M.N_E)            # grid of estimates (log s)
EXTRA_BOUNDS = dict(g_first=(0.05, 1.0), ref=M.BOX)
FIRST = 1                                              # order code: 1 test first, 2 test second


# ── CDF of one interval's estimate on E ──────────────────────────────────────

def cdf_single(mu, s):
    """Estimate tn_mean(m, s), m ~ N(mu, s): F(e) = Phi((h(e) - mu) / s), h the inverse of tn_mean."""
    g, m = M._inverse_tn(s)
    f = ndtr((np.interp(E, g, m) - np.asarray(mu, float)[..., None]) / s)
    f[..., -1] = 1.0
    return f


def cdf_fusion(mu_a, mu_v, sa, sv):
    w_a, w_v, sc = M.weights(sa, sv)
    return cdf_single(w_a * np.asarray(mu_a) + w_v * np.asarray(mu_v), sc)


def cdf_switching(mu_a, mu_v, sa, sv, p_v):
    return (1 - p_v) * cdf_single(mu_a, sa) + p_v * cdf_single(mu_v, sv)


# ── comparison with contraction of the first interval ────────────────────────

def p_order(f_test, f_std, order, g, ref):
    """P(test estimate > standard estimate) after contraction of the first interval. f_test, f_std: (n, N_E)
    CDFs on E (rows paired); order: (n,) 1 = test first, 2 = test second."""
    c = (1 - g) * ref
    y = 0.5 * (E[1:] + E[:-1])
    df = np.diff(np.concatenate([np.zeros((len(f_std), 1)), f_std], axis=1), axis=1)[:, 1:]
    y0_mass = f_std[:, :1]                                     # mass at the lower edge (practically 0)
    phi = np.where(np.asarray(order)[:, None] == FIRST, (y[None, :] - c) / g, g * y[None, :] + c)
    ft = np.stack([np.interp(phi[i], E, f_test[i], left=0.0, right=1.0) for i in range(len(f_test))])
    ft0 = np.array([np.interp((E[0] - c) / g if o == FIRST else g * E[0] + c, E, f_test[i], left=0.0, right=1.0)
                    for i, o in enumerate(order)])
    return np.sum((1 - ft) * df, axis=1) + (1 - ft0) * y0_mass[:, 0]


def _rows_cdf(keys, make):
    """CDFs for every row from one CDF per unique key (rows share stimulus levels)."""
    uniq, idx = np.unique(np.round(np.asarray(keys, float), 10), axis=0, return_inverse=True)
    return np.stack([make(*np.atleast_1d(k)) for k in uniq])[idx.ravel()]


def p_unimodal_order(log_t, order, s, g, ref):
    f_t = _rows_cdf(log_t, lambda t: cdf_single(t, s))
    f_s = np.repeat(cdf_single(M.LOG_STD, s)[None], len(f_t), axis=0)
    return p_order(f_t, f_s, order, g, ref)


def p_main_order(model, prm, noise, log_t, log_sv, order):
    key = F._noise_key(noise)
    sa, sv = prm[f"sigma_a_{key}"], prm["sigma_v"]
    if model == "fusion":
        cdf = lambda ma, mv: cdf_fusion(ma, mv, sa, sv)
    elif model == "switching":
        p_v = prm[f"p_switch_{key}"]
        cdf = lambda ma, mv: cdf_switching(ma, mv, sa, sv, p_v)
    elif model == "causal":
        cn = M.CausalNoiseLevel(sa, sv, prm.get("p_c", prm.get(f"p_c_{key}")))
        cdf = cn.cdf
    else:
        raise ValueError(model)
    f_t = _rows_cdf(log_t, lambda t: cdf(t, t))
    f_s = _rows_cdf(log_sv, lambda s: cdf(M.LOG_STD, s))
    return p_order(f_t, f_s, order, prm["g_first"], prm["ref"])


# ── design, likelihood, fit ──────────────────────────────────────────────────

def design(pid):
    """Binomial cells as fitting.design, split by presentation order (column order)."""
    uni, _ = D.load_unimodal(pid)
    main, _ = D.load(pid, "main")
    cells = lambda d, by: (d[d["order"].isin([1, 2])].assign(log_t=np.log(d["testDurS"].astype(float)).round(10))
                           .groupby(by + ["order", "log_t"])["chose_test"].agg(k="sum", n="count").reset_index())
    ua = cells(uni[uni.task == "unimodal_auditory"], ["noise"]) if not uni.empty else pd.DataFrame()
    uv = cells(uni[uni.task == "unimodal_visual"], []) if not uni.empty else pd.DataFrame()
    mm = cells(main, ["noise", "conflict"]) if not main.empty else pd.DataFrame()
    if not mm.empty:
        mm["log_sv"] = np.log(D.STANDARD_S + mm["conflict"])
    base = F.design(pid)
    return F.Design(pid, ua, uv, mm, base.files, int(ua.n.sum() + uv.n.sum() + mm.n.sum()))


def param_names(model, order=True):
    return F.param_names(model) + (["g_first", "ref"] if order else [])


def cell_probabilities(model, prm, des):
    prm = dict(prm)
    prm.setdefault("g_first", 1.0)
    prm.setdefault("ref", M.LOG_STD)
    g, ref, lam = prm["g_first"], prm["ref"], prm["lapse"]
    out = {}
    if not des.uni_a.empty:
        p = np.empty(len(des.uni_a))
        for noise, c in des.uni_a.groupby("noise"):
            i = des.uni_a.index.get_indexer(c.index)
            p[i] = p_unimodal_order(c.log_t.to_numpy(), c.order.to_numpy(), prm[f"sigma_a_{F._noise_key(noise)}"], g, ref)
        out["uni_a"] = lam / 2 + (1 - lam) * p
    if not des.uni_v.empty:
        out["uni_v"] = lam / 2 + (1 - lam) * p_unimodal_order(des.uni_v.log_t.to_numpy(), des.uni_v.order.to_numpy(),
                                                              prm["sigma_v"], g, ref)
    if not des.main.empty:
        p = np.empty(len(des.main))
        for noise, c in des.main.groupby("noise"):
            i = des.main.index.get_indexer(c.index)
            p[i] = p_main_order(model, prm, noise, c.log_t.to_numpy(), c.log_sv.to_numpy(), c.order.to_numpy())
        out["main"] = lam / 2 + (1 - lam) * p
    return out


def neg_log_lik(theta, model, des, names, parts=False):
    probs = cell_probabilities(model, dict(zip(names, theta)), des)
    ll = {k: F._binom_ll(p, getattr(des, k).k.to_numpy(), getattr(des, k).n.to_numpy()) for k, p in probs.items()}
    return ll if parts else -sum(ll.values())


def _bounds(names):
    return [EXTRA_BOUNDS[n] if n in EXTRA_BOUNDS else b for n, b in zip(names, F._bounds(names))]


def _starts(des, model, n_starts, seed):
    """The main fits' starts for the shared parameters, plus a spread of (g_first, ref)."""
    base = F._starts(des, F.param_names(model), n_starts, seed)
    gs = [0.7, 0.5, 0.9, 0.35, 0.8, 0.6, 0.45, 0.95]
    refs = M.LOG_STD + np.array([0.0, -0.2, 0.2, -0.4, 0.1, 0.3, -0.1, 0.4])
    return [np.concatenate([b, [gs[i % 8], refs[i % 8]]]) for i, b in enumerate(base)]


def fit_path(pid, model, state):
    return FIT_DIR / state / f"{pid}_{model}.json"


def fit(pid, model, n_starts=F.N_STARTS, seed=0, refit=False, verbose=True):
    """Fit one participant (saved; reused until the data change)."""
    des = design(pid)
    if des.main.empty:
        return None
    state = "complete" if D.status(pid)["complete"] else "partial"
    names = param_names(model)
    key = F._key(des, model, names, n_starts, seed, dict(order_model="contraction_v1", e_grid=len(E)))
    path = fit_path(pid, model, state)
    if not refit and path.exists():
        rec = json.loads(path.read_text())
        if rec.get("fingerprint") == key:
            return rec
    t0, bounds, starts = time.time(), _bounds(names), []
    for x0 in _starts(des, model, n_starts, seed):
        r = minimize(neg_log_lik, x0, args=(model, des, names), method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=500))
        starts.append(dict(start=[float(v) for v in x0], x=[float(v) for v in r.x], nll=float(r.fun),
                           converged=bool(r.success), n_eval=int(r.nfev)))
        if verbose:
            print(f"[order fit] {pid} {model}: start {len(starts)}/{n_starts} -log L {r.fun:.2f}", flush=True)
    best = min(starts, key=lambda s: s["nll"])
    k, n = len(names), des.n_trials
    rec = dict(participant=pid, status=state, model=model, fingerprint=key, params=dict(zip(names, best["x"])),
               ref_ms=float(1000 * np.exp(best["x"][names.index("ref")])), nll=best["nll"], n_params=k,
               n_trials=n, aic=2 * k + 2 * best["nll"], bic=k * np.log(n) + 2 * best["nll"],
               loglik_parts=neg_log_lik(np.array(best["x"]), model, des, names, parts=True),
               n_near_best=int(sum(s["nll"] < best["nll"] + 0.5 for s in starts)), starts=starts,
               data_files=des.files, fitted=dt.datetime.now().isoformat(timespec="seconds"),
               seconds=round(time.time() - t0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    for other in ("complete", "partial"):
        if other != state:
            fit_path(pid, model, other).unlink(missing_ok=True)
    path.write_text(json.dumps(rec, indent=2))
    return rec


def _job(args):
    return fit(*args)


def fit_all(pids=None, models=M.MODELS, workers=1, refit=False):
    pids = pids or [p for p in D.participants() if D.status(p)["main_parts"]]
    jobs = [(p, m, F.N_STARTS, 0, refit) for p in pids for m in models]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            recs = list(ex.map(_job, jobs))
    else:
        recs = [_job(j) for j in jobs]
    return table([r for r in recs if r is not None])


def table(recs=None):
    """One row per saved order-model fit, with the matching main fit's AIC (no order) for comparison."""
    if recs is None:
        recs = [json.loads(p.read_text()) for p in sorted(FIT_DIR.glob("*/*.json"))]
    rows = []
    for r in recs:
        base = F.fit(r["participant"], r["model"], verbose=False)
        rows.append(dict(participant=r["participant"], status=r["status"], model=r["model"], nll=r["nll"],
                         aic=r["aic"], aic_no_order=base["aic"], d_aic_order=r["aic"] - base["aic"],
                         n_near_best=r["n_near_best"], seconds=r["seconds"], ref_ms=r["ref_ms"], **r["params"]))
    return pd.DataFrame(rows)


def check(pid="KJ", tol=2e-3):
    """g_first = 1 must give the main fits' -log L at their parameters (order-split cells, same P)."""
    des = design(pid)
    out = {}
    for model in M.MODELS:
        rec = F.fit(pid, model, verbose=False)
        prm = dict(rec["params"], g_first=1.0, ref=M.LOG_STD)
        names = list(prm)
        nll = neg_log_lik(np.array([prm[n] for n in names]), model, des, names)
        out[model] = (nll, rec["nll"])
        assert abs(nll - rec["nll"]) < tol * abs(rec["nll"]), (model, nll, rec["nll"])
    return out


def main(argv=None):
    """python -m avdur_analysis.order_models [--pids KJ] [--models fusion] [--workers 6] [--refit]"""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pids", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=list(M.MODELS), choices=list(M.MODELS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args(argv)
    t = fit_all(a.pids, tuple(a.models), a.workers, a.refit)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()


# ── predictions and figure ───────────────────────────────────────────────────

def predicted_pf(pid, model):
    """Psychometric fit (per condition and order) to the order model's expected answers in the main
    experiment, the same way the data are fitted (psychometric.py, lapse per noise level)."""
    from . import psychometric as P
    rec = fit(pid, model, verbose=False)
    des = design(pid)
    p = cell_probabilities(model, rec["params"], des)["main"]
    mm = des.main
    noise = np.where(np.isclose(mm.noise, D.NOISE_LOW), "low", "high")
    cond = [f"AV_{n}_{int(round(1000 * c)):+d}|{P.ORDER_NAMES[int(o)]}" for n, c, o in zip(noise, mm.conflict, mm.order)]
    agg = pd.DataFrame(dict(condition=cond, x=mm.log_t - M.LOG_STD, k=mm.n * p, n=mm.n.astype(float)))
    f = P.fit_cells(agg, pd.Series({c: c.split("_")[1] for c in agg.condition.unique()}))
    rows = []
    for c, mu, sig in zip(f.conditions, f.mu, f.sigma):
        base, _, order = c.partition("|")
        rows.append(dict(participant=pid, model=model, noise=D.NOISE_LOW if "_low_" in base else D.NOISE_HIGH,
                         conflict_ms=int(base.split("_")[2]), order=order, sigma=sig,
                         shift_ms=float(P.derived(mu, sig)["shift_ms"])))
    return pd.DataFrame(rows)


FIG_DIR = D.RESULTS / "figures" / "order_models"
ORDERS = (("test_first", "Test first"), ("test_second", "Test second"))
TESTED_MS = (50.0, 2000.0)


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"{name}.{ext}")
    return fig


def fitted_pids(model="switching"):
    return [p for p in D.participants() if fit_path(p, model, "complete").exists() or fit_path(p, model, "partial").exists()]


def predictions(pids, models=M.MODELS):
    return pd.concat([predicted_pf(p, m) for p in pids for m in models], ignore_index=True)


def _ylim(ax, values, log=False):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(v):
        return
    if log:
        ax.set_yscale("log")
        ax.set_ylim(max(v.min() / 1.6, 0.01), v.max() * 1.6)
        from . import style as St
        St.plain_log_y(ax)
    else:
        pad = 0.15 * (v.max() - v.min()) + 25
        ax.set_ylim(min(v.min(), 0) - pad, max(v.max(), 0) + pad)


def fig_predictions(who="all", col="shift_ms", models=M.MODELS, errorbar="auto", save=True, width=15.0,
                    grid=False):
    """Data and order-model predictions, one column per noise level x order (test first | test second), so
    the flatter test-first curves and the smaller test-second conflict effect are visible directly.
    Points: the data fitted per condition and order (participant: bootstrap 95% CI; group: mean +/- SEM over
    participants). Lines: each model's expected answers fitted the same way (v1 model colours and line
    styles). col: "shift_ms" (PSE shift) or "sigma" (psychometric sigma, log axis). The y range of a row
    follows the estimates; a PSE outside the tested durations (50-2000 ms) is not drawn (the title says how
    many), and intervals may run off the panel. width: figure width in inches (narrower = larger text).
    grid: for one participant or the group, 2 x 2 panels (rows: noise level, columns: order)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from . import psychometric as P
    from . import style as St
    pids = fitted_pids(models[0])
    meas = P.collect("main_order", pids)
    meas = meas.assign(ok=meas.pse_ms.between(*TESTED_MS) & meas.at_bound.fillna("").astype(str).eq(""))
    pred = predictions(pids, models)
    group = who == "group"
    rows = ["group"] if group else (pids if who == "all" else [who])
    mode = ("sem" if group else "ci") if errorbar == "auto" else errorbar
    log = col == "sigma"
    grid = grid and len(rows) == 1
    if grid:
        fig, ax2 = plt.subplots(2, 2, figsize=(width, 0.72 * width), sharex=True)
        axes = np.array(list(ax2.ravel()), dtype=object).reshape(1, 4)
    else:
        fig, axes = plt.subplots(len(rows), 4, figsize=(width, width / 15 * 3.5 * len(rows) + 0.6), squeeze=False,
                                 sharex=True)
    for row, who_r in zip(axes, rows):
        sel = pids if who_r == "group" else [who_r]
        mt_all, pr_all = meas[meas.participant.isin(sel)], pred[pred.participant.isin(sel)]
        seen, hidden = [], int((~mt_all.ok).sum())
        for j, (noise, (order, olabel)) in enumerate([(n, o) for n in (D.NOISE_LOW, D.NOISE_HIGH) for o in ORDERS]):
            ax = row[j]
            color = St.NOISE_COLORS[noise]
            mt = mt_all[np.isclose(mt_all.noise, noise) & (mt_all.order == order) & mt_all.ok]
            pr = pr_all[np.isclose(pr_all.noise, noise) & (pr_all.order == order)]
            for m in models:
                line = pr[pr.model == m].groupby("conflict_ms")[col].mean()
                sty = St.MODEL_STYLE[m]
                ax.plot(line.index, line.to_numpy(), color=sty["color"], ls=sty["ls"], lw=1.8, zorder=3)
                seen += list(line.to_numpy())
            mfc = color if order == "test_first" else "white"
            for c, g in mt.groupby("conflict_ms"):
                if group:
                    v = g[col].to_numpy()
                    y = v.mean()
                    e = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 and mode == "sem" else None
                else:
                    r = g.iloc[0]
                    y, e = r[col], [[r[col] - r[f"{col}_lo"]], [r[f"{col}_hi"] - r[col]]]
                ax.errorbar(c, y, yerr=e, fmt="o", color=color, mfc=mfc, ms=6, capsize=2, lw=1.1, zorder=4)
                seen.append(y)
            ax.axvline(0, **St.GUIDE)
            if not log:
                ax.axhline(0, **St.GUIDE)
            if who_r == rows[0]:
                ax.set_title(f"{St.NOISE_LABELS[noise].split(' (')[0]}, {olabel.lower()}",
                             fontsize=plt.rcParams["axes.titlesize"] * 0.8)
        for ax in row:
            _ylim(ax, seen, log)
        label = f"Group (N = {len(pids)})" if group else who_r
        if hidden:
            label += f"\n({hidden} off-range PSEs not drawn)"
        ylab = "PSE shift (ms)" if col == "shift_ms" else "σ (log units, psychometric)"
        row[0].set_ylabel(f"{label}\n{ylab}")
        if grid:
            row[2].set_ylabel(ylab)
        for ax in row[1:]:
            ax.tick_params(labelleft=True)
    for ax in (axes[0][2:] if grid else axes[-1]):
        ax.set_xlabel("Conflict (ms)")
    handles = [Line2D([], [], color="0.3", marker="o", ls="", label="Data, test first"),
               Line2D([], [], color="0.3", marker="o", mfc="white", ls="", label="Data, test second")]
    handles += [Line2D([], [], color=St.MODEL_STYLE[m]["color"], ls=St.MODEL_STYLE[m]["ls"], lw=1.8,
                       label=St.MODEL_STYLE[m]["label"] + " + contraction") for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=5 if width >= 14 and not grid else 3,
               bbox_to_anchor=(0.5, -0.06 if grid else -0.02 - (0.03 if width >= 14 else 0.09) / len(rows)))
    bars = "bootstrap 95% CI" if mode == "ci" else "SEM over participants" if mode == "sem" else str(mode)
    fig.suptitle(f"Order models: data (error bars: {bars}) and predictions", y=1.0)
    fig.tight_layout()
    return _save(fig, f"fig_predictions_{col}_{who}", save)


def fig_fit_summary(save=True, layout="row"):
    """A: how much contraction improves each model (AIC with order minus without; negative: better).
    B: model comparison with order, delta AIC vs forced fusion per participant and summed (95% interval from
    resampling participants). C, D: the fitted contraction weight g_first and reference duration ref.
    layout: "row" (1 x 4) or "grid" (2 x 2, for a page)."""
    import matplotlib.pyplot as plt
    from . import model_figures as MF
    from . import style as St
    t = table()
    pids = [p for p in D.participants() if p in set(t.participant)]
    if layout == "grid":
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1.5, 1, 1]))
    x = np.arange(len(pids))
    h = 0.8 / len(M.MODELS)
    for j, m in enumerate(M.MODELS):
        sty = St.MODEL_STYLE[m]
        tm = t[t.model == m].set_index("participant").reindex(pids)
        axes[0].bar(x + (j - 1) * h, tm.d_aic_order, width=h, color=sty["color"], alpha=0.85, label=sty["label"])
        axes[2].scatter(x + (j - 1) * h, tm.g_first, color=sty["color"], marker=sty["marker"], s=45, zorder=3)
        axes[3].scatter(x + (j - 1) * h, tm.ref_ms, color=sty["color"], marker=sty["marker"], s=45, zorder=3)
    axes[0].set_ylabel("ΔAIC: with order - without")
    axes[0].set_title("Gain from contraction", fontsize=plt.rcParams["axes.titlesize"] * 0.8)
    axes[0].legend(fontsize=10, loc="lower left")
    dl = MF.delta_aic(t).drop(columns="fusion")
    g = MF.group_delta(t).set_index("model").drop(index="fusion")
    xs = np.arange(len(pids) + 1)
    h2 = 0.8 / dl.shape[1]
    for j, m in enumerate(dl.columns):
        sty = St.MODEL_STYLE[m]
        vals = list(dl[m].reindex(pids)) + [g.loc[m, "summed"]]
        xx = xs + (j - (dl.shape[1] - 1) / 2) * h2
        axes[1].bar(xx, vals, width=h2, color=sty["color"], alpha=0.85)
        axes[1].errorbar(xx[-1], g.loc[m, "summed"], yerr=[[g.loc[m, "summed"] - g.loc[m, "lo"]],
                         [g.loc[m, "hi"] - g.loc[m, "summed"]]], color="black", capsize=5, lw=1.4)
    axes[1].axvline(len(pids) - 0.5, color="0.7", lw=1)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(pids + ["Summed"])
    axes[1].set_ylabel("ΔAIC vs forced fusion")
    axes[1].set_title("Models with contraction (negative: better than fusion)",
                      fontsize=plt.rcParams["axes.titlesize"] * 0.8)
    axes[2].axhline(1, **St.GUIDE)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("g_first (1 = no contraction)")
    axes[3].axhline(1000 * D.STANDARD_S, **St.GUIDE)
    axes[3].set_ylabel("Reference duration ref (ms)")
    for ax in (axes[0], axes[2], axes[3]):
        ax.set_xticks(x)
        ax.set_xticklabels(pids)
    for ax in axes[:2]:
        ax.axhline(0, color="black", lw=1)
    St.panel_labels(axes)
    fig.tight_layout()
    return _save(fig, "fig_fit_summary", save)


def fig_aic_heatmap(save=True):
    """Model comparison as delta AIC vs forced fusion, per participant plus the sum over participants:
    A without order (the main fits), B with memory contraction (each model compared with forced fusion that
    also has contraction). Both panels share one colour scale (as in model_figures.fig2_aic_heatmap); negative (red) is better than fusion."""
    import matplotlib.pyplot as plt
    from . import model_figures as MF
    from . import style as St
    t = table()
    status = t.drop_duplicates("participant").set_index("participant").status
    panels = [("aic_no_order", "Without order"), ("aic", "With memory contraction")]
    dls = []
    for crit, _ in panels:
        dl = MF.delta_aic(t, crit=crit)
        dl.loc["Summed"] = dl.sum()
        dls.append(dl)
    lim = max(10.0, max(np.nanmax(np.abs(d.drop(index="Summed").to_numpy())) for d in dls))
    n_r, n_c = dls[0].shape
    fig, axes = plt.subplots(1, 2, figsize=(2.0 + 3.4 * n_c, 1.4 + 0.65 * n_r), sharey=True)
    for ax, dl, (_, title) in zip(axes, dls, panels):
        im = ax.imshow(dl.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(dl.to_numpy()):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=12,
                    fontweight="bold" if dl.index[i] == "Summed" else "normal",
                    color="white" if abs(v) > 0.6 * lim else "black")
        ax.axhline(n_r - 1.5, color="black", lw=1.5)
        ax.set_xticks(range(n_c))
        ax.set_xticklabels([St.MODEL_STYLE[m]["label"].replace(" ", "\n") for m in dl.columns])
        ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"] * 0.85)
    axes[0].set_yticks(range(n_r))
    axes[0].set_yticklabels([f"{p} ({status[p]})" if p in status else p for p in dls[0].index])
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("ΔAIC vs forced fusion\n(negative: better)")
    St.panel_labels(axes)
    return _save(fig, "fig_aic_heatmap", save)
