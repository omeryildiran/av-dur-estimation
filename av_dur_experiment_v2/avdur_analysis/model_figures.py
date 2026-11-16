"""Modelling figures (fits from fitting.py, data from psychometric.py).

    fig1_optimal_integration()   observed bimodal sigma against the optimal-integration prediction
    fig2_aic_heatmap()           delta AIC of every model and participant, relative to forced fusion
    fig3_delta_aic_bars()        the same as bars, one row per participant
    fig4_group_delta_aic()       summed delta AIC with a 95% interval from resampling participants
    fig5_pse_models(who)         measured and model-predicted PSE shift against conflict (low | high noise)
    fig6_sigma_models(who)       the same for sigma
    fig7_posterior_common(who)   causal inference: posterior probability of a common cause against conflict

Model predictions of PSE and sigma: the model's P(test longer) at the participant's tested cells (same
trial counts, lapse included) is fitted with the same psychometric function as the data (fractional
counts), i.e. what the data pipeline would find with unlimited trials. v1 simulated data and refitted.
who: participant ID, "group" (mean +/- SEM over participants) or "all" (a row per participant).
Figures go to results/figures/models/.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import data as D
from . import fitting as F
from . import models as M
from . import psychometric as P
from . import style as St
from .raw_figures import _err

FIG_DIR = D.RESULTS / "figures" / "models"
PRED_DIR = D.RESULTS / "model_fits" / "predictions"
REF = "fusion"


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"{name}.{ext}")
    return fig


def fits(pids=None, models=M.MODELS, **kw):
    """Table of the saved joint fits (fitted now if missing)."""
    return F.fit_all(pids, models, **kw)


def delta_aic(table, ref=REF, crit="aic"):
    wide = table.pivot(index="participant", columns="model", values=crit)
    wide = wide.loc[[p for p in D.participants() if p in wide.index]]
    return wide.sub(wide[ref], axis=0)[[m for m in M.MODELS if m in wide.columns]]


# ── predicted psychometric functions ─────────────────────────────────────────

def _labels(noise, conflict):
    return [f"AV_{'low' if np.isclose(n, D.NOISE_LOW) else 'high'}_{int(round(1000 * c)):+d}"
            for n, c in zip(noise, conflict)]


def predicted_pf(pid, model):
    """Psychometric fit to the model's expected answers at the participant's cells (cached)."""
    rec = F.fit(pid, model, verbose=False)
    path = PRED_DIR / rec["status"] / f"{pid}_{model}_pf.csv"
    if path.exists():
        t = pd.read_csv(path)
        if (t.get("fingerprint") == rec["fingerprint"]).all():
            return t
    des = F.design(pid)
    p = F.cell_probabilities(model, rec["params"], des)["main"]
    agg = pd.DataFrame(dict(condition=_labels(des.main.noise, des.main.conflict), x=des.main.log_t - M.LOG_STD,
                            k=des.main.n * p, n=des.main.n.astype(float)))
    key = pd.Series({c: c.split("_")[1] for c in agg.condition.unique()})
    f = P.fit_cells(agg, key)
    rows = []
    for c, mu, sig, lam in zip(f.conditions, f.mu, f.sigma, f.lapse):
        noise = D.NOISE_LOW if c.split("_")[1] == "low" else D.NOISE_HIGH
        dv = P.derived(mu, sig)
        rows.append(dict(participant=pid, model=model, condition=c, noise=noise, conflict_ms=int(c.split("_")[2]),
                         mu=mu, sigma=sig, lapse=lam, shift_ms=float(dv["shift_ms"]), pse_ms=float(dv["pse_ms"]),
                         fingerprint=rec["fingerprint"]))
    t = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(path, index=False)
    return t


def predictions(pids, models=M.MODELS):
    return pd.concat([predicted_pf(p, m) for p in pids for m in models], ignore_index=True)


# ── figure 1: optimal integration ────────────────────────────────────────────

def optimal_table(observed="matched"):
    """Per participant and noise level: predicted optimal sigma_AV from the unimodal sigmas and the observed
    bimodal sigma (psychometric sigmas throughout; the sqrt 2 cancels). observed: "matched" (0 conflict) or
    "mean" (mean over conflicts). 95% intervals from the bootstrap samples."""
    rows = []
    for pid in D.participants():
        bu, bm = P.load_boot(pid, "unimodal"), P.load_boot(pid, "main")
        if bu is None or bm is None or "V" not in list(bu["conditions"]):
            continue
        uc, mc = list(bu["conditions"]), list(bm["conditions"])
        sv = bu["sigma"][:, uc.index("V")]
        for noise, tag in ((D.NOISE_LOW, "low"), (D.NOISE_HIGH, "high")):
            sa = bu["sigma"][:, uc.index(f"A_{tag}")]
            pred = sa * sv / np.sqrt(sa ** 2 + sv ** 2)
            cols = [i for i, c in enumerate(mc) if c.startswith(f"AV_{tag}_")]
            obs = bm["sigma"][:, mc.index(f"AV_{tag}_+0")] if observed == "matched" else bm["sigma"][:, cols].mean(1)
            u, m = P.collect("unimodal", [pid]), P.collect("main", [pid])
            sa0, sv0 = u.set_index("condition").loc[[f"A_{tag}", "V"], "sigma"]
            mm = m[np.isclose(m.noise, noise)]
            obs0 = mm.loc[mm.conflict_ms == 0, "sigma"].iloc[0] if observed == "matched" else mm.sigma.mean()
            rows.append(dict(participant=pid, noise=noise, sigma_a=sa0, sigma_v=sv0,
                             predicted=sa0 * sv0 / np.hypot(sa0, sv0), observed=obs0, better_cue=min(sa0, sv0),
                             predicted_lo=np.percentile(pred, 2.5), predicted_hi=np.percentile(pred, 97.5),
                             observed_lo=np.percentile(obs, 2.5), observed_hi=np.percentile(obs, 97.5)))
    return pd.DataFrame(rows)


def fig1_optimal_integration(observed="matched", save=True):
    t = optimal_table(observed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    hi = 1.25 * max(t.observed.max(), t.predicted_hi.max())          # intervals running off the top are cut
    for ax, noise in zip(axes, (D.NOISE_LOW, D.NOISE_HIGH)):
        tn = t[np.isclose(t.noise, noise)]
        color = St.NOISE_COLORS[noise]
        ax.plot([0, hi], [0, hi], **St.GUIDE)
        ax.errorbar(tn.predicted, tn.observed, xerr=[tn.predicted - tn.predicted_lo, tn.predicted_hi - tn.predicted],
                    yerr=[tn.observed - tn.observed_lo, tn.observed_hi - tn.observed], fmt="o", color=color, ms=9,
                    capsize=4, lw=1.4)
        for r in tn.itertuples():
            ax.annotate(r.participant, (r.predicted, r.observed), textcoords="offset points", xytext=(6, 4), fontsize=11)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.set_title(St.NOISE_LABELS[noise])
        ax.set_xlabel("Predicted optimal σ_AV")
        ax.set_ylabel("Observed σ_AV" + (" (0 conflict)" if observed == "matched" else " (mean over conflicts)"))
    St.panel_labels(axes)
    fig.suptitle("Psychometric σ; error bars: bootstrap 95% CI. Above the diagonal: less precise than optimal", y=1.02,
                 fontsize=13)
    fig.tight_layout()
    return _save(fig, f"fig1_optimal_integration_{observed}", save)


# ── figures 2-4: model comparison ────────────────────────────────────────────

def fig2_aic_heatmap(table, crit="aic", save=True):
    dl = delta_aic(table, crit=crit)
    fig, ax = plt.subplots(figsize=(2.2 + 1.6 * dl.shape[1], 1.2 + 0.7 * dl.shape[0]))
    lim = max(10.0, np.nanmax(np.abs(dl.to_numpy())))
    im = ax.imshow(dl.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(dl.to_numpy()):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=12,
                color="white" if abs(v) > 0.6 * lim else "black")
    ax.set_xticks(range(dl.shape[1]))
    ax.set_xticklabels([St.MODEL_STYLE[m]["label"] for m in dl.columns])
    ax.set_yticks(range(dl.shape[0]))
    ax.set_yticklabels([f"{p} ({table[table.participant == p].status.iloc[0]})" for p in dl.index])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"Δ{crit.upper()} vs forced fusion\n(negative: better)")
    fig.tight_layout()
    return _save(fig, f"fig2_{crit}_heatmap", save)


def fig3_delta_aic_bars(table, crit="aic", save=True):
    dl = delta_aic(table, crit=crit).drop(columns=REF)
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.8 * len(dl)))
    y = np.arange(len(dl))
    h = 0.8 / dl.shape[1]
    for j, m in enumerate(dl.columns):
        sty = St.MODEL_STYLE[m]
        ax.barh(y + (j - (dl.shape[1] - 1) / 2) * h, dl[m], height=h, color=sty["color"], alpha=0.85, label=sty["label"])
    ax.axvline(0, color="black", lw=1)
    for x in (-2, 2):
        ax.axvline(x, **St.GUIDE)
    ax.set_yticks(y)
    ax.set_yticklabels(dl.index)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ{crit.upper()} vs forced fusion (negative: better than fusion)")
    St.legend_outside(ax, "right")
    fig.tight_layout()
    return _save(fig, f"fig3_delta_{crit}_bars", save)


def group_delta(table, crit="aic", n_boot=10000, seed=0):
    """Summed delta (vs forced fusion) per model with a 95% interval from resampling participants."""
    dl = delta_aic(table, crit=crit)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(dl), size=(n_boot, len(dl)))
    rows = []
    for m in dl.columns:
        v = dl[m].to_numpy()
        boots = v[idx].sum(1)
        rows.append(dict(model=m, summed=v.sum(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                         n_participants=len(v), n_better=int((v < 0).sum()) if m != REF else 0))
    return pd.DataFrame(rows)


def fig4_group_delta_aic(table, crit="aic", save=True):
    g = group_delta(table, crit).set_index("model").drop(index=REF)
    dl = delta_aic(table, crit=crit)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for i, m in enumerate(g.index):
        sty = St.MODEL_STYLE[m]
        ax.bar(i, g.loc[m, "summed"], color=sty["color"], alpha=0.8, width=0.6)
        ax.errorbar(i, g.loc[m, "summed"], yerr=[[g.loc[m, "summed"] - g.loc[m, "lo"]], [g.loc[m, "hi"] - g.loc[m, "summed"]]],
                    color="black", capsize=8, lw=1.6)
        ax.scatter(np.full(len(dl), i) + np.linspace(-0.12, 0.12, len(dl)), dl[m], color="0.3", s=18, zorder=3)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([St.MODEL_STYLE[m]["label"] for m in g.index])
    ax.set_ylabel(f"Summed Δ{crit.upper()} vs forced fusion")
    ax.set_title(f"N = {len(dl)}; bars: 95% CI over participants; dots: participants", fontsize=12)
    fig.tight_layout()
    return _save(fig, f"fig4_group_delta_{crit}", save)


# ── figures 5-6: measured vs predicted PSE and sigma ─────────────────────────

def _measured(pids):
    m = P.collect("main", pids)
    return m[m.participant.isin(pids)]


def _models_panels(axes, meas, pred, col, who, mode):
    group = who == "group"
    for ax, noise in zip(axes, (D.NOISE_LOW, D.NOISE_HIGH)):
        mn, pn = meas[np.isclose(meas.noise, noise)], pred[np.isclose(pred.noise, noise)]
        conflicts = sorted(mn.conflict_ms.unique())
        for k, c in enumerate(conflicts):
            rows = mn[mn.conflict_ms == c]
            rgba = St.conflict_rgba(noise, k, len(conflicts))
            if group:
                y, e = _err(rows[col], mode)
                yerr = None if np.isnan(e) else e
            else:
                r = rows.iloc[0]
                y = r[col]
                yerr = [[y - r[f"{col}_lo"]], [r[f"{col}_hi"] - y]] if mode == "ci" else None
            ax.errorbar(c, y, yerr=yerr, fmt="o", color=rgba, mec=rgba, ms=8, capsize=5, lw=1.6, zorder=5)
        for m in M.MODELS:
            pm = pn[pn.model == m].groupby("conflict_ms")[col]
            sty = St.MODEL_STYLE[m]
            mean = pm.mean()
            ax.plot(mean.index, mean.to_numpy(), color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=5,
                    mfc="white", lw=1.8, zorder=4)
            if group and pm.size().min() > 1:
                sem = pm.std(ddof=1) / np.sqrt(pm.size())
                ax.fill_between(mean.index, mean - sem, mean + sem, color=sty["color"], alpha=0.12, lw=0)
        ax.axvline(0, **St.GUIDE)
        ax.set_xlabel("Conflict: visual - auditory standard (ms)")
        ax.set_title(St.NOISE_LABELS[noise], fontsize=plt.rcParams["axes.titlesize"] * 0.8)
    if col == "shift_ms":
        for ax in axes:
            ax.axhline(0, **St.GUIDE)


def _model_handles():
    hs = [Line2D([], [], color="0.3", marker="o", lw=0, label="Data")]
    return hs + [Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"], mfc="white", label=s["label"])
                 for s in St.MODEL_STYLE.values()]


def _fig_models(col, ylabel, who, errorbar, save, name):
    table = fits()
    pids = [p for p in D.participants() if p in set(table.participant)]
    meas, pred = _measured(pids), predictions(pids)
    if who == "all":
        fig, axes = plt.subplots(len(pids), 2, figsize=(12, 3.8 * len(pids)), squeeze=False, sharex=True)
        for row, pid in zip(axes, pids):
            _models_panels(row, meas[meas.participant == pid], pred[pred.participant == pid], col, pid,
                           "ci" if errorbar == "auto" else errorbar)
            row[0].set_ylabel(f"{pid}\n{ylabel}")
        fig.legend(handles=_model_handles(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout()
        return _save(fig, f"{name}_all", save)
    group = who == "group"
    mode = ("sem" if group else "ci") if errorbar == "auto" else errorbar
    sel = pids if group else [who]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    _models_panels(axes, meas[meas.participant.isin(sel)], pred[pred.participant.isin(sel)], col, who, mode)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(handles=_model_handles(), loc="upper left", fontsize=12)
    St.panel_labels(axes)
    label = f"Group (N = {len(sel)})" if group else who
    fig.suptitle(f"{label}; data error bars: {'SEM' if mode == 'sem' else 'bootstrap 95% CI' if mode == 'ci' else mode}",
                 y=1.02)
    fig.tight_layout()
    return _save(fig, f"{name}_{who}", save)


def fig5_pse_models(who="group", errorbar="auto", save=True):
    return _fig_models("shift_ms", "PSE shift (ms)", who, errorbar, save, "fig5_pse_models")


def fig6_sigma_models(who="group", errorbar="auto", save=True):
    return _fig_models("sigma", "σ (log units, psychometric)", who, errorbar, save, "fig6_sigma_models")


# ── figure 7: posterior probability of a common cause ────────────────────────

def posterior_by_conflict(pid, n_gh=64, p_c="shared"):
    """Causal-inference fit: mean posterior P(C=1) for the standard interval (m_A around 500 ms, m_V around
    500 ms + conflict) at every tested conflict and noise level, by Gauss-Hermite quadrature."""
    rec = F.fit(pid, "causal", p_c=p_c, verbose=False)
    prm = rec["params"]
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    x, w = np.sqrt(2) * x, w / np.sqrt(np.pi)
    conflicts = sorted(F.design(pid).main.conflict.unique())
    rows = []
    for noise, tag in ((D.NOISE_LOW, "low"), (D.NOISE_HIGH, "high")):
        sa, sv = prm[f"sigma_a_{tag}"], prm["sigma_v"]
        pc = prm.get("p_c", prm.get(f"p_c_{tag}"))
        for c in conflicts:
            ma = (M.LOG_STD + sa * x)[:, None]
            mv = (np.log(D.STANDARD_S + c) + sv * x)[None, :]
            post = float(w @ M.posterior_common(ma, mv, sa, sv, pc) @ w)
            rows.append(dict(participant=pid, noise=noise, conflict_ms=int(round(1000 * c)), posterior=post, prior=pc))
    return pd.DataFrame(rows)


def fig7_posterior_common(who="all", save=True):
    pids = [p for p in D.participants() if F.design(p).main.size]
    post = pd.concat([posterior_by_conflict(p) for p in pids], ignore_index=True)

    def panel(ax, t, group=False):
        for noise in (D.NOISE_LOW, D.NOISE_HIGH):
            tn = t[np.isclose(t.noise, noise)]
            g = tn.groupby("conflict_ms")["posterior"]
            mean = g.mean()
            color = St.NOISE_COLORS[noise]
            ax.plot(mean.index, mean.to_numpy(), "o-", color=color, ms=6, label=St.NOISE_LABELS[noise])
            if group and g.size().min() > 1:
                sem = g.std(ddof=1) / np.sqrt(g.size())
                ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.15, lw=0)
        ax.axhline(t.prior.mean(), color="0.4", ls="--", lw=1.2, label="Prior p_c" + (" (mean)" if group else ""))
        ax.set_ylim(0, 1.02)
        ax.axvline(0, **St.GUIDE)
        ax.set_xlabel("Conflict (ms)")
        ax.set_ylabel("P(common cause | m)")

    if who == "all":
        fig, axes = St.grid(len(pids), ncols=min(4, len(pids)), panel=(4.4, 3.6), sharey=True)
        for ax, pid in zip(axes, pids):
            panel(ax, post[post.participant == pid])
            ax.set_title(pid)
        axes[0].legend(fontsize=10, loc="lower left")
        fig.tight_layout()
        return _save(fig, "fig7_posterior_common_all", save)
    fig, ax = plt.subplots(figsize=(7, 5))
    panel(ax, post if who == "group" else post[post.participant == who], group=who == "group")
    ax.legend(fontsize=11, loc="lower left")
    ax.set_title("Group (mean ± SEM)" if who == "group" else who)
    fig.tight_layout()
    return _save(fig, f"fig7_posterior_common_{who}", save)


def param_table(table):
    cols = ["participant", "status", "model", "nll", "aic", "bic", "n_params", "n_near_best", "at_bound", "lapse",
            "sigma_a_low", "sigma_a_high", "sigma_v", "p_c", "p_switch_low", "p_switch_high"]
    return table[[c for c in cols if c in table.columns]]
