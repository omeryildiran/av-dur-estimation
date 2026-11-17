"""Raw-data figures (no observer models).

    fig1_unimodal(who)          psychometric curves of the unimodal blocks | JND or Weber fraction
    fig2_bimodal(who)           psychometric curves across conflicts: low noise | high noise
    fig3_conflict_effect(who)   PSE shift | sigma against conflict, by noise level
    fig4_staircases(pid)        every staircase of the main experiment (or the unimodal blocks)

who: a participant ID, "group" (curves averaged over participants, pooled data points, mean +/- SEM), or
"all" (one panel, or one row, per participant). errorbar: "auto" (participant: bootstrap 95% CI; group:
SEM), "ci", "sem", "sd" or None. Figures are saved to results/figures/raw/ (save=False to skip).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import data as D
from . import psychometric as P
from . import style as St

FIG_DIR = D.RESULTS / "figures" / "raw"
BIN_WIDTH = 0.1                     # log units: the data points are binned in log(test / standard)
MIN_BIN_N = 3
GROUP = "group"
ALL = "all"


# ── helpers ───────────────────────────────────────────────────────────────────

def binned(d, min_n=MIN_BIN_N):
    """Proportion 'test longer' in bins of log(test / standard); x at the bin's mean test duration (ms)."""
    b = np.floor(d["x"] / BIN_WIDTH).astype(int)
    g = d.assign(b=b, ms=1000 * d["testDurS"]).groupby("b").agg(ms=("ms", "mean"), p=("chose_test", "mean"),
                                                               n=("chose_test", "size"))
    return g[g.n >= min_n]


def _curve_ms(rows, ms):
    """Mean psychometric curve over the rows (one per participant) at test durations ms."""
    x = np.log(ms / (1000 * D.STANDARD_S))
    return np.mean([P.predict(x, r.mu, r.sigma, r.lapse) for r in rows.itertuples()], axis=0)


def _points(ax, b, color, scale=1.0, **kw):
    ax.scatter(b.ms, b.p, s=scale * (12 + 2.2 * np.sqrt(b.n) * 4), color=color, alpha=0.75,
               edgecolors="white", linewidths=0.6, zorder=4, **kw)


def _pf_axes(ax, title=None):
    St.duration_axis(ax)
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, **St.GUIDE)
    ax.axvline(1000 * D.STANDARD_S, **St.GUIDE)
    ax.set_xlabel("Test duration (ms)")
    ax.set_ylabel('P("test longer")')
    if title:
        ax.set_title(title)


def _err(values, how):
    """(centre, half-width) of per-participant values for group error bars."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) < 2 or how is None:
        return v.mean(), np.nan
    sd = v.std(ddof=1)
    if how == "sd":
        return v.mean(), sd
    sem = sd / np.sqrt(len(v))
    if how == "ci":
        from scipy.stats import t
        return v.mean(), t.ppf(0.975, len(v) - 1) * sem
    return v.mean(), sem


def _mode(errorbar, group):
    if errorbar == "auto":
        return "sem" if group else "ci"
    if not group and errorbar in ("sem", "sd"):
        raise ValueError("one participant: errorbar 'ci' (bootstrap) or None")
    return errorbar


def _pids_with(analysis, pids, complete_only):
    t = P.collect(analysis, pids)
    if complete_only:
        t = t[t.status == "complete"]
    return t


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"{name}.{ext}")
    return fig


def _who_label(who, table):
    if who == GROUP:
        n = table.participant.nunique()
        return f"Group (N = {n}: {', '.join(sorted(table.participant.unique()))})"
    s = table[table.participant == who].status.iloc[0]
    return f"{who} ({s})"


# ── figure 1: unimodal ────────────────────────────────────────────────────────

MEASURES = {"weber": ("weber", "Weber fraction"), "jnd": ("jnd_ms", "JND (ms)"),
            "sigma": ("sigma", "σ (log units, psychometric)")}


def _unimodal_curves(ax, table, trials, scale=1.0):
    ms = np.geomspace(40, 2300, 300)
    for cond, sty in St.UNIMODAL_STYLE.items():
        rows = table[table.condition == cond]
        if rows.empty:
            continue
        ax.plot(ms, _curve_ms(rows, ms), color=sty["color"], label=sty["label"], zorder=3)
        _points(ax, binned(trials[trials.condition == cond]), sty["color"], scale)


def fig1_unimodal(who="MZ", measure="weber", errorbar="auto", pids=None, complete_only=False, save=True):
    table = _pids_with("unimodal", pids, complete_only)
    col, ylab = MEASURES[measure]
    if who == ALL:
        parts = sorted(table.participant.unique(), key=D.participants().index)
        fig, axes = St.grid(len(parts), ncols=4, panel=(4.2, 3.6), sharey=True)
        for ax, pid in zip(axes, parts):
            t = table[table.participant == pid]
            _unimodal_curves(ax, t, D.load_unimodal(pid)[0], scale=0.6)
            _pf_axes(ax, f"{pid} ({t.status.iloc[0]})")
        handles = [Line2D([], [], color=s["color"], label=s["label"]) for s in St.UNIMODAL_STYLE.values()]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout()
        return _save(fig, "fig1_unimodal_all", save)

    group = who == GROUP
    mode = _mode(errorbar, group)
    t = table if group else table[table.participant == who]
    if t.empty:
        raise ValueError(f"no unimodal data for {who}")
    trials = (pd.concat([D.load_unimodal(p)[0] for p in t.participant.unique()]) if group
              else D.load_unimodal(who)[0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    _unimodal_curves(a1, t, trials)
    _pf_axes(a1)
    a1.legend(loc="upper left")
    conds = [c for c in St.UNIMODAL_STYLE if c in set(t.condition)]
    for i, cond in enumerate(conds):
        color = St.UNIMODAL_STYLE[cond]["color"]
        rows = t[t.condition == cond]
        if group:
            jitter = np.linspace(-0.12, 0.12, len(rows)) if len(rows) > 1 else [0.0]
            a2.scatter(i + np.asarray(jitter), rows[col], color=color, alpha=0.35, s=30, zorder=2)
            m, e = _err(rows[col], mode)
            a2.errorbar(i, m, yerr=None if np.isnan(e) else e, fmt="o", color=color, ms=10, capsize=6, lw=2, zorder=3)
        else:
            r = rows.iloc[0]
            yerr = [[r[col] - r[f"{col}_lo"]], [r[f"{col}_hi"] - r[col]]] if mode == "ci" else None
            a2.errorbar(i, r[col], yerr=yerr, fmt="o", color=color, ms=10, capsize=6, lw=2)
            if r.get("beyond_range", False):
                a2.annotate("extrapolated", (i, r[col]), textcoords="offset points", xytext=(8, 0), fontsize=10)
    a2.set_xticks(range(len(conds)))
    a2.set_xticklabels([St.UNIMODAL_STYLE[c]["label"].replace(", ", "\n") for c in conds])
    a2.set_xlim(-0.6, len(conds) - 0.4)
    a2.set_ylabel(ylab)
    if measure != "sigma":
        a2.set_yscale("log")
        lo, hi = a2.get_ylim()
        ticks = [v for v in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
                 if lo <= v <= hi]
        a2.set_yticks(ticks)
        a2.set_yticklabels([f"{v:g}" for v in ticks])
        a2.minorticks_off()
    St.panel_labels([a1, a2])
    fig.suptitle(_who_label(who, t), y=1.02)
    fig.tight_layout()
    return _save(fig, f"fig1_unimodal_{who}_{measure}", save)


# ── figure 2: bimodal psychometric curves ─────────────────────────────────────

def _bimodal_panel(ax, t, trials, noise, scale=1.0, points=True):
    ms = np.geomspace(40, 2300, 300)
    tn = t[np.isclose(t.noise, noise)]
    conflicts = sorted(tn.conflict_ms.unique())
    for k, c in enumerate(conflicts):
        color = St.conflict_color(noise, k, len(conflicts))
        ax.plot(ms, _curve_ms(tn[tn.conflict_ms == c], ms), color=color, lw=2.2 * scale + 0.4, zorder=3,
                label=f"{c:+d} ms")
        if points:
            sel = trials[np.isclose(trials.noise, noise) & (trials.conflict_ms == c)]
            _points(ax, binned(sel), color, scale * 0.6)
    _pf_axes(ax)


def fig2_bimodal(who="MZ", pids=None, complete_only=False, points=True, save=True):
    table = _pids_with("main", pids, complete_only)
    noises = (D.NOISE_LOW, D.NOISE_HIGH)
    if who == ALL:
        parts = sorted(table.participant.unique(), key=D.participants().index)
        fig, axes = plt.subplots(len(parts), 2, figsize=(11, 4.0 * len(parts)), squeeze=False, sharey=True)
        for row, pid in zip(axes, parts):
            t = table[table.participant == pid]
            trials = D.load(pid, "main")[0]
            for ax, noise in zip(row, noises):
                _bimodal_panel(ax, t, trials, noise, scale=0.7, points=points)
                ax.set_title(f"{pid} ({t.status.iloc[0]}): {St.NOISE_LABELS[noise].split(' (')[0].lower()}")
            row[1].set_ylabel("")
        for ax in axes[-1]:
            St.legend_outside(ax, "below", ncol=3, title="Conflict, visual - auditory (ms)",
                              fontsize=plt.rcParams["legend.fontsize"] * 0.9)
        fig.tight_layout()
        return _save(fig, "fig2_bimodal_all", save)

    group = who == GROUP
    t = table if group else table[table.participant == who]
    if t.empty:
        raise ValueError(f"no main-experiment data for {who}")
    trials = (pd.concat([D.load(p, "main")[0] for p in t.participant.unique()]) if group
              else D.load(who, "main")[0])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, noise in zip(axes, noises):
        _bimodal_panel(ax, t, trials, noise, points=points)
        ax.set_title(St.NOISE_LABELS[noise])
        ax.legend(loc="lower right", ncol=2, title="Conflict (ms)", fontsize=10, title_fontsize=11,
                  handlelength=1.2, columnspacing=0.8)
    St.panel_labels(axes)
    fig.suptitle(_who_label(who, t), y=1.02)
    fig.tight_layout()
    return _save(fig, f"fig2_bimodal_{who}", save)


# ── figure 3: effect of conflict on PSE and sigma ─────────────────────────────

def _with_baseline(t, pid, baseline):
    """Per-condition PSE shift (and bootstrap CI) against the standard or the participant's 0-conflict PSE."""
    t = t.copy()
    if baseline == "standard":
        t["y_pse"], t["y_pse_lo"], t["y_pse_hi"] = t.shift_ms, t.shift_ms_lo, t.shift_ms_hi
        return t
    boot = P.load_boot(pid, "main")
    conds = list(boot["conditions"])
    pse_b = 1000 * D.STANDARD_S * np.exp(boot["mu"])
    for noise in t.noise.unique():
        base = t[np.isclose(t.noise, noise) & (t.conflict_ms == 0)].iloc[0]
        j0 = conds.index(base.condition)
        for i in t.index[np.isclose(t.noise, noise)]:
            j = conds.index(t.at[i, "condition"])
            diff = pse_b[:, j] - pse_b[:, j0]
            t.at[i, "y_pse"] = t.at[i, "pse_ms"] - base.pse_ms
            t.at[i, "y_pse_lo"], t.at[i, "y_pse_hi"] = np.percentile(diff, [2.5, 97.5])
    return t


def _conflict_panels(a_pse, a_sig, t, mode, group, sigma, unimodal=None, small=False):
    offsets = {D.NOISE_LOW: -6.0, D.NOISE_HIGH: 6.0}
    sig_col = "sigma" if sigma == "pf" else "sigma_single"
    div = 1.0 if sigma == "pf" else np.sqrt(2)
    for noise in (D.NOISE_LOW, D.NOISE_HIGH):
        tn = t[np.isclose(t.noise, noise)]
        conflicts = sorted(tn.conflict_ms.unique())
        for k, c in enumerate(conflicts):
            rows = tn[tn.conflict_ms == c]
            rgba = St.conflict_rgba(noise, k, len(conflicts))
            x = c + offsets[noise]
            for ax, col, lo, hi in ((a_pse, "y_pse", "y_pse_lo", "y_pse_hi"),
                                    (a_sig, sig_col, f"{sig_col}_lo", f"{sig_col}_hi")):
                if group:
                    m, e = _err(rows[col], mode)
                    yerr = None if np.isnan(e) else e
                else:
                    r = rows.iloc[0]
                    m = r[col]
                    yerr = [[m - r[lo]], [r[hi] - m]] if mode == "ci" else None
                ax.errorbar(x, m, yerr=yerr, fmt="o", color=rgba, mec=rgba, ms=5 if small else 8,
                            capsize=3 if small else 6, lw=1.6, zorder=4)
        if unimodal is not None:
            u = unimodal[unimodal.condition == ("A_low" if noise == D.NOISE_LOW else "A_high")]
            if not u.empty:
                a_sig.axhline(u["sigma"].mean() / div, color=St.NOISE_COLORS[noise], ls=":", lw=1.4, zorder=1)
    if unimodal is not None and (unimodal.condition == "V").any():
        a_sig.axhline(unimodal.loc[unimodal.condition == "V", "sigma"].mean() / div, color=St.VISUAL_COLOR,
                      ls=":", lw=1.4, zorder=1)
    for ax in (a_pse, a_sig):
        ax.axvline(0, **St.GUIDE)
        ax.set_xlabel("Conflict: visual - auditory standard (ms)")
    a_pse.axhline(0, **St.GUIDE)


def fig3_conflict_effect(who="MZ", errorbar="auto", baseline="standard", sigma="pf", unimodal_ref=True,
                         pids=None, complete_only=False, save=True):
    """baseline: "standard" (PSE - 500 ms, v1) or "matched" (PSE - the participant's 0-conflict PSE).
    sigma: "pf" (psychometric sigma) or "single" (/ sqrt 2, one interval's noise as in the models).
    unimodal_ref: dotted lines at the unimodal sigma (auditory per noise level, visual black)."""
    table = _pids_with("main", pids, complete_only)
    uni = P.collect("unimodal", pids) if unimodal_ref else None
    ylab_pse = "PSE shift (ms)" if baseline == "standard" else "PSE - PSE at 0 conflict (ms)"
    ylab_sig = "σ (log units, psychometric)" if sigma == "pf" else "σ single interval (log units)"
    handles = [Line2D([], [], color=St.NOISE_COLORS[n], marker="o", lw=0, label=St.NOISE_LABELS[n])
               for n in (D.NOISE_LOW, D.NOISE_HIGH)]
    if unimodal_ref:
        handles.append(Line2D([], [], color="0.3", ls=":", label="Unimodal σ"))
    if who == ALL:
        parts = sorted(table.participant.unique(), key=D.participants().index)
        fig, axes = plt.subplots(len(parts), 2, figsize=(12, 3.6 * len(parts)), squeeze=False, sharex=True)
        mode = _mode(errorbar, False)
        for (a_pse, a_sig), pid in zip(axes, parts):
            t = _with_baseline(table[table.participant == pid], pid, baseline)
            u = uni[uni.participant == pid] if uni is not None else None
            _conflict_panels(a_pse, a_sig, t, mode, False, sigma, u, small=True)
            a_pse.set_ylabel(ylab_pse)
            a_sig.set_ylabel(ylab_sig)
            a_pse.set_title(f"{pid} ({t.status.iloc[0]})", loc="left")
        for ax in axes[:-1].ravel():
            ax.set_xlabel("")
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout()
        return _save(fig, f"fig3_conflict_all_{baseline}_{sigma}", save)

    group = who == GROUP
    mode = _mode(errorbar, group)
    t = table if group else table[table.participant == who]
    if t.empty:
        raise ValueError(f"no main-experiment data for {who}")
    t = pd.concat([_with_baseline(t[t.participant == p], p, baseline) for p in t.participant.unique()])
    u = None if uni is None else uni[uni.participant.isin(t.participant.unique())]   # same participants as the main data
    fig, (a_pse, a_sig) = plt.subplots(1, 2, figsize=(14, 5.4))
    _conflict_panels(a_pse, a_sig, t, mode, group, sigma, u)
    a_pse.set_ylabel(ylab_pse)
    a_sig.set_ylabel(ylab_sig)
    a_sig.legend(handles=handles, loc="upper left", fontsize=12)
    St.panel_labels([a_pse, a_sig])
    bars = {"ci": "bootstrap 95% CI" if not group else "95% CI over participants", "sem": "SEM", "sd": "SD",
            None: "no error bars"}[mode]
    fig.suptitle(f"{_who_label(who, t)}; error bars: {bars}", y=1.02)
    fig.tight_layout()
    return _save(fig, f"fig3_conflict_{who}_{baseline}_{sigma}", save)


# ── figure 4: staircases ──────────────────────────────────────────────────────

STAIR_STYLE = {0: dict(marker="o", ls="-"), 1: dict(marker="s", ls="--"), 2: dict(marker="^", ls="-."),
               3: dict(marker="D", ls=":")}


def _stair_panel(ax, d, color, title, part_len=None):
    rules = sorted(r for r in d.current_stair.dropna().unique() if r not in ("lapse_rate", "warmup"))
    for i, rule in enumerate(rules):
        s = d[d.current_stair == rule]
        n = np.arange(1, len(s) + 1)
        ms = 1000 * s.testDurS.to_numpy()
        sty = STAIR_STYLE[i % 4]
        ax.plot(n, ms, color=color, ls=sty["ls"], lw=1.0, alpha=0.6, zorder=2)
        longer = s.chose_test.to_numpy() == 1
        ax.scatter(n[longer], ms[longer], marker=sty["marker"], s=16, color=color, zorder=3)
        ax.scatter(n[~longer], ms[~longer], marker=sty["marker"], s=16, facecolors="white", edgecolors=color,
                   linewidths=0.9, zorder=3)
        rev = s.stair_is_reversal.eq(True).to_numpy()
        ax.scatter(n[rev], ms[rev], marker="x", s=22, color="0.25", lw=0.9, zorder=4)
    if part_len:
        top = max((d.current_stair == r).sum() for r in rules) if rules else 0
        for b in range(part_len, top, part_len):
            ax.axvline(b + 0.5, color="0.75", lw=0.8, zorder=0)
    ax.axhline(1000 * D.STANDARD_S, **St.GUIDE)
    ax.set_yscale("log")
    ax.set_yticks([50, 100, 200, 500, 1000])
    ax.set_yticklabels(["50", "100", "200", "500", "1000"])
    ax.minorticks_off()
    ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"] * 0.75)
    return rules


def fig4_staircases(pid="MZ", task="main", save=True):
    """Test duration of every staircase trial in order (filled: answered 'test longer'; open: 'standard
    longer'; x: reversal). Main: rows noise level, columns conflict; grey lines separate the parts.
    Unimodal ("unimodal"): auditory low, auditory high, visual. Easy trials are not shown."""
    if task == "main":
        d, _ = D.load(pid, "main", cleaned=False)
        if d.empty:
            raise ValueError(f"{pid} has no main-experiment data")
        conflicts = sorted(d.conflict_ms.unique())
        fig, axes = plt.subplots(2, len(conflicts), figsize=(2.3 * len(conflicts), 6.2), sharey=True, sharex=True)
        per_part = int(d[d.kind == "staircase"].groupby(["session_file", "condition", "current_stair"]).size().max())
        for row, noise in zip(axes, (D.NOISE_LOW, D.NOISE_HIGH)):
            for k, (ax, c) in enumerate(zip(row, conflicts)):
                sel = d[np.isclose(d.noise, noise) & (d.conflict_ms == c) & (d.kind == "staircase")]
                rules = _stair_panel(ax, sel, St.conflict_color(noise, k, len(conflicts)), f"{c:+d} ms", per_part)
            row[0].set_ylabel(f"{'Low' if noise == D.NOISE_LOW else 'High'} noise\nTest (ms)")
        for ax in axes[-1]:
            ax.set_xlabel("Trial")
        parts = ", ".join(sorted(d.block.dropna().unique()))
        name = f"fig4_staircases_{pid}_main"
    else:
        d, _ = D.load_unimodal(pid, cleaned=False)
        if d.empty:
            raise ValueError(f"{pid} has no unimodal data")
        conds = [c for c in St.UNIMODAL_STYLE if c in set(d.condition)]
        fig, axes = plt.subplots(1, len(conds), figsize=(5.0 * len(conds), 4.2), sharey=True, squeeze=False)
        for ax, cond in zip(axes[0], conds):
            sel = d[(d.condition == cond) & (d.kind == "staircase")]
            rules = _stair_panel(ax, sel, St.UNIMODAL_STYLE[cond]["color"], St.UNIMODAL_STYLE[cond]["label"])
            ax.set_xlabel("Trial")
        axes[0][0].set_ylabel("Test (ms)")
        parts = "unimodal blocks"
        name = f"fig4_staircases_{pid}_unimodal"
    handles = [Line2D([], [], color="0.3", marker=STAIR_STYLE[i]["marker"], ls=STAIR_STYLE[i]["ls"], label=r)
               for i, r in enumerate(rules)]
    handles += [Line2D([], [], color="0.3", marker="o", lw=0, label='"test longer"'),
                Line2D([], [], color="0.3", marker="o", lw=0, mfc="white", label='"standard longer"'),
                Line2D([], [], color="0.25", marker="x", lw=0, label="reversal")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.06), fontsize=11)
    fig.suptitle(f"{pid} ({'complete' if D.status(pid)['complete'] else 'partial'}): staircases, {parts}", y=1.01)
    fig.tight_layout()
    return _save(fig, name, save)
