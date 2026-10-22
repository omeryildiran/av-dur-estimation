"""Psychometric fit of one finished block, checks on attention/precision, and a plot.

Used by the combined session (avdur.run_blocks) while the participant rests: the fit runs in
a background process, never during trials. Checks concern only precision (sigma) and easy-
trial accuracy, never the PSE or the bias (SCREENING_CHECKS in experiment_settings.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_settings as S
from avdur import psychometrics as pm

LABELS = {"noise_calibration": "Noise calibration", "unimodal_auditory": "Unimodal auditory", "unimodal_visual": "Unimodal visual", "crossmodal": "Cross-modal"}


def _condition_name(task, noise):
    return "visual" if noise is None else f"noise {noise:g}"


def block_fit(task, csv_path, plot_dir=None):
    """Fit one block's CSV; returns a dict with per-condition parameters, warnings and the plot path."""
    if hasattr(os, "nice"):
        os.nice(5)                                   # stay out of the way of the window process
    df = pd.read_csv(csv_path)
    df = df[df["responses"].isin([1, 2])].copy()
    df["cond"] = df["audNoise"] if task != "unimodal_visual" else "visual"
    df = df[df["kind"] != "practice"]
    result = dict(task=task, csv=str(csv_path), n_trials=int(len(df)), conditions=[], warnings=[], plot=None)
    if df.empty:
        result["warnings"].append("no answered trials")
        return result
    fit = pm.fit(df, "cond", "lognormal", lapse_max=S.BIAS["lapse_max"])
    result["at_bound"] = list(fit["at_bound"])
    for _, p in fit["params"].iterrows():
        noise = None if p.condition == "visual" else float(p.condition)
        d = df[df["cond"] == p.condition]
        easy = d[d["kind"] == "lapse"]["is_correct"]
        std_ms = 1000 * float(d["standardDur"].iloc[0])
        entry = dict(noise=noise, mu=float(p.mu), sigma=float(p.sigma), lapse=float(p.lapse), n=int(p.n),
                     pse_ratio=pm.pse_ratio(p.mu, "lognormal"), pse_ms=std_ms * pm.pse_ratio(p.mu, "lognormal"),
                     standard_ms=std_ms,
                     easy_correct=float(easy.mean()) if len(easy) else float("nan"), easy_n=int(len(easy)),
                     sigma_at_bound=bool(p.sigma >= 2.999), v1_sigma_range=S.V1_SIGMA_RANGE.get((task, noise)))
        if task == "crossmodal":
            entry["correction_s"] = S.STANDARD_DUR_S - S.STANDARD_DUR_S * entry["pse_ratio"]
        checks = S.SCREENING_CHECKS.get(task, {})
        check = checks.get(noise) or (checks.get("high") if noise not in (None, S.NOISE_LOW) else None)
        name = _condition_name(task, noise)
        if check:
            if "max_sigma" in check and entry["sigma"] > check["max_sigma"]:
                result["warnings"].append(f"{LABELS[task]}, {name}: sigma {entry['sigma']:.2f} > {check['max_sigma']}")
            if "min_easy_correct" in check and entry["easy_n"] and entry["easy_correct"] < check["min_easy_correct"]:
                result["warnings"].append(f"{LABELS[task]}, {name}: easy trials {entry['easy_correct']:.0%} correct "
                                          f"< {check['min_easy_correct']:.0%}")
        result["conditions"].append(entry)
    if plot_dir is not None:
        result["plot"] = str(plot(task, df, result, Path(plot_dir), Path(csv_path).stem))
    return result


def plot(task, df, result, folder, stem):
    """P(test judged longer) against the test duration (ms), one curve per condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    colors = {None: "tab:green", 0.1: "tab:blue", 1.2: "tab:red"}
    ms = np.linspace(30, 1000, 400)
    for entry in result["conditions"]:
        noise, std = entry["noise"], entry["standard_ms"]
        c = colors.get(noise, "black")
        d = df[df["cond"] == ("visual" if noise is None else noise)]
        x = 1000 * d["testDurS"]
        bins = np.unique(np.quantile(x, np.linspace(0, 1, 11)))      # about equal trials per point
        bins[0], bins[-1] = bins[0] - 1e-6, bins[-1] + 1e-6
        pts = pd.DataFrame({"x": x, "y": d["chose_test"]}).groupby(pd.cut(x, bins), observed=True).agg(
            x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
        ax.scatter(pts.x, pts.y, s=10 + 2 * pts.n, color=c, alpha=0.6)
        ax.plot(ms, pm.predict(np.log(ms / std), entry["mu"], entry["sigma"], entry["lapse"]), color=c,
                label=f"{_condition_name(task, noise)}: σ {entry['sigma']:.2f}, PSE {entry['pse_ms']:.0f} ms, "
                      f"λ {entry['lapse']:.2f}")
    std = result["conditions"][0]["standard_ms"] if result["conditions"] else 500
    ax.axhline(0.5, color="0.85", lw=1)
    ax.axvline(std, color="0.6", lw=1, ls=":")
    what = "Auditory test" if task == "crossmodal" else "Test"
    ax.set(xlabel=f"{what} duration (ms)   [standard {std:.0f} ms]", ylabel="P(test judged longer)",
           xlim=(0, 1000), ylim=(-0.03, 1.03), title=f"{LABELS[task]}  ({result['n_trials']} trials)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = folder / f"{stem}_fit.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary_text(result):
    """Terminal lines for one block's fit."""
    lines = [f"── {LABELS[result['task']]} fit ({result['n_trials']} trials) ──"]
    for e in result["conditions"]:
        v1 = e["v1_sigma_range"]
        ref = f"  [v1 σ {v1[0]:.2f}–{v1[2]:.2f}, median {v1[1]:.2f}]" if v1 else ""
        if result["task"] == "unimodal_auditory" and e["noise"] not in (None, S.NOISE_LOW):
            ref = f"  [participant's calibrated level; target σ {S.NOISE_CALIBRATION['target_sigma']}]"
        extra = f", visual correction {e['correction_s']:+.3f} s" if "correction_s" in e else ""
        bound = " (AT THE FIT BOUND: about chance)" if e["sigma_at_bound"] else ""
        easy = f"{e['easy_correct']:.0%} of {e['easy_n']}" if e["easy_n"] else "none"
        lines.append(f"   {_condition_name(result['task'], e['noise']):9s} σ {e['sigma']:.2f}{bound}, "
                     f"PSE {e['pse_ms']:.0f} ms, lapse {e['lapse']:.2f}, easy trials {easy}{extra}{ref}")
    lines += [f"   WARNING: {w}" for w in result["warnings"]] or ["   checks passed"]
    if result.get("plot"):
        lines.append(f"   plot: {result['plot']}")
    return "\n".join(lines)


# ── main-experiment block: PSE against conflict ─────────────────────────────────

def main_block_fit(csv_path, plot_dir=None):
    """PSE shift per noise level and conflict, the slope of PSE on conflict (the weight given to
    the visual duration), and a plot. Pooled over presentation orders, lognormal, lapse <= 0.1."""
    df = pd.read_csv(csv_path)
    df = df[df["responses"].isin([1, 2]) & (df["kind"] != "practice")].copy()
    df["conflict_ms"] = (1000 * df["conflict_nominal"]).round().astype(int)
    result = dict(task="main", csv=str(csv_path), n_trials=int(len(df)), block=str(df["block"].iloc[0]) if len(df) else "",
                  conditions=[], slopes={}, warnings=[], plot=None)
    for noise, d in df.groupby("audNoise"):
        fit = pm.fit(d, "conflict_ms", "lognormal", lapse_max=S.BIAS["lapse_max"])["params"]
        std_ms = 1000 * float(d["standardDur"].iloc[0])
        rows = [dict(noise=float(noise), conflict_ms=int(p.condition), pse_shift_ms=std_ms * (np.exp(p.mu) - 1),
                     sigma=float(p.sigma), n=int(p.n)) for _, p in fit.iterrows()]
        result["conditions"] += rows
        if len(rows) > 1:
            x = np.array([r["conflict_ms"] for r in rows]); y = np.array([r["pse_shift_ms"] for r in rows])
            result["slopes"][float(noise)] = float(np.polyfit(x, y, 1)[0])
    if plot_dir is not None and result["conditions"]:
        result["plot"] = str(plot_main(result, Path(plot_dir), Path(csv_path).stem))
    return result


def plot_main(result, folder, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(result["conditions"])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    lim = max(300, t.conflict_ms.abs().max() + 50)
    ax.plot([-lim, lim], [-lim, lim], ":", color="0.6", label="visual capture (slope 1)")
    ax.axhline(0, color="0.85", lw=1)
    for noise, g in t.groupby("noise"):
        c = {0.1: "tab:blue", 1.2: "tab:red"}.get(noise, "black")
        g = g.sort_values("conflict_ms")
        ax.plot(g.conflict_ms, g.pse_shift_ms, "o-", color=c,
                label=f"noise {noise:g}: slope {result['slopes'].get(noise, np.nan):.2f}")
    ax.set(xlabel="Conflict: visual − auditory (ms)", ylabel="PSE shift (ms)", xlim=(-lim, lim),
           title=f"Main experiment {result['block']}  ({result['n_trials']} trials)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = folder / f"{stem}_fit.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main_summary_text(result):
    lines = [f"── Main experiment {result['block']} fit ({result['n_trials']} trials) ──"]
    for noise in sorted({c["noise"] for c in result["conditions"]}):
        cells = [c for c in result["conditions"] if c["noise"] == noise]
        shifts = ", ".join(f"{c['conflict_ms']:+d}: {c['pse_shift_ms']:+.0f}" for c in sorted(cells, key=lambda c: c["conflict_ms"]))
        lines.append(f"   noise {noise:g}: PSE shift (ms) by conflict  {shifts};  slope {result['slopes'].get(noise, float('nan')):.2f}")
    if result.get("plot"):
        lines.append(f"   plot: {result['plot']}")
    return "\n".join(lines)


def fit_after_session(task, csv_path, plot_dir):
    """Fit and plot a finished session from its own button; returns the terminal text."""
    if task == "main":
        return main_summary_text(main_block_fit(csv_path, plot_dir))
    return summary_text(block_fit(task, csv_path, plot_dir))
