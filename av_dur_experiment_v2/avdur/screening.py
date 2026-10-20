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

LABELS = {"unimodal_auditory": "Unimodal auditory", "unimodal_visual": "Unimodal visual", "crossmodal": "Cross-modal"}


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
        entry = dict(noise=noise, mu=float(p.mu), sigma=float(p.sigma), lapse=float(p.lapse), n=int(p.n),
                     pse_ratio=pm.pse_ratio(p.mu, "lognormal"),
                     easy_correct=float(easy.mean()) if len(easy) else float("nan"), easy_n=int(len(easy)),
                     sigma_at_bound=bool(p.sigma >= 2.999), v1_sigma_range=S.V1_SIGMA_RANGE.get((task, noise)))
        if task == "crossmodal":
            entry["correction_s"] = S.STANDARD_DUR_S - S.STANDARD_DUR_S * entry["pse_ratio"]
        check = S.SCREENING_CHECKS.get(task, {}).get(noise)
        name = _condition_name(task, noise)
        if check:
            if entry["sigma"] > check["max_sigma"]:
                result["warnings"].append(f"{LABELS[task]}, {name}: sigma {entry['sigma']:.2f} > {check['max_sigma']}")
            if entry["easy_n"] and entry["easy_correct"] < check["min_easy_correct"]:
                result["warnings"].append(f"{LABELS[task]}, {name}: easy trials {entry['easy_correct']:.0%} correct "
                                          f"< {check['min_easy_correct']:.0%}")
        result["conditions"].append(entry)
    if plot_dir is not None:
        result["plot"] = str(plot(task, df, result, Path(plot_dir), Path(csv_path).stem))
    return result


def plot(task, df, result, folder, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.linspace(np.log(0.05), np.log(1.95), 300)
    colors = {None: "tab:green", 0.1: "tab:blue", 1.2: "tab:red"}
    for entry in result["conditions"]:
        noise = entry["noise"]
        d = df[df["cond"] == ("visual" if noise is None else noise)]
        x = np.log(d["testDurS"] / d["standardDur"])
        bins = np.unique(np.quantile(x, np.linspace(0, 1, 11)))      # about equal trials per point
        bins[0], bins[-1] = bins[0] - 1e-9, bins[-1] + 1e-9
        g = pd.DataFrame({"x": x, "y": d["chose_test"]}).groupby(pd.cut(x, bins), observed=True)
        pts = g.agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
        c = colors.get(noise, "black")
        label = (f"{_condition_name(task, noise)}: σ {entry['sigma']:.2f}, PSE ×{entry['pse_ratio']:.2f}, "
                 f"λ {entry['lapse']:.2f}")
        ax.scatter(np.exp(pts.x), pts.y, s=12 + pts.n, color=c, alpha=0.6)
        ax.plot(np.exp(xs), pm.predict(xs, entry["mu"], entry["sigma"], entry["lapse"]), color=c, label=label)
    ax.axhline(0.5, color="0.8", lw=1)
    ax.axvline(1.0, color="0.8", lw=1)
    xlabel = "Auditory test / visual standard duration" if task == "crossmodal" else "Test / standard duration"
    ax.set(xlabel=xlabel, ylabel="P(test judged longer)", ylim=(-0.03, 1.03),
           title=f"{LABELS[task]}  ({result['n_trials']} trials)")
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
        extra = f", visual correction {e['correction_s']:+.3f} s" if "correction_s" in e else ""
        bound = " (AT THE FIT BOUND: about chance)" if e["sigma_at_bound"] else ""
        easy = f"{e['easy_correct']:.0%} of {e['easy_n']}" if e["easy_n"] else "none"
        lines.append(f"   {_condition_name(result['task'], e['noise']):9s} σ {e['sigma']:.2f}{bound}, "
                     f"PSE ×{e['pse_ratio']:.2f}, lapse {e['lapse']:.2f}, easy trials {easy}{extra}{ref}")
    lines += [f"   WARNING: {w}" for w in result["warnings"]] or ["   checks passed"]
    if result.get("plot"):
        lines.append(f"   plot: {result['plot']}")
    return "\n".join(lines)
