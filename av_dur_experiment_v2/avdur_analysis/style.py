"""Figure style: v1's plot_style.py (fonts, ticks, panel labels) scaled for multi-panel figures, and v1's
colour code.

Colours (v1 article): low auditory noise red, high auditory noise blue, visual black; conflicts are shades
of the noise colour (Reds / Blues, lighter = more negative); models: forced fusion green dashed squares,
causal inference purple solid circles, cue switching navy dotted triangles.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from . import data as D

NOISE_COLORS = {D.NOISE_LOW: "#d62728", D.NOISE_HIGH: "#1f77b4"}
VISUAL_COLOR = "#111111"
NOISE_CMAPS = {D.NOISE_LOW: "Reds", D.NOISE_HIGH: "Blues"}
NOISE_LABELS = {D.NOISE_LOW: f"Low auditory noise ({D.NOISE_LOW:.1f})",
                D.NOISE_HIGH: f"High auditory noise ({D.NOISE_HIGH:.1f})"}
UNIMODAL_STYLE = {"A_low": dict(color=NOISE_COLORS[D.NOISE_LOW], label="Auditory, low noise"),
                  "A_high": dict(color=NOISE_COLORS[D.NOISE_HIGH], label="Auditory, high noise"),
                  "V": dict(color=VISUAL_COLOR, label="Visual")}
MODEL_STYLE = {"fusion": dict(color="#2ca02c", ls="--", marker="s", label="Forced fusion"),
               "causal": dict(color="#600c70", ls="-", marker="o", label="Causal inference"),
               "switching": dict(color="#080529", ls=":", marker="^", label="Cue switching")}
GUIDE = dict(color="0.6", ls="--", lw=1.0, zorder=0)
PANEL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DURATION_TICKS_MS = [50, 100, 200, 500, 1000, 2000]


def setup(scale=1.0):
    """v1 rcParams; scale < 1 for figures with many panels (v1: 18 pt labels)."""
    fs = 18 * scale
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs * 1.05, "xtick.labelsize": fs * 0.85,
        "ytick.labelsize": fs * 0.85, "legend.fontsize": fs * 0.75, "axes.linewidth": 1.5,
        "xtick.major.width": 1.5, "ytick.major.width": 1.5, "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 7, "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "black", "legend.fancybox": False, "axes.grid": False, "figure.dpi": 100,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.2})


def conflict_color(noise, k, n):
    """Shade of the noise colour for the k-th of n conflicts (v1 started at 0.25; 0.4 keeps the lightest
    shade visible on white)."""
    cmap = plt.get_cmap(NOISE_CMAPS[noise])
    return cmap(0.4 + 0.55 * k / max(n - 1, 1))


def conflict_rgba(noise, k, n):
    """v1's PSE-vs-conflict points: the noise colour with alpha rising over the conflicts (v1 from 0.25;
    from 0.45 here so the most negative conflict stays visible)."""
    return (*mcolors.to_rgb(NOISE_COLORS[noise]), float(np.linspace(0.45, 1.0, n)[k]))


def panel_labels(axes, labels=None, x=-0.14, y=1.06, **kw):
    for ax, lab in zip(np.ravel(axes), labels or PANEL_LABELS):
        ax.text(x, y, lab, transform=ax.transAxes, fontweight="bold", va="top",
                fontsize=plt.rcParams["axes.titlesize"] * 1.1, **kw)


def duration_axis(ax, lo=40, hi=2300):
    """Log-scaled test-duration axis in ms with plain tick labels."""
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks([t for t in DURATION_TICKS_MS if lo <= t <= hi])
    ax.set_xticklabels([str(t) for t in DURATION_TICKS_MS if lo <= t <= hi])
    ax.minorticks_off()


def legend_outside(ax, where="right", **kw):
    """Legend box outside the axes (right or below), so it never covers data."""
    if where == "below":
        return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=kw.pop("ncol", 3), **kw)
    return ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, **kw)


def grid(n, ncols=4, panel=(4.0, 3.4), **kw):
    """Figure with n panels on a grid; unused panels hidden. Returns (fig, flat axes[:n])."""
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False, **kw)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]
