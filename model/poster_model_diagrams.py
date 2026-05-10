"""
Poster model diagrams for VSS — audiovisual duration estimation & causal inference.

Generates minimalist, vector-style schematic figures (SVG + PDF + PNG preview)
of the model families: Fusion, Causal Inference (Averaging / Selection /
Probability Matching), and Heuristic cue switching.

Style: thin lines, no axes, muted scientific palette, lots of whitespace.
All output is fully editable in Illustrator (no rasterized elements).

Run as a script or import the helpers from a notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch, Circle, Rectangle


# -----------------------------------------------------------------------------
# Style constants
# -----------------------------------------------------------------------------

C_AUD   = "#2b2b2b"   # auditory: dark warm gray
C_VIS   = "#C0553D"   # visual: muted red/orange
C_FUSED = "#111111"   # fused estimate: near black
C_GRAY  = "#9a9a9a"   # dashed / probabilistic
C_LIGHT = "#e8e8e8"   # very light gray fills
C_PANEL = "#f6f4f1"   # soft warm panel background
C_RULE  = "#d8d4cf"   # very faint divider/baseline
C_NODE  = "#111111"

LW_HAIR   = 0.55
LW_THIN   = 0.85
LW_NORMAL = 1.10
LW_BOLD   = 1.55

FONT_LABEL  = dict(fontsize=9.0)
FONT_TITLE  = dict(fontsize=11.0, weight="semibold")
FONT_SMALL  = dict(fontsize=7.8)
FONT_TAG    = dict(fontsize=7.6, weight="medium")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,        # editable text in Illustrator
    "ps.fonttype":  42,
    "svg.fonttype": "none",    # keep text as text in SVG
    "axes.linewidth": LW_THIN,
})


# -----------------------------------------------------------------------------
# Reusable visual primitives
# -----------------------------------------------------------------------------

def _clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def draw_gaussian(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    width: float = 1.2,
    height: float = 0.55,
    color: str = C_AUD,
    fill: bool = True,
    alpha_fill: float = 0.18,
    lw: float = LW_NORMAL,
    n: int = 160,
    label: Optional[str] = None,
    label_dy: float = -0.18,
    baseline: bool = True,
    baseline_pad: float = 0.10,
) -> None:
    """Gaussian bump with vertical alpha gradient fill and a hairline baseline."""
    xs = np.linspace(-3, 3, n)
    ys = np.exp(-0.5 * xs ** 2)
    ys = ys / ys.max() * height
    xs = xs / 6.0 * width + x
    ys = ys + y

    if fill:
        # Vertical gradient: denser near the baseline, fading toward the apex.
        layers = 24
        for i in range(layers):
            t = i / layers
            top = y + (1 - t) * height
            a = alpha_fill * (1 - t) ** 1.4
            ax.fill_between(xs, y, np.minimum(ys, top),
                            color=color, alpha=a, linewidth=0)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round")
    if baseline:
        ax.plot([x - width / 2 - baseline_pad, x + width / 2 + baseline_pad],
                [y, y], color=C_RULE, lw=LW_HAIR, solid_capstyle="round",
                zorder=1)
    if label is not None:
        ax.text(x, y + label_dy, label, ha="center", va="top",
                color=color, **FONT_LABEL)


def draw_pill(ax, xy, text, *, color="#3a3a3a",
              face="#f3efe8", edge="#cdc6bb", pad_x=0.10, pad_y=0.05,
              fontsize=7.6):
    """Soft pill label — use for scenario tags like 'common cause'."""
    from matplotlib.patches import FancyBboxPatch
    txt = ax.text(xy[0], xy[1], text, ha="center", va="center",
                  color=color, fontsize=fontsize, weight="medium", zorder=4)
    # estimate text bbox via renderer
    fig = ax.figure
    fig.canvas.draw()
    bb = txt.get_window_extent().transformed(ax.transData.inverted())
    w = (bb.width + 2 * pad_x); h = (bb.height + 2 * pad_y)
    box = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.10",
                         facecolor=face, edgecolor=edge,
                         lw=LW_HAIR, zorder=3)
    ax.add_patch(box)
    txt.set_zorder(5)


def draw_arrow(
    ax: plt.Axes,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    *,
    color: str = C_NODE,
    lw: float = LW_NORMAL,
    style: str = "-|>",
    dashed: bool = False,
    mutation: float = 8.0,
    shrinkA: float = 0.0,
    shrinkB: float = 2.0,
    curve: float = 0.0,
) -> None:
    """Solid or dashed arrow between two points. `curve` bends via rad= arg3."""
    cs = f"arc3,rad={curve}"
    arr = FancyArrowPatch(
        p0, p1,
        arrowstyle=style,
        connectionstyle=cs,
        mutation_scale=mutation,
        color=color, lw=lw,
        linestyle=(0, (3, 2)) if dashed else "-",
        shrinkA=shrinkA, shrinkB=shrinkB,
        joinstyle="round", capstyle="round",
    )
    ax.add_patch(arr)


def draw_probabilistic_arrow(ax, p0, p1, *, color=C_GRAY, lw=LW_THIN, **kw):
    """Dashed arrow signalling stochastic / probabilistic flow."""
    draw_arrow(ax, p0, p1, color=color, lw=lw, dashed=True, **kw)


def draw_branch(
    ax: plt.Axes,
    src: Tuple[float, float],
    targets: Sequence[Tuple[float, float]],
    *,
    color: str = C_NODE,
    lw: float = LW_NORMAL,
    dashed: bool = False,
    style: str = "-|>",
) -> None:
    """One source → multiple targets. Useful for branching decisions."""
    for t in targets:
        draw_arrow(ax, src, t, color=color, lw=lw, dashed=dashed, style=style)


def draw_merge(
    ax: plt.Axes,
    sources: Sequence[Tuple[float, float]],
    target: Tuple[float, float],
    *,
    color: str = C_NODE,
    lw: float = LW_NORMAL,
    weighted: bool = True,
    weight_labels: Optional[Sequence[str]] = None,
) -> None:
    """Multiple sources → one target. Optional small weight tags on incoming arrows."""
    for i, s in enumerate(sources):
        draw_arrow(ax, s, target, color=color, lw=lw)
        if weighted and weight_labels is not None and i < len(weight_labels):
            mx, my = (s[0] + target[0]) / 2, (s[1] + target[1]) / 2
            ax.text(mx + 0.04, my + 0.06, weight_labels[i],
                    ha="left", va="bottom", color="#555", **FONT_SMALL)
    # merge node
    ax.add_patch(Circle(target, 0.045, facecolor="white",
                        edgecolor=color, lw=lw, zorder=5))


def draw_threshold_gate(
    ax: plt.Axes,
    center: Tuple[float, float],
    *,
    width: float = 0.55,
    height: float = 0.28,
    color: str = C_NODE,
    lw: float = LW_NORMAL,
    label: Optional[str] = "switch",
) -> None:
    """A small step/threshold glyph used as a hard gate."""
    cx, cy = center
    # Bounding box
    rect = Rectangle((cx - width / 2, cy - height / 2), width, height,
                     facecolor="white", edgecolor=color, lw=lw)
    ax.add_patch(rect)
    # Step function inside the box
    xs = np.array([-1.0, -0.05, -0.05, 1.0]) * (width * 0.4) + cx
    ys = np.array([-1.0, -1.0,   1.0,   1.0]) * (height * 0.3) + cy
    ax.plot(xs, ys, color=color, lw=lw)
    if label:
        ax.text(cx, cy - height / 2 - 0.06, label,
                ha="center", va="top", color="#404040", **FONT_SMALL)


def draw_node(ax, xy, *, r=0.05, color=C_NODE, lw=LW_NORMAL, fill="white"):
    ax.add_patch(Circle(xy, r, facecolor=fill, edgecolor=color, lw=lw, zorder=5))


def draw_panel_title(ax, x, y, text, *, kicker: bool = False):
    if kicker:
        ax.text(x, y + 0.18, text.upper(), ha="center", va="bottom",
                fontsize=7.6, weight="bold", color="#7a7268",
                family="sans-serif")
    else:
        ax.text(x, y, text, ha="center", va="bottom", **FONT_TITLE)


def draw_soft_panel(ax, xy, w, h, *, face=C_PANEL, edge="#ece7df",
                    radius=0.18):
    """Faint rounded background panel — used to highlight a model family region."""
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(xy, w, h,
                         boxstyle=f"round,pad=0.0,rounding_size={radius}",
                         facecolor=face, edgecolor=edge,
                         lw=LW_HAIR, zorder=0)
    ax.add_patch(box)


def draw_section_rule(ax, x0, x1, y, *, color=C_RULE, lw=LW_HAIR):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=0)


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------

def panel_shared_encoding(ax: plt.Axes, *, x0: float = 0.0, y0: float = 0.0,
                          width: float = 4.0) -> None:
    """Top: shared sensory encoding — two overlapping Gaussians."""
    cx = x0 + width / 2
    # two gaussians slightly offset
    draw_gaussian(ax, cx - 0.55, y0, width=1.3, height=0.55,
                  color=C_AUD, label="Auditory")
    draw_gaussian(ax, cx + 0.55, y0, width=1.3, height=0.55,
                  color=C_VIS, label="Visual")
    ax.text(cx, y0 + 0.78, "Internal measurements", ha="center", va="bottom",
            **FONT_TITLE)
    ax.text(cx, y0 - 0.42, r"$m_a\sim\mathcal{N}(\log s_a,\sigma_a^2)$   "
                            r"$m_v\sim\mathcal{N}(\log s_v,\sigma_v^2)$",
            ha="center", va="top", **FONT_SMALL)


def panel_fusion(ax: plt.Axes, *, cx: float, cy: float) -> None:
    """Mandatory fusion: weighted avg into one estimate distribution."""
    draw_panel_title(ax, cx, cy + 1.55, "Fusion")
    # two input gaussians
    draw_gaussian(ax, cx - 0.55, cy + 0.85, width=0.9, height=0.36, color=C_AUD)
    draw_gaussian(ax, cx + 0.55, cy + 0.85, width=0.9, height=0.36, color=C_VIS)
    # weighted merge
    merge_pt = (cx, cy + 0.30)
    draw_merge(ax,
               sources=[(cx - 0.55, cy + 0.78), (cx + 0.55, cy + 0.78)],
               target=merge_pt,
               weight_labels=[r"$w_a$", r"$w_v$"])
    # downstream estimate gaussian
    draw_arrow(ax, merge_pt, (cx, cy - 0.10))
    draw_gaussian(ax, cx, cy - 0.55, width=1.1, height=0.45,
                  color=C_FUSED)
    ax.text(cx - 0.95, cy - 0.45, r"$\hat s_{\,fused}$",
            ha="right", va="center", color=C_FUSED, **FONT_LABEL)


def panel_causal_inference(ax: plt.Axes, *, cx: float, cy: float) -> None:
    """Centerpiece: C=1 vs C=2, then averaging / selection / prob-matching."""
    draw_panel_title(ax, cx, cy + 2.55, "Causal inference")

    # ---- top: two scenarios
    # P(C=1) — overlapping distributions
    x1 = cx - 1.20
    draw_gaussian(ax, x1 - 0.18, cy + 1.55, width=0.85, height=0.36, color=C_AUD)
    draw_gaussian(ax, x1 + 0.18, cy + 1.55, width=0.85, height=0.36, color=C_VIS)
    ax.text(x1, cy + 1.10, r"$P(C\!=\!1\mid m_a,m_v)$",
            ha="center", va="top", **FONT_SMALL)
    draw_pill(ax, (x1, cy + 2.10), "common cause")

    # P(C=2) — separated distributions
    x2 = cx + 1.20
    draw_gaussian(ax, x2 - 0.50, cy + 1.55, width=0.75, height=0.36, color=C_AUD)
    draw_gaussian(ax, x2 + 0.50, cy + 1.55, width=0.75, height=0.36, color=C_VIS)
    ax.text(x2, cy + 1.10, r"$P(C\!=\!2\mid m_a,m_v)$",
            ha="center", va="top", **FONT_SMALL)
    draw_pill(ax, (x2, cy + 2.10), "separate causes")

    # ---- branch into 3 submodels
    branch_src = (cx, cy + 0.85)
    draw_arrow(ax, (x1, cy + 1.05), branch_src, curve=-0.15, lw=LW_THIN)
    draw_arrow(ax, (x2, cy + 1.05), branch_src, curve= 0.15, lw=LW_THIN)
    draw_node(ax, branch_src)

    sub_y = cy + 0.05
    sub_xs = [cx - 1.40, cx, cx + 1.40]
    for sx in sub_xs:
        draw_arrow(ax, branch_src, (sx, sub_y + 0.20), lw=LW_THIN)

    # A. Averaging — smooth blend
    sx = sub_xs[0]
    ax.text(sx, sub_y + 0.05, "Averaging", ha="center", va="top", **FONT_LABEL)
    # blend visual: smooth gradient bar with rounded caps
    bar_x = np.linspace(sx - 0.48, sx + 0.48, 140)
    for i, bx in enumerate(bar_x):
        t = i / (len(bar_x) - 1)
        col = _blend_hex(C_AUD, C_VIS, t)
        ax.plot([bx, bx], [sub_y - 0.30, sub_y - 0.16], color=col,
                lw=1.0, solid_capstyle="butt")
    # outline pill around the gradient
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (sx - 0.50, sub_y - 0.31), 1.00, 0.16,
        boxstyle="round,pad=0.0,rounding_size=0.07",
        facecolor="none", edgecolor="#cdc6bb", lw=LW_HAIR, zorder=4))
    ax.text(sx, sub_y - 0.40,
            r"$(1{-}p)\,\hat s_{C=1} + p\,\hat s_{C=2}$",
            ha="center", va="top", **FONT_SMALL)

    # B. Selection — hard gate
    sx = sub_xs[1]
    ax.text(sx, sub_y + 0.05, "Selection", ha="center", va="top", **FONT_LABEL)
    draw_threshold_gate(ax, (sx, sub_y - 0.25), label=None)
    ax.text(sx, sub_y - 0.50, "hard threshold",
            ha="center", va="top", **FONT_SMALL)

    # C. Probability matching — stochastic switch
    sx = sub_xs[2]
    ax.text(sx, sub_y + 0.05, "Probability matching",
            ha="center", va="top", **FONT_LABEL)
    draw_node(ax, (sx, sub_y - 0.20), r=0.04)
    draw_probabilistic_arrow(ax, (sx, sub_y - 0.20),
                             (sx - 0.30, sub_y - 0.50), curve=-0.25)
    draw_probabilistic_arrow(ax, (sx, sub_y - 0.20),
                             (sx + 0.30, sub_y - 0.50), curve= 0.25)
    ax.text(sx, sub_y - 0.62, "stochastic switch",
            ha="center", va="top", **FONT_SMALL)

    # ---- terminal estimates per submodel
    for sx in sub_xs:
        draw_arrow(ax, (sx, sub_y - 0.78), (sx, sub_y - 1.05), lw=LW_THIN)
        draw_gaussian(ax, sx, sub_y - 1.40, width=0.85, height=0.34,
                      color=C_FUSED)


def panel_heuristic(ax: plt.Axes, *, cx: float, cy: float) -> None:
    """Heuristic cue switching — stochastic pick of one cue per trial."""
    draw_panel_title(ax, cx, cy + 1.55, "Heuristic cue switching")
    draw_gaussian(ax, cx - 0.55, cy + 0.85, width=0.9, height=0.36, color=C_AUD)
    draw_gaussian(ax, cx + 0.55, cy + 0.85, width=0.9, height=0.36, color=C_VIS)

    sw = (cx, cy + 0.30)
    draw_node(ax, sw, r=0.05)
    draw_probabilistic_arrow(ax, (cx - 0.55, cy + 0.78), sw)
    draw_probabilistic_arrow(ax, (cx + 0.55, cy + 0.78), sw)
    ax.text(cx + 0.10, cy + 0.36, "p / 1−p",
            ha="left", va="bottom", color="#555", **FONT_SMALL)

    draw_arrow(ax, sw, (cx, cy - 0.10))
    draw_gaussian(ax, cx, cy - 0.55, width=1.1, height=0.45,
                  color=C_FUSED, label="one cue per trial")


def panel_estimated_duration(ax: plt.Axes, *, cx: float, cy: float,
                             width: float = 4.0) -> None:
    draw_gaussian(ax, cx, cy, width=1.6, height=0.55, color=C_FUSED)
    ax.text(cx, cy - 0.20, "Estimated duration",
            ha="center", va="top", **FONT_TITLE)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _blend_hex(c1: str, c2: str, t: float) -> Tuple[float, float, float]:
    from matplotlib.colors import to_rgb
    a = np.array(to_rgb(c1)); b = np.array(to_rgb(c2))
    return tuple((1 - t) * a + t * b)


def _save(fig: plt.Figure, outdir: str, stem: str, *, png: bool = True) -> None:
    os.makedirs(outdir, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                    transparent=True, bbox_inches="tight", pad_inches=0.05)
    if png:
        fig.savefig(os.path.join(outdir, f"{stem}.png"),
                    transparent=True, bbox_inches="tight", pad_inches=0.05,
                    dpi=300)


# -----------------------------------------------------------------------------
# Top-level figures
# -----------------------------------------------------------------------------

def make_master_figure(outdir: str = "figures_poster") -> plt.Figure:
    """Single panoramic figure: encoding → 3 model families → estimated duration."""
    fig = plt.figure(figsize=(14.0, 9.0))
    ax = fig.add_axes([0, 0, 1, 1])
    _clean_axis(ax)
    ax.set_xlim(0, 14.0); ax.set_ylim(0, 9.0)

    # --- soft background panel under the centerpiece (CI)
    draw_soft_panel(ax, (4.55, 1.95), 4.90, 4.95)

    # --- top: shared encoding (centered)
    panel_shared_encoding(ax, x0=5.00, y0=8.00, width=4.0)
    # arrow into model families
    draw_arrow(ax, (7.00, 7.20), (7.00, 6.65), lw=LW_NORMAL)

    # faint section rule between encoding and model families
    draw_section_rule(ax, 1.0, 13.0, 7.45)

    # --- middle row: three families side by side; CI in the center, larger
    panel_fusion(ax,            cx=2.30, cy=4.50)
    panel_causal_inference(ax,  cx=7.00, cy=3.90)
    panel_heuristic(ax,         cx=11.70, cy=4.50)

    # arrows from each family down toward the final estimate
    final_y = 0.90
    final_x = 7.00
    # fusion → final (gentle bezier)
    draw_arrow(ax, (2.30, 3.65), (final_x - 0.30, final_y + 0.40),
               color=C_GRAY, lw=LW_THIN, curve=0.18)
    # CI submodels → final (three converging arrows)
    ci_terminal_y = 3.90 + 0.05 - 1.40 - 0.34
    for sx, curv in ((7.00 - 1.40, 0.05), (7.00, 0.0), (7.00 + 1.40, -0.05)):
        draw_arrow(ax, (sx, ci_terminal_y),
                   (final_x, final_y + 0.40),
                   color=C_GRAY, lw=LW_THIN, curve=curv)
    # heuristic → final
    draw_arrow(ax, (11.70, 3.65), (final_x + 0.30, final_y + 0.40),
               color=C_GRAY, lw=LW_THIN, curve=-0.18)

    # --- bottom: estimated duration
    panel_estimated_duration(ax, cx=final_x, cy=final_y)

    _save(fig, outdir, "poster_models_master")
    return fig


def make_panel_fusion(outdir: str = "figures_poster") -> plt.Figure:
    fig = plt.figure(figsize=(4.2, 5.2))
    ax = fig.add_axes([0, 0, 1, 1]); _clean_axis(ax)
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 5.2)
    panel_shared_encoding(ax, x0=0.60, y0=4.30, width=3.0)
    draw_section_rule(ax, 0.4, 3.8, 3.45)
    draw_arrow(ax, (2.10, 3.40), (2.10, 3.20), lw=LW_NORMAL)
    panel_fusion(ax, cx=2.10, cy=1.50)
    _save(fig, outdir, "poster_models_fusion")
    return fig


def make_panel_causal(outdir: str = "figures_poster") -> plt.Figure:
    fig = plt.figure(figsize=(7.0, 7.4))
    ax = fig.add_axes([0, 0, 1, 1]); _clean_axis(ax)
    ax.set_xlim(0, 7.0); ax.set_ylim(0, 7.4)
    draw_soft_panel(ax, (0.55, 0.30), 5.90, 5.45)
    panel_shared_encoding(ax, x0=1.50, y0=6.45, width=4.0)
    draw_section_rule(ax, 0.4, 6.6, 5.85)
    draw_arrow(ax, (3.50, 5.65), (3.50, 5.20), lw=LW_NORMAL)
    panel_causal_inference(ax, cx=3.50, cy=2.55)
    _save(fig, outdir, "poster_models_causal_inference")
    return fig


def make_panel_heuristic(outdir: str = "figures_poster") -> plt.Figure:
    fig = plt.figure(figsize=(4.2, 5.2))
    ax = fig.add_axes([0, 0, 1, 1]); _clean_axis(ax)
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 5.2)
    panel_shared_encoding(ax, x0=0.60, y0=4.30, width=3.0)
    draw_section_rule(ax, 0.4, 3.8, 3.45)
    draw_arrow(ax, (2.10, 3.40), (2.10, 3.20), lw=LW_NORMAL)
    panel_heuristic(ax, cx=2.10, cy=1.50)
    _save(fig, outdir, "poster_models_heuristic")
    return fig


def make_all(outdir: str = "figures_poster") -> None:
    make_master_figure(outdir)
    make_panel_fusion(outdir)
    make_panel_causal(outdir)
    make_panel_heuristic(outdir)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "figures_poster")
    make_all(out)
    print(f"Saved poster diagrams to: {out}")
