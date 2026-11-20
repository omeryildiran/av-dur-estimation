"""Figures, numbers and tables for order_model_explainer.tex (run from anywhere):

    python docs/order_model_explainer/make_figures.py

Reads the saved order-model fits (results/model_fits_order/) and the order-split psychometric fits; writes
figures/*.pdf, numbers.tex (macros) and tables.tex next to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from avdur_analysis import data as D, model_figures as MF, models as M, order_models as O
from avdur_analysis import psychometric as P, style as St

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
St.setup(scale=0.75)
NUM = {}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# ── 1. schematic: a trial and the contraction ────────────────────────────────

def fig_schematic():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for y, first, second in ((1.9, "Test", "Standard"), (0.0, "Standard", "Test")):
        for x0, lab in ((0.0, first), (1.6, second)):
            col = "#cfe3f3" if lab == "Test" else "#f6d5d5"
            a.add_patch(Rectangle((x0, y), 1.2, 0.7, fc=col, ec="0.2"))
            a.text(x0 + 0.6, y + 0.35, lab, ha="center", va="center", fontsize=12)
        a.text(-0.15, y + 0.35, "test first" if first == "Test" else "test second", ha="right", va="center",
               fontsize=12, style="italic")
        a.add_patch(FancyArrowPatch((0.6, y + 0.72), (3.35, y + 0.55), connectionstyle="arc3,rad=-0.35",
                                    arrowstyle="-|>", mutation_scale=12, color="0.35"))
        a.text(1.9, y + 1.02, "held in memory", ha="center", fontsize=9, color="0.35")
        a.text(3.45, y + 0.35, "which was\nlonger?", va="center", fontsize=11)
    a.text(0.6, -0.35, "interval 1", ha="center", fontsize=10, color="0.4")
    a.text(2.2, -0.35, "interval 2", ha="center", fontsize=10, color="0.4")
    a.set_xlim(-1.6, 4.5)
    a.set_ylim(-0.55, 3.05)
    a.axis("off")
    a.set_title("A. One trial, two orders", loc="left")

    g, ref, e1, e2 = 0.5, 400.0, 700.0, 500.0
    e1c = np.exp(g * np.log(e1) + (1 - g) * np.log(ref))
    b.set_xscale("log")
    b.set_xlim(250, 1000)
    b.set_ylim(-1, 1.6)
    b.axhline(0, color="0.3", lw=1)
    b.set_xticks([300, 400, 500, 700, 1000])
    b.set_xticklabels(["300", "400", "500", "700", "1000"])
    b.minorticks_off()
    b.set_yticks([])
    for side in ("left", "right", "top"):
        b.spines[side].set_visible(False)
    b.plot([ref], [0], "D", color="0.3", ms=9)
    b.text(ref, -0.45, "memory\nreference", ha="center", va="top", fontsize=10)
    b.plot([e1], [0.9], "o", color="#1f77b4", ms=10)
    b.text(e1, 1.15, "interval 1\nas perceived", ha="center", fontsize=10)
    b.plot([e1c], [0], "o", color="#1f77b4", mfc="white", ms=10, mew=2)
    b.add_patch(FancyArrowPatch((e1, 0.8), (e1c * 1.01, 0.12), arrowstyle="-|>", mutation_scale=14, color="#1f77b4"))
    b.text(e1c, -0.45, "interval 1\nas compared", ha="center", va="top", fontsize=10, color="#1f77b4")
    b.plot([e2], [0.9], "s", color="0.5", ms=9)
    b.text(e2, 1.15, "interval 2\n(unchanged)", ha="center", fontsize=10, color="0.4")
    b.set_xlabel("Duration (ms, log scale)")
    b.set_title(f"B. Contraction with g = {g}: 700 ms becomes {e1c:.0f} ms", loc="left")
    fig.tight_layout()
    save(fig, "fig1_schematic")


# ── 2. what contraction predicts ─────────────────────────────────────────────

def _pse_sigma(log_t, p):
    """PSE (log) and psychometric sigma from a curve (25/50/75% points; Gaussian in log t)."""
    x25, x50, x75 = np.interp([0.25, 0.5, 0.75], p, log_t)
    return x50, (x75 - x25) / (2 * 0.6744897501960817)


def fig_predictions_toy():
    s, g, ref = 0.20, 0.5, np.log(0.4)
    log_t = np.linspace(np.log(0.15), np.log(1.6), 600)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    p0 = O.p_unimodal_order(log_t, np.ones_like(log_t), s, 1.0, M.LOG_STD)
    p1 = O.p_unimodal_order(log_t, np.ones_like(log_t), s, g, ref)
    p2 = O.p_unimodal_order(log_t, np.full_like(log_t, 2), s, g, ref)
    ms = 1000 * np.exp(log_t)
    a.plot(ms, p0, color="0.6", lw=2, label="No contraction (either order)")
    a.plot(ms, p1, color="#1f77b4", lw=2.2, label="Test first")
    a.plot(ms, p2, color="#1f77b4", lw=2.2, ls="--", label="Test second")
    St.duration_axis(a, 150, 1600)
    a.axhline(0.5, **St.GUIDE)
    a.axvline(500, **St.GUIDE)
    a.set_xlabel("Test duration (ms)")
    a.set_ylabel('P("test longer")')
    a.set_title(f"A. One modality: σ = {s}, g = {g}, ref = 400 ms", loc="left")
    a.legend(loc="upper left", fontsize=10)
    (m0, s0), (m1, s1), (m2, s2) = (_pse_sigma(log_t, p) for p in (p0, p1, p2))
    NUM.update(toyGamma=f"{g}", toySigmaRatio=f"{s2 / s1:.3f}", toySigmaFirst=f"{s1:.3f}", toySigmaSecond=f"{s2:.3f}",
               toySigmaNone=f"{s0:.3f}", toyPseFirst=f"{1000 * np.exp(m1):.0f}", toyPseSecond=f"{1000 * np.exp(m2):.0f}",
               toyPseFirstFormula=f"{1000 * np.exp((M.LOG_STD - (1 - g) * ref) / g):.0f}",
               toyPseSecondFormula=f"{1000 * np.exp(g * M.LOG_STD + (1 - g) * ref):.0f}",
               toySigmaFirstFormula=f"{s * np.sqrt(1 + g * g) / g:.3f}", toySigmaSecondFormula=f"{s * np.sqrt(1 + g * g):.3f}")

    sa, sv = 0.30, 0.25
    prm = dict(sigma_a_high=sa, sigma_v=sv, ref=M.LOG_STD)
    conflicts = np.array([-450, -250, -167, -83, 0, 83, 167, 250, 450])
    lt = np.linspace(np.log(0.06), np.log(1.9), 500)
    curves = {}
    for lab, gg, order in (("none", 1.0, 1), ("first", g, 1), ("second", g, 2)):
        shifts = []
        for c in conflicts:
            pp = O.p_main_order("fusion", dict(prm, g_first=gg), D.NOISE_HIGH, lt, np.full_like(lt, np.log(0.5 + c / 1000)),
                                np.full(len(lt), order))
            shifts.append(1000 * np.exp(_pse_sigma(lt, pp)[0]) - 500)
        curves[lab] = np.array(shifts)
    b.plot(conflicts, curves["none"], "o-", color="0.6", lw=2, label="No contraction")
    b.plot(conflicts, curves["first"], "o-", color="#2ca02c", lw=2, label="Test first")
    b.plot(conflicts, curves["second"], "o--", color="#2ca02c", mfc="white", lw=2, label="Test second")
    b.axhline(0, **St.GUIDE)
    b.axvline(0, **St.GUIDE)
    b.set_xlabel("Conflict: visual - auditory standard (ms)")
    b.set_ylabel("PSE shift (ms)")
    b.set_title(f"B. Forced fusion, g = {g}, ref = 500 ms", loc="left")
    b.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig2_predictions_toy")
    i = list(conflicts).index(250)
    NUM.update(toyShiftNone=f"{curves['none'][i]:.0f}", toyShiftFirst=f"{curves['first'][i]:.0f}",
               toyShiftSecond=f"{curves['second'][i]:.0f}")


# ── 3. the evidence in the data ──────────────────────────────────────────────

def fig_evidence(pids):
    uo, mo = P.collect("unimodal_order", pids), P.collect("main_order", pids)
    ok = lambda t: t[t.pse_ms.between(50, 2000) & t.at_bound.fillna("").astype(str).eq("")]
    wide = lambda t, col: t.pivot_table(index=["participant", "base_condition"], columns="order", values=col).dropna()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    styles = {"A_low": St.NOISE_COLORS[D.NOISE_LOW], "A_high": St.NOISE_COLORS[D.NOISE_HIGH], "V": St.VISUAL_COLOR}
    su, sm = wide(uo, "sigma"), wide(mo, "sigma")
    for (pid, cond), r in sm.iterrows():
        c = St.NOISE_COLORS[D.NOISE_LOW if "_low_" in cond else D.NOISE_HIGH]
        a.scatter(r.test_first, r.test_second, s=16, color=c, alpha=0.35, lw=0)
    for (pid, cond), r in su.iterrows():
        a.scatter(r.test_first, r.test_second, s=70, color=styles[cond], edgecolor="white", zorder=3)
        a.annotate(pid, (r.test_first, r.test_second), xytext=(4, 2), textcoords="offset points", fontsize=8)
    lim = (0.02, 6)
    a.plot(lim, lim, color="0.5", lw=1, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    for ax_set in (a.set_xticks, a.set_yticks):
        ax_set(ticks)
    a.set_xticklabels([f"{t:g}" for t in ticks])
    a.set_yticklabels([f"{t:g}" for t in ticks])
    a.minorticks_off()
    a.set_xlabel("σ with the test first")
    a.set_ylabel("σ with the test second")
    a.set_title("A. Precision: below the line = sharper with the test second", loc="left")
    for lab, c in (("Auditory, low noise", styles["A_low"]), ("Auditory, high noise", styles["A_high"]),
                   ("Visual", styles["V"])):
        a.scatter([], [], s=70, color=c, label=lab)
    a.scatter([], [], s=16, color="0.5", alpha=0.5, label="Bimodal conditions")
    a.legend(fontsize=9, loc="upper left")
    NUM.update(nUniBelow=str(int((su.test_second < su.test_first).sum())), nUni=str(len(su)),
               nBiBelow=str(int((sm.test_second < sm.test_first).sum())), nBi=str(len(sm)),
               medRatioUni=f"{np.median(su.test_second / su.test_first):.2f}",
               medRatioBi=f"{np.median(sm.test_second / sm.test_first):.2f}")

    pu = wide(ok(uo), "shift_ms")
    p0 = wide(ok(mo[mo.conflict_ms == 0]), "shift_ms")
    for (pid, cond), r in p0.iterrows():
        c = St.NOISE_COLORS[D.NOISE_LOW if "_low_" in cond else D.NOISE_HIGH]
        b.scatter(r.test_first, r.test_second, s=40, color=c, marker="s", alpha=0.6, lw=0)
    for (pid, cond), r in pu.iterrows():
        b.scatter(r.test_first, r.test_second, s=70, color=styles[cond], edgecolor="white", zorder=3)
        b.annotate(pid, (r.test_first, r.test_second), xytext=(4, 2), textcoords="offset points", fontsize=8)
    b.axhline(0, color="0.5", lw=1)
    b.axvline(0, color="0.5", lw=1)
    b.set_xlabel("PSE shift with the test first (ms)")
    b.set_ylabel("PSE shift with the test second (ms)")
    b.set_title("B. Bias: opposite signs (top-left, bottom-right)", loc="left")
    b.scatter([], [], s=40, color="0.5", marker="s", alpha=0.6, label="Bimodal, 0 conflict")
    b.legend(fontsize=9, loc="upper right")
    both = np.concatenate([pu.to_numpy(), p0.to_numpy()])
    NUM.update(nOpposite=str(int((np.sign(both[:, 0]) != np.sign(both[:, 1])).sum())), nPse=str(len(both)))
    fig.tight_layout()
    save(fig, "fig3_evidence")


# ── 4. results ───────────────────────────────────────────────────────────────

def fig_results(pids):
    fig = O.fig_fit_summary(save=False, layout="grid")
    save(fig, "fig4_fit_summary")
    fig = O.fig_predictions("KJ", save=False, width=10, grid=True)
    save(fig, "fig5_predictions_KJ")
    fig = O.fig_predictions("group", save=False, width=10, grid=True)
    save(fig, "fig6_predictions_group")


def tables(pids):
    t = O.table()
    t = t[t.participant.isin(pids)]
    lab = {m: St.MODEL_STYLE[m]["label"] for m in M.MODELS}
    lines = [r"\begin{tabular}{ll rrrr}", r"\toprule",
             r"Participant & Model & AIC without order & AIC with order & Gain & $g$ \quad ref (ms) \\", r"\midrule"]
    for pid in pids:
        for m in M.MODELS:
            r = t[(t.participant == pid) & (t.model == m)].iloc[0]
            lines.append(f"{pid if m == M.MODELS[0] else ''} & {lab[m]} & {r.aic_no_order:.1f} & {r.aic:.1f} & "
                         f"{r.d_aic_order:.1f} & {r.g_first:.2f} \\quad {r.ref_ms:.0f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    dl = MF.delta_aic(t)
    g = MF.group_delta(t).set_index("model")
    comp = [r"\begin{tabular}{l rr}", r"\toprule", r"Participant & Causal inference & Cue switching \\", r"\midrule"]
    for pid in pids:
        comp.append(f"{pid} & {dl.loc[pid, 'causal']:+.1f} & {dl.loc[pid, 'switching']:+.1f} \\\\")
    comp += [r"\midrule",
             f"Summed & {g.loc['causal', 'summed']:+.1f} [{g.loc['causal', 'lo']:.0f}, {g.loc['causal', 'hi']:.0f}] & "
             f"{g.loc['switching', 'summed']:+.1f} [{g.loc['switching', 'lo']:.0f}, {g.loc['switching', 'hi']:.0f}] \\\\",
             r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables.tex").write_text("% written by make_figures.py\n\\newcommand{\\TableFits}{%\n" + "\n".join(lines)
                                     + "}\n\\newcommand{\\TableComparison}{%\n" + "\n".join(comp) + "}\n")
    NUM.update(gainMin=f"{-t.d_aic_order.max():.0f}", gainMax=f"{-t.d_aic_order.min():.0f}",
               gMin=f"{t.g_first.min():.2f}", gMax=f"{t.g_first.max():.2f}",
               refMin=f"{t.ref_ms.min():.0f}", refMax=f"{t.ref_ms.max():.0f}",
               sumCausal=f"{g.loc['causal', 'summed']:.0f}", sumSwitching=f"{g.loc['switching', 'summed']:.0f}",
               nPids=str(len(pids)), pids=", ".join(pids))


def checks():
    out = O.check("KJ")
    NUM["checkMaxDiff"] = f"{max(abs(a - b) for a, b in out.values()):.4f}"


def main():
    pids = O.fitted_pids()
    fig_schematic()
    fig_predictions_toy()
    fig_evidence(pids)
    fig_results(pids)
    tables(pids)
    checks()
    tex = "% written by make_figures.py\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in NUM.items())
    (HERE / "numbers.tex").write_text(tex)
    print(NUM)


if __name__ == "__main__":
    main()
