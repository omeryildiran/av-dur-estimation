"""Figures for the PI summary (6 Nov 2026) from the three simulations in this folder."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
from avdur.staircase import Staircase
import experiment_settings as S

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # categorical, fixed order
GREY, INK, MUTED = "#8a8a85", "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
rmse = lambda e: np.sqrt(np.mean(np.square(e)))

def dots(ax, table, designs, xlabels, highlight):
    """table[design] = values per x position; dot plot with slight offsets."""
    n = len(designs); off = np.linspace(-0.22, 0.22, n)
    for i, d in enumerate(designs):
        hl = d == highlight
        ax.plot(np.arange(len(xlabels)) + off[i], table[d], "o", ms=8 if hl else 6.5, color=C[i],
                mec="white", mew=1.5, label=d + ("  (chosen)" if hl else ""), zorder=3 if hl else 2)
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels)
    ax.set_ylim(bottom=0)

# ── 1. unimodal ───────────────────────────────────────────────────────────
u = pd.read_csv(HERE / "unimodal_sim.csv")
order = ["4x35 (now)", "2D1U+1D3U x70", "3D1U+1D2U x70", "2D1U+1D2U x70"]
names = {"4x35 (now)": "4 staircases x 35 (before)", "2D1U+1D3U x70": "2D1U + 1D3U x 70",
         "3D1U+1D2U x70": "3D1U + 1D2U x 70", "2D1U+1D2U x70": "2D1U + 1D2U x 70"}
cells = [("auditory", 0.3), ("auditory", 0.6), ("auditory", 1.2), ("visual", 0.6), ("visual", 1.0)]
xl = [f"{t[:3]}. σ {s}" for t, s in cells]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, col, scale, lab in ((axs[0], "err_ln_sigma", 100, "σ error (RMS, % of σ)"), (axs[1], "err_pse_ms", 1, "PSE error (RMS, ms)")):
    tab = {names[d]: [scale * rmse(u[(u.task == t) & (u.sigma == s) & (u.design == d)][col]) for t, s in cells] for d in order}
    dots(ax, tab, [names[d] for d in order], xl, "2D1U + 1D3U x 70"); ax.set_ylabel(lab)
axs[0].legend(loc="lower right", fontsize=8.5)
fig.suptitle("Unimodal blocks: 140 staircase + 20 easy trials per condition (300 simulated listeners per cell)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "fig1_unimodal_staircases.png", dpi=200); plt.close(fig)

# ── 2. main: shared vs per-condition, 1D2U vs 1D3U ────────────────────────
m = pd.read_csv(HERE / "main_shared_sim.csv")
m["grp"] = np.where(m.conflict.abs() > 0.3, "±450 ms", "0 to ±250 ms")
order = ["shared, 2D1U+1D2U", "per condition, 2D1U+1D2U", "per condition, 2D1U+1D3U", "shared, 2D1U+1D3U"]
names = {"shared, 2D1U+1D2U": "v1: 2 staircases shared by all conditions", "per condition, 2D1U+1D2U": "2D1U + 1D2U per condition",
         "per condition, 2D1U+1D3U": "2D1U + 1D3U per condition", "shared, 2D1U+1D3U": "2D1U + 1D3U shared"}
cells = [(0.3, "0 to ±250 ms"), (0.3, "±450 ms"), (0.8, "0 to ±250 ms"), (0.8, "±450 ms")]
xl = [f"σ {s}\n{g}" for s, g in cells]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
for ax, col, scale, lab in ((axs[0], "e_sig", 100, "σ error (RMS, % of σ)"), (axs[1], "e_pse", 1, "PSE error (RMS, ms)")):
    tab = {names[d]: [scale * rmse(m[(m.sigma == s) & (m.grp == g) & (m.design == d)][col]) for s, g in cells] for d in order}
    dots(ax, tab, [names[d] for d in order], xl, "2D1U + 1D3U per condition"); ax.set_ylabel(lab)
axs[1].legend(loc="upper left", fontsize=8.5)
fig.suptitle("Main experiment: 9 conflicts x 102 staircase trials + 96 easy per noise level (100 listeners; PSE pulled 30% / 70% to vision)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "fig2_main_shared_vs_per_condition.png", dpi=200); plt.close(fig)

# ── 3. change between days ────────────────────────────────────────────────
d = pd.read_csv(HERE / "drift_sim.csv")
order = ["continue", "continue+step", "restart", "MCS narrow", "MCS wide"]
names = {"continue": "staircases continue", "continue+step": "continue, step back to 80 ms",
         "restart": "staircases restart each part", "MCS narrow": "constant stimuli 275-900 ms",
         "MCS wide": "constant stimuli 150-1650 ms"}
scen = ["no change", "PSE +80 ms on day 2", "sigma x1.5 on day 2", "sigma x0.67 on day 2"]
xl = ["no change", "PSE +80 ms", "σ x1.5", "σ x0.67"]
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.1), sharex=True)
for r, sig in enumerate((0.3, 0.8)):
    for c, (col, scale, lab) in enumerate((("e_sig", 100, "σ error (RMS, %)"), ("e_pse", 1, "PSE error (RMS, ms)"))):
        ax = axs[r, c]
        tab = {names[k]: [scale * rmse(d[(d.sigma == sig) & (d.scen == s) & (d.design == k)][col]) for s in scen] for k in order}
        dots(ax, tab, [names[k] for k in order], xl, names["continue+step"]); ax.set_ylabel(lab)
        ax.set_title(f"listener σ {sig}", loc="left", fontsize=10, color=MUTED)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.8)
fig.suptitle("Part 1 on day 1, parts 2-3 on day 2, listener changes on day 2 (main design, 100 listeners per cell)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(HERE / "fig3_day_change_and_constant_stimuli.png", dpi=200); plt.close(fig)

# ── 4. example staircases over the three parts ───────────────────────────
rng = np.random.default_rng(4)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (sig, pse1, pse2, title) in zip(axs, ((0.3, 650, 650, "σ 0.3, PSE 650 ms (e.g. +450 ms conflict), no change"),
                                             (0.8, 400, 520, "σ 0.8, PSE 400 ms on day 1, 520 ms on day 2"))):
    for k, (rule, start) in enumerate((("2D1U", "floor"), ("1D3U", "ceiling"))):
        st = Staircase(rule, S.TASKS["main"]["floor"], start=start, **S.STAIRCASE)
        lv = []
        for part in range(3):
            if part:
                st.reset_step()
            mu = np.log((pse1 if part == 0 else pse2) / 500)
            for _ in range(17):
                x = np.log(1 + st.level); lv.append(500 * (1 + st.level))
                st.update(rng.random() < 0.015 + 0.97 * norm.cdf((x - mu) / sig))
        ax.plot(np.arange(1, 52), lv, "-o", ms=3.5, lw=2, color=C[k], label=f"{rule} (target {'70.7' if k == 0 else '20.6'}% 'test longer')")
        z = norm.ppf(0.707 if k == 0 else 0.206)                 # where this rule converges (lapse ignored)
        ax.plot([1, 17.5], [pse1 * np.exp(z * sig)] * 2, color=C[k], lw=1, ls=":")
        ax.plot([17.5, 51], [pse2 * np.exp(z * sig)] * 2, color=C[k], lw=1, ls=":",
                label="its target level" if k == 0 else None)
    ax.plot([1, 17.5], [pse1] * 2, color=INK, lw=1, ls="--"); ax.plot([17.5, 51], [pse2] * 2, color=INK, lw=1, ls="--", label="listener's PSE")
    for b in (17.5, 34.5):
        ax.axvline(b, color=GREY, lw=0.8)
    ax.set_yscale("log"); ax.set_yticks([25, 50, 100, 200, 500, 1000, 2000]); ax.set_yticklabels([25, 50, 100, 200, 500, 1000, 2000])
    ax.minorticks_off(); ax.set_xlabel("trial of this staircase (parts 1 | 2 | 3)"); ax.set_title(title, loc="left", fontsize=9.5, color=MUTED)
axs[0].set_ylabel("test duration (ms, log)"); axs[0].legend(loc="lower right", fontsize=8.5)
fig.suptitle("One condition's two staircases over the three parts (17 trials each per part)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "fig4_example_staircases.png", dpi=200); plt.close(fig)
print("done")
