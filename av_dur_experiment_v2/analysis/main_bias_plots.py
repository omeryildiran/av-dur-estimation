"""One participant's main experiment, all parts pooled: PSE shift and sigma against conflict, and the
psychometric fit of every condition. Lognormal fit, lapse <= 0.1 shared per noise level.

    python analysis/main_bias_plots.py MZ            # figures to analysis/pilot_main/
    python analysis/main_bias_plots.py MZ --refit    # refit (otherwise the saved fits are reused)

Left out: warm-up trials (load_trials), trials with a dropped frame or a late first frame (more than half
a frame, so sound and flash were misaligned), and answers faster than 100 ms after the prompt.
Prediction from the unimodal blocks (reliability weighting): w_V = sA^2 / (sA^2 + sV^2), and the PSE of
the standard moves by w_V of the visual stretch in log duration: PSE = 500 ms x ((500 + conflict) / 500)^w_V.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment_settings as S  # noqa: E402
from avdur import psychometrics as pm, storage  # noqa: E402

PID = sys.argv[1] if len(sys.argv) > 1 else "MZ"
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "pilot_main"
N_BOOT = 200
INK, MUTED, GREY = "#0b0b0b", "#52514e", "#9a9993"
COL = {S.NOISE_LOW: "#2a78d6", S.NOISE_HIGH: "#eb6834"}          # categorical slots 1-2 (validated pair)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})


def clean(d):
    d = d[d.responses.isin([1, 2])]
    glitch = (d.n_dropped_frames.fillna(0) > 0) | (d.flip0_error_s.abs() > 0.5 / S.EXPECTED_FRAME_RATE_HZ)
    fast = d.response_rts < 0.1
    return d[~glitch & ~fast], int(glitch.sum()), int(fast.sum())


def fit_one(d):
    p = pm.fit(d.assign(c="x"), "c", "lognormal", lapse_max=0.1)["params"].iloc[0]
    return float(p.mu), float(p.sigma), float(p.lapse)


def boot(d, rng):
    """Percentile CI of PSE (ms) and sigma: resample trials within the condition."""
    pse, sig = [], []
    for _ in range(N_BOOT):
        mu, s, _ = fit_one(d.iloc[rng.integers(len(d), size=len(d))])
        pse.append(500 * np.exp(mu)); sig.append(s)
    return np.percentile(pse, [2.5, 97.5]), np.percentile(sig, [2.5, 97.5])


def unimodal_sigmas(pid):
    out = {}
    for task in ("unimodal_auditory", "unimodal_visual"):
        d = storage.load_trials(task, pid, DATA)
        if d.empty:
            continue
        d, _, _ = clean(d)
        d = d.assign(cond=d.audNoise.fillna(-1))
        for _, p in pm.fit(d, "cond", "lognormal", lapse_max=0.1)["params"].iterrows():
            out["visual" if p.condition == -1 else float(p.condition)] = float(p.sigma)
    return out


def main():
    rng = np.random.default_rng(1)
    m = storage.load_trials("main", PID, DATA)
    n_all = len(m)
    m, n_glitch, n_fast = clean(m)
    m = m[m.kind.isin(["staircase", "lapse"])]
    m["conflict_ms"] = (1000 * m.conflict_nominal).round().astype(int)
    rows = []
    saved = OUT / f"{PID}_main_fits.csv"
    if saved.exists() and "--refit" not in sys.argv:          # the bootstrap takes a few minutes
        rows = pd.read_csv(saved).to_dict("records")
    for (noise, c), d in ([] if rows else m.groupby(["audNoise", "conflict_ms"])):
        mu, s, lam = fit_one(d)
        (plo, phi), (slo, shi) = boot(d, rng)
        rows.append(dict(noise=noise, conflict_ms=c, pse_ms=500 * np.exp(mu), pse_lo=plo, pse_hi=phi,
                         sigma=s, sigma_lo=slo, sigma_hi=shi, lapse=lam, n=len(d)))
    fits = pd.DataFrame(rows)
    uni = unimodal_sigmas(PID)
    OUT.mkdir(parents=True, exist_ok=True)
    fits.round(4).to_csv(OUT / f"{PID}_main_fits.csv", index=False)

    # ── figure 1: PSE shift and sigma against conflict ──────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    cs = np.linspace(-450, 450, 200)
    a1.plot(cs, cs, color=GREY, lw=1, ls="--")
    a1.text(430, 430, "vision only", color=MUTED, ha="right", va="bottom", fontsize=9)
    a1.axhline(0, color=GREY, lw=1, ls="--")
    a1.text(430, 8, "sound only", color=MUTED, ha="right", va="bottom", fontsize=9)
    for noise, g in fits.groupby("noise"):
        col, off = COL.get(noise, INK), (-6 if noise == S.NOISE_LOW else 6)
        a1.errorbar(g.conflict_ms + off, g.pse_ms - 500, yerr=[g.pse_ms - g.pse_lo, g.pse_hi - g.pse_ms], fmt="o",
                    ms=6, color=col, lw=1.2, capsize=0, label=f"noise {noise:g}")
        slope = np.polyfit(g.conflict_ms, g.pse_ms - 500, 1)[0]
        a1.plot(g.conflict_ms, np.polyval(np.polyfit(g.conflict_ms, g.pse_ms - 500, 1), g.conflict_ms), color=col,
                lw=1, alpha=0.6)
        a1.text(-440, (230 if noise == S.NOISE_LOW else 300), f"noise {noise:g}: slope {slope:.2f}", color=col, fontsize=9)
        if noise in uni and "visual" in uni:
            w = uni[noise] ** 2 / (uni[noise] ** 2 + uni["visual"] ** 2)
            a1.plot(cs, 500 * ((500 + cs) / 500) ** w - 500, color=col, lw=1.5, ls=":",
                    label=f"predicted from unimodal σ (w_V {w:.2f})")
        a2.errorbar(g.conflict_ms + off, g.sigma, yerr=[g.sigma - g.sigma_lo, g.sigma_hi - g.sigma], fmt="o", ms=6,
                    color=col, lw=1.2, capsize=0, label=f"noise {noise:g}")
        if noise in uni:
            a2.axhline(uni[noise], color=col, lw=1, ls=":")
    if "visual" in uni:
        a2.axhline(uni["visual"], color=INK, lw=1, ls=":")
        a2.text(-440, uni["visual"], f"visual alone {uni['visual']:.2f}", color=MUTED, va="bottom", fontsize=9)
    for noise in (S.NOISE_LOW, S.NOISE_HIGH):
        if noise in uni:
            a2.text(440, uni[noise], f"auditory alone, noise {noise:g}: {uni[noise]:.2f}", color=COL[noise],
                    ha="right", va="bottom", fontsize=9)
    a1.set(xlabel="conflict: visual standard minus auditory standard (ms)", ylabel="PSE minus 500 ms (ms)",
           title=f"{PID}: PSE shift (95% bootstrap CI)", xticks=sorted(fits.conflict_ms.unique()))
    a2.set(xlabel="conflict (ms)", ylabel="σ (log duration)", title="σ of the audiovisual judgement",
           xticks=sorted(fits.conflict_ms.unique()), ylim=(0, max(1.0, fits.sigma_hi.max() * 1.05)))
    a1.tick_params(axis="x", labelrotation=45); a2.tick_params(axis="x", labelrotation=45)
    a1.legend(loc="lower right", fontsize=8.5)
    a2.legend(loc="upper right", fontsize=8.5)
    fig.text(0.01, 0.005, f"{len(m)} trials (3 parts pooled; warm-up, {n_glitch} timing-glitch and {n_fast} "
             f"answers < 100 ms left out, of {n_all}). Lognormal fit, lapse ≤ 0.1.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / f"{PID}_main_pse_sigma.png", dpi=160)

    # ── figure 2: psychometric fits, one panel per noise level, one curve per conflict ──
    # diverging colour by conflict: blue = flash shorter than the sound, red = longer, grey = none
    neg = ["#184f95", "#2a78d6", "#5598e7", "#86b6ef"]                # -450, -250, -167, -83
    pos = ["#f2a5a4", "#e8716f", "#e34948", "#a92a29"]                # +83, +167, +250, +450
    conflicts = sorted(fits.conflict_ms.unique())
    ccol = dict(zip(conflicts, neg + ["#77766f"] + pos))
    noises = sorted(fits.noise.unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(6.2 * len(noises), 4.9), sharey=True)
    ms = np.geomspace(40, 2200, 400)
    for ax, noise in zip(np.atleast_1d(axes), noises):
        for c in conflicts:
            d = m[(m.audNoise == noise) & (m.conflict_ms == c) & (m.kind == "staircase")]
            f = fits[(fits.noise == noise) & (fits.conflict_ms == c)].iloc[0]
            col = ccol[c]
            xs = 1000 * d.testDurS
            bins = np.unique(np.quantile(xs, np.linspace(0, 1, 7)))
            bins[0] -= 1e-6; bins[-1] += 1e-6
            pts = pd.DataFrame({"x": xs, "y": d.chose_test}).groupby(pd.cut(xs, bins), observed=True).agg(
                x=("x", lambda v: float(np.exp(np.log(v).mean()))), y=("y", "mean"), n=("y", "size"))
            ax.scatter(pts.x, pts.y, s=6 + pts.n, color=col, alpha=0.45, lw=0, zorder=2)
            ax.plot(ms, pm.predict(np.log(ms / 500), np.log(f.pse_ms / 500), f.sigma, f.lapse), color=col,
                    lw=2.4 if c == 0 else 1.8, zorder=3, label=f"{c:+d} ms  (PSE {f.pse_ms:.0f})" if c else
                    f"0 ms  (PSE {f.pse_ms:.0f})")
        ax.axvline(500, color=GREY, lw=0.9, ls=":")
        ax.axhline(0.5, color="#d6d5cf", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 200, 500, 1000, 2000]); ax.set_xticklabels(["50", "100", "200", "500", "1000", "2000"])
        ax.minorticks_off()
        ax.set_xlim(40, 2200)
        a = f"auditory alone σ {uni[noise]:.2f}" if noise in uni else ""
        ax.set_title(f"noise {noise:g}" + (f"   ({a})" if a else ""), loc="left", fontsize=10.5)
        ax.set_xlabel("test duration (ms; standard 500)")
        ax.legend(title="conflict (visual − auditory standard)", loc="upper left", fontsize=8.3, title_fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel("P(test judged longer)")
    fig.suptitle(f"{PID}: main experiment, all parts — lines: lognormal fits; dots: staircase trials, binned; "
                 "dotted: standard", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"{PID}_main_psychometrics.png", dpi=160)
    print(fits.round(3).to_string(index=False))
    print("unimodal sigma:", {k: round(v, 3) for k, v in uni.items()})
    print(f"left out: {n_glitch} timing glitches, {n_fast} answers < 100 ms (of {n_all})")


if __name__ == "__main__":
    main()
