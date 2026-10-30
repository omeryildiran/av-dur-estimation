"""Presentation order: test first vs test second, in every task (v2 P101-P105 and v1 bimodal).

    python analysis/order_effect_plots.py        # figures in analysis/order_effect_figures/

1. order_sigma.pdf: psychometric sigma (lognormal fit, lapse <= 0.1) by order, per participant, for the v2
   unimodal auditory (low, high noise), unimodal visual and cross-modal 500 ms blocks.
2. order_calibration.pdf: v2 noise calibration, % correct by noise level and order (fixed test durations).
3. order_v1_bimodal.pdf: v1 main experiment, PSE shift against conflict by order (group mean +- SE) and
   each participant's PSE-vs-conflict slope by order. If the standard (which carries the conflict) is
   replaced by a remembered reference when it comes first, the slope is smaller on test-second trials.
"""

import glob
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
V2, V1 = HERE.parent / "data_solid", HERE.parents[1] / "av_dur_data"
OUT = HERE / "order_effect_figures"
COL = {1: "tab:red", 2: "tab:blue"}
LAB = {1: "test first", 2: "test second"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def fit_pf(t, s, y):
    """Lognormal PF on ln(test / standard): (mu, sigma)."""
    x, y = np.log(np.asarray(t, float) / np.asarray(s, float)), np.asarray(y, float)

    def nll(p):
        pr = np.clip(p[2] / 2 + (1 - p[2]) * norm.cdf((x - p[0]) / np.exp(p[1])), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    r = min((minimize(nll, x0, method="L-BFGS-B", bounds=[(-1.5, 1.5), (np.log(.03), np.log(5)), (0, .1)])
             for x0 in ([0, -1, .02], [0, 0, .05], [-.3, -.5, .02])), key=lambda r: r.fun)
    return float(r.x[0]), float(np.exp(r.x[1]))


def v2(task):
    fs = [f for f in sorted(glob.glob(str(V2 / "P10[1-5]" / task / "*.csv")))]
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f).assign(pid=Path(f).parts[-3]) for f in fs], ignore_index=True)
    return d[d.kind.isin(["staircase", "lapse"])]


def fig_sigma():
    panels = []
    a = v2("unimodal_auditory")
    a = a.assign(lev=np.where(a.audNoise > 0.11, "high", "low"))
    panels += [("unimodal auditory, low noise", a[a.lev == "low"]), ("unimodal auditory, high noise", a[a.lev == "high"])]
    panels.append(("unimodal visual", v2("unimodal_visual")))
    x = v2("crossmodal")
    x = x[np.isclose(x.standardDur, 0.5)]
    x = x.assign(lev=np.where(x.audNoise > 0.11, "high", "low"))
    panels += [("cross-modal 500 ms, low noise", x[x.lev == "low"]), ("cross-modal 500 ms, high noise", x[x.lev == "high"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 3), sharey=True)
    for ax, (title, d) in zip(axes, panels):
        pids = sorted(d.pid.unique())
        for i, pid in enumerate(pids):
            s = {}
            for o in (1, 2):
                g = d[(d.pid == pid) & (d.order == o)]
                s[o] = fit_pf(g.testDurS, g.standardDur, g.chose_test)[1] if len(g) > 20 else np.nan
            ax.plot([i - 0.1, i + 0.1], [s[1], s[2]], color="0.7", lw=0.8)
            for k, o in enumerate((1, 2)):
                ax.scatter(i - 0.1 + 0.2 * k, s[o], color=COL[o], s=22, zorder=3, label=LAB[o] if i == 0 else None)
        ax.set_xticks(range(len(pids))); ax.set_xticklabels(pids, rotation=45)
        ax.set(title=title, yscale="log")
        ax.title.set_fontsize(8.5)
    axes[0].set_ylabel("psychometric sigma (log scale)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "order_sigma.pdf"); fig.savefig(OUT / "order_sigma.png", dpi=130)
    plt.close(fig)


def fig_calibration():
    d = v2("noise_calibration")
    d = d[d.kind == "staircase"]
    bins = np.exp(np.linspace(np.log(0.1), np.log(2.0), 9))
    pids = sorted(d.pid.unique())
    fig, axes = plt.subplots(1, len(pids), figsize=(12, 2.8), sharey=True)
    for ax, pid in zip(axes, pids):
        g = d[d.pid == pid]
        for o in (1, 2):
            h = g[g.order == o]
            b = pd.cut(h.audNoise, bins)
            m = h.groupby(b, observed=True).is_correct.agg(["mean", "size"])
            m = m[m["size"] >= 5]
            mid = [np.sqrt(i.left * i.right) for i in m.index]
            ax.plot(mid, 100 * m["mean"], "o-", color=COL[o], ms=3, label=LAB[o])
        ax.axhline(70.7, color="0.8", lw=0.8)
        ax.set(title=pid, xscale="log", xlabel="noise level", ylim=(20, 102))
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.2, 0.5, 1, 2]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("% correct (bins with >= 5 trials)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "order_calibration.pdf"); fig.savefig(OUT / "order_calibration.png", dpi=130)
    plt.close(fig)


def fig_v1():
    d = pd.read_csv(V1 / "all_bimodal.csv")
    d["chose_test"] = (d.responses == d.order).astype(float)
    d["conf"] = np.round(d.conflictDur * 12) / 12
    rows = []
    for (pid, noise, conf, o), g in d.groupby(["participantID", "audNoise", "conf", "order"]):
        if len(g) >= 15:
            rows.append(dict(pid=pid, noise=noise, conf=conf, order=o, pse=1000 * 0.5 * (np.exp(fit_pf(g.testDurS, g.standardDur, g.chose_test)[0]) - 1)))
    p = pd.DataFrame(rows)
    p["shift"] = p.pse - p.groupby(["pid", "noise", "order"]).pse.transform(lambda s: s[p.loc[s.index, "conf"].abs() < 1e-6].mean())
    slopes = p.groupby(["pid", "noise", "order"]).apply(lambda g: np.polyfit(1000 * g.conf, g.pse, 1)[0] if len(g) >= 4 else np.nan).rename("slope").reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, noise in zip(axes[:2], (0.1, 1.2)):
        for o in (1, 2):
            g = p[(p.noise == noise) & (p.order == o)].groupby("conf")["shift"].agg(["mean", "sem", "size"])
            ax.errorbar(1000 * g.index + (o - 1.5) * 6, g["mean"], g["sem"], color=COL[o], marker="o", ms=3, capsize=2, label=LAB[o])
        ax.plot([-250, 250], [-250, 250], color="0.85", lw=0.8, ls="--")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set(title=f"v1 bimodal, noise {noise}", xlabel="conflict (visual - auditory, ms)", ylabel="PSE shift (ms)", ylim=(-150, 150))
    axes[0].legend(frameon=False, fontsize=7)
    ax = axes[2]
    for k, noise in enumerate((0.1, 1.2)):
        s = slopes[slopes.noise == noise].pivot(index="pid", columns="order", values="slope").dropna()
        for _, r in s.iterrows():
            ax.plot([2 * k, 2 * k + 1], [r[1], r[2]], color="0.6", lw=0.8, marker="o", ms=3)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["low:\nfirst", "low:\nsecond", "high:\nfirst", "high:\nsecond"])
    ax.set(title="slope of PSE vs conflict, per participant", ylabel="ms per ms")
    fig.tight_layout()
    fig.savefig(OUT / "order_v1_bimodal.pdf"); fig.savefig(OUT / "order_v1_bimodal.png", dpi=130)
    plt.close(fig)
    t = slopes.pivot_table(index=["pid", "noise"], columns="order", values="slope").dropna()
    for noise, g in t.groupby(level=1):
        dlt = g[1] - g[2]
        print(f"v1 noise {noise}: slope test-first {g[1].mean():.2f}, test-second {g[2].mean():.2f}; "
              f"first - second {dlt.mean():+.2f} (SE {dlt.std(ddof=1) / np.sqrt(len(dlt)):.2f}, n {len(dlt)}, larger with test first: {(dlt > 0).sum()})")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_sigma()
    fig_calibration()
    fig_v1()
