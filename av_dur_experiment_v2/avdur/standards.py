"""Cross-modal task at several standards: the plan per participant and the bias by standard.

The 500 ms standard (STANDARD_DUR_S) keeps the full cross-modal design and alone sets the main
experiment's correction (bias.load_crossmodal). The other standards (CROSSMODAL_STANDARDS) use a
shorter design; each standard is its own session, labelled S300 / S500 / S800.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import experiment_settings as S
from avdur import design, psychometrics as pm, storage

CFG = S.CROSSMODAL_STANDARDS


def label(standard_s):
    return f"S{round(1000 * standard_s)}"


def plan(pid, root=None, several=None):
    """[(standard_s, design override or None, block label)] in running order for this participant."""
    several = CFG["enabled"] if several is None else several
    if not several:
        return [(S.STANDARD_DUR_S, None, None)]
    extra = list(CFG["standards"])
    if design.participant_number(pid, root) % 2 == 0:
        extra = extra[::-1]
    return [(S.STANDARD_DUR_S, None, label(S.STANDARD_DUR_S))] + [(s, dict(CFG["extra_design"]), label(s)) for s in extra]


def done_standards(pid, root=None):
    """Standards (s) of this participant's completed cross-modal sessions (older sessions: 500 ms)."""
    return {round(float(m.get("standard_s", S.STANDARD_DUR_S)), 4) for m in storage.sessions(pid, "crossmodal", root)}


def remaining(pid, root=None, several=None):
    done = done_standards(pid, root)
    return [p for p in plan(pid, root, several) if round(p[0], 4) not in done]


def bias_table(pid, root=None, n_boot=200, rng=None):
    """Bias (standard - auditory PSE) per standard and noise level, from every completed cross-modal session."""
    rng = rng or np.random.default_rng(0)
    d = storage.load_trials("crossmodal", pid, root)
    if d.empty:
        return pd.DataFrame()
    d = d[d["responses"].isin([1, 2])]
    rows = []
    for (std, noise), g in d.groupby([d["standardDur"].round(4), "audNoise"]):
        g = g.assign(c="x")
        p = pm.fit(g, "c", "lognormal", lapse_max=S.BIAS["lapse_max"])["params"].iloc[0]
        pse = std * np.exp(p.mu)
        row = dict(standard_ms=1000 * std, noise=float(noise), n=int(len(g)), pse_ms=1000 * pse,
                   bias_ms=1000 * (std - pse), bias_fraction=(std - pse) / std, sigma=float(p.sigma), lapse=float(p.lapse))
        if n_boot:
            mus = pm.bootstrap(g, "c", "lognormal", n_boot, rng, lapse_max=S.BIAS["lapse_max"])[:, 0]
            b = 1000 * std * (1 - np.exp(mus))
            row.update(ci_lo_ms=float(np.percentile(b, 2.5)), ci_hi_ms=float(np.percentile(b, 97.5)))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_text(table):
    lines = ["── Audiovisual bias by standard (standard − auditory PSE; positive: vision seems shorter) ──"]
    for _, r in table.sort_values(["noise", "standard_ms"]).iterrows():
        ci = f"  95% CI {r.ci_lo_ms:+.0f} to {r.ci_hi_ms:+.0f}" if "ci_lo_ms" in r and pd.notna(r.ci_lo_ms) else ""
        lines.append(f"   noise {r.noise:<5g} standard {r.standard_ms:4.0f} ms: bias {r.bias_ms:+5.0f} ms "
                     f"({100 * r.bias_fraction:+5.1f}%){ci}; σ {r.sigma:.2f}")
    for noise, g in table.groupby("noise"):
        if g.standard_ms.nunique() > 1:
            slope = np.polyfit(g.standard_ms, g.bias_ms, 1)[0]
            lines.append(f"   noise {noise:g}: bias changes by {1000 * slope:+.0f} ms per second of standard "
                         f"(0 = constant offset; about {1000 * g.bias_fraction.mean():+.0f} = proportional)")
    return "\n".join(lines)


def plot(table, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for noise, g in table.groupby("noise"):
        c = "tab:blue" if noise == S.NOISE_LOW else "tab:red"
        g = g.sort_values("standard_ms")
        err = [g.bias_ms - g.ci_lo_ms, g.ci_hi_ms - g.bias_ms] if "ci_lo_ms" in g else None
        axes[0].errorbar(g.standard_ms, g.bias_ms, yerr=err, fmt="o-", color=c, capsize=4, label=f"noise {noise:g}")
        axes[1].plot(g.standard_ms, 100 * g.bias_fraction, "o-", color=c, label=f"noise {noise:g}")
    axes[0].set(xlabel="Visual standard (ms)", ylabel="Bias (ms)", title="Constant offset: flat line")
    axes[1].set(xlabel="Visual standard (ms)", ylabel="Bias (% of the standard)", title="Proportional bias: flat line")
    for ax in axes:
        ax.axhline(0, color="0.8", lw=1)
        ax.legend(frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Audiovisual duration bias by standard (positive: vision seems shorter)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def report(pid, root=None, n_boot=200):
    """Table, terminal text and plot (in <PID>/fits/) once a participant has several standards."""
    table = bias_table(pid, root, n_boot)
    if table.empty or table.standard_ms.nunique() < 2:
        return None
    png = Path(root or storage.DATA_ROOT) / pid / "fits" / f"{pid}_crossmodal_standards_{storage.timestamp()}_fit.png"
    plot(table, png)
    table.to_csv(png.with_suffix(".csv"), index=False)
    return summary_text(table) + f"\n   plot: {png}"
