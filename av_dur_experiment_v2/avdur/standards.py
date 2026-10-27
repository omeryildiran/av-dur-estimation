"""Cross-modal task at several standards: the plan per participant and the bias by standard.

The 500 ms standard (STANDARD_DUR_S) keeps the full cross-modal design and alone sets the main
experiment's correction (bias.load_crossmodal). The other standards (CROSSMODAL_STANDARDS) use a
shorter design. All of a participant's standards run interleaved in one session (block label
e.g. S300-S500-S800); with the 500 ms standard only, the session is as before (no block label).
Until 27 Oct 2026 each standard was its own session (S300 / S500 / S800: P101, P104).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import experiment_settings as S
from avdur import psychometrics as pm, storage

CFG = S.CROSSMODAL_STANDARDS


def label(standard_s):
    return f"S{round(1000 * standard_s)}"


def session_standards(meta):
    """Standards (s) of one cross-modal session (interleaved sessions have several)."""
    return [float(v) for v in meta.get("standards_s") or [meta.get("standard_s", S.STANDARD_DUR_S)]]


def extra_participants(root=None):
    """Participants with a completed cross-modal session at an extra standard (not STANDARD_DUR_S)."""
    return sorted({m["participant"] for m in storage.sessions(None, "crossmodal", root)
                   if any(abs(v - S.STANDARD_DUR_S) > 1e-9 for v in session_standards(m))})


def wants_extra(pid, root=None):
    """Default for this participant: the extra standards while fewer than CFG["n_participants"]
    participants have done one (a participant who has one keeps getting the rest)."""
    if not CFG["enabled"]:
        return False
    n = CFG.get("n_participants")
    if n is None:
        return True
    have = extra_participants(root)
    return pid in have or len(have) < n


def design_for(standard_s):
    """None (the full design) for STANDARD_DUR_S, the shorter extra design for the others."""
    return None if abs(standard_s - S.STANDARD_DUR_S) < 1e-9 else dict(CFG["extra_design"])


def session_plan(stds):
    """One session for these standards: (standard_s, design, block label, [(standard_s, design)] interleaved).
    500 ms (when included) is the session's own standard; a single standard runs alone."""
    stds = sorted(stds, key=lambda v: (abs(v - S.STANDARD_DUR_S) > 1e-9, v))
    lab = "-".join(label(v) for v in sorted(stds)) if len(stds) > 1 or abs(stds[0] - S.STANDARD_DUR_S) > 1e-9 else None
    return (stds[0], design_for(stds[0]), lab, [(v, design_for(v)) for v in stds[1:]])


def planned_standards(pid, root=None, several=None):
    """Standards (s) for this participant. several: True all standards, False 500 ms only, None the
    default (wants_extra)."""
    several = wants_extra(pid, root) if several is None else several
    return [S.STANDARD_DUR_S] + (list(CFG["standards"]) if several else [])


def plan(pid, root=None, several=None):
    """[one session plan] (see session_plan): every planned standard interleaved in one session."""
    return [session_plan(planned_standards(pid, root, several))]


def done_standards(pid, root=None):
    """Standards (s) of this participant's completed cross-modal sessions (older sessions: 500 ms)."""
    return {round(v, 4) for m in storage.sessions(pid, "crossmodal", root) for v in session_standards(m)}


def remaining_standards(pid, root=None, several=None):
    done = done_standards(pid, root)
    return [v for v in planned_standards(pid, root, several) if round(v, 4) not in done]


def remaining(pid, root=None, several=None):
    """[one session plan] for the standards not done yet, or []."""
    todo = remaining_standards(pid, root, several)
    return [session_plan(todo)] if todo else []


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
