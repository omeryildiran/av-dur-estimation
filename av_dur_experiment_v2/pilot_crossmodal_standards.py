"""Pilot: is the audiovisual duration bias the same at different standard durations?

    python pilot_crossmodal_standards.py --pid P102                  # lab CRT, real run (about 20 min)
    python pilot_crossmodal_standards.py --pid P102 --screen 1       # choose the participant's screen
    python pilot_crossmodal_standards.py --pid P99 --simulate        # check the pipeline without a window

The cross-modal task (auditory test vs. visual standard) as in the experiment, run in one
block per standard duration (default 300, 500, 800 ms; random block order), both noise
levels interleaved within a block. Per condition: two 1D1U staircases (they settle on the
PSE, one starting at each end) and a few easy trials. Data go to data_pilot/<PID>/crossmodal/
(one session per block; its settings snapshot records the standard). At the end, each
standard and noise level is fitted (lognormal, lapse <= 0.1, bootstrap CI) and the bias is
reported in ms and as a fraction of the standard, with a plot:
  - the same bias in ms at every standard  -> a constant offset (what the experiment assumes);
  - the same fraction of the standard      -> a proportional bias (the correction should scale).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur import design, psychometrics as pm, storage  # noqa: E402
from avdur.session import Session  # noqa: E402

PILOT_ROOT = ROOT / "data_pilot"
BASE_CROSSMODAL = dict(S.TASKS["crossmodal"])


def configure(standard_s, trials_per_staircase, lapse_per_level, noise_levels):
    """Settings for one block: the cross-modal task with this standard (read at run time everywhere)."""
    S.STANDARD_DUR_S = float(standard_s)
    S.TASKS["crossmodal"] = dict(BASE_CROSSMODAL, staircases=("1D1U", "1D1U_b"),
                                 trials_per_staircase=trials_per_staircase,
                                 lapse_trials_per_level=lapse_per_level, noise_levels=tuple(noise_levels))


def fit_block(csv_path, standard_s, n_boot, rng):
    d = pd.read_csv(csv_path)
    d = d[d["responses"].isin([1, 2])]
    rows = []
    for noise, g in d.groupby("audNoise"):
        g = g.assign(c="x")
        p = pm.fit(g, "c", "lognormal", lapse_max=S.BIAS["lapse_max"])["params"].iloc[0]
        boot = pm.bootstrap(g, "c", "lognormal", n_boot, rng, lapse_max=S.BIAS["lapse_max"])[:, 0] if n_boot else None
        pse = standard_s * np.exp(p.mu)
        row = dict(standard_ms=1000 * standard_s, noise=float(noise), n=int(len(g)), pse_ms=1000 * pse,
                   bias_ms=1000 * (standard_s - pse), bias_fraction=(standard_s - pse) / standard_s,
                   sigma=float(p.sigma), lapse=float(p.lapse),
                   easy_correct=float(g[g["kind"] == "lapse"]["is_correct"].mean()))
        if boot is not None:
            b = 1000 * (standard_s - standard_s * np.exp(boot))
            row.update(bias_ci_lo_ms=float(np.percentile(b, 2.5)), bias_ci_hi_ms=float(np.percentile(b, 97.5)))
        rows.append(row)
    return rows


def report(table, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for noise, g in table.groupby("noise"):
        c = {0.1: "tab:blue", 1.2: "tab:red"}.get(noise, "black")
        g = g.sort_values("standard_ms")
        err = None
        if "bias_ci_lo_ms" in g:
            err = [g.bias_ms - g.bias_ci_lo_ms, g.bias_ci_hi_ms - g.bias_ms]
        axes[0].errorbar(g.standard_ms, g.bias_ms, yerr=err, fmt="o-", color=c, capsize=4, label=f"noise {noise:g}")
        axes[1].plot(g.standard_ms, 100 * g.bias_fraction, "o-", color=c, label=f"noise {noise:g}")
    axes[0].axhline(0, color="0.8", lw=1)
    axes[1].axhline(0, color="0.8", lw=1)
    axes[0].set(xlabel="Visual standard (ms)", ylabel="Bias = standard − auditory PSE (ms)",
                title="Constant offset: flat line")
    axes[1].set(xlabel="Visual standard (ms)", ylabel="Bias (% of the standard)", title="Proportional bias: flat line")
    for ax in axes:
        ax.legend(frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Audiovisual duration bias at different standards (positive: vision seems shorter)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def summary(table):
    lines = ["", "── Audiovisual bias by standard (bias = standard − auditory PSE; positive: vision seems shorter) ──"]
    for _, r in table.sort_values(["noise", "standard_ms"]).iterrows():
        ci = (f"  95% CI {r.bias_ci_lo_ms:+.0f} to {r.bias_ci_hi_ms:+.0f}" if "bias_ci_lo_ms" in r and pd.notna(r.bias_ci_lo_ms)
              else "")
        lines.append(f"   noise {r.noise:<4g} standard {r.standard_ms:4.0f} ms: PSE {r.pse_ms:4.0f} ms, bias {r.bias_ms:+5.0f} ms"
                     f" ({100 * r.bias_fraction:+5.1f}%){ci}; σ {r.sigma:.2f}, lapse {r.lapse:.2f}, easy {r.easy_correct:.0%}")
    for noise, g in table.groupby("noise"):
        if len(g) > 1:
            slope_ms = np.polyfit(g.standard_ms, g.bias_ms, 1)[0]
            lines.append(f"   noise {noise:g}: bias changes by {1000 * slope_ms:+.0f} ms per second of standard "
                         f"(0 = constant offset; equal to the mean fraction × 1000 = proportional, here "
                         f"{1000 * g.bias_fraction.mean():+.0f})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--monitor", default=S.DEFAULT_MONITOR, choices=list(S.MONITOR_PROFILES))
    ap.add_argument("--screen", type=int, help="screen number for the window (default: from the monitor profile)")
    ap.add_argument("--standards", type=float, nargs="+", default=[0.3, 0.5, 0.8], help="standard durations (s)")
    ap.add_argument("--noise", type=float, nargs="+", default=list(S.NOISE_LEVELS))
    ap.add_argument("--trials-per-staircase", type=int, default=30)
    ap.add_argument("--easy-per-level", type=int, default=3, help="easy trials per level (2 levels) and condition")
    ap.add_argument("--simulate", action="store_true", help="simulated observer, no window")
    ap.add_argument("--autotest", action="store_true", help="automatic answers on the real display and sound card")
    ap.add_argument("--max-trials", type=int, help="simulate/autotest only: first N trials of each block")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    pid = design.normalize_pid(args.pid)
    seed = args.seed if args.seed is not None else int(time.time()) % 2**31
    rng = np.random.default_rng(seed)
    standards = list(rng.permutation(args.standards))
    root = PILOT_ROOT if not (args.simulate or args.autotest) else PILOT_ROOT / ("simulated" if args.simulate else "autotest")

    if args.simulate:
        from avdur.presenter_sim import SimulatedPresenter
        presenter = SimulatedPresenter(seed=seed)
    else:
        from avdur.presenter_psychopy import PsychoPyPresenter
        presenter = PsychoPyPresenter(args.monitor, autotest_seed=seed if args.autotest else None, screen=args.screen)

    print(f"[{pid}] cross-modal pilot: standards {', '.join(f'{1000 * s:.0f} ms' for s in standards)} (this order), "
          f"noise {args.noise}; data in {root}", flush=True)
    rows, sessions = [], []
    for k, std in enumerate(standards):
        configure(std, args.trials_per_staircase, args.easy_per_level, args.noise)
        last = k == len(standards) - 1
        session = Session("crossmodal", pid, presenter, seed=seed + k, root=root, monitor=args.monitor,
                          simulate=args.simulate, autotest=args.autotest, max_trials=args.max_trials,
                          solid_root=False, close_at_end=last, verbose=True)
        if not session.run():
            print("Stopped before the end; blocks finished so far are saved and fitted below.", flush=True)
            break
        session.meta["pilot"] = dict(name="crossmodal_standards", standard_s=std, block=k + 1, of=len(standards),
                                     block_order_ms=[1000 * s for s in standards])
        session.save_meta()
        sessions.append((std, session.csv_path))
        rows += fit_block(session.csv_path, std, args.n_boot, rng)
        print(summary(pd.DataFrame(rows[-len(args.noise):])), flush=True)
        if not last:
            presenter.wait_screen(f"Part {k + 1} of {len(standards)} is done.\n\nPlease take a short break.",
                                  lambda: True, S.BLOCK_REST_MIN_S)
    if not rows:
        return 1
    table = pd.DataFrame(rows)
    stamp = storage.timestamp()
    out = root / pid / "fits"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{pid}_crossmodal_pilot_{stamp}.csv", index=False)
    png = out / f"{pid}_crossmodal_pilot_{stamp}_fit.png"
    report(table, png)
    print(summary(table) + f"\n   table: {out / f'{pid}_crossmodal_pilot_{stamp}.csv'}\n   plot:  {png}", flush=True)
    if not (args.simulate or args.autotest):
        if sys.platform == "win32":
            os.startfile(str(png))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(png)], check=False)
    return 0 if len(sessions) == len(standards) else 3


if __name__ == "__main__":
    sys.exit(main())
