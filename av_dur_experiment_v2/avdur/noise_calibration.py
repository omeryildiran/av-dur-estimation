"""Auditory noise calibration: the participant's high noise level.

Unimodal auditory trials (500 ms standard) with the test fixed at exp(+/-delta) x the standard,
where delta = z(target_correct) x target_sigma: a listener whose auditory sigma equals
target_sigma answers target_correct of these trials correctly. Two interleaved 2-down-1-up
staircases on ln(noise) (2 correct in a row: louder noise; 1 error: softer) converge on the
noise level where that is so, i.e. where the participant's sigma = target_sigma. One starts
easy, one hard; order and test direction are balanced. A few easy catch trials at low noise
check attention.

Estimate: the geometric mean of (1) the staircases (mean ln noise over the second half of each,
averaged) and (2) a fit of sigma(noise) = exp(a + b ln noise) to every staircase trial, solved
for sigma = target. Combining the two was the most accurate estimator in simulation. The result
is saved as <data folder>/<PID>/noise/<PID>_noise_<time>.json and used as the participant's
high noise level by every later task (low noise stays NOISE_LOW).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import experiment_settings as S
from avdur import storage
from avdur.design import TrialSpec
from avdur.session import Session
from avdur.staircase import Staircase

C = S.NOISE_CALIBRATION
DELTA = float(norm.ppf(C["target_correct"]) * C["target_sigma"])          # ln(test / standard)
LEVELS = (float(np.exp(DELTA) - 1), float(np.exp(-DELTA) - 1))           # test longer / shorter


class NoiseCalibration(Session):
    def __init__(self, pid, presenter, **kwargs):
        kwargs.pop("noise_high", None)
        kwargs["practice"] = False                         # it starts easy; no separate practice
        super().__init__("noise_calibration", pid, presenter, **kwargs)
        self.schedule_task = "unimodal_auditory"
        self.calibration = None

    # ── design ────────────────────────────────────────────────────────────────
    def _build_trials(self, rng, first_order):
        n = C["trials_per_staircase"]
        slots = []
        for name in self.cfg["staircases"]:
            combos = [(o, lv) for o in (1, 2) for lv in LEVELS]
            for k in range(n):
                order, level = combos[k % 4]
                slots.append(TrialSpec(self.task, None, 0.0, "staircase", name, level, order))
        for k in range(C["catch_trials"]):
            slots.append(TrialSpec(self.task, S.NOISE_LOW, 0.0, "lapse", None, (-0.9, 0.9)[k % 2], 1 + (k // 2) % 2))
        rng.shuffle(slots)
        return slots

    def _make_stairs(self, rng):
        lo, hi = C["bounds"]
        stairs = {}
        for name, start in zip(self.cfg["staircases"], C["start_noise"]):
            # level = -ln(noise): the staircase's "down" (2 correct) makes the noise louder
            st = Staircase("2D1U", floor=-np.log(hi), ceiling=-np.log(lo), init_step=C["init_step_ln"],
                           step_factor=C["step_factor"], n_step_changes=C["n_step_changes"])
            st.level, st.start = float(-np.log(start)), f"noise {start:g}"
            stairs[(None, 0.0, name)] = st
        return stairs

    def _trial_level(self, spec, stair):
        return float(spec.fixed_level)

    def _trial_noise(self, spec, stair):
        return round(float(np.exp(-stair.level)), 4) if stair is not None else spec.noise

    def _update_stair(self, stair, chose_test, is_correct):
        stair.update(bool(is_correct))

    def progress(self, rows, n_total, t0, stairs):
        import time
        n = len(rows)
        elapsed = time.monotonic() - t0 if t0 else 0.0
        catch = [r["is_correct"] for r in rows if r["kind"] == "lapse"]
        noise = "  ".join(f"{name}: noise {np.exp(-st.level):.2f} ({st.reversals} reversals)"
                          for (_, _, name), st in stairs.items())
        return (f"[{self.pid} noise calibration] {n}/{n_total} trials, {elapsed / 60:.1f} min | {noise} | "
                f"catch trials {sum(catch)}/{len(catch)}")

    # ── result ────────────────────────────────────────────────────────────────
    def _after_run(self, meta, rows):
        result = estimate(pd.DataFrame(rows))
        result.update(participant=self.pid, session_csv=self.csv_path.name,
                      created=dt.datetime.now().isoformat(timespec="seconds"))
        folder = Path(self.root) / self.pid / "noise"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.pid}_noise_{storage.timestamp()}.json"
        try:
            result["plot"] = str(plot(pd.DataFrame(rows), result, Path(self.root) / self.pid / "fits",
                                      self.csv_path.stem))
        except Exception as err:                          # a plot problem must not lose the calibration
            result["plot"] = None
            result["warnings"].append(f"plot failed: {err}")
        storage.write_json(path, result)
        if self.solid_root is not False and storage.is_solid(meta, self.root):
            storage.copy_to_solid([path], self.pid, "noise", self.solid_root)
        result["file"] = str(path)
        meta["noise_calibration"] = {k: v for k, v in result.items() if k != "trace"}
        self.calibration = result


def _sigma_fit(d):
    """sigma(noise) = exp(a + b ln noise), lapse 0.02, fitted to correct/incorrect at +/-DELTA."""
    ln_n = np.log(d["audNoise"].to_numpy(float))
    y = d["is_correct"].to_numpy(float)

    def nll(p):
        sigma = np.exp(p[0] + p[1] * ln_n)
        pc = np.clip(0.01 + 0.98 * norm.cdf(DELTA / sigma), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(pc) + (1 - y) * np.log(1 - pc))

    best = min((minimize(nll, x0, method="L-BFGS-B", bounds=[(-4.0, 3.0), (0.01, 3.0)])
                for x0 in ([-1.0, 0.5], [-0.5, 1.0], [-1.5, 0.2], [0.0, 0.1])), key=lambda r: r.fun)
    a, b = best.x
    return float(a), float(b)


def estimate(df):
    lo, hi = C["bounds"]
    stair = df[df["kind"] == "staircase"].sort_values("trial_num")
    warnings, per_stair = [], {}
    for name, s in stair.groupby("current_stair"):
        rev = s["stair_is_reversal"].astype(str).isin(["True", "1"]).to_numpy()
        ln_n = np.log(s["audNoise"].to_numpy(float))
        per_stair[name] = dict(noise=float(np.exp(ln_n[len(ln_n) // 2:].mean())), reversals=int(rev.sum()),
                               last_noise=float(np.exp(ln_n[-1])))
        if rev.sum() < 3:
            warnings.append(f"staircase {name}: only {int(rev.sum())} reversals")
    noise_stairs = float(np.clip(np.exp(np.mean([np.log(v["noise"]) for v in per_stair.values()])), lo, hi))
    a, b = _sigma_fit(stair)
    noise_fit = float(np.clip(np.exp((np.log(C["target_sigma"]) - a) / b), lo, hi))
    fit_valid = b > 0.05                    # a flat sigma(noise) fit says nothing about where the target is reached
    noise_high = round(float(np.clip(np.sqrt(noise_stairs * noise_fit) if fit_valid else noise_stairs, lo, hi)), 3)
    catch = df[df["kind"] == "lapse"]["is_correct"]
    notes = [] if fit_valid else ["the fit found no change of sigma with noise; the staircases alone set the level"]
    sigma_at_low = float(np.exp(a + b * np.log(lo)))
    if noise_high >= hi / 1.25:
        notes.append(f"near the upper limit ({hi}): a good listener; their high-noise sigma will be somewhat "
                     f"below the target {C['target_sigma']}")
    if noise_high <= lo * 1.25 or sigma_at_low > 1.5 * C["target_sigma"]:
        warnings.append(f"near the lower limit ({lo}; fitted sigma there {sigma_at_low:.2f}): sigma "
                        f"{C['target_sigma']} is not reached even in the softest noise (poor auditory duration discrimination)")
    if fit_valid and abs(np.log(noise_fit / noise_stairs)) > np.log(2.5):
        warnings.append(f"staircases ({noise_stairs:.2f}) and fit ({noise_fit:.2f}) disagree by more than x2.5")
    if len(catch) and catch.mean() < 5 / 6:
        warnings.append(f"easy catch trials only {int(catch.sum())}/{len(catch)} correct")
    trace = {name: s["audNoise"].round(4).tolist() for name, s in stair.groupby("current_stair")}
    return dict(noise_high=noise_high, target_sigma=C["target_sigma"], target_correct=C["target_correct"],
                delta_ln=DELTA, test_ms=[round(500 * np.exp(DELTA)), round(500 * np.exp(-DELTA))],
                noise_staircases=noise_stairs, staircases=per_stair, noise_fit=noise_fit,
                sigma_fit=dict(a=a, b=b, sigma_at_low=float(np.exp(a + b * np.log(S.NOISE_LOW))),
                               sigma_at_result=float(np.exp(a + b * np.log(noise_high)))),
                catch_correct=float(catch.mean()) if len(catch) else float("nan"), n_catch=int(len(catch)),
                fit_valid=bool(fit_valid), n_trials=int(len(df)), bounds=list(C["bounds"]), warnings=warnings,
                notes=notes, trace=trace)


def plot(df, result, folder, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    stair = df[df["kind"] == "staircase"].sort_values("trial_num")
    for (name, s), c in zip(stair.groupby("current_stair"), ("tab:purple", "tab:orange")):
        ax1.plot(s["trial_num"], s["audNoise"], "o-", ms=4, color=c, lw=1, label=f"staircase {name}")
        wrong = s["is_correct"] == 0
        ax1.scatter(s["trial_num"][wrong], s["audNoise"][wrong], s=30, facecolors="white", edgecolors=c, zorder=3)
    ax1.axhline(result["noise_high"], color="black", ls="--", lw=1, label=f"result {result['noise_high']:.2f}")
    ax1.axhline(S.NOISE_HIGH_DEFAULT, color="0.6", ls=":", lw=1, label=f"v1 high noise {S.NOISE_HIGH_DEFAULT}")
    ax1.set(yscale="log", xlabel="Trial number", ylabel="Noise level", title="Staircases (open: error)")
    ax1.legend(frameon=False, fontsize=9)
    n = stair["audNoise"].to_numpy(float)
    bins = np.unique(np.quantile(np.log(n), np.linspace(0, 1, 7)))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    g = pd.DataFrame({"ln": np.log(n), "y": stair["is_correct"]}).groupby(pd.cut(np.log(n), bins), observed=True).agg(
        ln=("ln", "mean"), y=("y", "mean"), k=("y", "size"))
    ax2.scatter(np.exp(g["ln"]), g["y"], s=10 + 4 * g["k"], color="tab:blue", alpha=0.7)
    xs = np.exp(np.linspace(np.log(C["bounds"][0]), np.log(C["bounds"][1]), 200))
    sig = np.exp(result["sigma_fit"]["a"] + result["sigma_fit"]["b"] * np.log(xs))
    ax2.plot(xs, 0.01 + 0.98 * norm.cdf(DELTA / sig), color="tab:blue", label="fit")
    ax2.axhline(C["target_correct"], color="0.7", ls=":", lw=1)
    ax2.axvline(result["noise_high"], color="black", ls="--", lw=1)
    ax2.set(xscale="log", ylim=(0.3, 1.02), xlabel="Noise level",
            ylabel=f"Correct at {result['test_ms'][1]} / {result['test_ms'][0]} vs 500 ms",
            title=f"High noise level {result['noise_high']:.2f} (target σ {C['target_sigma']}; fit alone {result['noise_fit']:.2f})")
    ax2.legend(frameon=False)
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ticks = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2]          # plain numbers on the log axes, not 10^x
    for axis in (ax1.yaxis, ax2.xaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = folder / f"{stem}_fit.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary_text(result):
    st = "; ".join(f"{k}: {v['noise']:.2f} ({v['reversals']} reversals)" for k, v in result["staircases"].items())
    lines = [f"── Noise calibration: high noise level {result['noise_high']:.3f} "
             f"(target σ {result['target_sigma']}; v1 used {S.NOISE_HIGH_DEFAULT}) ──",
             f"   staircases {st}; fit {result['noise_fit']:.2f}; catch trials {result['catch_correct']:.0%} of "
             f"{result['n_catch']}; fitted σ at the chosen level {result['sigma_fit']['sigma_at_result']:.2f}"]
    lines += [f"   note: {n}" for n in result.get("notes", [])]
    lines += [f"   WARNING: {w}" for w in result["warnings"]] or ["   checks passed"]
    if result.get("plot"):
        lines.append(f"   plot: {result['plot']}")
    return "\n".join(lines)
