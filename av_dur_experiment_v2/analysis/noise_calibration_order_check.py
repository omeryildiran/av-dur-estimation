"""Why the noise calibration chose 1.7-1.8 for P103 and P105, and what a cap does (28 Oct 2026).

    python analysis/noise_calibration_order_check.py            # fits only (seconds)
    python analysis/noise_calibration_order_check.py --simulate # + calibration variants (~2.5 min)

Per participant, the noise calibration and the unimodal auditory block are fitted together with
a separate sigma(noise) and PSE for each presentation order:
    P(test longer) = lapse/2 + (1 - lapse) Phi((x - mu_o) / sigma_o(n)),  ln sigma_o(n) = a_o + b_o ln n
In loud noise most listeners are near chance with the test first and sharp with it second. The
calibration scores % correct at a fixed +-0.41 pooled over orders, which then barely changes with
noise, so its staircases drift; the unimodal block fits one sigma over both orders and a wide
range of test durations, which the test-first trials blow up.

"pooled sigma" = the single-Gaussian sigma (lapse <= 0.1) that best fits the 50:50 order mixture
over 80-950 ms tests. It is lower than what the unimodal QUEST+ reports (that puts the tests at
the extremes, where the mixture is furthest from one Gaussian), so read it as a lower bound.

--simulate runs the calibration (avdur.noise_calibration's staircases and estimate) on each
participant's fitted model: start points, trials per staircase and the upper cap.
"""

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment_settings as S  # noqa: E402
from avdur import noise_calibration as NC  # noqa: E402
from avdur.staircase import Staircase  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data"
UNIMODAL = {"P101": "P101_unimodal_auditory_20261026-154210", "P102": "P102_unimodal_auditory_20261021-185440",
            "P103": "P103_unimodal_auditory_20261023-123954", "P104": "P104_unimodal_auditory_20261026-110537",
            "P105": "P105_unimodal_auditory_20261028-173836"}      # P102's later file is an 8-trial aborted run
XS = np.log(np.linspace(0.08, 0.95, 40) / 0.5)
DELTA = NC.DELTA


def load(pid):
    cal = pd.read_csv(sorted(glob.glob(str(DATA / pid / "noise_calibration" / "*.csv")))[-1])
    u = pd.read_csv(DATA / pid / "unimodal_auditory" / f"{UNIMODAL[pid]}.csv")
    d = pd.concat([cal, u])
    return d[d.kind.isin(["staircase", "lapse"])]


def fit_order(d):
    """[a1, b1, mu1, a2, b2, mu2, lapse]; 1 = test first, 2 = test second."""
    x = np.log(d.testDurS / d.standardDur).to_numpy()
    ln_n = np.log(d.audNoise.to_numpy())
    second = (d.order == 2).to_numpy()
    y = d.chose_test.to_numpy(float)

    def nll(p):
        a, b, mu = (np.where(second, p[k + 3], p[k]) for k in range(3))
        pr = np.clip(p[6] / 2 + (1 - p[6]) * norm.cdf((x - mu) / np.exp(a + b * ln_n)), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    bounds = [(-4, 3), (-0.5, 3), (-1.5, 1.5)] * 2 + [(0, 0.1)]
    starts = ([-1, .5, 0, -1, .5, 0, .02], [-.5, 1, 0, -1.5, .3, 0, .05], [0, 1.5, -.2, -1, 1, .1, .02])
    return min((minimize(nll, x0, method="L-BFGS-B", bounds=bounds) for x0 in starts), key=lambda r: r.fun).x


def p_longer(p, x, n, order):
    a, b, mu = p[3:6] if order == 2 else p[0:3]
    return p[6] / 2 + (1 - p[6]) * norm.cdf((x - mu) / np.exp(a + b * np.log(n)))


def sigma_by_order(p, n):
    return np.exp(p[0] + p[1] * np.log(n)), np.exp(p[3] + p[4] * np.log(n))


def pooled_sigma(p, n):
    target = 0.5 * (p_longer(p, XS, n, 1) + p_longer(p, XS, n, 2))

    def loss(q):
        pr = np.clip(q[2] / 2 + (1 - q[2]) * norm.cdf((XS - q[0]) / np.exp(q[1])), 1e-6, 1 - 1e-6)
        return -np.sum(target * np.log(pr) + (1 - target) * np.log(1 - pr))

    q = min((minimize(loss, x0, method="L-BFGS-B", bounds=[(-2, 2), (np.log(.03), np.log(10)), (0, .1)])
             for x0 in ([0, -1, .02], [0, .5, .05])), key=lambda r: r.fun).x
    return float(np.exp(q[1]))


def correct_at_delta(p, n, order):
    return 0.5 * (p_longer(p, DELTA, n, order) + 1 - p_longer(p, -DELTA, n, order))


def run_calibration(p, n_per, starts, hi, rng):
    """avdur.noise_calibration's design, staircases and estimate on the fitted listener, with upper cap hi."""
    C = S.NOISE_CALIBRATION
    old = C["bounds"]
    C["bounds"] = (old[0], hi)
    try:
        stairs = {}
        for name, start in zip("AB", starts):
            st = Staircase("2D1U", floor=-np.log(hi), ceiling=-np.log(old[0]), init_step=C["init_step_ln"],
                           step_factor=C["step_factor"], n_step_changes=C["n_step_changes"])
            st.level = float(-np.log(min(start, hi)))
            stairs[name] = st
        combos = [(o, lv) for o in (1, 2) for lv in NC.LEVELS]
        slots = [("staircase", name, combos[k % 4]) for name in stairs for k in range(n_per)]
        slots += [("lapse", None, (1 + (k // 2) % 2, (-0.9, 0.9)[k % 2])) for k in range(C["catch_trials"])]
        rng.shuffle(slots)
        rows = []
        for i, (kind, name, (order, level)) in enumerate(slots):
            n = float(np.exp(-stairs[name].level)) if kind == "staircase" else S.NOISE_LOW
            x = float(np.log(1 + level))
            correct = (rng.random() < p_longer(p, x, n, order)) == (x > 0)
            reversal = False
            if kind == "staircase":
                stairs[name].update(correct)
                reversal = stairs[name].is_reversal
            rows.append(dict(kind=kind, trial_num=i, current_stair=name, stair_is_reversal=reversal, audNoise=n,
                             testDurS=0.5 * np.exp(x), standardDur=0.5, is_correct=float(correct), order=order))
        return NC.estimate(pd.DataFrame(rows))["noise_high"]
    finally:
        C["bounds"] = old


VARIANTS = [("2.0 cap, start 0.3/1.6, 100 per staircase", 100, (0.3, 1.6), 2.0),
            ("2.0 cap, both start at 0.5", 100, (0.5, 0.5), 2.0),
            ("2.0 cap, 200 per staircase", 200, (0.3, 1.6), 2.0),
            ("1.5 cap", 100, (0.3, 1.6), 1.5),
            ("1.2 cap, both start at 0.5 (since 28 Oct)", 100, (0.5, 0.5), 1.2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--runs", type=int, default=150)
    args = ap.parse_args()
    levels = (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
    for pid in UNIMODAL:
        p = fit_order(load(pid))
        chosen = json.loads(sorted((DATA / pid / "noise").glob("*.json"))[-1].read_text())["noise_high"]
        print(f"\n{pid}: calibrated {chosen}; PSE test first {500 * np.exp(p[2]):.0f} ms, test second "
              f"{500 * np.exp(p[5]):.0f} ms; lapse {p[6]:.2f}")
        print("   noise                        " + "  ".join(f"{v:>9}" for v in levels))
        sig = [sigma_by_order(p, v) for v in levels]
        print("   sigma test first / second    " + "  ".join(f"{s1:4.2f}/{s2:4.2f}" for s1, s2 in sig))
        print("   % correct at +-0.41, ditto   " + "  ".join(f"{correct_at_delta(p, v, 1):4.2f}/{correct_at_delta(p, v, 2):4.2f}"
                                                       for v in levels))
        print("   pooled sigma (lower bound)   " + "  ".join(f"{pooled_sigma(p, v):9.2f}" for v in levels))
        if args.simulate:
            for name, n_per, starts, hi in VARIANTS:
                res = np.array([run_calibration(p, n_per, starts, hi, np.random.default_rng([k, n_per, int(10 * hi),
                                                                                              int(10 * starts[1])]))
                                for k in range(args.runs)])
                q = np.quantile(res, [0.1, 0.5, 0.9])
                print(f"   {name:<44} level {q[1]:.2f} (80%: {q[0]:.2f}-{q[2]:.2f}), pooled sigma there "
                      f"{pooled_sigma(p, q[1]):.2f}")


if __name__ == "__main__":
    main()
