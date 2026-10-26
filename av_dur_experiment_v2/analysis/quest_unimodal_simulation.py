"""QUEST+ vs the four staircases of the unimodal blocks: how many trials for the same sigma?

    python analysis/quest_unimodal_simulation.py          # about 8 min on 8 cores
    python analysis/quest_unimodal_simulation.py --quick

The unimodal design per condition: 2D1U and 3D1U from the floor, 1D2U and 1D3U from the ceiling,
35 trials each, plus 14 easy trials (7 at each end) = 154 trials. QUEST+ (standard version: mu and
sigma, lapse as nuisance; see quest_simulation.py) runs with the same easy trials. Each run is 154
trials and is evaluated after 60 ... 154 trials, as in quest_simulation.py; the fit is the
experiment's (lognormal, lapse 0-0.1).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quest_simulation as qs  # noqa: E402

from avdur.staircase import Staircase  # noqa: E402
import experiment_settings as S  # noqa: E402

OUT = qs.OUT
BUDGETS = (60, 80, 100, 120, 154)
N_PER_STAIR, N_EASY = 35, 14
RULES = (("2D1U", "floor"), ("1D2U", "ceiling"), ("3D1U", "floor"), ("1D3U", "ceiling"))
TASK = {  # floor (test/standard - 1) and easy levels from TASKS
    "auditory": dict(floor=S.TASKS["unimodal_auditory"]["floor"], easy=S.TASKS["unimodal_auditory"]["lapse_levels"]),
    "visual": dict(floor=S.TASKS["unimodal_visual"]["floor"], easy=S.TASKS["unimodal_visual"]["lapse_levels"]),
}
X_GRID = np.linspace(np.log(1 + S.TASKS["unimodal_auditory"]["floor"]), np.log(1 + S.STAIRCASE["ceiling"]), 60)


def schedule(rng, task):
    kinds = np.array(["s"] * (4 * N_PER_STAIR) + ["e"] * N_EASY)
    rng.shuffle(kinds)
    lo, hi = TASK[task]["easy"]
    easy = iter(np.log(1 + np.r_[[lo] * (N_EASY // 2), [hi] * (N_EASY // 2)][rng.permutation(N_EASY)]))
    return kinds, easy, rng.integers(1, 3, len(kinds))


def run_staircases(obs, rng, task):
    kinds, easy, orders = schedule(rng, task)
    stairs = [Staircase(rule, TASK[task]["floor"], S.STAIRCASE["ceiling"], start=start) for rule, start in RULES]
    which = iter(np.repeat(np.arange(4), N_PER_STAIR)[rng.permutation(4 * N_PER_STAIR)])
    xs, ys = [], []
    for t, k in enumerate(kinds):
        if k == "e":
            x = next(easy); y = obs.answer(x, t, orders[t], rng)
        else:
            st = stairs[next(which)]
            x = np.log(1 + st.level); y = obs.answer(x, t, orders[t], rng); st.update(y)
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys, float)


def run_quest(obs, rng, task):
    kinds, easy, orders = schedule(rng, task)
    q = qs.QuestPlus(x_grid=X_GRID if task == "auditory" else X_GRID[X_GRID >= np.log(1 + TASK[task]["floor"]) - 1e-9])
    xs, ys = [], []
    for t, k in enumerate(kinds):
        x = next(easy) if k == "e" else q.next_x(rng)
        y = obs.answer(x, t, orders[t], rng)
        q.update(x, y)
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys, float)


def simulate_one(args):
    scenario, task, rep, obs_kw, seed = args
    rows = []
    for method, fn in (("staircase", run_staircases), ("quest", run_quest)):
        rng = np.random.default_rng([seed, 0 if method == "staircase" else 1])
        obs = qs.Observer(**obs_kw)
        xs, ys = fn(obs, rng, task)
        for b in BUDGETS:
            mu, sig = qs.fit_ml(xs[:b], ys[:b])
            rows.append(dict(scenario=scenario, task=task, rep=rep, method=method, budget=b, true_mu=obs_kw["mu"],
                             true_sigma=obs_kw["sigma"], true_lapse=obs_kw["lapse"], mu_hat=mu, sigma_hat=sig,
                             mu_hat_lapse_known=mu, sigma_hat_lapse_known=sig,
                             share_near_pse=float(np.mean(np.abs(xs[:b] - obs_kw["mu"]) < obs_kw["sigma"]))))
    return rows


def jobs(n_rep, rng):
    out = []
    grids = {"auditory": (0.2, 0.35, 0.75, 1.1), "visual": (0.4, 0.62, 1.0, 1.5)}
    for task, sigmas in grids.items():
        for s in sigmas:
            for lapse in (0.0, 0.05, 0.1):
                for r in range(n_rep):
                    out.append((f"{task} s{s} l{lapse}", task, r,
                                dict(mu=float(rng.uniform(-0.15, 0.15)), sigma=s, lapse=lapse), int(rng.integers(2**31))))
        for r in range(n_rep):                                   # stress tests at a typical sigma
            s = sigmas[2]
            out.append((f"{task} lapse 0.15", task, r, dict(mu=float(rng.uniform(-0.15, 0.15)), sigma=s, lapse=0.15),
                        int(rng.integers(2**31))))
            out.append((f"{task} learning", task, r, dict(mu=float(rng.uniform(-0.15, 0.15)), sigma=1.3 * s,
                                                         sigma_end=0.7 * s, drift_mu=0.1, lapse=0.05), int(rng.integers(2**31))))
            out.append((f"{task} order bias", task, r, dict(mu=float(rng.uniform(-0.15, 0.15)), sigma=s, lapse=0.05,
                                                           order_bias=0.25), int(rng.integers(2**31))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()
    js = jobs(4 if args.quick else args.reps, np.random.default_rng(7))
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(simulate_one, js, chunksize=4) for r in chunk]
    df = pd.DataFrame(rows)
    tag = "_quick" if args.quick else ""
    df.to_csv(OUT / f"unimodal_raw{tag}.csv", index=False)
    summ = qs.summarise(df)
    summ.to_csv(OUT / f"unimodal_summary{tag}.csv", index=False)
    print(f"{len(js)} observers x 2 methods in {time.time() - t0:.0f} s")
    print(summ.pivot_table(index=["scenario", "budget"], columns="method",
                           values=["sigma_rmse_ln", "pse_mae_ms"]).round(3).to_string())


if __name__ == "__main__":
    main()
