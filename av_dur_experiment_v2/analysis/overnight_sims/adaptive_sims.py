"""Unimodal and cross-modal blocks: staircases vs QUEST+ (PSE and slope) vs QUEST+ (PSE only).

    python analysis/overnight_sims/adaptive_sims.py --reps 250 --workers 2

Each simulated run is 200 trials of one condition (staircase or QUEST+ trials plus easy trials,
7% of the trials, at the task's easy levels, randomly interleaved) and is evaluated after its
first 40 ... 200 trials, with the experiment's fit (lognormal, lapse 0-0.1). Observers:

  aud_low   unimodal auditory, low noise    sigma U(0.12, 0.45), PSE ln-ratio N(0, 0.08)
  aud_high  unimodal auditory, high noise   sigma U(0.50, 1.10)
  visual    unimodal visual                 sigma U(0.30, 1.00)
  xm_low    cross-modal, low noise          sigma U(0.15, 0.90), PSE ln-ratio U(-0.60, -0.05) (bias)
  xm_high   cross-modal, high noise         sigma U(0.40, 1.10), PSE ln-ratio U(-0.60, 0.05)

lapse U(0, 0.1) everywhere; robustness sets (aud_high and xm_low only): lapse 0.15, learning
(sigma falls by 40% and the PSE drifts by 0.1 over the run), order bias 0.15.

Methods:
  st4    2D1U + 3D1U from the floor, 1D2U + 1D3U from the ceiling (the old unimodal design)
  stxm   1D1U from each end + 2D1U (floor) + 1D2U (ceiling) (the current cross-modal design)
  st11   two 1D1U, one from each end (the 300/800 ms cross-modal design)
  q_ms   QUEST+ targeting PSE and slope, lapse as nuisance (the current unimodal design)
  q_mu   QUEST+ targeting the PSE only
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import quest_simulation as qs  # noqa: E402
from avdur.staircase import Staircase  # noqa: E402

OUT = HERE / "results"
N_RUN = 200
BUDGETS = (40, 60, 80, 100, 120, 150, 200)
EASY_SHARE = 0.07
FLOOR, CEIL = -0.9, 0.9
X_GRID = np.log(1 + np.geomspace(1 + FLOOR, 1 + CEIL, 60) - 1)
EASY = {"aud_low": (-0.9, 0.9), "aud_high": (-0.9, 0.9), "visual": (-0.8, 0.8), "xm_low": (-0.9, 0.9), "xm_high": (-0.9, 0.9)}
STAIRS = {"st4": (("2D1U", "floor"), ("1D2U", "ceiling"), ("3D1U", "floor"), ("1D3U", "ceiling")),
          "stxm": (("1D1U", "floor"), ("1D1U", "ceiling"), ("2D1U", "floor"), ("1D2U", "ceiling")),
          "st11": (("1D1U", "floor"), ("1D1U", "ceiling"))}
METHODS = ("st4", "stxm", "st11", "q_ms", "q_mu")


def observer(task, rng, variant):
    lam = rng.uniform(0, 0.1)
    if task == "aud_low":
        mu, s = rng.normal(0, 0.08), rng.uniform(0.12, 0.45)
    elif task == "aud_high":
        mu, s = rng.normal(0, 0.08), rng.uniform(0.5, 1.1)
    elif task == "visual":
        mu, s = rng.normal(0, 0.08), rng.uniform(0.3, 1.0)
    elif task == "xm_low":
        mu, s = rng.uniform(-0.6, -0.05), rng.uniform(0.15, 0.9)
    else:
        mu, s = rng.uniform(-0.6, 0.05), rng.uniform(0.4, 1.1)
    kw = dict(mu=float(mu), sigma=float(s), lapse=float(lam), n=N_RUN)
    if variant == "lapse0.15":
        kw["lapse"] = 0.15
    elif variant == "learning":
        kw.update(sigma=float(s) * 1.25, sigma_end=float(s) * 0.75, drift_mu=0.1)
    elif variant == "order_bias":
        kw["order_bias"] = 0.15
    return kw


def run(method, task, obs, rng):
    n_easy = int(round(EASY_SHARE * N_RUN)) // 2 * 2
    kinds = np.r_[["s"] * (N_RUN - n_easy), ["e"] * n_easy]
    rng.shuffle(kinds)
    lo, hi = EASY[task]
    easy = list(np.log(1 + rng.permutation(np.r_[[lo] * (n_easy // 2), [hi] * (n_easy // 2)])))
    orders = rng.integers(1, 3, N_RUN)
    xs, ys = [], []
    if method.startswith("q_"):
        q = qs.QuestPlus(x_grid=X_GRID, target="mu" if method == "q_mu" else "mu_sigma")
        for t, k in enumerate(kinds):
            x = easy.pop() if k == "e" else q.next_x()
            y = obs.answer(x, t, orders[t], rng)
            q.update(x, y)
            xs.append(x); ys.append(y)
    else:
        stairs = [Staircase(r, FLOOR, CEIL, start=s) for r, s in STAIRS[method]]
        n_s = N_RUN - n_easy
        which = list(rng.permutation(np.arange(n_s) % len(stairs)))
        for t, k in enumerate(kinds):
            if k == "e":
                x = easy.pop(); y = obs.answer(x, t, orders[t], rng)
            else:
                st = stairs[which.pop()]
                x = float(np.log(1 + st.level)); y = obs.answer(x, t, orders[t], rng); st.update(y)
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys, float)


def one(args):
    task, variant, rep, seed = args
    rng = np.random.default_rng(seed)
    kw = observer(task, rng, variant)
    rows = []
    for mi, method in enumerate(METHODS):
        if task.startswith("xm") and method == "st4":
            continue
        if not task.startswith("xm") and method in ("stxm", "st11"):
            continue
        r = np.random.default_rng([seed, mi])
        xs, ys = run(method, task, qs.Observer(**kw), r)
        for b in BUDGETS:
            mu, sig = qs.fit_ml(xs[:b], ys[:b])
            rows.append(dict(task=task, variant=variant, rep=rep, method=method, budget=b, true_mu=kw["mu"],
                             true_sigma=kw["sigma"] if variant != "learning" else kw["sigma"] * (0.75 / 1.25) ** 0.5,
                             lapse=kw["lapse"], mu_hat=mu, sigma_hat=sig,
                             near=float(np.mean(np.abs(xs[:b] - kw["mu"]) < kw["sigma"]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=250)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(2610)
    jobs = []
    for task in EASY:
        for variant in ("base", "lapse0.15", "learning", "order_bias"):
            if variant != "base" and task not in ("aud_high", "xm_low"):
                continue
            n = args.reps if variant == "base" else args.reps // 2
            jobs += [(task, variant, r, int(rng.integers(2**31))) for r in range(n)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(one, jobs, chunksize=2) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adaptive_raw.csv", index=False)
    print(f"{len(jobs)} observers in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
