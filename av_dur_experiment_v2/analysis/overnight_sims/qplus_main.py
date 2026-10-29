"""Main (bimodal) experiment: staircases vs QUEST+ in every condition, at 72 trials per condition.

    python analysis/overnight_sims/qplus_main.py --workers 8 --hours 2

9 conflicts (-450 ... +450 ms) x 2 noise levels, 72 trials per condition (64 adaptive + 8 easy at
the floor and ceiling, randomly interleaved). Per condition, one of:
  stair  2D1U from the floor + 1D2U from the ceiling, 32 trials each (the current main task)
  q_ms   QUEST+ targeting PSE and slope (lapse as nuisance), as the unimodal blocks
  q_mu   QUEST+ targeting the PSE only
QUEST+ runs in ln(test / standard) on whole-frame test durations 25-950 ms, with a flat prior over
PSE ln-ratio -2.5 ... 1 (the PSE at -450 ms can be near ln 0.3), sigma 0.05-2.5, lapse 0-0.1.

Same simulated participants as bimodal_recovery.py (same seeds and truths); each method gets its
own answers. Fits as fixed_sv.py: sigma_V fixed at the true value x exp(N(0, 0.2)), standing in for
the joint fit (one sigma_V shared with the unimodal and cross-modal blocks); sigma_A, lapse and the
model's own parameters free. Per condition the PSE is also fitted (lognormal PF).
"""

import argparse
import json
import logging
import os
import time
import warnings
from multiprocessing import get_context

import numpy as np

import avmodels as M
import bimodal_recovery as B
from fixed_sv import bads_fixed

N_COND = 72
N_EASY = 8
SV_ERR = 0.2
METHODS = ("stair", "q_ms", "q_mu")
CONFLICTS = B.CONFLICTS["9c"]
Q_T = np.unique(B.frames(np.geomspace(0.025, 0.95, 70)))     # candidate tests (s), whole frames


def new_quest(target):
    import quest_simulation as qs
    return qs.QuestPlus(x_grid=np.log(Q_T / B.STD), mu_grid=np.linspace(-2.5, 1.0, 106),
                        sigma_grid=np.exp(np.linspace(np.log(0.05), np.log(2.5), 22)),
                        lapse_grid=np.linspace(0, 0.1, 4), target=target)


def run_condition(method, gen, truth, ni, c, rng):
    from avdur.staircase import Staircase
    p_tab = M.p_longer(gen, truth, ni, B.STD, B.STD + c, B.LEVELS)
    p_at = lambda t: float(np.interp(t, B.LEVELS, p_tab))  # noqa: E731
    kinds = np.r_[["s"] * (N_COND - N_EASY), ["e"] * N_EASY]
    rng.shuffle(kinds)
    easy = list(rng.permutation(np.r_[[B.frames(B.STD * (1 + B.FLOOR))] * (N_EASY // 2), [B.frames(B.STD * (1 + B.CEIL))] * (N_EASY // 2)]))
    if method == "stair":
        stairs = [Staircase("2D1U", B.FLOOR, B.CEIL, start="floor"), Staircase("1D2U", B.FLOOR, B.CEIL, start="ceiling")]
        which = list(rng.permutation(np.arange(N_COND - N_EASY) % 2))
    else:
        q = new_quest("mu" if method == "q_mu" else "mu_sigma")
    tests, ys = [], []
    for k in kinds:
        if k == "e":
            t = float(easy.pop())
        elif method == "stair":
            st = stairs[which.pop()]
            t = max(float(B.frames(B.STD * (1 + st.level))), 1 / B.FR)
        else:
            t = float(B.STD * np.exp(q.next_x()))
        y = bool(rng.random() < p_at(t))
        if method != "stair":
            q.update(np.log(t / B.STD), y)                         # easy trials inform QUEST+ too, as in the unimodal blocks
        elif k != "e":
            st.update(y)
        tests.append(t); ys.append(y)
    return np.array(tests), np.array(ys, float), p_tab


def one(args):
    gen, it, seed = args
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    rng = np.random.default_rng(seed)
    truth = B.sample_truth(gen, rng)                              # same truth as bimodal_recovery / fixed_sv
    r2 = np.random.default_rng([seed, 11])
    sv = float(np.clip(truth[2] * np.exp(r2.normal(0, SV_ERR)), 0.05, 1.9))
    t0 = time.time()
    res = dict(gen=gen, iter=it, seed=seed, truth=truth, sv_fixed=sv, methods={})
    for mi, method in enumerate(METHODS):
        rr = np.random.default_rng([seed, 100 + mi])
        ni_all, c_all, t_all, y_all, pse = [], [], [], [], []
        for ni in B.NOISE:
            for c in CONFLICTS:
                t, y, p_tab = run_condition(method, gen, truth, ni, float(c), rr)
                ni_all.append(np.full(N_COND, ni)); c_all.append(np.full(N_COND, c)); t_all.append(t); y_all.append(y)
                ph, sh = B.fit_pf(t, y)
                pse.append(dict(noise=ni, conflict=float(c), pse_true=B.true_pse(p_tab, truth[0]), pse_hat=ph, sigma_hat=sh))
        data = M.Data(np.concatenate(ni_all), np.concatenate(c_all), np.concatenate(t_all), np.concatenate(y_all))
        fits = {"fusion": bads_fixed("fusion", data, sv, rr)}
        lam, sa1, _, sa2 = fits["fusion"][0]
        fits["ci"] = bads_fixed("ci", data, sv, rr, extra_x0=[np.r_[lam, sa1, 0.999, sa2]])
        fits["switch"] = bads_fixed("switch", data, sv, rr)
        out = {}
        for m, (x, fv) in fits.items():
            k = len(x) - 1
            out[m] = dict(params=[float(v) for v in x], nll=fv, k=k, aic=2 * k + 2 * fv, bic=k * np.log(data.n_trials) + 2 * fv)
        res["methods"][method] = dict(n_trials=data.n_trials, fits=out, pse=pse)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    path = B.OUT / ("qplus_main_pilot.json" if args.pilot else "qplus_main.json")
    results = [] if args.pilot or not path.exists() else json.loads(path.read_text())
    done = {(r["gen"], r["iter"]) for r in results}
    seeds = np.random.default_rng(20261028).integers(2**31, size=(len(B.MODELS), 10000))
    n_iter = 1 if args.pilot else args.n_iter
    jobs = [(g, i, int(seeds[k, i])) for i in range(n_iter) for k, g in enumerate(B.MODELS) if (g, i) not in done]
    print(f"{len(jobs)} participants, {args.workers} workers", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs), 1):
            results.append(r)
            tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(results)); os.replace(tmp, path)
            print(f"  {k}/{len(jobs)} {r['gen']} #{r['iter']} ({r['seconds']:.0f} s; {(time.time() - t0) / 60:.0f} min)", flush=True)
            if time.time() - t0 > args.hours * 3600:
                print("time limit reached; stopping", flush=True)
                pool.terminate()
                break


if __name__ == "__main__":
    main()
