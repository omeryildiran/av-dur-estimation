"""Bimodal recovery with the visual sigma fixed from the unimodal visual block.

    python analysis/overnight_sims/fixed_sv.py --workers 2 --hours 5

Same simulated participants and answers as bimodal_recovery.py (same seeds, so the same data),
for the designs 9c_60, 9c_90 and 7c_90. sigma_V is not fitted but set to the true value times
exp(N(0, 0.25)), the unimodal visual block's error (QUEST+, 70 trials: log RMSE about 0.26).
"""

import argparse
import json
import logging
import os
import time
import warnings
from multiprocessing import get_context

import numpy as np

import bimodal_recovery as B
import avmodels as M

DESIGNS = [("9c", 60, None), ("9c", 90, None), ("7c", 90, None)]
SV_ERR = 0.25
IV = 2                                                     # sigma_V in every model's parameter vector


def bads_fixed(model, data, sv, rng, extra_x0=(), n_starts=3):
    from pybads import BADS
    keep = [i for i in range(len(B.LB[model])) if i != IV]
    lb, ub, plb, pub = (np.array(d[model], float)[keep] for d in (B.LB, B.UB, B.PLB, B.PUB))
    full = lambda x: np.insert(np.asarray(x, float).ravel(), IV, sv)  # noqa: E731
    f = lambda x: M.nll(model, full(x), data)  # noqa: E731
    starts = [plb + rng.random(len(lb)) * (pub - plb) for _ in range(n_starts)] + [np.clip(x, lb + 1e-6, ub - 1e-6) for x in extra_x0]
    best = None
    for x0 in starts:
        try:
            r = BADS(f, x0, lb, ub, plb, pub, options=dict(display="off", uncertainty_handling=False, max_fun_evals=500)).optimize()
            x, fv = np.asarray(r["x"], float).ravel(), float(r["fval"])
        except Exception:
            continue
        if best is None or fv < best[1]:
            best = (x, fv)
    return full(best[0]), best[1]


def one(args):
    gen, it, seed = args
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    rng = np.random.default_rng(seed)                     # as bimodal_recovery.one: same truth, same answers
    truth = B.sample_truth(gen, rng)
    runs = {}
    for base in ("9c", "7c"):
        for ni in B.NOISE:
            for c in B.CONFLICTS[base]:
                runs[(base, ni, float(c))] = B.run_condition(gen, truth, ni, c, rng)
    r2 = np.random.default_rng([seed, 7])
    sv = float(np.clip(truth[2] * np.exp(r2.normal(0, SV_ERR)), 0.05, 1.9))
    t0 = time.time()
    res = dict(gen=gen, iter=it, seed=seed, truth=truth, sv_fixed=sv, designs={})
    for base, n, _ in DESIGNS:
        ni_all, c_all, t_all, y_all = [], [], [], []
        for ni in B.NOISE:
            for c in B.CONFLICTS[base]:
                t, y, _ = runs[(base, ni, float(c))]
                ni_all.append(np.full(n, ni)); c_all.append(np.full(n, c)); t_all.append(t[:n]); y_all.append(y[:n])
        data = M.Data(np.concatenate(ni_all), np.concatenate(c_all), np.concatenate(t_all), np.concatenate(y_all))
        fits = {}
        fus = bads_fixed("fusion", data, sv, r2)
        fits["fusion"] = fus
        lam, sa1, _, sa2 = fus[0]
        fits["ci"] = bads_fixed("ci", data, sv, r2, extra_x0=[np.r_[lam, sa1, 0.999, sa2]])
        fits["switch"] = bads_fixed("switch", data, sv, r2)
        out = {}
        for m, (x, fv) in fits.items():
            k = len(x) - 1
            out[m] = dict(params=[float(v) for v in x], nll=fv, k=k, aic=2 * k + 2 * fv, bic=k * np.log(data.n_trials) + 2 * fv)
        res["designs"][f"{base}_{n}"] = dict(n_trials=data.n_trials, fits=out)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=80)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hours", type=float, default=5)
    args = ap.parse_args()
    path = B.OUT / "fixed_sv.json"
    results = json.loads(path.read_text()) if path.exists() else []
    done = {(r["gen"], r["iter"]) for r in results}
    seeds = np.random.default_rng(20261028).integers(2**31, size=(len(B.MODELS), 10000))
    jobs = [(g, i, int(seeds[k, i])) for i in range(args.n_iter) for k, g in enumerate(B.MODELS) if (g, i) not in done]
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
