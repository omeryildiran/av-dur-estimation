"""Main (bimodal) experiment: how many trials per condition, and 9 conflicts (+-450 ms) vs 7 (+-250 ms)?

    python analysis/overnight_sims/bimodal_recovery.py --pilot      # timing
    python analysis/overnight_sims/bimodal_recovery.py --hours 7     # overnight (resumable)

Simulated participants answer the main task as it runs: per condition (noise x conflict) a
2D1U staircase from the floor and a 1D2U from the ceiling (avdur.staircase, main-task floor
-95%, ceiling +90%), plus easy trials at 25 and 950 ms, randomly interleaved, test durations
in whole frames (120 Hz). Each condition runs 200 trials; a smaller budget is the first n
trials of the same run (paired comparison, as stopping the session early).

Designs, all on the same simulated participant (same parameters, new answers per design):
  9c    9 conflicts -450 ... +450 ms (112.5 ms steps), at 60 / 90 / 120 / 154 trials per condition
  9c_w  9 conflicts, same total as 9c at 90, but trials moved from the small conflicts to the
        large ones (0 and +-112: x0.7, +-225: x1.0, +-338 and +-450: x1.2)
  7c    7 conflicts -250 ... +250 ms (the current design), at 90 and 154 trials per condition

Each data set is fitted by causal inference (CI), forced fusion and switching (avmodels: exact
quadrature likelihood, same models as monteCarloClass) with BADS: 3 starts each, CI also from
the fusion solution with p_c near 1 (fusion is CI at p_c = 1, so CI can never fit worse).
Selection by AIC and BIC. Per condition the PSE is also fitted (lognormal psychometric
function), to see how precisely the PSE-vs-conflict curve is measured.

No audiovisual bias is simulated (as if it were modelled exactly, see the report).
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import avmodels as M  # noqa: E402

OUT = HERE / "results"
STD = 0.5
FR = 120.0
NOISE = (0, 1)
FLOOR, CEIL = -0.95, 0.9
N_RUN = 200                                           # trials simulated per condition
EASY_SHARE = 14 / 154                                 # as the current design


def frames(s):
    return np.round(np.asarray(s) * FR) / FR


CONFLICTS = {"9c": frames(np.linspace(-0.45, 0.45, 9)), "7c": frames(np.linspace(-0.25, 0.25, 7))}
W9 = {0: 0.7, 1: 0.7, 2: 1.0, 3: 1.2, 4: 1.2}       # by |conflict| rank (0, 112, 225, 338, 450)
DESIGNS = [("9c", 60, None), ("9c", 90, None), ("9c", 120, None), ("9c", 154, None),
           ("9c", 90, W9), ("7c", 90, None), ("7c", 154, None)]
MODELS = ("ci", "fusion", "switch")
LB = {"ci": [0.0, 0.03, 0.03, 0.0, 0.03], "fusion": [0.0, 0.03, 0.03, 0.03],
      "switch": [0.0, 0.03, 0.03, 0.0, 0.03, 0.0]}
UB = {"ci": [0.25, 2.0, 2.0, 1.0, 2.0], "fusion": [0.25, 2.0, 2.0, 2.0], "switch": [0.25, 2.0, 2.0, 1.0, 2.0, 1.0]}
PLB = {"ci": [0.01, 0.1, 0.2, 0.1, 0.4], "fusion": [0.01, 0.1, 0.2, 0.4],
       "switch": [0.01, 0.1, 0.2, 0.1, 0.4, 0.1]}
PUB = {"ci": [0.1, 0.5, 1.0, 0.9, 1.2], "fusion": [0.1, 0.5, 1.0, 1.2], "switch": [0.1, 0.5, 1.0, 0.9, 1.2, 0.9]}


def sample_truth(gen, rng):
    lam = rng.uniform(0.001, 0.10)
    sa1, sa2 = rng.uniform(0.12, 0.45), rng.uniform(0.55, 1.0)
    sv = rng.uniform(0.30, 1.0)
    if gen == "fusion":
        return [lam, sa1, sv, sa2]
    if gen == "ci":
        return [lam, sa1, sv, rng.uniform(0.1, 0.9), sa2]
    return [lam, sa1, sv, rng.uniform(0.1, 0.9), sa2, rng.uniform(0.1, 0.9)]


LEVELS = frames(STD * (1 + np.linspace(FLOOR, CEIL, 400)))
LEVELS = np.unique(LEVELS[LEVELS > 0])


def run_condition(gen, truth, ni, c, rng):
    """200 trials of one condition: (test_s, chose_test) in the order they were run."""
    from avdur.staircase import Staircase
    p_tab = M.p_longer(gen, truth, ni, STD, STD + c, LEVELS)
    n_easy = int(round(EASY_SHARE * N_RUN)) // 2 * 2
    n_st = N_RUN - n_easy
    kinds = np.r_[["a"] * (n_st // 2), ["b"] * (n_st - n_st // 2), ["e"] * n_easy]
    rng.shuffle(kinds)
    easy = list(rng.permutation(np.r_[[frames(STD * (1 + FLOOR))] * (n_easy // 2), [frames(STD * (1 + CEIL))] * (n_easy // 2)]))
    stairs = {"a": Staircase("2D1U", FLOOR, CEIL, start="floor"), "b": Staircase("1D2U", FLOOR, CEIL, start="ceiling")}
    tests, ys = [], []
    for k in kinds:
        t = easy.pop() if k == "e" else float(frames(STD * (1 + stairs[k].level)))
        t = max(t, 1 / FR)
        p = np.interp(t, LEVELS, p_tab)
        y = bool(rng.random() < p)
        if k != "e":
            stairs[k].update(y)
        tests.append(t); ys.append(y)
    return np.array(tests), np.array(ys, float), p_tab


def true_pse(p_tab, lam):
    """Test duration (s) where P = 0.5 (lapse is symmetric, so the same with or without it)."""
    i = np.searchsorted(p_tab, 0.5)
    if i == 0 or i >= len(p_tab):
        return float("nan")
    return float(np.interp(0.5, p_tab[i - 1:i + 1], LEVELS[i - 1:i + 1]))


def fit_pf(t, y):
    """Per-condition lognormal PF (as the experiment's screening): PSE (s)."""
    from quest_simulation import fit_ml
    mu, sig = fit_ml(np.log(t / STD), y)
    return float(STD * np.exp(mu)), float(sig)


def bads_fit(model, data, rng, extra_x0=(), n_starts=3):
    from pybads import BADS
    lb, ub, plb, pub = (np.array(d[model], float) for d in (LB, UB, PLB, PUB))
    f = lambda x: M.nll(model, np.asarray(x, float).ravel(), data)  # noqa: E731
    starts = [plb + rng.random(len(lb)) * (pub - plb) for _ in range(n_starts)] + [np.clip(x, lb + 1e-6, ub - 1e-6) for x in extra_x0]
    best = None
    for x0 in starts:
        try:
            r = BADS(f, x0, lb, ub, plb, pub, options=dict(display="off", uncertainty_handling=False,
                                                          max_fun_evals=600)).optimize()
            x, fv = np.asarray(r["x"], float).ravel(), float(r["fval"])
        except Exception:
            continue
        if best is None or fv < best[1]:
            best = (x, fv)
    return best


def fit_models(data, rng):
    out = {}
    fus = bads_fit("fusion", data, rng)
    out["fusion"] = fus
    lam, sa1, sv, sa2 = fus[0]
    out["ci"] = bads_fit("ci", data, rng, extra_x0=[np.r_[lam, sa1, sv, 0.999, sa2]])
    out["switch"] = bads_fit("switch", data, rng)
    res = {}
    for m, (x, fv) in out.items():
        k = len(x)
        res[m] = dict(params=[float(v) for v in x], nll=fv, k=k, aic=2 * k + 2 * fv, bic=k * np.log(data.n_trials) + 2 * fv)
    return res


def one(args):
    gen, it, seed = args
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    rng = np.random.default_rng(seed)
    truth = sample_truth(gen, rng)
    t0 = time.time()
    res = dict(gen=gen, iter=it, seed=seed, truth=truth, designs={})
    runs = {}
    for base in ("9c", "7c"):
        for ni in NOISE:
            for c in CONFLICTS[base]:
                runs[(base, ni, float(c))] = run_condition(gen, truth, ni, c, rng)
    for base, n, w in DESIGNS:
        name = f"{base}{'_w' if w else ''}_{n}"
        cs = CONFLICTS[base]
        rank = {float(c): int(r) for c, r in zip(cs, np.abs(np.round(cs / (cs[1] - cs[0]))).astype(int))}
        ni_all, c_all, t_all, y_all, pse = [], [], [], [], []
        for ni in NOISE:
            for c in cs:
                t, y, p_tab = runs[(base, ni, float(c))]
                m = int(round(n * (w[rank[float(c)]] if w else 1)))
                t, y = t[:m], y[:m]
                ni_all.append(np.full(m, ni)); c_all.append(np.full(m, c)); t_all.append(t); y_all.append(y)
                pse_hat, sig_hat = fit_pf(t, y)
                pse.append(dict(noise=ni, conflict=float(c), n=m, pse_true=true_pse(p_tab, truth[0]),
                                pse_hat=pse_hat, sigma_hat=sig_hat))
        data = M.Data(np.concatenate(ni_all), np.concatenate(c_all), np.concatenate(t_all), np.concatenate(y_all))
        res["designs"][name] = dict(n_trials=data.n_trials, fits=fit_models(data, rng), pse=pse)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=60, help="simulated participants per generating model")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--hours", type=float, default=None, help="stop handing out new jobs after this long")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tag = "_pilot" if args.pilot else ""
    path = OUT / f"bimodal_recovery{tag}.json"
    results = json.loads(path.read_text()) if path.exists() and not args.pilot else []
    done = {(r["gen"], r["iter"]) for r in results}
    seeds = np.random.default_rng(20261028).integers(2**31, size=(len(MODELS), 10000))
    n_iter = 1 if args.pilot else args.n_iter
    jobs = [(g, i, int(seeds[k, i])) for i in range(n_iter) for k, g in enumerate(MODELS) if (g, i) not in done]
    print(f"{len(jobs)} participants to simulate and fit, {args.workers} workers", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        it = pool.imap_unordered(one, jobs)
        for k, r in enumerate(it, 1):
            results.append(r)
            tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(results)); os.replace(tmp, path)
            print(f"  {k}/{len(jobs)} {r['gen']} #{r['iter']} ({r['seconds']:.0f} s; {(time.time() - t0) / 60:.0f} min)", flush=True)
            if args.hours and time.time() - t0 > args.hours * 3600:
                print("time limit reached; stopping", flush=True)
                pool.terminate()
                break


if __name__ == "__main__":
    main()
