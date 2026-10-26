#!/usr/bin/env python3
"""
Model recovery at 154 / 120 / 100 trials per condition (v2 main-experiment design)
==================================================================================
Question: can the main experiment be shortened from 154 trials per condition without
losing the ability to tell causal inference (lognorm), forced fusion (fusionOnlyLogNorm)
and cue switching (switchingFree) apart?

Design (as the v2 main experiment):
  - 2 noise levels x 7 conflicts (-250 ... +250 ms), standard 500 ms.
  - 9 test durations per condition, -80% ... +80% of the standard (constant stimuli, as the
    repo's other recovery scripts; the staircases put more trials near the PSE).
  - 17 / 13 / 11 trials per test duration = 153 / 117 / 99 trials per condition. The smaller
    data sets are the first 13 / 11 trials of each cell of the same simulated participant, so
    the comparison between trial counts is paired.

Simulated participants (v2-like; noise label 1.2 = the participant's calibrated high noise):
  lapse U(0.001, 0.10), sigma_a low U(0.12, 0.45), sigma_a high U(0.55, 1.0),
  sigma_v U(0.30, 1.0), p_c U(0.1, 0.9), p_switch (each noise level) U(0.1, 0.9).

Unlike simulateMonteCarloData (which draws 3 x the template's trials per cell), the data here
have exactly the planned number of trials. Each data set is fitted by every model with
monteCarloClass (BADS, shared lapse, freeP_c False, as refit_three_models.sh); the final
log-likelihoods are evaluated with more Monte Carlo draws so that model selection is not
decided by simulation noise.

    python run_model_recovery_trials.py --pilot            # 1 participant per model, timing
    python run_model_recovery_trials.py --n_iter 40        # full run (resumable)

Results: model_recovery_trials/recovery_trials.json (+ summary printed).
Use the mathmod environment (pybads): ~/anaconda3/envs/mathmod/bin/python
"""

import argparse
import contextlib
import io
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

import monteCarloClass

MODELS = ["lognorm", "fusionOnlyLogNorm", "switchingFree"]
ABBR = {"lognorm": "CI", "fusionOnlyLogNorm": "Fusion", "switchingFree": "Switch"}
NOISE = (0.1, 1.2)
CONFLICTS = np.round(np.linspace(-0.25, 0.25, 7), 4)
STANDARD = 0.5
DELTAS = np.round(np.linspace(-0.8, 0.8, 9) * STANDARD, 5)
N_PER_CELL = {153: 17, 117: 13, 99: 11}          # trials per condition -> per test duration
OUT = "model_recovery_trials"


def sample_truth(gen, rng):
    lam = rng.uniform(0.001, 0.10)
    sa1, sa2 = rng.uniform(0.12, 0.45), rng.uniform(0.55, 1.0)
    sv = rng.uniform(0.30, 1.0)
    if gen == "fusionOnlyLogNorm":
        return [lam, sa1, sv, sa2]
    if gen == "lognorm":
        return [lam, sa1, sv, rng.uniform(0.1, 0.9), sa2]
    return [lam, sa1, sv, rng.uniform(0.1, 0.9), sa2, rng.uniform(0.1, 0.9)]


def template(n_per_cell):
    rows = []
    for noise in NOISE:
        for c in CONFLICTS:
            for d in DELTAS:
                t = STANDARD + d
                for rep in range(n_per_cell):
                    rows.append(dict(standardDur=STANDARD, testDurS=t, deltaDurS=d,
                                     logDeltaDur=np.log(t) - np.log(STANDARD),
                                     logDeltaDurMs=np.log(t * 1000) - np.log(STANDARD * 1000),
                                     audNoise=noise, conflictDur=c,
                                     unbiasedVisualStandardDur=STANDARD + c, unbiasedVisualTestDur=t,
                                     chose_test=0, chose_standard=1, responses=1, rep=rep))
    return pd.DataFrame(rows)


def _model(data, name, n_simul, n_starts):
    with contextlib.redirect_stdout(io.StringIO()):
        mc = monteCarloClass.OmerMonteCarlo(data)
    mc.modelName, mc.freeP_c, mc.sharedLambda = name, False, True
    mc.nSimul, mc.nStart, mc.optimizationMethod = n_simul, n_starts, "bads"
    return mc


def simulate(gen, params, n_full, rng, n_simul_gen=20000):
    """Exactly n_full trials per (noise, conflict, test) cell, drawn from the generating model."""
    data = template(n_full)
    mc = _model(data, gen, n_simul_gen, 1)
    p_of = {}
    for _, cell in mc.groupedData.iterrows():
        noise, c, d = cell["audNoise"], cell["conflictDur"], cell["deltaDurS"]
        pr = mc.getParamsCausal(np.asarray(params), noise, c)
        lam, sa, sv, pc, tmin, tmax = pr[:6]
        stims = (STANDARD, STANDARD + d, STANDARD + c, STANDARD + d)
        p_of[(noise, c, d)] = float(mc.probTestLonger_vectorized_mc(stims, sa, sv, pc, lam, tmin, tmax))
    p = np.array([p_of[(r.audNoise, r.conflictDur, r.deltaDurS)] for r in data.itertuples()])
    data["chose_test"] = (rng.random(len(data)) < p).astype(int)
    data["chose_standard"] = 1 - data["chose_test"]
    return data


def fit_all(data, n_simul_fit, n_simul_eval, n_starts):
    out = {}
    for m in MODELS:
        try:
            mc = _model(data, m, n_simul_fit, n_starts)
            with contextlib.redirect_stdout(io.StringIO()):
                fp = mc.fitCausalInferenceMonteCarlo(mc.groupedData)
                mc.nSimul = n_simul_eval
                nll = float(mc.nLLMonteCarloCausal(fp, mc.groupedData))
            k = len(fp)
            out[m] = dict(params=[float(v) for v in fp], nLL=nll, k=k,
                          AIC=2 * k + 2 * nll, BIC=k * np.log(len(data)) + 2 * nll)
        except Exception as err:                          # a failed fit counts as missing
            out[m] = dict(error=str(err)[:200])
    return out


def one(args):
    gen, it, seed, n_simul_fit, n_simul_eval, n_starts = args
    rng = np.random.default_rng(seed)
    np.random.seed(seed % 2**32)                      # monteCarloClass draws from the global generator
    truth = sample_truth(gen, rng)
    t0 = time.time()
    full = simulate(gen, truth, max(N_PER_CELL.values()), rng)
    res = dict(gen=gen, iter=it, seed=seed, truth=truth, fits={})
    for n_cond, n_cell in sorted(N_PER_CELL.items(), reverse=True):
        res["fits"][str(n_cond)] = fit_all(full[full.rep < n_cell].reset_index(drop=True), n_simul_fit,
                                           n_simul_eval, n_starts)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summary(results):
    lines = []
    for n_cond in sorted(N_PER_CELL, reverse=True):
        lines.append(f"\n{n_cond} trials per condition (rows: generating model; columns: chosen by AIC)")
        lines.append(f"{'':>8} " + " ".join(f"{ABBR[m]:>7}" for m in MODELS) + "   correct")
        for gen in MODELS:
            rs = [r for r in results if r["gen"] == gen]
            counts = {m: 0 for m in MODELS}
            n_ok = 0
            for r in rs:
                f = r["fits"][str(n_cond)]
                ok = {m: v["AIC"] for m, v in f.items() if "AIC" in v}
                if len(ok) == len(MODELS):
                    counts[min(ok, key=ok.get)] += 1
                    n_ok += 1
            pct = 100 * counts[gen] / n_ok if n_ok else float("nan")
            lines.append(f"{ABBR[gen]:>8} " + " ".join(f"{counts[m]:>7}" for m in MODELS) + f"   {pct:5.0f}% of {n_ok}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_iter", type=int, default=40, help="simulated participants per generating model")
    ap.add_argument("--nSimul", type=int, default=500, help="Monte Carlo draws per likelihood during fitting")
    ap.add_argument("--nSimul_eval", type=int, default=5000, help="draws for the final log-likelihood")
    ap.add_argument("--nStarts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.pilot:
        args.n_iter, tag = 1, "_pilot"
    else:
        tag = ""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"recovery_trials{tag}.json")
    results = json.load(open(path)) if os.path.exists(path) and not args.pilot else []
    done = {(r["gen"], r["iter"]) for r in results}
    seeds = np.random.default_rng(20261026).integers(2**31, size=(len(MODELS), 10000))
    jobs = [(g, i, int(seeds[k, i]), args.nSimul, args.nSimul_eval, args.nStarts)
            for i in range(args.n_iter) for k, g in enumerate(MODELS) if (g, i) not in done]
    print(f"{len(jobs)} simulated participants to fit ({len(done)} already done), {args.workers} workers", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs), 1):
            results.append(r)
            with open(path + ".tmp", "w") as f:
                json.dump(results, f)
            os.replace(path + ".tmp", path)
            print(f"  {k}/{len(jobs)} {ABBR[r['gen']]} #{r['iter']} ({r['seconds']:.0f} s; "
                  f"{(time.time() - t0) / 60:.0f} min so far)", flush=True)
    print(summary(results), flush=True)


if __name__ == "__main__":
    main()
