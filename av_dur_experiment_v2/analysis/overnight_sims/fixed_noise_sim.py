"""Fixed high noise instead of a calibration: how bad may a participant's high-noise auditory sigma be
before the main experiment stops telling the models apart?

    python analysis/overnight_sims/fixed_noise_sim.py --pilot
    python analysis/overnight_sims/fixed_noise_sim.py --n_iter 25 --workers 7    (resumable)

Simulated participants as in crossmodal_baseline.py (causal inference, fusion, switching; main experiment
as it runs from 29 Oct 2026: 9 conflicts x 2 noise levels, 2D1U + 1D2U x 50 and 10 easy trials per condition;
sigma_A and sigma_V known from the unimodal blocks with log error 0.2), without any audiovisual bias
(decided 31 Oct 2026), except that the high-noise auditory sigma is set on a grid. The grid is the
psychometric sigma the unimodal block reports (screening, fit plots); the models use the measurement
sigma, psychometric / sqrt(2). The low-noise sigma is drawn as before but kept below the high one.
Fits: fusion, CI (also started from fusion at p_c = 1) and switching, chosen by AIC.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import avmodels as M  # noqa: E402
import bias_recovery as BR  # noqa: E402
import bimodal_recovery as B  # noqa: E402
import crossmodal_baseline as XB  # noqa: E402

PSI_GRID = (0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0)   # psychometric sigma at high noise (unimodal block)
XB.BOUNDS["sa2"] = (0.03, 3.0, 0.2, 2.0)               # room for sigma_A up to 3.0 / sqrt(2) and above
ERR = XB.ERR


def one(args):
    try:
        return _one(*args)
    except Exception as err:
        return dict(gen=args[0], psi=args[1], iter=args[2], error=repr(err))


def _one(gen, psi, it, seed):
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    rng = np.random.default_rng(seed)
    truth = B.sample_truth(gen, rng)
    sa2 = psi / np.sqrt(2)
    truth[BR.SA2_INDEX[gen]] = sa2
    truth[1] = min(truth[1], 0.9 * sa2)                  # low noise never worse than high noise
    rd = np.random.default_rng([seed, 2])
    cols = [[], [], [], []]
    for ni in B.NOISE:
        for c in XB.CONFLICTS:
            t, y = XB.run_main_condition(gen, truth, None, ni, float(c), rd)
            for lst, v in zip(cols, (np.full(len(t), ni), np.full(len(t), float(c)), t, y)):
                lst.append(v)
    data = M.Data(*(np.concatenate(v) for v in cols))
    ru = np.random.default_rng([seed, 3])
    z = ru.normal(size=3)
    sv_hat = float(np.clip(truth[2] * np.exp(ERR * z[0]), 0.05, 1.9))
    sa_prior = (truth[1] * np.exp(ERR * z[1]), sa2 * np.exp(ERR * z[2]), ERR)
    t0 = time.time()
    rf = np.random.default_rng([seed, 4])
    fits = {}
    for model in XB.FIT_ORDER:
        f, names = XB.make_nll(model, data, sv_hat, sa_prior)
        x0s = []
        if model == "ci":
            fu = fits["fusion"]["params"]
            x0s.append([fu["lam"], fu["sa1"], 0.999, fu["sa2"]])
        prm, fv = XB.bads_fit(f, names, rf, x0s=x0s, n_random=3)
        fits[model] = dict(params=prm, nll=fv, aic=2 * len(prm) + 2 * fv)
    return dict(gen=gen, psi=psi, iter=it, seed=seed, truth=[float(v) for v in truth], fits=fits,
                seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=25)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    path = B.OUT / ("fixed_noise_sim_pilot.json" if args.pilot else "fixed_noise_sim.json")
    results = [] if args.pilot or not path.exists() else json.loads(path.read_text())
    done = {(r["gen"], r["psi"], r["iter"]) for r in results}
    seeds = np.random.default_rng(20261101).integers(2**31, size=(len(B.MODELS), len(PSI_GRID), 1000))
    n_iter = 1 if args.pilot else args.n_iter
    grid = (0.6, 3.0) if args.pilot else PSI_GRID
    jobs = [(g, psi, i, int(seeds[k, PSI_GRID.index(psi), i])) for i in range(n_iter) for psi in grid
            for k, g in enumerate(B.MODELS) if (g, psi, i) not in done]
    print(f"{len(jobs)} participants, {args.workers} workers", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs), 1):
            if "error" in r:
                print(f"  {k}/{len(jobs)} {r['gen']} {r['psi']} FAILED: {r['error']}", flush=True)
                continue
            results.append(r)
            tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(results)); os.replace(tmp, path)
            print(f"  {k}/{len(jobs)} {r['gen']} psi {r['psi']} ({r['seconds']:.0f} s; {(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
