"""Without the cross-modal block: can the audiovisual bias (c, d) be recovered from the main experiment?

    python analysis/overnight_sims/bias_recovery.py --workers 8 --hours 13

Bias model (avmodels.vis): ln v_hat = ln v - c[noise] - d ln(v / 0.5), in both intervals.
Under forced fusion c cancels exactly (it shifts the standard and the test alike); under causal
inference and switching it acts only through the perceived A-V discrepancy, so it is weakly
identified. d changes the visual weight and is better identified.

Simulated participants: CI, fusion or switching (ranges as bimodal_recovery.py) plus a bias
c_low U(0, 0.5), c_high U(-0.1, 0.4), d U(-0.3, 0.6) (P101, P102, P104 cross-modal: c 0.06-0.50,
d -0.23-0.60). The main experiment as proposed: 9 conflicts (-450 ... +450 ms) x 2 noise levels,
2D1U + 1D2U per condition plus easy trials (about 9%), run to 106 trials per condition and scored
at 70 (64 + 6 easy) and 106 (96 + 10 easy).

Fits as the joint fit would constrain them: the unimodal visual block measures sigma_V / (1 - d)
(the visual sigma on the biased scale), simulated as the true value x exp(N(0, 0.2)); sigma_V =
that x (1 - d). Each model is fitted (BADS) three ways: without a bias (c = d = 0), with d only
(c = 0; c barely changes the predictions), and with c_low, c_high, d free; the biased fits also
start from the no-bias solution, so they never fit worse.

--scenarios adds what the unimodal auditory block gives the joint fit (29 Oct 2026): sigma_A at each
noise level estimated as the true value x exp(N(0, err)) and added as a Gaussian penalty on
ln sigma_A (the unimodal likelihood, approximately); sigma_V is fixed with the same log error.
    python analysis/overnight_sims/bias_recovery.py --workers 7 --hours 8 --budgets 70 \
        --scenarios sa0.15:0.15 sa0.05:0.05 --out bias_recovery_sa
The data are the same simulated participants as the run without the penalty (same seeds and draws).
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

N_RUN, N_EASY = 106, 10
BUDGETS = (70, 106)
SV_ERR = 0.2
CONFLICTS = B.CONFLICTS["9c"]
# model parameters without sigma_V (fitted vector = these, then c_low, c_high, d when biased)
FREE = {"ci": ["lam", "sa1", "pc", "sa2"], "fusion": ["lam", "sa1", "sa2"], "switch": ["lam", "sa1", "ps1", "sa2", "ps2"]}
BOUNDS = dict(lam=(0, .25, .01, .1), sa1=(.03, 2, .1, .5), sa2=(.03, 2, .4, 1.2), pc=(0, 1, .1, .9), ps1=(0, 1, .1, .9),
              ps2=(0, 1, .1, .9), c1=(-.8, .8, -.1, .5), c2=(-.8, .8, -.1, .5), d=(-.8, .8, -.3, .5))


def full_params(model, named, sv):
    if model == "ci":
        return [named["lam"], named["sa1"], sv, named["pc"], named["sa2"]]
    if model == "fusion":
        return [named["lam"], named["sa1"], sv, named["sa2"]]
    return [named["lam"], named["sa1"], sv, named["ps1"], named["sa2"], named["ps2"]]


EXTRA = {"nobias": [], "d": ["d"], "bias": ["c1", "c2", "d"]}
SA2_INDEX = {"ci": 4, "fusion": 3, "switch": 4}         # sigma_A at high noise in the full parameter list


def make_nll(model, data, sv_eff, variant, sa_prior=None, fixed_c=None):
    """sa_prior: (sigma_A low, sigma_A high, log sd) from the unimodal auditory block, or None (sigma_A free).
    fixed_c: (c_low, c_high) used when c is not fitted (v1_bias_fits.py: the cross-modal values), else 0."""
    names = FREE[model] + EXTRA[variant]
    fc = fixed_c or (0.0, 0.0)

    def f(x):
        nm = dict(zip(names, np.asarray(x, float).ravel()))
        d = nm.get("d", 0.0)
        bias = (nm.get("c1", fc[0]), nm.get("c2", fc[1]), d) if variant != "nobias" or fixed_c else None
        out = M.nll(model, full_params(model, nm, sv_eff * (1 - d)), data, bias)
        if sa_prior is not None:
            out += 0.5 * sum((np.log(nm[k] / v) / sa_prior[2]) ** 2 for k, v in zip(("sa1", "sa2"), sa_prior[:2]))
        return out
    return f, names


def fit(model, data, sv_eff, variant, rng, x0s=(), sa_prior=None, fixed_c=None):
    from pybads import BADS
    f, names = make_nll(model, data, sv_eff, variant, sa_prior, fixed_c)
    lb, ub, plb, pub = (np.array([BOUNDS[n][i] for n in names], float) for i in range(4))
    starts = [plb + rng.random(len(lb)) * (pub - plb) for _ in range(2 if x0s else 3)] + [np.clip(x, lb + 1e-6, ub - 1e-6) for x in x0s]
    best = None
    for x0 in starts:
        try:
            r = BADS(f, x0, lb, ub, plb, pub, options=dict(display="off", uncertainty_handling=False, max_fun_evals=700)).optimize()
            x, fv = np.asarray(r["x"], float).ravel(), float(r["fval"])
        except Exception:
            continue
        if best is None or fv < best[1]:
            best = (x, fv)
    return dict(zip(names, map(float, best[0]))), best[1]


def run_condition(gen, truth, bias, ni, c, rng):
    from avdur.staircase import Staircase
    p_tab = M.p_longer(gen, truth, ni, B.STD, B.STD + c, B.LEVELS, bias)
    kinds = np.r_[["a"] * ((N_RUN - N_EASY) // 2), ["b"] * ((N_RUN - N_EASY) // 2), ["e"] * N_EASY]
    rng.shuffle(kinds)
    easy = list(rng.permutation(np.r_[[B.frames(B.STD * (1 + B.FLOOR))] * (N_EASY // 2), [B.frames(B.STD * (1 + B.CEIL))] * (N_EASY // 2)]))
    stairs = {"a": Staircase("2D1U", B.FLOOR, B.CEIL, start="floor"), "b": Staircase("1D2U", B.FLOOR, B.CEIL, start="ceiling")}
    tests, ys = [], []
    for k in kinds:
        t = float(easy.pop()) if k == "e" else max(float(B.frames(B.STD * (1 + stairs[k].level))), 1 / B.FR)
        y = bool(rng.random() < np.interp(t, B.LEVELS, p_tab))
        if k != "e":
            stairs[k].update(y)
        tests.append(t); ys.append(y)
    return np.array(tests), np.array(ys, float), p_tab


def one(args):
    gen, it, seed, cfg = args
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    rng = np.random.default_rng(seed)
    truth = B.sample_truth(gen, rng)                      # sigma_V here is the unimodal (biased-scale) sigma
    rb = np.random.default_rng([seed, 21])
    bias = (float(rb.uniform(0, .5)), float(rb.uniform(-.1, .4)), float(rb.uniform(-.3, .6)))
    sv_eff = truth[2]
    truth_m = list(truth); truth_m[2] = sv_eff * (1 - bias[2])  # the model's sigma_V
    z_v = rb.normal()                                     # the same draw as before 29 Oct (normal(0, 0.2) = 0.2 z)
    runs = {(ni, float(c)): run_condition(gen, truth_m, bias, ni, float(c), rb) for ni in B.NOISE for c in CONFLICTS}
    z_a = np.random.default_rng([seed, 22]).normal(size=2)
    sa_true = (truth_m[1], truth_m[SA2_INDEX[gen]])
    t0 = time.time()
    res = dict(gen=gen, iter=it, seed=seed, truth=truth_m, sv_eff=sv_eff, bias=bias)
    for name, sa_err, sv_err in cfg["scenarios"]:
        sv_hat = float(np.clip(sv_eff * np.exp(sv_err * z_v), 0.05, 1.9))
        sa_prior = None if sa_err is None else tuple(float(v * np.exp(sa_err * z)) for v, z in zip(sa_true, z_a)) + (sa_err,)
        out = dict(sv_hat=sv_hat, sa_prior=sa_prior, budgets={})
        for n in cfg["budgets"]:
            cols = [[], [], [], []]
            for (ni, c), (t, y, _) in runs.items():
                for lst, v in zip(cols, (np.full(n, ni), np.full(n, c), t[:n], y[:n])):
                    lst.append(v)
            data = M.Data(*(np.concatenate(v) for v in cols))
            fits = {}
            for model in B.MODELS:
                fits[model] = {}
                base = None
                for variant in ("nobias", "d", "bias"):
                    x0s = [] if base is None else [np.r_[[base[k] for k in FREE[model]], np.zeros(len(EXTRA[variant]))]]
                    prm, fv = fit(model, data, sv_hat, variant, rb, x0s=x0s, sa_prior=sa_prior)
                    base = base or prm
                    k = len(prm)
                    fits[model][variant] = dict(params=prm, nll=fv, aic=2 * k + 2 * fv, bic=k * np.log(data.n_trials) + 2 * fv)
            out["budgets"][str(n)] = dict(n_trials=data.n_trials, fits=fits)
        if name is None:                                  # the original run's layout
            res.update(sv_hat=sv_hat, budgets=out["budgets"])
        else:
            res.setdefault("scenarios", {})[name] = out
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hours", type=float, default=13)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--scenarios", nargs="+", default=[],
                    help="name:err, e.g. sa0.15:0.15 (sigma_A penalised and sigma_V fixed, both with log error err)")
    ap.add_argument("--out", default="bias_recovery")
    args = ap.parse_args()
    scenarios = [(nm, float(e), float(e)) for nm, e in (s.split(":") for s in args.scenarios)] or [(None, None, SV_ERR)]
    cfg = dict(budgets=args.budgets, scenarios=scenarios)
    path = B.OUT / (f"{args.out}_pilot.json" if args.pilot else f"{args.out}.json")
    results = [] if args.pilot or not path.exists() else json.loads(path.read_text())
    done = {(r["gen"], r["iter"]) for r in results}
    seeds = np.random.default_rng(20261029).integers(2**31, size=(len(B.MODELS), 10000))
    n_iter = 1 if args.pilot else args.n_iter
    jobs = [(g, i, int(seeds[k, i]), cfg) for i in range(n_iter) for k, g in enumerate(B.MODELS) if (g, i) not in done]
    print(f"{len(jobs)} participants, {args.workers} workers, {cfg}", flush=True)
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
