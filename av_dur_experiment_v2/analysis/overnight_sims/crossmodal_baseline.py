"""Is a separate cross-modal block worth it for the bimodal model fits, or can the audiovisual bias be ignored?

    python analysis/overnight_sims/crossmodal_baseline.py --pilot                   # timing, 1 participant per model
    python analysis/overnight_sims/crossmodal_baseline.py --workers 7 --hours 6.5   # overnight (resumable)

Simulated participants: causal inference, fusion or switching (parameters as bimodal_recovery.py) with
an audiovisual bias ln v_hat = ln v - c[noise] - d ln(v / 0.5) (avmodels.vis), drawn from the measured
cross-modal biases:
  c_low ~ N(0.26, 0.26), c_high ~ N(0.17, 0.30), correlation 0.8 (v1 cross-modal, 11 participants);
  d ~ N(0.25, 0.20) (v2 pilot with 300/500/800 ms standards, P101, P102, P104: the spread of the
  estimates without their standard errors); c in [-0.5, 1.0], d in [-0.3, 0.7] (redrawn outside).
bias_recovery.py drew narrower biases (c U(0, 0.5) and U(-0.1, 0.4), d U(-0.3, 0.6)).

Main experiment as it runs from 29 Oct 2026: 9 conflicts x 2 noise levels, per condition a 2D1U and a
1D2U staircase x 50 trials and 10 easy trials (110 per condition, 1980 trials). The unimodal blocks give
sigma_A (a penalty on ln sigma_A, log sd 0.2) and the unimodal visual sigma, sigma_V / (1 - d) (fixed,
log error 0.2), as from 150 + 20 QUEST+ trials (quest_simulation_results).

Cross-modal block as it ran (auditory test vs visual standard, both noise levels):
  500 ms: 1D1U pair + 2D1U + 1D2U x 45 trials + 20 easy per noise level (400 trials, ~23 min);
  300 and 800 ms: 1D1U pair x 40 + 6 easy per noise level (172 trials each, ~10 min each).
The answers come from the same measurements: P(auditory test longer) = Phi((ln a - m_v) /
sqrt(sigma_A^2 + sigma_V^2)), m_v = (1 - d) ln v + d ln 0.5 - c[noise], plus the lapse. Fitted by maximum
likelihood (one lapse, a slope per noise level): the 500 ms block alone gives c (d unknown, taken as 0);
all three blocks give c and d, with standard errors from the Hessian.

Transfer: the bias the cross-modal block measures is the bimodal one ("same"), or an independent draw
from the same distribution ("none": the bias of separate cues does not carry over to the combined
stimulus, the user's concern of 27 Oct 2026).

Bimodal fits (BADS), every model:
  ignore   c = d = 0 (the current plan)
  dfree    c = 0, d free (the current plan's sensitivity check)
  xm500    c fixed at the 500 ms block's estimates, d = 0
  xm3      c and d fixed at the three-standard estimates
  xm3p     c and d free with Gaussian priors from the three-standard estimates (roughly a joint fit)
  oracle   c and d fixed at the true values: the most any baseline could give
xm500, xm3 and xm3p for both transfer cases. Within a variant fusion is fitted first and CI also starts
from its solution with p_c near 1, so CI never fits worse than fusion; every variant but ignore also
starts from the same model's ignore fit.
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
from scipy.optimize import minimize  # noqa: E402
from scipy.special import ndtr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import avmodels as M  # noqa: E402
import bias_recovery as BR  # noqa: E402
import bimodal_recovery as B  # noqa: E402

N_STAIR, N_EASY = 50, 10                    # per staircase and per condition (main experiment from 29 Oct 2026)
CONFLICTS = B.CONFLICTS["9c"]
ERR = 0.2                                   # log error of sigma_A and sigma_V from the unimodal blocks
BIAS_MEAN, BIAS_SD, RHO = np.array([0.26, 0.17, 0.25]), np.array([0.26, 0.30, 0.20]), 0.8
C_RANGE, D_RANGE = (-0.5, 1.0), (-0.3, 0.7)
XM_FLOOR, XM_CEIL = -0.9, 0.9
SE_INFLATE = 1.2                            # prior sd = 1.2 x the Hessian SE (z-scores of the estimates had SD 1.02-1.26)
# per standard: staircase rules, trials per staircase and easy trials per end, per noise level
XM_DESIGN = {0.5: (("1D1U", "1D1U_b", "2D1U", "1D2U"), 45, 10), 0.3: (("1D1U", "1D1U_b"), 40, 3),
             0.8: (("1D1U", "1D1U_b"), 40, 3)}
BOUNDS = dict(BR.BOUNDS, c1=(-1.2, 1.2, -0.1, 0.6), c2=(-1.2, 1.2, -0.1, 0.6), d=(-0.8, 0.9, -0.2, 0.6))
FIT_ORDER = ("fusion", "ci", "switch")
TRANSFERS = ("same", "none")
VARIANTS = ["ignore", "dfree", "oracle"] + [f"{v}|{t}" for t in TRANSFERS for v in ("xm500", "xm3", "xm3p")]
OUT = B.OUT


def draw_bias(rng):
    while True:
        z = rng.normal(size=3)
        c1 = BIAS_MEAN[0] + BIAS_SD[0] * z[0]
        c2 = BIAS_MEAN[1] + BIAS_SD[1] * (RHO * z[0] + np.sqrt(1 - RHO ** 2) * z[1])
        d = BIAS_MEAN[2] + BIAS_SD[2] * z[2]
        if C_RANGE[0] <= min(c1, c2) and max(c1, c2) <= C_RANGE[1] and D_RANGE[0] <= d <= D_RANGE[1]:
            return float(c1), float(c2), float(d)


def run_main_condition(gen, truth_m, bias, ni, c, rng):
    from avdur.staircase import Staircase
    p_tab = M.p_longer(gen, truth_m, ni, B.STD, B.STD + c, B.LEVELS, bias)
    kinds = np.r_[["a"] * N_STAIR, ["b"] * N_STAIR, ["e"] * N_EASY]
    rng.shuffle(kinds)
    easy = list(rng.permutation(np.r_[[B.frames(B.STD * (1 + B.FLOOR))] * (N_EASY // 2),
                                      [B.frames(B.STD * (1 + B.CEIL))] * (N_EASY // 2)]))
    stairs = {"a": Staircase("2D1U", B.FLOOR, B.CEIL, start="floor"), "b": Staircase("1D2U", B.FLOOR, B.CEIL, start="ceiling")}
    tests, ys = [], []
    for k in kinds:
        t = float(easy.pop()) if k == "e" else max(float(B.frames(B.STD * (1 + stairs[k].level))), 1 / B.FR)
        y = bool(rng.random() < np.interp(t, B.LEVELS, p_tab))
        if k != "e":
            stairs[k].update(y)
        tests.append(t); ys.append(y)
    return np.array(tests), np.array(ys, float)


def p_crossmodal(la, lv, ni, sa, sv_m, lam, c1, c2, d):
    """P(auditory test longer than the visual standard); la, lv log durations, ni noise index (arrays)."""
    mv = (1 - d) * lv + d * M.LSTD - np.where(ni == 0, c1, c2)
    return lam / 2 + (1 - lam) * ndtr((la - mv) / np.hypot(np.where(ni == 0, sa[0], sa[1]), sv_m))


def run_crossmodal(sa, sv_m, lam, bias, rng):
    """Rows (noise index, standard s, test s, chose test) of the 500, 300 and 800 ms blocks."""
    from avdur.staircase import Staircase
    rows = []
    for v, (rules, n, easy_per_end) in XM_DESIGN.items():
        for ni in (0, 1):
            pair = ["floor", "ceiling"] if rng.random() < 0.5 else ["ceiling", "floor"]   # the 1D1U pair: one at each end
            stairs = {r: Staircase(r, XM_FLOOR, XM_CEIL, start=pair[k] if r.startswith("1D1U") else
                                   ("floor" if r == "2D1U" else "ceiling")) for k, r in enumerate(rules)}
            kinds = list(rng.permutation([r for r in rules for _ in range(n)] + ["e"] * (2 * easy_per_end)))
            easy = list(rng.permutation([XM_FLOOR] * easy_per_end + [XM_CEIL] * easy_per_end))
            for k in kinds:
                a = v * (1 + (easy.pop() if k == "e" else stairs[k].level))
                y = bool(rng.random() < p_crossmodal(np.log(a), np.log(v), np.array(ni), sa, sv_m, lam, *bias))
                if k != "e":
                    stairs[k].update(y)
                rows.append((ni, v, a, float(y)))
    return np.array(rows, float)


def _hessian(f, x, h=1e-3):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def fit_crossmodal(rows, with_d):
    """ML fit of (c1, c2[, d], s1, s2, lapse); returns estimates of (c1, c2, d) and their standard errors."""
    if not with_d:
        rows = rows[np.isclose(rows[:, 1], 0.5)]
    ni, lv, la, y = rows[:, 0].astype(int), np.log(rows[:, 1]), np.log(rows[:, 2]), rows[:, 3]
    names = ["c1", "c2"] + (["d"] if with_d else []) + ["s1", "s2", "lam"]

    def nll(x):
        p = dict(zip(names, x))
        mv = (1 - p.get("d", 0.0)) * lv + p.get("d", 0.0) * M.LSTD - np.where(ni == 0, p["c1"], p["c2"])
        pr = p["lam"] / 2 + (1 - p["lam"]) * ndtr((la - mv) / np.where(ni == 0, p["s1"], p["s2"]))
        pr = np.clip(pr, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log1p(-pr))

    bnd = dict(c1=(-1.5, 1.5), c2=(-1.5, 1.5), d=(-1.0, 0.95), s1=(0.03, 4.0), s2=(0.03, 4.0), lam=(0.0, 0.1))
    starts = [dict(c1=0.2, c2=0.1, d=0.2, s1=0.5, s2=1.0, lam=0.02), dict(c1=0.0, c2=0.0, d=0.0, s1=0.3, s2=0.6, lam=0.05),
              dict(c1=0.5, c2=0.4, d=0.4, s1=1.0, s2=1.5, lam=0.01)]
    best = min((minimize(nll, [s[k] for k in names], method="L-BFGS-B", bounds=[bnd[k] for k in names]) for s in starts),
               key=lambda r: r.fun)
    x = best.x
    k = len(names) - 1                          # standard errors with the lapse held at its estimate
    H = _hessian(lambda z: nll(np.r_[z, x[-1]]), x[:-1])
    try:
        w, V = np.linalg.eigh(H)
        cov = (V / np.maximum(w, 1e-6)) @ V.T
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    est = dict(zip(names, map(float, x)))
    se = dict(zip(names[:-1], map(float, se)))
    c = [est["c1"], est["c2"], est.get("d", 0.0)]
    c_se = [se["c1"], se["c2"], se.get("d", np.nan)]
    return dict(est=c, se=[float(np.clip(v, 0.02, 1.0)) if np.isfinite(v) else 1.0 for v in c_se],
                sigma=[est["s1"], est["s2"]], lam=est["lam"], n=int(len(rows)))


def make_nll(model, data, sv_hat, sa_prior, bias=None, prior=None, free_d=False):
    """bias: (c1, c2, d) fixed, or None (c = d = 0). prior: (means, sds) of c1, c2, d, which are then free.
    free_d: c = 0 and d free without a prior. sigma_V = sv_hat x (1 - d) (the unimodal block measures sigma_V / (1 - d))."""
    extra = ["c1", "c2", "d"] if prior is not None else (["d"] if free_d else [])
    names = BR.FREE[model] + extra
    mean, sd = (np.asarray(prior[0], float), np.asarray(prior[1], float)) if prior is not None else (None, None)

    def f(x):
        nm = dict(zip(names, np.asarray(x, float).ravel()))
        b = (nm["c1"], nm["c2"], nm["d"]) if prior is not None else ((0.0, 0.0, nm["d"]) if free_d else bias)
        d = 0.0 if b is None else b[2]
        out = M.nll(model, BR.full_params(model, nm, sv_hat * (1 - d)), data, b)
        out += 0.5 * sum((np.log(nm[k] / v) / sa_prior[2]) ** 2 for k, v in zip(("sa1", "sa2"), sa_prior[:2]))
        if prior is not None:
            out += 0.5 * float(np.sum(((np.array(b) - mean) / sd) ** 2))
        return out
    return f, names


def bads_fit(f, names, rng, x0s=(), n_random=3):
    from pybads import BADS
    lb, ub, plb, pub = (np.array([BOUNDS[n][i] for n in names], float) for i in range(4))
    starts = [plb + rng.random(len(lb)) * (pub - plb) for _ in range(n_random)]
    starts += [np.clip(np.asarray(x, float), lb + 1e-6, ub - 1e-6) for x in x0s]
    best = None
    for attempt in range(2):                                # a second round of random starts if every start failed
        for x0 in starts:
            try:
                r = BADS(f, x0, lb, ub, plb, pub, options=dict(display="off", uncertainty_handling=False, max_fun_evals=700)).optimize()
                x, fv = np.asarray(r["x"], float).ravel(), float(r["fval"])
            except Exception:
                continue
            if np.isfinite(fv) and (best is None or fv < best[1]):
                best = (x, fv)
        if best is not None:
            break
        starts = [plb + rng.random(len(lb)) * (pub - plb) for _ in range(3)]
    if best is None:
        return dict(zip(names, [float("nan")] * len(names))), float("inf")
    return dict(zip(names, map(float, best[0]))), best[1]


def one(args):
    try:
        return _one(*args)
    except Exception as err:                                # reported and skipped, so one failure cannot stop the run
        return dict(gen=args[0], iter=args[1], seed=args[2], error=repr(err), seconds=0.0)


def _one(gen, it, seed, cfg):
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    rng = np.random.default_rng(seed)
    truth = B.sample_truth(gen, rng)                        # sigma_V here is the unimodal (biased-scale) sigma
    rb = np.random.default_rng([seed, 31])
    bias = draw_bias(rb)
    bias_other = draw_bias(rb)                              # the cross-modal bias when it does not transfer
    sv_eff = truth[2]
    truth_m = list(truth); truth_m[2] = sv_eff * (1 - bias[2])
    sa_true = (truth_m[1], truth_m[BR.SA2_INDEX[gen]])
    rd = np.random.default_rng([seed, 32])
    cols = [[], [], [], []]
    for ni in B.NOISE:
        for c in CONFLICTS:
            t, y = run_main_condition(gen, truth_m, bias, ni, float(c), rd)
            for lst, v in zip(cols, (np.full(len(t), ni), np.full(len(t), float(c)), t, y)):
                lst.append(v)
    data = M.Data(*(np.concatenate(v) for v in cols))
    ru = np.random.default_rng([seed, 33])
    z = ru.normal(size=3)
    sv_hat = float(np.clip(sv_eff * np.exp(ERR * z[0]), 0.05, 1.9))
    sa_prior = tuple(float(v * np.exp(ERR * zz)) for v, zz in zip(sa_true, z[1:])) + (ERR,)
    xm = {}
    for tr, b in (("same", bias), ("none", bias_other)):
        rows = run_crossmodal(sa_true, truth_m[2], truth_m[0], b, np.random.default_rng([seed, 34 if tr == "same" else 35]))
        xm[tr] = dict(xm500=fit_crossmodal(rows, with_d=False), xm3=fit_crossmodal(rows, with_d=True), bias=list(b))
    t0 = time.time()
    rf = np.random.default_rng([seed, 36])
    fits = {}
    for variant in VARIANTS:
        kind, tr = (variant.split("|") + [None])[:2]
        spec = dict(bias=None, prior=None, free_d=False)
        if kind == "dfree":
            spec["free_d"] = True
        elif kind == "oracle":
            spec["bias"] = bias
        elif kind == "xm500":
            spec["bias"] = tuple(xm[tr]["xm500"]["est"][:2]) + (0.0,)
        elif kind == "xm3":
            spec["bias"] = tuple(xm[tr]["xm3"]["est"])
        elif kind == "xm3p":                              # Hessian SEs were 2-26% too small (60 simulated blocks)
            spec["prior"] = (xm[tr]["xm3"]["est"], [SE_INFLATE * s for s in xm[tr]["xm3"]["se"]])
        fits[variant] = {}
        for model in FIT_ORDER:
            f, names = make_nll(model, data, sv_hat, sa_prior, **spec)
            x0s = []
            n_extra = len(names) - len(BR.FREE[model])
            if variant != "ignore":
                base = fits["ignore"][model]["params"]
                x0s.append(np.r_[[base[k] for k in BR.FREE[model]], (xm[tr]["xm3"]["est"] if kind == "xm3p" else [0.0] * n_extra)])
            if model == "ci":                             # fusion = CI at p_c = 1
                fu = fits[variant]["fusion"]["params"]
                x0s.append(np.r_[[fu["lam"], fu["sa1"], 0.999, fu["sa2"]], [fu[k] for k in names[4:]]])
            prm, fv = bads_fit(f, names, rf, x0s=x0s, n_random=3 if variant == "ignore" else 1)
            k = len(prm)
            fits[variant][model] = dict(params=prm, nll=fv, aic=2 * k + 2 * fv)
    return dict(gen=gen, iter=it, seed=seed, truth=truth_m, sv_eff=sv_eff, bias=list(bias), sv_hat=sv_hat,
                sa_prior=list(sa_prior), n_trials=data.n_trials, xm=xm, fits=fits, seconds=round(time.time() - t0, 1),
                design=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=200)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--hours", type=float, default=6.5)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = dict(n_stair=N_STAIR, n_easy=N_EASY, err=ERR, bias_mean=BIAS_MEAN.tolist(), bias_sd=BIAS_SD.tolist(), rho=RHO)
    path = OUT / ("crossmodal_baseline_pilot.json" if args.pilot else "crossmodal_baseline.json")
    results = [] if args.pilot or not path.exists() else json.loads(path.read_text())
    done = {(r["gen"], r["iter"]) for r in results}
    seeds = np.random.default_rng(20261030).integers(2**31, size=(len(B.MODELS), 10000))
    n_iter = 1 if args.pilot else args.n_iter
    jobs = [(g, i, int(seeds[k, i]), cfg) for i in range(n_iter) for k, g in enumerate(B.MODELS) if (g, i) not in done]
    print(f"{len(jobs)} participants, {args.workers} workers, {cfg}", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs), 1):
            if "error" in r:
                print(f"  {k}/{len(jobs)} {r['gen']} #{r['iter']} FAILED: {r['error']}", flush=True)
                continue
            results.append(r)
            tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(results)); os.replace(tmp, path)
            print(f"  {k}/{len(jobs)} {r['gen']} #{r['iter']} ({r['seconds']:.0f} s; {(time.time() - t0) / 60:.0f} min)", flush=True)
            if time.time() - t0 > args.hours * 3600:
                print("time limit reached; stopping", flush=True)
                pool.terminate()
                break


if __name__ == "__main__":
    main()
