"""v1 main experiment: can the audiovisual bias (c, d) be fitted from bimodal data? A real-data check of the
simulations (bias_recovery.py), with v1's cross-modal block as an independent measure of c.

    python analysis/overnight_sims/v1_bias_fits.py --workers 7     # fits (resumable), then the summary
    python analysis/overnight_sims/v1_bias_fits.py --summary       # summary only

v1 (av_dur_data/): 11 participants, 7 conflicts (-250 ... +250 ms) x 2 noise levels (0.1, 1.2), about 1,000
trials per noise level. Every visual flash was lengthened by the participant's cross-modal correction
(VisualPSE, -0.18 ... +0.28 s); the fits use the visual durations as shown (avmodels.Data vis_offset), so the
c fitted from the main data should equal the cross-modal block's c = -ln(PSE / 500 ms) (auditory test vs
visual 500 ms standard). v1 measured the bias at 500 ms only, so d has no independent measure.

sigma_V is fixed at the unimodal visual fit (sigma_V (1 - d) in the model, as in bias_recovery.py); sigma_A
is free, or tied to the unimodal auditory fit (Gaussian penalty on ln sigma_A, log sd 0.2). The unimodal
psychometric sigma is sqrt(2) x the measurement sigma (a 2IFC compares two noisy intervals), so both are divided
by sqrt(2), as in the article's fixed-noise fits (runFitting_fixedUnimodalSigma.py). Until 30 Oct 2026 they were
not: sigma 41% too large (those results: results/v1_bias_fits_before_sqrt2.json). Each model (CI,
fusion, switching) is fitted without a bias (c = d = 0), with d, with c_low, c_high, d (bias_recovery.fit, BADS;
the biased fits also start from the no-bias solution), and with c fixed at the cross-modal values, d = 0
("xc") or free ("xc_d"): does the bias measured with A and V in separate intervals fit the main data? The boxcar prior spans the durations shown
(0.025-1.3 s; the corrected visual ones reach 1.27 s), and sigma_A at high noise may reach 4 (unimodal v1 values
reach 3.7).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V1 = HERE.parents[2] / "av_dur_data"
OUT = HERE / "results" / "v1_bias_fits.json"
NOISE = {0.1: 0, 1.2: 1}
SA_ERR = 0.2                      # v1 unimodal auditory: 154 staircase trials per noise level
PRIOR_S = (0.025, 1.3)
SCENARIOS = {"sigma_A free": None, "sigma_A tied": SA_ERR}
VARIANTS = {"nobias": ("nobias", False), "d": ("d", False), "bias": ("bias", False),
            "xc": ("nobias", True), "xc_d": ("d", True)}     # name: (bias_recovery variant, c fixed at cross-modal)


def load(name):
    d = pd.read_csv(V1 / f"all_{name}.csv")
    d["chose_test"] = (d.responses == d.order).astype(float)
    return d


def fit_pf(t, y, std=0.5):
    """Lognormal psychometric function, lapse 0-0.1: (mu, sigma, lapse)."""
    x = np.log(np.asarray(t, float) / std)
    y = np.asarray(y, float)

    def nll(p):
        pr = np.clip(p[2] / 2 + (1 - p[2]) * norm.cdf((x - p[0]) / np.exp(p[1])), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    r = min((minimize(nll, x0, method="L-BFGS-B", bounds=[(-1.5, 1.5), (np.log(.03), np.log(5)), (0, .1)])
             for x0 in ([0, -1, .02], [0, 0, .05], [-.3, -.5, .02])), key=lambda r: r.fun)
    return float(r.x[0]), float(np.exp(r.x[1])), float(r.x[2])


def baselines():
    """Per participant: unimodal sigma_A (low, high), sigma_V, and the cross-modal c (low, high) with bootstrap SE."""
    aud, vis, xm, bim = load("auditory"), load("visual"), load("crossmodal"), load("bimodal")
    rng = np.random.default_rng(1)
    out = {}
    for pid in sorted(bim.participantID.unique()):
        a, v, x = aud[aud.participantID == pid], vis[vis.participantID == pid], xm[xm.participantID == pid]
        b = dict(sa=[fit_pf(g.testDurS, g.chose_test)[1] for _, g in sorted(a.groupby("audNoise"))],
                 sv=fit_pf(v.testDurS, v.chose_test)[1], c=[], c_se=[], vis_offset=[])
        for noise, g in sorted(x.groupby("audNoise")):
            mu = fit_pf(g.testDurS, g.chose_test)[0]
            boot = [fit_pf(g.testDurS.to_numpy()[i], g.chose_test.to_numpy()[i])[0]
                    for i in (rng.integers(len(g), size=len(g)) for _ in range(200))]
            b["c"].append(-mu)
            b["c_se"].append(float(np.std(boot)))
        bm = bim[bim.participantID == pid]
        for noise, g in sorted(bm.groupby("audNoise")):
            b["vis_offset"].append(float(np.median(g.recordedDurVisualStandard - g.standardDur - g.conflictDur)))
        b["VisualPSE"] = [float(g.VisualPSE.median()) for _, g in sorted(bm.groupby("audNoise"))]
        out[pid] = b
    return out


def main_data(pid, offsets):
    import avmodels as M
    d = load("bimodal")
    d = d[d.participantID == pid]
    ni = d.audNoise.map(NOISE).to_numpy()
    off = np.array(offsets)[ni]
    keep = d.testDurS.to_numpy() + off >= 0.02           # visual test of at least 20 ms as shown
    conflict = np.round(d.conflictDur.to_numpy() * 12) / 12   # nominal: multiples of 1/12 s
    return M.Data(ni[keep], conflict[keep], d.testDurS.to_numpy()[keep], d.chose_test.to_numpy()[keep],
                  vis_offset={0: offsets[0], 1: offsets[1]}), int((~keep).sum())


def one(args):
    pid, scenario, base, done_fits = args
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    import avmodels as M
    import bias_recovery as BR
    import bimodal_recovery as B
    M.LO, M.HI = np.log(PRIOR_S[0]), np.log(PRIOR_S[1])
    BR.BOUNDS = dict(BR.BOUNDS, sa2=(.03, 4, .4, 1.5))   # v1 unimodal sigma_A at noise 1.2 reaches 3.7
    data, dropped = main_data(pid, base["vis_offset"])
    err = SCENARIOS[scenario]
    sa = [s / np.sqrt(2) for s in base["sa"]]            # psychometric sigma -> measurement sigma
    sv = base["sv"] / np.sqrt(2)
    sa_prior = None if err is None else (sa[0], sa[1], err)
    rng = np.random.default_rng([int(pid[1:]), len(scenario)])
    t0 = time.time()
    fits = {m: dict(v) for m, v in (done_fits or {}).items()}
    for model in B.MODELS:
        fits.setdefault(model, {})
        for name, (variant, xc) in VARIANTS.items():
            if name in fits[model]:
                continue
            first = fits[model].get("nobias", {}).get("params")
            x0s = [] if first is None else [np.r_[[first[k] for k in BR.FREE[model]], np.zeros(len(BR.EXTRA[variant]))]]
            prm, fv = BR.fit(model, data, sv, variant, rng, x0s=x0s, sa_prior=sa_prior,
                             fixed_c=tuple(base["c"]) if xc else None)
            k = len(prm)
            fits[model][name] = dict(params=prm, nll=fv, aic=2 * k + 2 * fv, bic=k * np.log(data.n_trials) + 2 * fv)
    return dict(pid=pid, scenario=scenario, n_trials=data.n_trials, dropped=dropped, fits=fits,
                seconds=round(time.time() - t0, 1))


def summary(results, base):
    rows = []
    for r in results:
        b = base[r["pid"]]
        best = {v: min(r["fits"], key=lambda m: r["fits"][m][v]["aic"]) for v in VARIANTS if all(v in f for f in r["fits"].values())}
        for model, fv in r["fits"].items():
            for variant, f in fv.items():
                p = f["params"]
                rows.append(dict(pid=r["pid"], scenario=r["scenario"], model=model, variant=variant, aic=f["aic"],
                                 best=model == best.get(variant), c1=p.get("c1"), c2=p.get("c2"), d=p.get("d"), pc=p.get("pc"),
                                 x_c1=b["c"][0], x_c2=b["c"][1], x_se1=b["c_se"][0], x_se2=b["c_se"][1]))
    df = pd.DataFrame(rows)
    print(f"{df.pid.nunique()} participants, about {np.mean([r['n_trials'] for r in results]):.0f} trials each\n")
    print("== per participant: cross-modal c (SE) vs c from the main data (c and d free, AIC-best model) ==")
    for scn, g in df[(df.variant == "bias") & df.best].groupby("scenario"):
        print(f"-- {scn}")
        print(g.assign(**{"xmodal c low": g.x_c1.round(2).astype(str) + " (" + g.x_se1.round(2).astype(str) + ")",
                          "main c low": g.c1.round(2), "xmodal c high": g.x_c2.round(2).astype(str) + " (" + g.x_se2.round(2).astype(str) + ")",
                          "main c high": g.c2.round(2), "main d": g.d.round(2)})
              [["pid", "model", "xmodal c low", "main c low", "xmodal c high", "main c high", "main d"]].to_string(index=False))
    print("\n== agreement of c, main vs cross-modal (22 values: 11 participants x 2 noise levels) ==")
    for (scn, model), g in df[df.variant == "bias"].groupby(["scenario", "model"]):
        est, tru = np.r_[g.c1, g.c2], np.r_[g.x_c1, g.x_c2]
        print(f"{scn:13} {model:7} r = {np.corrcoef(est, tru)[0, 1]:+.2f}  RMSE {np.sqrt(np.mean((est - tru) ** 2)):.2f}  "
              f"(cross-modal SD {tru.std():.2f}, mean SE {np.mean(np.r_[g.x_se1, g.x_se2]):.2f})")
    g = df[(df.variant == "bias") & df.best]
    for scn, h in g.groupby("scenario"):
        est, tru = np.r_[h.c1, h.c2], np.r_[h.x_c1, h.x_c2]
        print(f"{scn:13} AIC-best r = {np.corrcoef(est, tru)[0, 1]:+.2f}  RMSE {np.sqrt(np.mean((est - tru) ** 2)):.2f}")
    print("\n== AIC: winning model per variant, and the best AIC relative to no bias (negative = better) ==")
    print("   xc: c fixed at the cross-modal values, d = 0; xc_d: the same with d free")
    have = [v for v in VARIANTS if v in set(df.variant)]
    for scn, g in df.groupby("scenario"):
        w = g.pivot_table(index="pid", columns=["variant"], values="aic", aggfunc="min")
        t = pd.DataFrame({f"best {v}": g[(g.variant == v) & g.best].set_index("pid").model for v in have})
        for v in have[1:]:
            t[f"dAIC {v}"] = (w[v] - w["nobias"]).round(1)
        print(f"-- {scn}\n{t.to_string()}")
        print("   sum over participants: " + ", ".join(f"{v} {(w[v] - w['nobias']).sum():+.0f}" for v in have[1:]))
    print("\n== p_c (CI fit) by variant ==")
    ci = df[df.model == "ci"].pivot_table(index=["scenario", "pid"], columns="variant", values="pc")
    print(ci[have].round(2).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="one participant, sigma_A free (timing)")
    args = ap.parse_args()
    base_path = OUT.with_name("v1_baselines.json")
    if base_path.exists():
        base = json.loads(base_path.read_text())
    else:
        base = baselines()
        base_path.write_text(json.dumps(base, indent=1))
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    if not args.summary:
        prev = {(r["pid"], r["scenario"]): r for r in results}
        complete = {key for key, r in prev.items() if all(set(VARIANTS) <= set(f) for f in r["fits"].values())}
        jobs = [(pid, s, base[pid], prev.get((pid, s), {}).get("fits")) for s in SCENARIOS for pid in base
                if (pid, s) not in complete]
        results = [r for key, r in prev.items() if key in complete]
        done = complete
        if args.pilot:
            print(one(jobs[0]))
            return
        with get_context("spawn").Pool(args.workers) as pool:
            for r in pool.imap_unordered(one, jobs):
                results.append(r)
                OUT.write_text(json.dumps(results))
                print(f"{r['pid']} {r['scenario']}: {r['seconds']:.0f} s ({len(results)}/{len(jobs) + len(done)})", flush=True)
    summary(results, base)


if __name__ == "__main__":
    main()
