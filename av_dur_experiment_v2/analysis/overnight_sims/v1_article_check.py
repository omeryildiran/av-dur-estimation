"""Does avmodels reproduce the article's fixed-noise model comparison on v1 (switching best for 8 of 11)?

    python analysis/overnight_sims/v1_article_check.py

The article (av_dur_article.tex, "Fixed sensory noise") fixes sigma_A (low, high noise) and sigma_V at the
unimodal psychometric sigma / sqrt(2) (a 2IFC compares two noisy intervals) and fits the lapse and p_c or
p_switch by Monte Carlo (monteCarloClass, 2000 draws per condition). Its fits are in
unimodalSigma_modelFits_singleLapse/ (fusion, causal inference = lognorm, switching = switchingFree).
Here the same three models are fitted with avmodels (exact quadrature likelihood), lapse and p_c / p_switch free:
  article sigma, nominal    the article's sigma values; visual durations nominal (standard + conflict), as the article
  own sigma, nominal        sigma from v1_bias_fits.baselines (v1_baselines.json) / sqrt(2)
  own sigma, as shown       the same with the visual durations as shown (nominal + VisualPSE), as v1_bias_fits.py
The anonymous IDs are matched to the article's folders through participant_id_mapping.txt (private, local only).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "v1_article_check.json"
ARTICLE = ROOT / "unimodalSigma_modelFits_singleLapse"
ART_MODEL = {"fusion": "fusionOnlyLogNorm", "ci": "lognorm", "switch": "switchingFree"}
FREE = {"fusion": ["lam"], "ci": ["lam", "pc"], "switch": ["lam", "ps1", "ps2"]}
BND = {"lam": (0.0, 0.25), "pc": (0.0, 1.0), "ps1": (0.0, 1.0), "ps2": (0.0, 1.0)}
SETTINGS = ("article sigma, nominal", "own sigma, nominal", "own sigma, as shown")


def mapping():
    """Anonymous ID -> the article's folder name (from the private mapping file)."""
    out = {}
    for line in (ROOT / "participant_id_mapping.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("P") and parts[1][1:].isdigit():
            out.setdefault(parts[1], []).append(parts[0])
    return out


def article_fit(pid, names):
    """The article's fixed sigmas and AIC per model for this participant, or None."""
    for name in names:
        folder = ARTICLE / name
        if folder.exists():
            fits = {m: json.loads((folder / f"{name}_{a}_LapseFix_sharedPrior_fit.json").read_text()) for m, a in ART_MODEL.items()}
            s = fits["fusion"]["fixedSensorySigmas"]
            return dict(sa=[s["sigma_a_low"], s["sigma_a_high"]], sv=s["sigma_v"], aic={m: f["AIC"] for m, f in fits.items()})
    return None


def fit_fixed(model, data, sa, sv):
    import avmodels as M
    names = FREE[model]

    def full(x):
        nm = dict(zip(names, x))
        if model == "ci":
            return [nm["lam"], sa[0], sv, nm["pc"], sa[1]]
        if model == "fusion":
            return [nm["lam"], sa[0], sv, sa[1]]
        return [nm["lam"], sa[0], sv, nm["ps1"], sa[1], nm["ps2"]]

    f = lambda x: M.nll(model, full(x), data)  # noqa: E731
    starts = {"fusion": [[0.02], [0.08]], "ci": [[0.02, 0.2], [0.02, 0.6], [0.05, 0.99]],
              "switch": [[0.02, 0.2, 0.5], [0.02, 0.5, 0.8], [0.05, 0.8, 0.3]]}[model]
    best = min((minimize(f, x0, method="L-BFGS-B", bounds=[BND[n] for n in names]) for x0 in starts), key=lambda r: r.fun)
    return dict(zip(names, map(float, best.x))), float(best.fun)


def one(args):
    pid, setting, sa, sv, offsets = args
    warnings.filterwarnings("ignore")
    import avmodels as M
    import v1_bias_fits as V
    M.LO, M.HI = np.log(V.PRIOR_S[0]), np.log(V.PRIOR_S[1])
    data, _ = V.main_data(pid, offsets)
    fits = {}
    for model in ("fusion", "ci", "switch"):
        prm, nll = fit_fixed(model, data, sa, sv)
        fits[model] = dict(params=prm, nll=nll, aic=2 * len(prm) + 2 * nll)
    return dict(pid=pid, setting=setting, sa=list(sa), sv=sv, fits=fits)


def main():
    base = json.loads((HERE / "results" / "v1_baselines.json").read_text())
    names = mapping()
    art = {pid: article_fit(pid, names.get(pid, [])) for pid in base}
    jobs = []
    for pid, b in base.items():
        own_sa, own_sv = [s / np.sqrt(2) for s in b["sa"]], b["sv"] / np.sqrt(2)
        if art[pid]:
            jobs.append((pid, SETTINGS[0], art[pid]["sa"], art[pid]["sv"], [0.0, 0.0]))
        jobs.append((pid, SETTINGS[1], own_sa, own_sv, [0.0, 0.0]))
        jobs.append((pid, SETTINGS[2], own_sa, own_sv, b["vis_offset"]))
    with get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 7) as pool:
        res = pool.map(one, jobs)
    OUT.write_text(json.dumps(dict(results=res, article={p: a and a["aic"] for p, a in art.items()}), indent=1))
    label = {"fusion": "fusion", "ci": "CI", "switch": "switching"}
    print(f"{'pid':4} {'article (Monte Carlo)':22} " + " ".join(f"{s:24}" for s in SETTINGS))
    counts = {s: {} for s in ("article",) + SETTINGS}
    for pid in base:
        a = art[pid]
        wa = label[min(a["aic"], key=a["aic"].get)] if a else "-"
        counts["article"][wa] = counts["article"].get(wa, 0) + 1
        row = [f"{pid:4} {wa:22}"]
        for s in SETTINGS:
            r = next((r for r in res if r["pid"] == pid and r["setting"] == s), None)
            w = label[min(r["fits"], key=lambda m: r["fits"][m]["aic"])] if r else "-"
            counts[s][w] = counts[s].get(w, 0) + 1
            row.append(f"{w:24}")
        print(" ".join(row))
    print("\nwinners: " + " | ".join(f"{s}: {c}" for s, c in counts.items()))
    print("\nsigma (article vs own / sqrt 2): " + "; ".join(
        f"{pid} sA_low {art[pid]['sa'][0]:.2f}/{base[pid]['sa'][0] / np.sqrt(2):.2f} sA_high {art[pid]['sa'][1]:.2f}/"
        f"{base[pid]['sa'][1] / np.sqrt(2):.2f} sV {art[pid]['sv']:.2f}/{base[pid]['sv'] / np.sqrt(2):.2f}"
        for pid in base if art[pid]))


if __name__ == "__main__":
    main()
