"""Order-balanced re-analysis of the bimodal experiment (used by balancedOrderAnalysis.ipynb).

Pipeline
--------
1. ``build_trials``     Raw bimodal session files -> one session-labelled trial table.
                        Each participant's sessions go through ``loadData`` so the 17 ms
                        timing criterion and all derived columns match the published analysis.
2. ``make_balanced``    Keep correctly timed sessions only (code version C), then equalise
                        test-first and test-second trials in every participant x noise x
                        conflict cell by random subsampling.
3. ``fit_pse``          Per-cell psychometric fits (free mu, sigma, lambda; lognormal PF).
4. ``run_model_fits``   Quick fits of forced fusion, causal inference and probabilistic cue
                        switching, sensory noise fixed from the unimodal tasks, one shared
                        lapse rate (the published "fixed sensory noise" set-up).
5. ``model_pse``        Model-predicted PSE for every cell from the fitted parameters.

Run ``python balanced_order_tools.py`` to compute and cache everything under
``balanced_order_analysis/``; the notebook then loads the caches.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import contextlib
import io
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

REPO = Path(__file__).resolve().parent
GDRIVE = Path(
    "/Users/ofy204/Library/CloudStorage/GoogleDrive-omerulfaruk97@gmail.com/My Drive/"
    "MyReposDrive/obsidian_Notes/Landy Omer Re 1"
)
RAW_DIRS = [GDRIVE / "MegaData", GDRIVE / "av-dur-estimation" / "data", REPO / "data"]
OUT_DIR = REPO / "balanced_order_analysis"
SHARED_FILE = REPO / "av_dur_data" / "all_bimodal.csv"

NOMINAL_CONFLICTS = np.array([-0.25, -0.167, -0.083, 0.0, 0.083, 0.167, 0.25])
PARTICIPANTS = {
    "as": "P01", "dt": "P02", "hh": "P03", "ip": "P04", "ln2": "P07", "mh": "P08",
    "ml": "P09", "mt": "P10", "oy": "P11", "qs": "P12", "sx": "P13",
}
AFFECTED = ("hh", "ip", "ln2", "mh", "ml", "qs", "sx")

MODELS = ("fusionOnlyLogNorm", "lognorm", "switchingFree")
MODEL_LABELS = {
    "fusionOnlyLogNorm": "Forced fusion",
    "lognorm": "Causal inference",
    "switchingFree": "Probabilistic cue switching",
}
K_FREE = {"fusionOnlyLogNorm": 1, "lognorm": 2, "switchingFree": 3}

# The bimodal sessions that make up the analysed data (identical to av_dur_data/all_bimodal.csv).
# version  A: visual test in the second interval got only half the bias correction (2-5 June 2025)
#          B: visual test in the second interval ignored the test duration (7-11 June 2025)
#          C: correct code (from 12 June 2025)
# Columns: code, session stamp, version, use the hand-edited file, expected rows, note.
SESSIONS = [
    ("as", "2025-06-20_18h35.36", "C", False, 616, ""),
    ("as", "2025-06-21_12h03.50", "C", False, 924, ""),
    ("as", "2025-06-21_13h15.23", "C", False, 616, ""),
    ("dt", "2025-06-12_13h17.06", "C", False, 14, "aborted start"),
    ("dt", "2025-06-12_13h19.10", "C", False, 616, ""),
    ("dt", "2025-06-13_15h12.38", "C", True, 336, "1D2U staircase removed by hand"),
    ("dt", "2025-06-14_14h11.21", "C", False, 924, ""),
    ("dt", "2025-06-14_15h11.25", "C", False, 336, "1D2U staircase re-run"),
    ("hh", "2025-06-02_11h26.04", "A", False, 616, ""),
    ("hh", "2025-06-03_11h16.46", "A", False, 616, ""),
    ("hh", "2025-06-05_11h27.28", "A", False, 924, ""),
    ("hh", "2025-06-13_12h36.28", "C", False, 228, "make-up"),
    ("hh", "2025-06-13_12h56.33", "C", False, 120, "make-up"),
    ("ip", "2025-06-04_12h35.40", "A", False, 616, ""),
    ("ip", "2025-06-17_13h31.09", "C", False, 312, "make-up; bias corrections swapped between noise levels"),
    ("ip", "2025-06-17_15h12.53", "C", False, 616, ""),
    ("ip", "2025-06-18_11h25.30", "C", False, 924, ""),
    ("ln2", "2025-06-04_11h36.41", "A", False, 616, ""),
    ("ln2", "2025-06-07_13h18.21", "B", False, 616, ""),
    ("ln2", "2025-06-12_13h54.19", "C", False, 14, "aborted start"),
    ("ln2", "2025-06-12_14h20.21", "C", False, 924, ""),
    ("ln2", "2025-06-14_15h40.10", "C", False, 296, "make-up"),
    ("ln2", "2025-06-14_16h02.00", "C", False, 352, "make-up"),
    ("mh", "2025-06-11_16h47.50", "B", False, 616, ""),
    ("mh", "2025-06-17_14h05.00", "C", False, 312, "make-up"),
    ("mh", "2025-06-17_14h23.07", "C", False, 616, ""),
    ("mh", "2025-06-18_13h01.18", "C", False, 924, ""),
    ("ml", "2025-06-04_14h04.53", "A", False, 616, ""),
    ("ml", "2025-06-10_14h01.35", "B", False, 616, ""),
    ("ml", "2025-06-11_12h05.11", "B", False, 924, ""),
    ("ml", "2025-06-13_13h05.16", "C", False, 384, "make-up"),
    ("ml", "2025-06-13_13h27.50", "C", False, 136, "make-up"),
    ("ml", "2025-06-13_13h36.42", "C", False, 232, "make-up"),
    ("mt", "2025-06-16_13h23.41", "C", False, 616, ""),
    ("mt", "2025-06-20_13h10.00", "C", False, 616, ""),
    ("mt", "2025-06-20_13h49.47", "C", False, 924, ""),
    ("oy", "2025-06-19_19h28.31", "C", False, 1232, ""),
    ("oy", "2025-06-21_22h27.34", "C", False, 939, ""),
    ("qs", "2025-06-09_15h37.06", "B", False, 616, ""),
    ("qs", "2025-06-20_12h23.20", "C", False, 616, ""),
    ("qs", "2025-06-21_14h24.51", "C", False, 924, ""),
    ("qs", "2025-06-21_16h18.43", "C", False, 376, "make-up"),
    ("sx", "2025-06-10_11h11.59", "B", False, 616, ""),
    ("sx", "2025-06-11_15h08.46", "B", False, 616, ""),
    ("sx", "2025-06-12_15h39.04", "C", False, 924, ""),
    ("sx", "2025-06-13_16h17.37", "C", False, 544, "make-up"),
]


@contextlib.contextmanager
def _chdir(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def snap_conflict(values):
    values = np.asarray(values, dtype=float)
    return NOMINAL_CONFLICTS[np.abs(values[:, None] - NOMINAL_CONFLICTS).argmin(1)]


def _find_session_file(code, stamp, edited):
    hits = []
    for folder in RAW_DIRS:
        hits += sorted(folder.glob(f"*mainExpAvDurEstimate_{stamp}*.csv"))
    hits = [
        h for h in hits
        if "_fit" not in h.name
        and ("_edited" in h.name) == edited
        and h.name.split("_mainExp")[0].lower().startswith(code[:2])
    ]
    if not hits:
        raise FileNotFoundError(f"No raw file for {code} {stamp} in {RAW_DIRS}")
    return hits[0]


def _read_session(path):
    d = pd.read_csv(path)
    d = d[~d["order"].isna() & ~d["conflictDur"].isna()].copy()
    return d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")])


def build_trials(force=False):
    """Session-labelled, timing-filtered trial table for the 11 analysed participants."""
    import loadData

    cache = OUT_DIR / "trials_labeled.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    work = OUT_DIR / "_loaddata"
    (work / "data").mkdir(parents=True, exist_ok=True)
    frames = []
    for code, pid in PARTICIPANTS.items():
        parts = []
        for c, stamp, version, edited, n_expected, note in SESSIONS:
            if c != code:
                continue
            path = _find_session_file(code, stamp, edited)
            d = _read_session(path)
            if len(d) != n_expected:
                raise ValueError(f"{path.name}: {len(d)} rows, expected {n_expected}")
            d["participantID"] = pid
            d["code"] = code
            d["session"] = stamp
            d["codeVersion"] = version
            d["buggySession"] = version in ("A", "B")
            d["sessionNote"] = note
            d["sourceFile"] = path.name
            parts.append(d)
        raw = pd.concat(parts, ignore_index=True)
        fname = f"{code}_labeled.csv"
        raw.to_csv(work / "data" / fname, index=False)
        with _chdir(work), contextlib.redirect_stdout(io.StringIO()):
            processed, _ = loadData.loadData(fname, verbose=False)
        frames.append(processed)

    trials = pd.concat(frames, ignore_index=True)
    trials["conflict_nominal"] = snap_conflict(trials["conflictPreMod"])
    trials["affected"] = trials["code"].isin(AFFECTED)
    trials["swappedBiasCorrection"] = trials["sessionNote"].str.contains("swapped", na=False)
    trials["oneFrameShortTest"] = (
        (trials["code"] == "ml") & (trials["session"] == "2025-06-04_14h04.53") & (trials["order"] == 2)
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(cache, index=False)
    return trials


def validate_against_shared(trials):
    """Per participant x order trial counts: rebuilt table vs av_dur_data/all_bimodal.csv."""
    shared = pd.read_csv(SHARED_FILE)
    a = trials.groupby(["participantID", "order"]).size().rename("rebuilt")
    b = shared.groupby(["participantID", "order"]).size().rename("shared")
    table = pd.concat([a, b], axis=1).fillna(0).astype(int)
    table["match"] = table["rebuilt"] == table["shared"]
    return table


def make_balanced(trials, seed=0):
    """Correctly timed sessions only, then equal test-first/test-second counts per cell."""
    fixed = trials[~trials["buggySession"]]
    rng = np.random.default_rng(seed)
    keep = []
    for _, cell in fixed.groupby(["code", "audNoise", "conflict_nominal"], sort=True):
        first, second = cell[cell["order"] == 1], cell[cell["order"] == 2]
        m = min(len(first), len(second))
        if m == 0:
            continue
        keep.append(first.sample(m, random_state=int(rng.integers(2**31))))
        keep.append(second.sample(m, random_state=int(rng.integers(2**31))))
    return pd.concat(keep).sort_index()


def fit_pse(trials, dataset):
    """Free mu/sigma/lambda lognormal PF per participant x noise x nominal conflict."""
    from free_psychometric_refits import _condition_counts, fit_pf_counts, mu_to_shift_s

    rows = []
    for (code, noise, conflict), cell in trials.groupby(["code", "audNoise", "conflict_nominal"]):
        x, n_chose, n_total = _condition_counts(cell)
        fit = fit_pf_counts(x, n_chose, n_total)
        rows.append(dict(
            dataset=dataset, code=code, participantID=PARTICIPANTS[code], audNoise=float(noise),
            conflict=float(conflict), conflict_ms=int(round(conflict * 1000)),
            n=len(cell), n_test_first=int((cell["order"] == 1).sum()),
            n_test_second=int((cell["order"] == 2).sum()),
            mu=fit["mu"], sigma=fit["sigma"], lambda_=fit["lambda_"],
            pse_ms=1000 * mu_to_shift_s(fit["mu"]),
        ))
    return pd.DataFrame(rows)


# ── Quick model fits ──────────────────────────────────────────────────────────

def _fixed_sigmas(code):
    from runFitting_fixedUnimodalSigma import load_fixed_sigmas

    return load_fixed_sigmas(REPO / "data" / f"{code}_auditory_fits.mat",
                             REPO / "data" / f"{code}_visual_fits.mat")


def _make_fitter(data, code, model, n_simul):
    from monteCarloClass import OmerMonteCarlo

    with contextlib.redirect_stdout(io.StringIO()):
        fitter = OmerMonteCarlo(data, dataName=code)
    fitter.modelName = model
    fitter.sharedLambda = True
    fitter.freeP_c = False
    fitter.nSimul = n_simul
    return fitter


_GRIDS = {
    "lognorm": [[0.01, 0.05, 0.1, 0.2, 0.3], [0.05, 0.25, 0.5, 0.75, 0.95, 0.999]],
    "switchingFree": [[0.02, 0.08, 0.2], [0.05, 0.3, 0.6, 0.9], [0.05, 0.3, 0.6, 0.9]],
}


def _fit_one(task):
    """Grid search + bounded Powell on a common-random-number likelihood, then one
    re-evaluation with more Monte Carlo samples for the reported AIC."""
    from runFitting_fixedUnimodalSigma import free_bounds, full_vector

    dataset, code, model, data, n_simul, n_simul_final, seed, out_path = task
    started = time.time()
    sigmas = _fixed_sigmas(code)
    fitter = _make_fitter(data, code, model, n_simul)
    bounds = free_bounds(model)
    n_eval = 0

    def objective(free):
        nonlocal n_eval
        n_eval += 1
        free = np.clip(np.atleast_1d(np.asarray(free, dtype=float)), bounds[:, 0], bounds[:, 1])
        np.random.seed(seed)
        return float(fitter.nLLMonteCarloCausal(full_vector(model, free, sigmas), fitter.groupedData))

    if model == "fusionOnlyLogNorm":
        res = minimize_scalar(lambda lam: objective([lam]), bounds=tuple(bounds[0]),
                              method="bounded", options={"xatol": 1e-3})
        best = np.array([res.x])
    else:
        candidates = [np.array(c, dtype=float) for c in itertools.product(*_GRIDS[model])]
        values = [objective(c) for c in candidates]
        x0 = candidates[int(np.argmin(values))]
        res = minimize(objective, x0, method="Powell", bounds=bounds,
                       options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 150})
        best = np.clip(res.x, bounds[:, 0], bounds[:, 1])
        if objective(best) > min(values):
            best = x0

    nll_opt = objective(best)
    fitter.nSimul = n_simul_final
    np.random.seed(seed + 1)
    full = full_vector(model, best, sigmas)
    nll = float(fitter.nLLMonteCarloCausal(full, fitter.groupedData))
    k = K_FREE[model]
    result = dict(
        dataset=dataset, code=code, participantID=PARTICIPANTS[code], model=model,
        modelLabel=MODEL_LABELS[model], freeParams=best.tolist(), fittedParams=full.tolist(),
        nLL=nll, nLL_optimizer=nll_opt, AIC=2 * k + 2 * nll, k=k,
        n_conditions=int(len(fitter.groupedData)), n_trials=int(len(data)),
        sigmas=sigmas, nSimulFit=n_simul, nSimulFinal=n_simul_final, nEval=n_eval,
        seconds=round(time.time() - started, 1),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def run_model_fits(datasets, n_jobs=7, n_simul=1000, n_simul_final=5000, seed=0, force=False):
    """Fit all models to every participant in every dataset; cached per fit as JSON."""
    results, tasks = [], []
    for name, trials in datasets.items():
        for code in PARTICIPANTS:
            data = trials[trials["code"] == code]
            for model in MODELS:
                out = OUT_DIR / "fits" / name / f"{code}_{model}.json"
                if out.exists() and not force:
                    results.append(json.loads(out.read_text()))
                else:
                    tasks.append((name, code, model, data, n_simul, n_simul_final, seed, out))
    if tasks:
        # slowest models first so the pool stays busy
        tasks.sort(key=lambda t: MODELS.index(t[2]), reverse=True)
        print(f"Fitting {len(tasks)} participant x model x dataset combinations on {n_jobs} workers ...")
        started = time.time()
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_fit_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                print(f"  [{i}/{len(tasks)}] {r['dataset']:8s} {r['code']:3s} {r['modelLabel']:28s} "
                      f"AIC={r['AIC']:9.1f}  ({r['seconds']:.0f}s, {time.time() - started:.0f}s total)",
                      flush=True)
    return pd.DataFrame(results)


# ── Model-predicted PSEs ──────────────────────────────────────────────────────

def _predict_one(task):
    from free_psychometric_refits import fit_pf_counts, mu_to_shift_s

    row, data, cells, n_simul, grid = task
    fitter = _make_fitter(data, row["code"], row["model"], n_simul)
    params = np.asarray(row["fittedParams"], dtype=float)
    out = []
    for noise, conflict in cells:
        lam, sig_a, sig_v, p, t_min, t_max = fitter.getParamsCausal(params, noise, conflict)[:6]
        np.random.seed(0)
        p_longer = np.array([
            fitter.probTestLonger_vectorized_mc((0.5, t, 0.5 + conflict, t), sig_a, sig_v, p, lam, t_min, t_max)
            for t in grid
        ])
        fit = fit_pf_counts(np.log(grid / 0.5), p_longer * n_simul, np.full(grid.size, float(n_simul)))
        out.append(dict(dataset=row["dataset"], code=row["code"], participantID=row["participantID"],
                        model=row["model"], modelLabel=row["modelLabel"], audNoise=float(noise),
                        conflict=float(conflict), conflict_ms=int(round(conflict * 1000)),
                        pse_ms=1000 * mu_to_shift_s(fit["mu"])))
    return out


def model_pse(fits, datasets, n_jobs=7, n_simul=4000):
    """Model-predicted PSE per participant x noise x nominal conflict, for every fit."""
    grid = np.exp(np.linspace(np.log(0.15), np.log(1.4), 41))
    tasks = []
    for _, row in fits.iterrows():
        data = datasets[row["dataset"]]
        data = data[data["code"] == row["code"]]
        cells = sorted({(float(n), float(c)) for n, c in zip(data["audNoise"], data["conflict_nominal"])})
        tasks.append((row.to_dict(), data, cells, n_simul, grid))
    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for chunk in pool.map(_predict_one, tasks):
            rows.extend(chunk)
    return pd.DataFrame(rows)


def main(n_jobs=7):
    OUT_DIR.mkdir(exist_ok=True)
    trials = build_trials(force=True)
    check = validate_against_shared(trials)
    print(check.to_string())
    if not check["match"].all():
        raise RuntimeError("Rebuilt trial table does not match av_dur_data/all_bimodal.csv")
    balanced = make_balanced(trials)
    datasets = {"full": trials, "balanced": balanced}
    pse = pd.concat([fit_pse(d, name) for name, d in datasets.items()], ignore_index=True)
    pse.to_csv(OUT_DIR / "pse_fits.csv", index=False)
    fits = run_model_fits(datasets, n_jobs=n_jobs)
    fits.to_csv(OUT_DIR / "model_fits.csv", index=False)
    preds = model_pse(fits, datasets, n_jobs=n_jobs)
    preds.to_csv(OUT_DIR / "model_pse.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
