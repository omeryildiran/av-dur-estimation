#!/usr/bin/env python3
"""Fit the three main AV models with sensory noise fixed from unimodal fits.

The auditory and visual psychometric-fit sigmas describe the difference between
two noisy intervals.  They are therefore divided by sqrt(2) before being used
as the single-interval sensory-noise parameters in the AV models.

Only the genuinely free parameters are optimized:

    fusionOnlyLogNorm : shared lapse
    lognorm           : shared lapse, shared p_common
    switchingFree     : shared lapse, p_visual_low, p_visual_high

The saved ``fittedParams`` remain full model vectors so that the existing model
prediction and plotting code can load them without modification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

import loadData
from monteCarloClass import OmerMonteCarlo


MAIN_MODELS = ("fusionOnlyLogNorm", "lognorm", "switchingFree")


def participant_paths(participant: str, data_dir: Path) -> tuple[str, Path, Path, Path]:
    """Resolve participant case while retaining the repository's HH filename."""
    raw = Path(participant).stem
    if raw.lower().endswith("_all"):
        raw = raw[:-4]

    candidates = [raw, raw.lower(), raw.upper()]
    for candidate in dict.fromkeys(candidates):
        data_path = data_dir / f"{candidate}_all.csv"
        auditory_path = data_dir / f"{candidate.lower()}_auditory_fits.mat"
        visual_path = data_dir / f"{candidate.lower()}_visual_fits.mat"
        if data_path.exists() and auditory_path.exists() and visual_path.exists():
            return candidate, data_path, auditory_path, visual_path

    raise FileNotFoundError(
        f"Could not find the AV data and both unimodal fits for participant {participant!r} "
        f"under {data_dir}."
    )


def load_fixed_sigmas(auditory_path: Path, visual_path: Path) -> dict[str, float]:
    auditory = np.asarray(loadmat(auditory_path)["fittedParams"], dtype=float).ravel()
    visual = np.asarray(loadmat(visual_path)["fittedParams"], dtype=float).ravel()
    if auditory.size < 3 or visual.size < 2:
        raise ValueError("Unexpected unimodal fittedParams layout in the MAT files.")

    # Auditory MAT layout: [lambda, sigma_high_noise, sigma_low_noise, ...].
    # AV vector layout: sigma_a_low (audNoise=.1), sigma_v, sigma_a_high (audNoise=1.2).
    return {
        "sigma_a_low": float(auditory[2] / np.sqrt(2)),
        "sigma_v": float(visual[1] / np.sqrt(2)),
        "sigma_a_high": float(auditory[1] / np.sqrt(2)),
    }


def free_bounds(model: str) -> np.ndarray:
    if model == "fusionOnlyLogNorm":
        return np.asarray([(0.001, 0.4)], dtype=float)
    if model == "lognorm":
        return np.asarray([(0.001, 0.4), (0.0, 1.0)], dtype=float)
    if model == "switchingFree":
        return np.asarray([(0.001, 0.4), (0.0, 1.0), (0.0, 1.0)], dtype=float)
    raise ValueError(f"Unsupported model {model!r}; choose from {MAIN_MODELS}.")


def full_vector(model: str, free: np.ndarray, sigmas: dict[str, float]) -> np.ndarray:
    lapse = float(free[0])
    sa_low = sigmas["sigma_a_low"]
    sv = sigmas["sigma_v"]
    sa_high = sigmas["sigma_a_high"]
    if model == "fusionOnlyLogNorm":
        return np.asarray([lapse, sa_low, sv, sa_high])
    if model == "lognorm":
        return np.asarray([lapse, sa_low, sv, free[1], sa_high])
    if model == "switchingFree":
        return np.asarray([lapse, sa_low, sv, free[1], sa_high, free[2]])
    raise ValueError(model)


def optimize_model(
    fitter: OmerMonteCarlo,
    model: str,
    sigmas: dict[str, float],
    optimizer: str,
    n_starts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, list[dict]]:
    bounds = free_bounds(model)

    def objective(free: np.ndarray) -> float:
        return float(fitter.nLLMonteCarloCausal(full_vector(model, free, sigmas), fitter.groupedData))

    if optimizer == "bads":
        try:
            from pybads import BADS
        except ImportError as exc:
            raise RuntimeError(
                "--optimizer bads requires pybads in the active environment."
            ) from exc

    best_free = None
    best_nll = np.inf
    history = []
    lb, ub = bounds[:, 0], bounds[:, 1]
    plb = lb + 0.2 * (ub - lb)
    pub = ub - 0.2 * (ub - lb)

    for start in range(1, n_starts + 1):
        x0 = rng.uniform(plb, pub)
        started = time.time()
        if optimizer == "bads":
            result = BADS(
                objective, x0, lb, ub, plb, pub, options={"display": "off"}
            ).optimize()
            x = np.asarray(result.x, dtype=float)
            nll = float(result.fval)
            success = True
            message = "BADS completed"
        else:
            result = minimize(objective, x0, method="Powell", bounds=bounds)
            x = np.asarray(result.x, dtype=float)
            nll = float(result.fun)
            success = bool(result.success)
            message = str(result.message)

        # Re-evaluate once at the returned point because the likelihood is a
        # Monte Carlo approximation and the optimizer's cached value is noisy.
        nll_checked = objective(np.clip(x, lb, ub))
        history.append(
            {
                "start": start,
                "initialFreeParams": x0.tolist(),
                "fittedFreeParams": x.tolist(),
                "optimizerNLL": nll,
                "checkedNLL": nll_checked,
                "success": success,
                "message": message,
                "elapsedSeconds": time.time() - started,
            }
        )
        print(
            f"  start {start}/{n_starts}: checked nLL={nll_checked:.3f}, "
            f"free={np.round(x, 5)}"
        )
        if np.isfinite(nll_checked) and nll_checked < best_nll:
            best_nll = nll_checked
            best_free = np.clip(x, lb, ub)

    if best_free is None:
        raise RuntimeError(f"All optimization starts failed for {model}.")
    return full_vector(model, best_free, sigmas), best_nll, history


def save_result(
    output_dir: Path,
    participant_id: str,
    model: str,
    fitted: np.ndarray,
    nll: float,
    n_conditions: int,
    sigmas: dict[str, float],
    source_files: tuple[Path, Path],
    n_simul: int,
    optimizer: str,
    history: list[dict],
    overwrite: bool,
) -> Path:
    participant_dir = output_dir / participant_id
    participant_dir.mkdir(parents=True, exist_ok=True)
    path = participant_dir / f"{participant_id}_{model}_LapseFix_sharedPrior_fit.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")

    k = {"fusionOnlyLogNorm": 1, "lognorm": 2, "switchingFree": 3}[model]
    log_likelihood = -float(nll)
    payload = {
        "participantID": participant_id,
        "modelType": f"{model}_LapseFix_sharedPrior",
        "fittedParams": fitted.tolist(),
        "AIC": float(2 * k - 2 * log_likelihood),
        "BIC": float(np.log(n_conditions) * k - 2 * log_likelihood),
        "logLikelihood": log_likelihood,
        "n_conditions": int(n_conditions),
        "nFreeParameters": k,
        "sharedLambda": True,
        "freeP_c": False,
        "fixedSensorySigmas": sigmas,
        "unimodalSigmaSourceFiles": [str(path) for path in source_files],
        "nSimul": int(n_simul),
        "optimizer": optimizer,
        "nStartHistory": history,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("participant", help="Participant ID (for example as or HH).")
    parser.add_argument(
        "--models", nargs="+", choices=MAIN_MODELS, default=list(MAIN_MODELS)
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("unimodalSigma_modelFits_singleLapse")
    )
    parser.add_argument("--n-simul", type=int, default=3000)
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--optimizer", choices=("bads", "scipy"), default="bads")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_simul < 10:
        parser.error("--n-simul must be at least 10")
    if args.n_starts < 1:
        parser.error("--n-starts must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    resolved_id, data_path, auditory_path, visual_path = participant_paths(
        args.participant, args.data_dir
    )
    participant_id = resolved_id.lower()
    sigmas = load_fixed_sigmas(auditory_path, visual_path)
    print(f"Participant: {participant_id}")
    print(f"Fixed sensory sigmas: {sigmas}")

    # loadData expects a filename relative to ./data.
    if args.data_dir.resolve() == Path("data").resolve():
        data, data_name = loadData.loadData(data_path.name)
    else:
        old_cwd = Path.cwd()
        try:
            os.chdir(args.data_dir.parent)
            data, data_name = loadData.loadData(data_path.name)
        finally:
            os.chdir(old_cwd)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # Monte Carlo draws inside OmerMonteCarlo use the legacy RNG.
    for model in args.models:
        print(f"\nFitting {model} with one shared lapse ...")
        fitter = OmerMonteCarlo(data, dataName=data_name)
        fitter.modelName = model
        fitter.sharedLambda = True
        fitter.freeP_c = False
        fitter.nSimul = args.n_simul
        fitted, nll, history = optimize_model(
            fitter, model, sigmas, args.optimizer, args.n_starts, rng
        )
        saved = save_result(
            args.output_dir,
            participant_id,
            model,
            fitted,
            nll,
            len(fitter.groupedData),
            sigmas,
            (auditory_path, visual_path),
            args.n_simul,
            args.optimizer,
            history,
            args.overwrite,
        )
        print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
