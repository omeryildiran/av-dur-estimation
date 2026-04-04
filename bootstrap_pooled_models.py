"""Bootstrap pooled model psychometric fits in parallel.

For each model, loads the simulated data and its psychometric fit,
then runs parametric bootstrap using fitMainClass.paramBootstrap.
All 3 models run in parallel via multiprocessing.

Usage:
    python bootstrap_pooled_models.py [--nboots 200]
"""

import argparse
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from fitMainClass import fitPychometric as fitPsychometric


MODELS = {
    "lognorm": "all_lognorm_LapseFree_sharedPrior_simulated.csv",
    "fusionOnlyLogNorm": "all_fusionOnlyLogNorm_LapseFree_sharedPrior_simulated.csv",
    "switchingFree": "all_switchingFree_LapseFree_sharedPrior_simulated.csv",
}

FIT_DIR = "psychometric_fits_simulated/all"
SIM_DIR = "simulated_data/all"
OUT_DIR = "bootstrapped_params/all"


def run_bootstrap(args):
    model_name, n_boots = args
    sim_path = os.path.join(SIM_DIR, MODELS[model_name])
    fit_path = os.path.join(FIT_DIR, f"all_{model_name}_psychometricFits.json")

    print(f"[{model_name}] Loading simulated data from {sim_path}")
    sim_data = pd.read_csv(sim_path)
    fitter = fitPsychometric(sim_data, intensityVar="logDurRatio")

    with open(fit_path) as f:
        fit_params = json.load(f)["fitParams"]

    print(f"[{model_name}] Starting {n_boots} bootstrap iterations...")
    t0 = time.time()
    boots = fitter.paramBootstrap(fit_params, nBoots=n_boots)
    elapsed = time.time() - t0
    print(f"[{model_name}] Done in {elapsed/60:.1f} min "
          f"({elapsed/n_boots:.1f}s/iter), shape={boots.shape}")

    out_path = os.path.join(OUT_DIR, f"all_{model_name}_psychometric_bootstrapped_params.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(boots.tolist(), f)
    print(f"[{model_name}] Saved to {out_path}")
    return model_name, boots.shape, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nboots", type=int, default=200,
                        help="Number of bootstrap iterations per model")
    args = parser.parse_args()

    parser.add_argument("--sequential", action="store_true",
                        help="Run models one at a time (default: parallel)")
    args = parser.parse_args()

    jobs = [(name, args.nboots) for name in MODELS]

    if args.sequential:
        print(f"Running {args.nboots} bootstraps for {len(MODELS)} models sequentially")
        results = [run_bootstrap(j) for j in jobs]
    else:
        n_workers = min(len(MODELS), mp.cpu_count() - 1)
        print(f"Running {args.nboots} bootstraps for {len(MODELS)} models "
              f"using {n_workers} parallel workers")
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(run_bootstrap, jobs)
    print("\n=== SUMMARY ===")
    total = 0
    for name, shape, elapsed in results:
        total += elapsed
        print(f"  {name:25s}: {shape}, {elapsed/60:.1f} min")
    print(f"  {'TOTAL':25s}: {total/60:.1f} min")


if __name__ == "__main__":
    main()
