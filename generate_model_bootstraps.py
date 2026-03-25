import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

import loadData
import monteCarloClass


PARTICIPANT_IDS = ["as", "oy", "dt", "HH", "ip", "ln2", "mh", "ml", "mt", "qs", "sx"]
MODELS = ["lognorm", "switchingFree", "fusionOnlyLognorm"]
N_BOOTS = 1000
N_WORKERS = max(1, cpu_count() - 1)


def save_bootstraps(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values)
    with open(path, "w") as f:
        json.dump(arr.tolist(), f, indent=2)
    return arr.shape


def process_job(job):
    participant_id, model_name, n_boots = job
    csv_name = f"{participant_id}_all.csv"
    sim_fit_path = Path("psychometric_fits_simulated") / participant_id / f"{participant_id}_{model_name}_psychometricFits.json"
    if not sim_fit_path.exists():
        return (participant_id, model_name, False, "missing simulated fit")

    try:
        with open(sim_fit_path, "r") as f:
            sim_fit = json.load(f)

        data, data_name = loadData.loadData(csv_name, verbose=False)
        mc_fitter = monteCarloClass.OmerMonteCarlo(data)
        mc_fitter.freeP_c = False
        mc_fitter.sharedLambda = True
        mc_fitter.dataName = data_name

        boot_values = mc_fitter.paramBootstrap(sim_fit["fitParams"], nBoots=n_boots)
        out_path = Path("bootstrapped_params") / participant_id / f"{participant_id}_{model_name}_sharedPrior_bootstrapped_params.json"
        shape = save_bootstraps(out_path, boot_values)
        return (participant_id, model_name, True, f"{out_path} shape={shape}")
    except Exception as e:
        return (participant_id, model_name, False, str(e))


def main():
    print(f"Generating model bootstraps for {len(PARTICIPANT_IDS)} participants x {len(MODELS)} models")
    print(f"Bootstrap iterations per fit: {N_BOOTS}")
    print(f"Using {N_WORKERS} worker processes out of {cpu_count()} CPUs")

    jobs = [(participant_id, model_name, N_BOOTS) for participant_id in PARTICIPANT_IDS for model_name in MODELS]
    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(process_job, jobs)

    print("\n=== SUMMARY ===")
    ok = sum(1 for _, _, success, _ in results if success)
    print(f"Successful: {ok}/{len(results)}")
    for participant_id, model_name, success, info in results:
        status = "OK" if success else "FAIL"
        print(f"{status:4} {participant_id:>4}  {model_name:<18} {info}")


if __name__ == "__main__":
    main()
