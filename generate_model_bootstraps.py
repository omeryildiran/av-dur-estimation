import json
import os
from pathlib import Path

import numpy as np

import loadData
import monteCarloClass


PARTICIPANT_IDS = ["as", "oy", "dt", "HH", "ip", "ln2", "mh", "ml", "mt", "qs", "sx"]
MODELS = ["lognorm", "switchingFree", "fusionOnlyLognorm"]
N_BOOTS = 1000


def save_bootstraps(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values)
    with open(path, "w") as f:
        json.dump(arr.tolist(), f, indent=2)
    return arr.shape


def main():
    print(f"Generating model bootstraps for {len(PARTICIPANT_IDS)} participants x {len(MODELS)} models")
    print(f"Bootstrap iterations per fit: {N_BOOTS}")

    results = []
    for participant_id in PARTICIPANT_IDS:
        csv_name = f"{participant_id}_all.csv"
        print(f"\n=== Participant {participant_id} ({csv_name}) ===")
        data, data_name = loadData.loadData(csv_name, verbose=False)
        mc_fitter = monteCarloClass.OmerMonteCarlo(data)
        mc_fitter.freeP_c = False
        mc_fitter.sharedLambda = True
        mc_fitter.dataName = data_name

        for model_name in MODELS:
            sim_fit_path = Path("psychometric_fits_simulated") / participant_id / f"{participant_id}_{model_name}_psychometricFits.json"
            if not sim_fit_path.exists():
                print(f"  Missing simulated fit for {model_name}: {sim_fit_path}")
                results.append((participant_id, model_name, False, "missing simulated fit"))
                continue

            with open(sim_fit_path, "r") as f:
                sim_fit = json.load(f)

            print(f"  Bootstrapping {model_name} from {sim_fit_path}")
            try:
                boot_values = mc_fitter.paramBootstrap(sim_fit["fitParams"], nBoots=N_BOOTS)
                out_path = Path("bootstrapped_params") / participant_id / f"{participant_id}_{model_name}_sharedPrior_bootstrapped_params.json"
                shape = save_bootstraps(out_path, boot_values)
                print(f"    Saved {out_path} shape={shape}")
                results.append((participant_id, model_name, True, str(shape)))
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append((participant_id, model_name, False, str(e)))

    print("\n=== SUMMARY ===")
    ok = sum(1 for _, _, success, _ in results if success)
    print(f"Successful: {ok}/{len(results)}")
    for participant_id, model_name, success, info in results:
        status = "OK" if success else "FAIL"
        print(f"{status:4} {participant_id:>4}  {model_name:<18} {info}")


if __name__ == "__main__":
    main()
