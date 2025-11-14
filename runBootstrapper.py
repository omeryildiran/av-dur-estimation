import os
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

def saveBootstrappedParams(mc_fitter=None, dataBoots=None, dataName=None):
    """
    Save bootstrapped parameters to JSON file, preserving array structure
    """
    participantID = dataName.split(".csv")[0]


    modelType="dataFit"


    filename = f"{participantID.split('_')[0]}_{modelType}_bootstrapped_params.json"
    filename = os.path.join("bootstrapped_params", participantID.split('_')[0], filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert numpy array to list while preserving structure
    if dataBoots is None:
        print("Warning: dataBoots is None, saving empty array")
        dataBoots_serializable = []
    elif isinstance(dataBoots, np.ndarray):
        # Convert the entire numpy array to a nested list, preserving all dimensions
        dataBoots_serializable = dataBoots.tolist()
    elif isinstance(dataBoots, (list, tuple)):
        # If it's already a list/tuple, convert any numpy arrays within it
        dataBoots_serializable = []
        for item in dataBoots:
            if isinstance(item, np.ndarray):
                dataBoots_serializable.append(item.tolist())
            else:
                dataBoots_serializable.append(item)
    else:
        # For other types, try to convert directly
        try:
            if hasattr(dataBoots, 'tolist'):
                dataBoots_serializable = dataBoots.tolist()
            else:
                dataBoots_serializable = [dataBoots]
        except Exception as e:
            print(f"Warning: Could not serialize dataBoots: {e}")
            dataBoots_serializable = []

    try:
        with open(filename, 'w') as f:
            json.dump(dataBoots_serializable, f, indent=2)
        print(f"Bootstrapped parameters saved to {filename}")
        print(f"Saved array shape: {np.array(dataBoots_serializable).shape if dataBoots_serializable else 'empty'}")
        return True
    except Exception as e:
        print(f"Error saving bootstrapped parameters: {e}")
        return False



# # Try to load existing bootstrapped parameters to test
# dataBoots = loadBootstrappedParams(mc_fitter, dataName)
# print(f"Original dataBoots shape: {dataBoots.shape}")



# List of all participant filenameshape
participant_files = [
    "as_all.csv", "oy_all.csv", "dt_all.csv", "HH_all.csv", 
    "ip_all.csv", "ln1_all.csv", "ln2_all.csv", "mh_all.csv", 
    "ml_all.csv", "mt_all.csv", "qs_all.csv", "sx_all.csv"
]
#participant_files=["dt_all.csv"]  # For testing only

#from monteCarloClass import MonteCarloFitter
import monteCarloClass

nBoots = 50

def SimpleNamespace(**kwargs):
    class Namespace:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    return Namespace(**kwargs)


def process_participant(dataName):
    """
    Process a single participant file - this function will be run in parallel
    """
    import monteCarloClass
    import loadData
    import psychometricFitLoader as pfl_data
    
    try:
        print(f"Processing {dataName}...")
        filename = dataName
        data, dataName = loadData.loadData(filename)

        mc_fitter = monteCarloClass.OmerMonteCarlo(data)
        participant_id = dataName.split("_")[0]
        mc_fitter.dataFit =mc_fitter.fitMultipleStartingPoints(data,1)
        #a = mc_fitter.dataFit["parameters"]
        #mc_fitter.dataFit = SimpleNamespace(**mc_fitter.dataFit)
        #mc_fitter.dataFit.x = a

        dataBoots = mc_fitter.paramBootstrap(mc_fitter.dataFit.x, nBoots=nBoots)
        #print(f"Bootstrapped data shape for {dataName}: {dataBoots.shape}")
        
        saveBootstrappedParams(mc_fitter=None, dataBoots=dataBoots, dataName=dataName)
        print(f"Completed {dataName}")
        return (dataName, True, None)
    except Exception as e:
        print(f"Error processing {dataName}: {e}")
        return (dataName, False, str(e))


import loadData

if __name__ == '__main__':
    # Determine number of cores to use (leave 1 core free for system)
    n_cores = max(1, cpu_count() - 1)
    print(f"Using {n_cores} cores out of {cpu_count()} available")
    print(f"Processing {len(participant_files)} participants with {nBoots} bootstrap iterations each\n")
    
    # Run in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(process_participant, participant_files)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    print(f"Successfully processed: {successful}/{len(participant_files)}")
    if failed > 0:
        print(f"Failed: {failed}")
        for dataName, success, error in results:
            if not success:
                print(f"  - {dataName}: {error}")

    