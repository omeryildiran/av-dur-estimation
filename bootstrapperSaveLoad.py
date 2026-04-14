import os
import json
import numpy as np
import pandas as pd
def saveBootstrappedParams(mc_fitter, dataBoots, dataName):
    """
    Save bootstrapped parameters to JSON file, preserving array structure
    """
    participantID = dataName.split(".csv")[0]
    modelType = mc_fitter.modelName

    if mc_fitter.sharedLambda:
        modelType += "_LapseFix"
    else:
        modelType += "_LapseFree"

    if mc_fitter.freeP_c:
        modelType += "_contextualPrior"
    else:
        modelType += "_sharedPrior"

    modelType="lognorm_LapseFree_sharedPrior"


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

def loadBootstrappedParams(mc_fitter, dataName):
    """
    Load bootstrapped parameters from JSON file, restoring original array structure
    """
    participantID = dataName.split(".csv")[0]
    modelType = mc_fitter.modelName


    filename = f"{participantID.split('_')[0]}_dataFit_bootstrapped_params.json"
    filename = os.path.join("bootstrapped_params", participantID.split('_')[0], filename)
    
    if not os.path.exists(filename):
        print(f"Bootstrapped parameters file not found: {filename}")
        return None
    
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            if not content:
                print(f"Warning: Empty file {filename}")
                return None
            
            dataBoots_list = json.loads(content)
        
        print(f"Loaded bootstrapped parameters from {filename}")
        
        # Convert the list back to numpy array, preserving the original structure
        if isinstance(dataBoots_list, list):
            if len(dataBoots_list) == 0:
                print("Warning: Loaded empty array")
                return np.array([])
            else:
                dataBoots_array = np.array(dataBoots_list)
                print(f"Loaded array shape: {dataBoots_array.shape}")
                return dataBoots_array
        else:
            # If for some reason it's not a list, try to convert anyway
            print("Warning: Loaded data is not a list, attempting conversion")
            return np.array(dataBoots_list)
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filename}: {e}")
        print("The file might be corrupted. Consider deleting it and regenerating the bootstrap data.")
        return None
    except Exception as e:
        print(f"Error loading bootstrapped parameters: {e}")
        return None# Save the current dataBoots array
#saveBootstrappedParams(mc_fitter, dataBoots, dataName)

# # Try to load existing bootstrapped parameters to test
# dataBoots = loadBootstrappedParams(mc_fitter, dataName)
# print(f"Original dataBoots shape: {dataBoots.shape}")

# List of all participant filenames
# participant_files = [
#     "as_all.csv", "oy_all.csv", "dt_all.csv", "HH_all.csv", 
#     "ip_all.csv", "ln_all.csv", "LN01_all.csv", "mh_all.csv", 
#     "ml_all.csv", "mt_all.csv", "qs_all.csv", "sx_all.csv"
# ]

# from monteCarloClass import OmerMonteCarlo as MonteCarloFitter
# # Load participant data
# from loadData import loadData


# for dataName in participant_files:
#     filename = file
#     data, dataName = loadData(filename)
#     mc_fitter = MonteCarloFitter(data)

#     participant_id = dataName.split("_")[0]
#     # Load or generate bootstrapped parameters for each participant
#     dataBoots = loadBootstrappedParams(mc_fitter, dataName)
#     if dataBoots is None:
#         print(f"No existing bootstrapped parameters for {dataName}, generating new ones...")
#         # Here you would generate new bootstrapped parameters as needed
#         # For demonstration, we'll create a dummy array
#         dataBoots = np.random.rand(100, len(mc_fitter.paramNames))  # Example shape
#         saveBootstrappedParams(mc_fitter, dataBoots, dataName)
#     else:
#         print(f"Bootstrapped parameters for {dataName} loaded successfully.")