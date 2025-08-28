import os
import json
def loadFitResults(dataName, modelName="lognorm"):
    participantID = dataName.split('_')[0]
    filepath = os.path.join("model_fits", participantID, f"{participantID}_{modelName}_fit.json")
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"No saved fit results found for {dataName} with model {modelName}.")
        return None
