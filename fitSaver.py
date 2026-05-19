import os
import numpy as np
import json


def _resolveSaveLocation(rootDir, participantID, noiseLabel):
    """Build the output directory, splitting by noiseLabel when provided."""
    if noiseLabel is None:
        return os.path.join(rootDir, participantID)
    return os.path.join(f"{rootDir}_byNoise", noiseLabel, participantID)


def saveFitResultsSingle(fitter, fittedParams, dataName, noiseLabel=None):
    participantID = dataName.split('_')[0]
    save_dir = _resolveSaveLocation("model_fits", participantID, noiseLabel)
    modelType = fitter.modelName

    if fitter.sharedLambda:
        modelType += "_LapseFix"
    else:
        modelType += "_LapseFree"

    if fitter.freeP_c:
        modelType += "_contextualPrior"
    else:
        modelType += "_sharedPrior"



    filename = f"{participantID}_{modelType}_fit.json"

    filepath = os.path.join(save_dir, filename)

    # make directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # calculate AIC and BIC
    k = fitter.getActualParameterCount()  # actual fitted parameters only
    n=len(fitter.groupedData) # number of data points
    fitter.logLikelihood= -fitter.nLLMonteCarloCausal(fittedParams, fitter.groupedData)
    aic= 2*k - 2*fitter.logLikelihood
    bic= np.log(n)*k - 2*fitter.logLikelihood

    

    # store results in a dictionary
    results_dict = {
        "participantID": participantID,
        "modelType": modelType,
        "noiseLabel": noiseLabel,
        "fittedParams": fittedParams.tolist() if isinstance(fittedParams, np.ndarray) else fittedParams,
        "AIC": float(aic),
        "BIC": float(bic),
        "logLikelihood": fitter.logLikelihood,
        "n_conditions": n
    }

    # Save JSON
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"✅ Saved fit for {participantID} ({modelType}) to: {filepath}")

def saveSimulatedData(fitter, dataName, noiseLabel=None):
    print(f"Generating SimData based on model  {fitter.modelName}")
    # generate simulated data
    simulatedData= fitter.simulateMonteCarloData(fitter.modelFit, fitter.data)

    if simulatedData is None or simulatedData.empty:
        print("No simulated data to save.")
        return

    participantID = dataName.split('_')[0]
    save_dir = _resolveSaveLocation("simulated_data", participantID, noiseLabel)
    modelType = fitter.modelName
    
    if fitter.sharedLambda:
        modelType += "_LapseFix"
    else:
        modelType += "_LapseFree"
    
    if fitter.freeP_c:
        modelType += "_contextualPrior"
    else:
        modelType += "_sharedPrior"

    filename = f"{participantID}_{modelType}_simulated.csv"

    filepath = os.path.join(save_dir, filename)

    # make directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save CSV
    simulatedData.to_csv(filepath, index=False)
        
    print(f"✅ Saved simulated data for {participantID} ({modelType}) to: {filepath}")

#saveFitResultsSingle(fittedParams, dataName, modelType="lognormal")