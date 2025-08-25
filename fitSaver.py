import os
import numpy as np
import json
def saveFitResultsSingle(fitter,fittedParams, dataName,modelType="lognormal"):
    participantID = dataName.split('_')[0]
    save_dir = os.path.join("model_fits", participantID)

    filename = f"{participantID}_{modelType}_fit.json"
    filepath = os.path.join(save_dir, filename)

    # make directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # calculate AIC and BIC
    k=len(fittedParams) # number of parameters
    n=len(fitter.groupedData) # number of data points
    fitter.logLikelihood= -fitter.nLLMonteCarloCausal(fittedParams, fitter.groupedData)
    aic= 2*k - 2*fitter.logLikelihood
    bic= np.log(n)*k - 2*fitter.logLikelihood

    # store results in a dictionary
    results_dict = {
        "participantID": participantID,
        "modelType": modelType,
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


#saveFitResultsSingle(fittedParams, dataName, modelType="lognormal")