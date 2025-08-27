
if __name__ == "__main__":
    # Main script to run the Monte Carlo fitting for the causal inference model
    #arguments: data file name, model type (e.g., "lognorm" or "gauss"), number of simulations, optimization method, number of starts
    # Example usage: python runFitting.py mt_all.csv lognorm 2000 bad
    
    # take arguments from command line
    import sys
# take arguments from command line
    dataFiles = sys.argv[1].split(',') if len(sys.argv) > 1 else ["mt_all.csv"]
    modelName = sys.argv[2] if len(sys.argv) > 2 else "lognorm"
    nSimul = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    optimMethod = sys.argv[4] if len(sys.argv) > 4 else "bads"
    nStarts = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    print(f"Data file: {dataFiles}, Model: {modelName}, Simulations: {nSimul}, Optimizer: {optimMethod}, Starts: {nStarts}")



    import os
    import numpy as np
    import pandas as pd
    import loadData
    import monteCarloClass
    import time
    import fitSaver

    for dataFile in dataFiles:
        print(f"\n=== Processing file: {dataFile} ===")

        # Load the data
        data, dataName = loadData.loadData(dataFile)

        intensityVariable = "deltaDurS"
        sensoryVar = "audNoise"
        standardVar = "standardDur"
        conflictVar = "conflictDur"
        #s
        visualStandardVar = "unbiasedVisualStandardDur"
        visualTestVar = "unbiasedVisualTestDur"
        audioStandardVar = "standardDur"
        audioTestVar = "testDurS"


        ## Initialize the Monte Carlo fitter
        mc_fitter = monteCarloClass.OmerMonteCarlo(data)

        # Define variables in the fitter
        print("Data shape:", data.shape)
        print("\nConflict range:", data["conflictDur"].min(), "to", data["conflictDur"].max())
        print("Standard duration:", data["standardDur"].unique())
        print("Audio noise levels:", sorted(data["audNoise"].unique()))
        print("Visual test duration range:", data["recordedDurVisualTest"].min(), "to", data["recordedDurVisualTest"].max())
        print("t_min, t_max:", mc_fitter.t_min, mc_fitter.t_max)
        print("Log t_min, Log t_max:", np.log(mc_fitter.t_min), np.log(mc_fitter.t_max))

        # Set fitter parameters
        mc_fitter.nSimul = nSimul
        mc_fitter.optimizationMethod= optimMethod  # Use BADS for optimization
        mc_fitter.nStart = nStarts # Number of random starts for optimization
        mc_fitter.modelName = modelName  # Set measurement distribution to Gaussian

        # Fit the model and time it
        timeStart = time.time()
        print(f"\nFitting Causal Inference Model for {dataName} with {len(mc_fitter.groupedData)} unique conditions")
        fittedParams = mc_fitter.fitCausalInferenceMonteCarlo(mc_fitter.groupedData)
        print(f"\nFitted parameters for {dataName}: {fittedParams}")
        print(f"Time taken to fit: {time.time() - timeStart:.2f} seconds")
        mc_fitter.modelFit= fittedParams
        mc_fitter.logLikelihood= -mc_fitter.nLLMonteCarloCausal(fittedParams, mc_fitter.groupedData)

        # Save the fit results
        fitSaver.saveFitResultsSingle(mc_fitter, fittedParams, dataName, modelType=modelName)