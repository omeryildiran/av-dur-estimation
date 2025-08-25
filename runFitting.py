if __name__ == "__main__":
    # Main script to run the Monte Carlo fitting for the causal inference model
    #arguments: data file name, model type (e.g., "lognorm" or "gauss"), number of simulations, optimization method, number of starts
    # Example usage: python runFitting.py mt_all.csv lognorm 2000 bad
    
    # take arguments from command line
    import sys
    import multiprocessing as mp
    from functools import partial
    
    dataFiles = sys.argv[1].split(',') if len(sys.argv) > 1 else ["mt_all.csv"]
    modelName = sys.argv[2] if len(sys.argv) > 2 else "all"  # "all" to run all models
    nSimul = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    optimMethod = sys.argv[4] if len(sys.argv) > 4 else "bads"
    nStarts = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    
    # Define all models
    all_models = ["lognorm", "logLinearMismatch", "gaussian"]
    models_to_run = all_models if modelName == "all" else [modelName]
    
    print(f"Data files: {dataFiles}, Models: {models_to_run}, Simulations: {nSimul}, Optimizer: {optimMethod}, Starts: {nStarts}")

    import os
    import numpy as np
    import pandas as pd
    import loadData
    import monteCarloClass
    import time
    import fitSaver

    def fit_model_for_data(args):
        """Function to fit a single model for a single data file"""
        dataFile, model, nSimul, optimMethod, nStarts = args
        
        print(f"\n=== Processing file: {dataFile} with model: {model} ===")
        
        # Load the data
        data, dataName = loadData.loadData(dataFile)
        
        ## Initialize the Monte Carlo fitter
        mc_fitter = monteCarloClass.OmerMonteCarlo(data)
        
        # Set fitter parameters
        mc_fitter.nSimul = nSimul
        mc_fitter.optimizationMethod = optimMethod
        mc_fitter.nStart = nStarts
        mc_fitter.modelName = model
        
        # Fit the model and time it
        timeStart = time.time()
        print(f"\nFitting {model} model for {dataName} with {len(mc_fitter.groupedData)} unique conditions")
        
        try:
            fittedParams = mc_fitter.fitCausalInferenceMonteCarlo(mc_fitter.groupedData)
            print(f"\nFitted parameters for {dataName} ({model}): {fittedParams}")
            print(f"Time taken to fit: {time.time() - timeStart:.2f} seconds")
            
            mc_fitter.modelFit = fittedParams
            mc_fitter.logLikelihood = -mc_fitter.nLLMonteCarloCausal(fittedParams, mc_fitter.groupedData)
            
            # Save the fit results
            fitSaver.saveFitResultsSingle(mc_fitter, fittedParams, dataName, modelType=model)
            
            return f"SUCCESS: {dataFile} - {model}"
        except Exception as e:
            return f"ERROR: {dataFile} - {model}: {str(e)}"

    # Create all combinations of data files and models
    tasks = [(dataFile, model, nSimul, optimMethod, nStarts) 
             for dataFile in dataFiles 
             for model in models_to_run]
    
    # Run in parallel
    n_processes = min(len(tasks), mp.cpu_count())  # Use available CPUs but don't exceed task count
    print(f"\nRunning {len(tasks)} tasks across {n_processes} processes...")
    
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(fit_model_for_data, tasks)
    
    # Print results summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY:")
    print("="*50)
    for result in results:
        print(result)