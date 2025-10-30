
def process_single_file(args):
    """
    Process a single data file. This function will be called in parallel.
    
    Args:
        args: tuple containing (dataFile, modelName, nSimul, optimMethod, nStarts, integrationMethod)
    
    Returns:
        tuple: (dataFile, success, error_message)
    """
    dataFile, modelName, nSimul, optimMethod, nStarts, freeP_c, integrationMethod = args
    
    import os
    import numpy as np
    import pandas as pd
    import loadData
    import monteCarloClass
    import time
    import fitSaver
    
    try:
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

        # Set fitter parameters
        mc_fitter.nSimul = nSimul
        mc_fitter.optimizationMethod= optimMethod  # Use BADS for optimization
        mc_fitter.nStart = nStarts # Number of random starts for optimization
        mc_fitter.modelName = modelName  # Set measurement distribution to Gaussian
        mc_fitter.integrationMethod= integrationMethod # "numerical" or "analytical"
        mc_fitter.freeP_c = freeP_c  # Free prior probability of common cause
        print(f"Model name set to: {mc_fitter.modelName}")
        print(f"Shared lambda: {mc_fitter.sharedLambda}")
        print(f"Free P(C=1): {mc_fitter.freeP_c}")
        
        # Fit the model and time it
        timeStart = time.time()
        print(f"\nFitting Causal Inference Model for {dataName} with {len(mc_fitter.groupedData)} unique conditions")
        fittedParams = mc_fitter.fitCausalInferenceMonteCarlo(mc_fitter.groupedData)
        print(f"\nFitted parameters for {dataName}: {fittedParams}")
        print(f"Time taken to fit: {time.time() - timeStart:.2f} seconds")
        mc_fitter.modelFit= fittedParams
        mc_fitter.logLikelihood= -mc_fitter.nLLMonteCarloCausal(fittedParams, mc_fitter.groupedData)

        # Save the fit results
        fitSaver.saveFitResultsSingle(mc_fitter, fittedParams, dataName)

        # Optionally, generate and save simulated data based on the fitted model
        fitSaver.saveSimulatedData(mc_fitter, dataName)
        print(f"=== Finished processing file: {dataFile} ===\n")
        
        return (dataFile, True, None)
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing {dataFile}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return (dataFile, False, error_msg)


if __name__ == "__main__":
    # Main script to run the Monte Carlo fitting for the causal inference model
    #arguments: data file name, model type (e.g., "lognorm" or "gauss"), number of simulations, optimization method, number of starts
    # Example usage: python runFitting.py mt_all.csv lognorm 2000 bad
    
    #python runFitting.py "as_all.csv,oy_all.csv,dt_all.csv,HH_all.csv,ip_all.csv,ln_all.csv,LN01_all.csv,mh_all.csv,ml_all.csv,mt_all.csv,qs_all.csv,sx_all.csv" "lognorm" 500 "bads" 5
    #python runFitting.py "as_all.csv,oy_all.csv,dt_all.csv,HH_all.csv,ip_all.csv,ln2_all.csv, ln1_all.csv,mh_all.csv,ml_all.csv,mt_all.csv,qs_all.csv,sx_all.csv" "probabilityMatchingLogNorm" 2500 "bads" 10 True "analytical"
    # NEW: Add number of cores as optional argument
    #python runFitting.py "file1.csv,file2.csv" "lognorm" 500 "bads" 5 "analytical" 4
    
    #"probabilityMatching" 
    # "probabilityMatchingLogNorm"  # for log-space

    # take arguments from command line
    import sys
    import os
    import time
    from multiprocessing import Pool, cpu_count
    
    # Parse arguments
    dataFiles = sys.argv[1].split(',') if len(sys.argv) > 1 else ["mt_all.csv"]
    # Strip whitespace from file names
    dataFiles = [f.strip() for f in dataFiles]
    
    modelName = sys.argv[2] if len(sys.argv) > 2 else "lognorm"
    nSimul = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    optimMethod = sys.argv[4] if len(sys.argv) > 4 else "bads"
    nStarts = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    freeP_c = sys.argv[6] if len(sys.argv) > 6 else False # "numerical" or "analytical"
    integrationMethod= sys.argv[7] if len(sys.argv) > 7 else "analytical" # "numeric  # Whether to fit free prior probability of common cause

    n_cores = int(sys.argv[8]) if len(sys.argv) > 8 else min(cpu_count(), len(dataFiles))  # Use all available cores by default, but not more than files
    print(f"Data files: {dataFiles}")
    print(f"Model: {modelName}, Simulations: {nSimul}, Optimizer: {optimMethod}")
    print(f" Free P(C=1): {freeP_c}")
    print(f"Starts: {nStarts}, Integration: {integrationMethod}")
    print(f"Number of cores to use: {n_cores} (Available: {cpu_count()})")
    print(f"Processing {len(dataFiles)} files in parallel...\n")

    # Prepare arguments for parallel processing
    args_list = [(dataFile, modelName, nSimul, optimMethod, nStarts, integrationMethod,freeP_c) 
                 for dataFile in dataFiles]
    
    # Start timing
    overall_start = time.time()
    
    # Run in parallel
    if n_cores > 1 and len(dataFiles) > 1:
        with Pool(processes=n_cores) as pool:
            results = pool.map(process_single_file, args_list)
    else:
        # Fall back to sequential processing if only 1 core or 1 file
        print("Running sequentially (single core or single file)...\n")
        results = [process_single_file(args) for args in args_list]
    
    # Print summary
    overall_time = time.time() - overall_start
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total time: {overall_time:.2f} seconds")
    print(f"Average time per file: {overall_time/len(dataFiles):.2f} seconds")
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"\nSuccessfully processed: {len(successful)}/{len(dataFiles)} files")
    if successful:
        for dataFile, _, _ in successful:
            print(f"  ✅ {dataFile}")
    
    if failed:
        print(f"\nFailed: {len(failed)} files")
        for dataFile, _, error_msg in failed:
            print(f"  ❌ {dataFile}")
            if error_msg:
                print(f"     Error: {error_msg}")
