import numpy as np
import pandas as pd
import loadData
import monteCarloClass
import time

fileName="dt_all.csv"
"""
["as_all.csv", "oy_all.csv", "dt_all.csv", "HH_all.csv", "ip_all.csv", "ln_all.csv", 
"LN01_all.csv", "mh_all.csv", "ml_all.csv", "mt_all.csv", "qs_all.csv", "sx_all.csv"]

"""
#for fileName in ["as_all.csv", "oy_all.csv", "dt_all.csv", "HH_all.csv", "ip_all.csv", "ln_all.csv","LN01_all.csv", 
#                  "mh_all.csv", "ml_all.csv", "mt_all.csv", "qs_all.csv", "sx_all.csv"]:
for fileName in ["all_main.csv"]:
    data, dataName = loadData.loadData(fileName)

    intensityVariable = "deltaDurS"
    sensoryVar = "audNoise"
    standardVar = "standardDur"
    conflictVar = "conflictDur"
    #s
    visualStandardVar = "unbiasedVisualStandardDur"
    visualTestVar = "unbiasedVisualTestDur"
    audioStandardVar = "standardDur"
    audioTestVar = "testDurS"


    # Instantiate the Monte Carlo class
    mc_fitter = monteCarloClass.OmerMonteCarlo(data)


    # fit parameters
    #mc_fitter.nSimul = 1000
    mc_fitter.optimizationMethod= "bads"  # Use BADS for optimization
    mc_fitter.nStart = 1 # Number of random starts for optimization
    mc_fitter.modelName = "lognorm"  # Set measurement distribution to Gaussian
    mc_fitter.sharedLambda = False  # Use separate lapse rates for each condition
    mc_fitter.freeP_c = 0  # Allow different prior widths for conflict conditions

    mc_fitter.dataName = dataName

    import os
    import json
    import loadResults

    res=loadResults.loadFitResults(mc_fitter,dataName, modelName=mc_fitter.modelName)
    print(f"Loaded saved fit results: {res}")
    mc_fitter.modelFit= res['fittedParams']
    mc_fitter.logLikelihood= res['logLikelihood']
    mc_fitter.aic= res['AIC']
    mc_fitter.bic= res['BIC']
    mc_fitter.nDataPoints= res['n_conditions']


    import fitSaver
    # save simulation data for plotting
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

    filename = f"{participantID.split('_')[0]}_{modelType}_simulated.csv"
    filename = os.path.join("simulated_data",participantID.split('_')[0], filename)

    fitSaver.saveSimulatedData(mc_fitter, mc_fitter.dataName)
    mc_fitter.simulatedData= pd.read_csv(filename)
    print(f"Simulated data saved to {filename}")

