#proceed to simulate and save to {filename}
import fitSaver
import os
def loadSimulatedData(mc_fitter, dataName,participantID, modelType=mc_fitter.modelName):
    participantID = dataName.split(".csv")[0]
    #modelType = mc_fitter.modelName

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
    try:
        simulatedData= pd.read_csv(filename)
        print(f"Loaded saved simulated data from {filename}")
        return simulatedData
    except FileNotFoundError:
        print(f"No saved simulated data found at {filename}")
        
def generateAndSaveSimulatedData(mc_fitter, dataName, participantID, modelType=mc_fitter.modelName):
    filename = f"{participantID.split('_')[0]}_{modelType}_simulated.csv"
    filename = os.path.join("simulated_data",participantID.split('_')[0], filename
    fitSaver.saveSimulatedData(mc_fitter, mc_fitter.dataName)
    mc_fitter.simulatedData= pd.read_csv(filename)
    print(f"Simulated data saved to {filename}")
    return mc_fitter.simulatedData
# #proceed to simulate and save to {filename}
# mc_fitter.simulatedData=loadSimulatedData(mc_fitter, mc_fitter.dataName, participantID=dataName.split(".csv")[0], modelType=mc_fitter.modelName)
