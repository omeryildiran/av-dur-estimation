"""
Psychometric Function Fit Saver
Fits psychometric functions to participant data and saves results to JSON files
"""
import os
import json
import numpy as np
from fitMainClass import fitPychometric
from loadData import loadData


def fit_and_save_psychometric(dataName, nStart=1, save_dir="psychometric_fits"):
    """
    Fit psychometric function to participant data and save results
    
    Parameters:
    -----------
    dataName : str
        Name of the CSV file in the data folder
    nStart : int
        Number of random starting points for optimization
    save_dir : str
        Directory to save fit results
    
    Returns:
    --------
    dict : Fit results including parameters and metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataName}")
    print(f"{'='*60}")
    
    # Load data using the loadData function
    data, dataName = loadData(dataName)
    
    # Create the fitting model instance
    fit_model = fitPychometric(data, allIndependent=True, sharedSigma=False)
    
    # Access unique values from the model
    uniqueSensory = fit_model.uniqueSensory
    uniqueStandard = fit_model.uniqueStandard
    uniqueConflict = fit_model.uniqueConflict
    
    # Group data - use fit_model.data which has the logDurRatio column
    groupArgs = ['logDurRatio', 'audNoise', 'standardDur', 'conflictDur']
    grouped_data = fit_model.groupByChooseTest(fit_model.data, groupArgs)
    
    print(f"Total conditions: {len(grouped_data)}")
    print(f"Unique noise levels: {uniqueSensory}")
    print(f"Unique standards: {uniqueStandard}")
    print(f"Unique conflicts: {uniqueConflict}")
    
    # Fit the model
    print(f"\nFitting psychometric functions with {nStart} starting points...")
    best_fit = fit_model.fitMultipleStartingPoints(nStart=nStart)
    
    # Extract participant ID
    participantID = dataName.split('_')[0]
    
    # Prepare save directory
    participant_dir = os.path.join(save_dir, participantID)
    os.makedirs(participant_dir, exist_ok=True)
    
    # Calculate number of parameters and data points
    n_params = len(best_fit.x)
    n_conditions = len(grouped_data)
    
    # Calculate AIC and BIC
    log_likelihood = -best_fit.fun
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_conditions) * n_params - 2 * log_likelihood
    
    # Prepare results dictionary
    results = {
        "participantID": participantID,
        "dataName": dataName,
        "n_params": n_params,
        "n_conditions": n_conditions,
        "n_trials": int(len(fit_model.data)),
        "parameters": best_fit.x.tolist(),
        "log_likelihood": float(log_likelihood),
        "AIC": float(aic),
        "BIC": float(bic),
        "success": bool(best_fit.success),
        "message": best_fit.message,
        "uniqueSensory": uniqueSensory.tolist(),
        "uniqueStandard": uniqueStandard.tolist(),
        "uniqueConflict": uniqueConflict,
        "nStart": nStart
    }
    
    # Save to JSON
    filename = f"{participantID}_psychometric_fit.json"
    filepath = os.path.join(participant_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Saved psychometric fit to: {filepath}")
    print(f"   Params: {n_params}, Conditions: {n_conditions}, Trials: {len(fit_model.data)}")
    print(f"   AIC: {aic:.2f}, BIC: {bic:.2f}, LogLik: {log_likelihood:.2f}")
    
    return results


def batch_fit_participants(dataNames, nStart=1, save_dir="psychometric_fits"):
    """
    Fit psychometric functions for multiple participants
    
    Parameters:
    -----------
    dataNames : list of str
        List of CSV filenames to process
    nStart : int
        Number of random starting points for optimization
    save_dir : str
        Directory to save fit results
    
    Returns:
    --------
    dict : Dictionary mapping participant IDs to their fit results
    """
    all_results = {}
    successful = []
    failed = []
    
    for i, dataName in enumerate(dataNames, 1):
        print(f"\n\n{'#'*60}")
        print(f"# Participant {i}/{len(dataNames)}: {dataName}")
        print(f"{'#'*60}")
        
        try:
            results = fit_and_save_psychometric(dataName, nStart=nStart, save_dir=save_dir)
            participantID = results["participantID"]
            all_results[participantID] = results
            successful.append(dataName)
            
        except Exception as e:
            print(f"\n❌ ERROR processing {dataName}: {str(e)}")
            failed.append(dataName)
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n\n{'='*60}")
    print(f"BATCH FITTING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful)}/{len(dataNames)}")
    print(f"❌ Failed: {len(failed)}/{len(dataNames)}")
    
    if failed:
        print(f"\nFailed participants:")
        for name in failed:
            print(f"  - {name}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    dataNames = [
        "as_all.csv", "oy_all.csv", "dt_all.csv", "HH_all.csv",
        "ip_all.csv", "ln_all.csv", "LN01_all.csv", "mh_all.csv",
        "ml_all.csv", "mt_all.csv", "qs_all.csv", "sx_all.csv"
    ]
    
    results = batch_fit_participants(dataNames, nStart=1, save_dir="psychometric_fits_data")
