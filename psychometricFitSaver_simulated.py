"""
Psychometric Function Fit Saver for SIMULATED DATA
Fits psychometric functions to simulated data and saves results to JSON files
"""
import os
import json
import numpy as np
import pandas as pd
import fitMain


def fit_and_save_psychometric_simulated(sim_filepath, nStart=1, save_dir="psychometric_fits_simulated"):
    """
    Fit psychometric function to simulated data and save results
    
    Parameters:
    -----------
    sim_filepath : str
        Full path to the simulated CSV file (e.g., "simulated_data/oy/oy_lognorm_LapseFree_sharedPrior_simulated.csv")
    nStart : int
        Number of random starting points for optimization
    save_dir : str
        Directory to save fit results
    
    Returns:
    --------
    dict : Fit results including parameters and metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing: {sim_filepath}")
    print(f"{'='*60}")
    
    # Extract info from filename
    # Example: "simulated_data/oy/oy_lognorm_LapseFree_sharedPrior_simulated.csv"
    filename = os.path.basename(sim_filepath)
    parts = filename.replace('_simulated.csv', '').split('_')
    participantID = parts[0]
    model_type = '_'.join(parts[1:])  # e.g., "lognorm_LapseFree_sharedPrior"
    
    # Load the simulated data directly
    data = pd.read_csv(sim_filepath)
    
    # Add required derived columns if not present
    if 'chose_test' not in data.columns:
        data['chose_test'] = (data['responses'] == data['order']).astype(int)
    if 'chose_standard' not in data.columns:
        data['chose_standard'] = (data['responses'] != data['order']).astype(int)
    
    # Add delta_dur_percents if using deltaDurS (simulated data uses deltaDurS)
    if 'delta_dur_percents' not in data.columns and 'deltaDurS' in data.columns:
        data['delta_dur_percents'] = data['deltaDurS'] / data['standardDur']
    
    # Get unique values for metadata
    uniqueSensory = sorted(data['audNoise'].unique())
    uniqueStandard = sorted(data['standardDur'].unique())
    uniqueConflict = sorted(data['conflictDur'].unique())
    
    # Determine which intensity variable to use
    if 'delta_dur_percents' in data.columns:
        intensity_var = 'delta_dur_percents'
    elif 'deltaDurS' in data.columns:
        intensity_var = 'deltaDurS'
    else:
        raise ValueError("No suitable intensity variable found in simulated data")
    
    # Group data
    groupArgs = [intensity_var, 'audNoise', 'standardDur', 'conflictDur']
    
    # Define groupByChooseTest inline to avoid dependency issues
    def groupByChooseTest(x, groupArgs):
        grouped = x.groupby(groupArgs).agg(
            num_of_chose_test=('chose_test', 'sum'),
            total_responses=('responses', 'count'),
            num_of_chose_standard=('chose_standard', 'sum'),
        ).reset_index()
        grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']
        return grouped
    
    grouped_data = groupByChooseTest(data, groupArgs)
    
    print(f"Model type: {model_type}")
    print(f"Total conditions: {len(grouped_data)}")
    print(f"Unique noise levels: {uniqueSensory}")
    print(f"Unique standards: {uniqueStandard}")
    print(f"Unique conflicts: {uniqueConflict}")
    
    # Set global variables in fitMain for compatibility
    fitMain.data = data
    fitMain.uniqueSensory = np.array(uniqueSensory)
    fitMain.uniqueStandard = np.array(uniqueStandard)
    fitMain.uniqueConflict = uniqueConflict
    fitMain.sensoryVar = 'audNoise'
    fitMain.standardVar = 'standardDur'
    fitMain.conflictVar = 'conflictDur'
    fitMain.intensityVariable = intensity_var
    fitMain.nLambda = len(uniqueStandard)
    fitMain.nSigma = len(uniqueSensory)
    fitMain.nMu = len(uniqueConflict) * len(uniqueSensory)
    fitMain.allIndependent = True  # Required by fitMain
    fitMain.sharedSigma = False    # Required by fitMain
    
    # Fit the model
    print(f"\nFitting psychometric functions with {nStart} starting points...")
    best_fit = fitMain.fitMultipleStartingPoints(data, nStart=nStart)
    
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
        "model_type": model_type,
        "simulated_file": sim_filepath,
        "n_params": n_params,
        "n_conditions": n_conditions,
        "n_trials": int(len(data)),
        "parameters": best_fit.x.tolist(),
        "log_likelihood": float(log_likelihood),
        "AIC": float(aic),
        "BIC": float(bic),
        "success": bool(best_fit.success),
        "message": best_fit.message,
        "uniqueSensory": uniqueSensory,
        "uniqueStandard": uniqueStandard,
        "uniqueConflict": uniqueConflict,
        "nStart": nStart
    }
    
    # Save to JSON with model type in filename
    filename = f"{participantID}_{model_type}_simulated_psychometric_fit.json"
    filepath = os.path.join(participant_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Saved psychometric fit to: {filepath}")
    print(f"   Params: {n_params}, Conditions: {n_conditions}, Trials: {len(data)}")
    print(f"   AIC: {aic:.2f}, BIC: {bic:.2f}, LogLik: {log_likelihood:.2f}")
    
    return results


def batch_fit_simulated(participant_ids, model_type="lognorm_LapseFree_sharedPrior", 
                        nStart=1, save_dir="psychometric_fits_simulated",
                        sim_base_dir="simulated_data"):
    """
    Fit psychometric functions for multiple participants' simulated data
    
    Parameters:
    -----------
    participant_ids : list of str
        List of participant IDs (e.g., ['as', 'oy', 'dt'])
    model_type : str
        Type of simulated model to fit (e.g., "lognorm_LapseFree_sharedPrior")
    nStart : int
        Number of random starting points for optimization
    save_dir : str
        Directory to save fit results
    sim_base_dir : str
        Base directory containing simulated data
    
    Returns:
    --------
    dict : Dictionary mapping participant IDs to their fit results
    """
    all_results = {}
    successful = []
    failed = []
    
    for i, pid in enumerate(participant_ids, 1):
        # Construct simulated file path
        sim_filename = f"{pid}_{model_type}_simulated.csv"
        sim_filepath = os.path.join(sim_base_dir, pid, sim_filename)
        
        print(f"\n\n{'#'*60}")
        print(f"# Participant {i}/{len(participant_ids)}: {pid}")
        print(f"# Model: {model_type}")
        print(f"{'#'*60}")
        
        # Check if file exists
        if not os.path.exists(sim_filepath):
            print(f"⚠️  File not found: {sim_filepath}")
            failed.append(pid)
            continue
        
        try:
            results = fit_and_save_psychometric_simulated(
                sim_filepath, 
                nStart=nStart, 
                save_dir=save_dir
            )
            all_results[pid] = results
            successful.append(pid)
            
        except Exception as e:
            print(f"\n❌ ERROR processing {pid}: {str(e)}")
            failed.append(pid)
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n\n{'='*60}")
    print(f"BATCH FITTING SUMMARY - {model_type}")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful)}/{len(participant_ids)}")
    print(f"❌ Failed: {len(failed)}/{len(participant_ids)}")
    
    if failed:
        print(f"\nFailed participants:")
        for pid in failed:
            print(f"  - {pid}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    participant_ids = [
        "as", "oy", "dt", "HH", "ip", "ln", "LN01", 
        "mh", "ml", "mt", "qs", "sx"
    ]
    
    model_type = "lognorm_LapseFree_sharedPrior"
    
    results = batch_fit_simulated(
        participant_ids=participant_ids,
        model_type=model_type,
        nStart=1,
        save_dir="psychometric_fits_simulated"
    )
