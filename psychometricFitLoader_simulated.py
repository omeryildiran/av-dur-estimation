"""
Psychometric Function Fit Loader for SIMULATED DATA
Load saved psychometric function fits from simulated data
"""
import os
import json
import numpy as np
import pandas as pd


def load_psychometric_fit_simulated(participantID, model_type="lognorm_LapseFree_sharedPrior", 
                                     save_dir="psychometric_fits_simulated"):
    """
    Load saved psychometric fit for a participant's simulated data
    
    Parameters:
    -----------
    participantID : str
        Participant ID (e.g., 'as', 'oy', 'dt')
    model_type : str
        Model type (e.g., "lognorm_LapseFree_sharedPrior")
    save_dir : str
        Directory where fits are saved
    
    Returns:
    --------
    dict : Fit results including parameters and metadata
    """
    filename = f"{participantID}_{model_type}_simulated_psychometric_fit.json"
    filepath = os.path.join(save_dir, participantID, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved fit found for participant {participantID} ({model_type}) at {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    results['parameters'] = np.array(results['parameters'])
    results['uniqueSensory'] = np.array(results['uniqueSensory'])
    results['uniqueStandard'] = np.array(results['uniqueStandard'])
    
    return results


def load_all_psychometric_fits_simulated(model_type="lognorm_LapseFree_sharedPrior",
                                          save_dir="psychometric_fits_simulated"):
    """
    Load all saved psychometric fits for a specific model type
    
    Parameters:
    -----------
    model_type : str
        Model type (e.g., "lognorm_LapseFree_sharedPrior")
    save_dir : str
        Directory where fits are saved
    
    Returns:
    --------
    dict : Dictionary mapping participant IDs to their fit results
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Save directory not found: {save_dir}")
    
    all_fits = {}
    participant_dirs = [d for d in os.listdir(save_dir) 
                       if os.path.isdir(os.path.join(save_dir, d))]
    
    for participantID in participant_dirs:
        try:
            fit = load_psychometric_fit_simulated(participantID, model_type, save_dir)
            all_fits[participantID] = fit
        except Exception as e:
            print(f"Warning: Could not load fit for {participantID}: {str(e)}")
    
    return all_fits


def get_fit_summary_simulated(model_type="lognorm_LapseFree_sharedPrior",
                               save_dir="psychometric_fits_simulated"):
    """
    Get a summary DataFrame of all saved fits for a model type
    
    Parameters:
    -----------
    model_type : str
        Model type (e.g., "lognorm_LapseFree_sharedPrior")
    save_dir : str
        Directory where fits are saved
    
    Returns:
    --------
    pd.DataFrame : Summary of all fits
    """
    all_fits = load_all_psychometric_fits_simulated(model_type, save_dir)
    
    summary_data = []
    for participantID, fit in all_fits.items():
        summary_data.append({
            'participantID': participantID,
            'model_type': fit['model_type'],
            'n_params': fit['n_params'],
            'n_conditions': fit['n_conditions'],
            'n_trials': fit['n_trials'],
            'AIC': fit['AIC'],
            'BIC': fit['BIC'],
            'log_likelihood': fit['log_likelihood'],
            'success': fit['success']
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('participantID')
    
    return df


def get_parameters_simulated(participantID, model_type="lognorm_LapseFree_sharedPrior",
                             save_dir="psychometric_fits_simulated"):
    """
    Get fitted parameters for a specific participant's simulated data
    
    Parameters:
    -----------
    participantID : str
        Participant ID
    model_type : str
        Model type
    save_dir : str
        Directory where fits are saved
    
    Returns:
    --------
    np.ndarray : Array of fitted parameters
    """
    fit = load_psychometric_fit_simulated(participantID, model_type, save_dir)
    return fit['parameters']


def compare_data_vs_simulated(participantID, model_type="lognorm_LapseFree_sharedPrior",
                               data_fits_dir="psychometric_fits_data",
                               sim_fits_dir="psychometric_fits_simulated"):
    """
    Compare psychometric fits between real data and simulated data
    
    Parameters:
    -----------
    participantID : str
        Participant ID
    model_type : str
        Model type for simulated data
    data_fits_dir : str
        Directory with real data fits
    sim_fits_dir : str
        Directory with simulated data fits
    
    Returns:
    --------
    dict : Comparison metrics
    """
    # Load data fit
    data_fit_path = os.path.join(data_fits_dir, participantID, f"{participantID}_psychometric_fit.json")
    with open(data_fit_path, 'r') as f:
        data_fit = json.load(f)
    
    # Load simulated fit
    sim_fit = load_psychometric_fit_simulated(participantID, model_type, sim_fits_dir)
    
    # Compare
    comparison = {
        'participantID': participantID,
        'model_type': model_type,
        'data_AIC': data_fit['AIC'],
        'simulated_AIC': sim_fit['AIC'],
        'delta_AIC': sim_fit['AIC'] - data_fit['AIC'],
        'data_BIC': data_fit['BIC'],
        'simulated_BIC': sim_fit['BIC'],
        'delta_BIC': sim_fit['BIC'] - data_fit['BIC'],
        'data_loglik': data_fit['log_likelihood'],
        'simulated_loglik': sim_fit['log_likelihood'],
        'data_n_params': data_fit['n_params'],
        'simulated_n_params': sim_fit['n_params'],
    }
    
    return comparison


def get_all_comparisons(participant_ids, model_type="lognorm_LapseFree_sharedPrior",
                        data_fits_dir="psychometric_fits_data",
                        sim_fits_dir="psychometric_fits_simulated"):
    """
    Get comparisons for all participants
    
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    comparisons = []
    
    for pid in participant_ids:
        try:
            comp = compare_data_vs_simulated(pid, model_type, data_fits_dir, sim_fits_dir)
            comparisons.append(comp)
        except Exception as e:
            print(f"Warning: Could not compare {pid}: {e}")
    
    return pd.DataFrame(comparisons)


if __name__ == "__main__":
    # Example usage
    
    # Load single participant's simulated fit
    try:
        fit = load_psychometric_fit_simulated('as', 'lognorm_LapseFree_sharedPrior')
        print(f"Loaded fit for 'as': {fit['n_params']} parameters, AIC={fit['AIC']:.2f}")
    except FileNotFoundError:
        print("No fits found yet. Run psychometricFitSaver_simulated.py first.")
    
    # Load all fits for a model type
    try:
        all_fits = load_all_psychometric_fits_simulated('lognorm_LapseFree_sharedPrior')
        print(f"\nLoaded {len(all_fits)} participant simulated fits")
        
        # Get summary
        summary = get_fit_summary_simulated('lognorm_LapseFree_sharedPrior')
        print("\nSimulated Fit Summary:")
        print(summary)
        
    except FileNotFoundError:
        print("No fits directory found. Run psychometricFitSaver_simulated.py first.")
