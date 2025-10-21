"""
Psychometric Function Fit Loader
Load saved psychometric function fits from JSON files
"""
import os
import json
import numpy as np
import pandas as pd


def load_psychometric_fit(participantID, save_dir="psychometric_fits_data"):
    """
    Load saved psychometric fit for a participant
    
    Parameters:
    -----------
    participantID : str
        Participant ID (e.g., 'as', 'oy', 'dt')
    save_dir : str
        Directory where fits are saved
    
    Returns:
    --------
    dict : Fit results including parameters and metadata
    """
    filepath = os.path.join(save_dir, participantID, f"{participantID}_psychometric_fit.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved fit found for participant {participantID} at {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    results['parameters'] = np.array(results['parameters'])
    results['uniqueSensory'] = np.array(results['uniqueSensory'])
    results['uniqueStandard'] = np.array(results['uniqueStandard'])
    
    return results


def load_all_psychometric_fits(save_dir="psychometric_fits_data"):
    """
    Load all saved psychometric fits
    
    Parameters:
    -----------
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
            fit = load_psychometric_fit(participantID, save_dir)
            all_fits[participantID] = fit
        except Exception as e:
            print(f"Warning: Could not load fit for {participantID}: {str(e)}")
    
    return all_fits


def get_fit_summary(save_dir="psychometric_fits"):
    """
    Get a summary DataFrame of all saved fits
    
    Parameters:
    -----------
    save_dir : str
        Directory where fits are saved
    
    Returns:
    --------
    pd.DataFrame : Summary of all fits
    """
    all_fits = load_all_psychometric_fits(save_dir)
    
    summary_data = []
    for participantID, fit in all_fits.items():
        summary_data.append({
            'participantID': participantID,
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


def get_parameters(participantID, save_dir="psychometric_fits_data"):
    """
    Get fitted parameters for a specific participant
    
    Parameters:
    -----------
    participantID : str
        Participant ID
    save_dir : str
        Directory where fits are saved
    
    Returns:
    --------
    np.ndarray : Array of fitted parameters
    """
    fit = load_psychometric_fit(participantID, save_dir)
    return fit['parameters']


def get_model_comparison(save_dir="psychometric_fits"):
    """
    Get model comparison metrics (AIC/BIC) for all participants
    
    Parameters:
    -----------
    save_dir : str
        Directory where fits are saved
    
    Returns:
    --------
    pd.DataFrame : Model comparison table sorted by AIC
    """
    summary = get_fit_summary(save_dir)
    comparison = summary[['participantID', 'AIC', 'BIC', 'log_likelihood', 'n_params']]
    comparison = comparison.sort_values('AIC')
    
    return comparison


if __name__ == "__main__":
    # Example usage
    
    # Load single participant
    try:
        fit = load_psychometric_fit('as')
        print(f"Loaded fit for 'as': {fit['n_params']} parameters, AIC={fit['AIC']:.2f}")
    except FileNotFoundError:
        print("No fits found yet. Run psychometricFitSaver.py first.")
    
    # Load all fits
    try:
        all_fits = load_all_psychometric_fits()
        print(f"\nLoaded {len(all_fits)} participant fits")
        
        # Get summary
        summary = get_fit_summary()
        print("\nFit Summary:")
        print(summary)
        
    except FileNotFoundError:
        print("No fits directory found. Run psychometricFitSaver.py first.")
