"""
Fit individual participant psychometric functions using CUMULATIVE NORMAL model.
This produces sigma values in the same units as the pooled bootstrap analysis.
Mu is FREE (not fixed to 0).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import scipy.io
import os

# Parameters
FIXED_MU = False  # Allow mu to be free
STANDARD_DURATION = 0.5  # 500 ms

def load_participant_data(participant_id, modality='auditory'):
    """Load data for a single participant."""
    if modality == 'auditory':
        filename = f"data/{participant_id}_auditory.csv"
    else:
        filename = f"data/{participant_id}_visual.csv"
    
    data = pd.read_csv(filename)
    
    # Clean data
    data = data[data['audNoise'] != 0]
    data = data[data['standardDur'] != 0]
    
    # Round values
    data = data.round({'audNoise': 2, 'conflictDur': 2, 'delta_dur_percents': 2})
    
    # Define chose_test
    data['chose_test'] = (data['responses'] == data['order']).astype(int)
    
    return data


def psychometric_function_cumulative_normal(delta_percent, lambda_, mu, sigma):
    """
    Cumulative normal psychometric function.
    
    Parameters:
    - delta_percent: (test - standard) / standard (proportion, e.g., 0.2 for 20%)
    - lambda_: lapse rate
    - mu: bias (PSE shift in proportion units)
    - sigma: discrimination threshold (in proportion units)
    
    Returns:
    - P(choose test as longer)
    """
    z = (delta_percent - mu) / sigma
    p_longer = norm.cdf(z)
    p = lambda_/2 + (1 - lambda_) * p_longer
    return p


def negative_log_likelihood(params, deltas, chose_test, total_responses, fixed_mu=False):
    """Compute negative log-likelihood."""
    if fixed_mu:
        lambda_, sigma = params
        mu = 0
    else:
        lambda_, mu, sigma = params
    
    p = psychometric_function_cumulative_normal(deltas, lambda_, mu, sigma)
    
    epsilon = 1e-9
    p = np.clip(p, epsilon, 1 - epsilon)
    
    log_likelihood = np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
    return -log_likelihood


def fit_psychometric_single_condition(data, noise_level=None, fixed_mu=False):
    """
    Fit psychometric function to a single condition.
    
    Returns: dict with lambda, mu, sigma
    """
    if noise_level is not None:
        df = data[data['audNoise'] == noise_level].copy()
    else:
        df = data.copy()
    
    # Group by delta_dur_percents
    grouped = df.groupby('delta_dur_percents').agg(
        chose_test=('chose_test', 'sum'),
        total_responses=('chose_test', 'count')
    ).reset_index()
    
    deltas = grouped['delta_dur_percents'].values
    chose_test = grouped['chose_test'].values
    total_responses = grouped['total_responses'].values
    
    # Initial guesses
    if fixed_mu:
        init_guess = [0.05, 0.3]  # lambda, sigma
        bounds = [(0, 0.25), (0.01, 3.0)]  # lambda, sigma
    else:
        init_guess = [0.05, 0.0, 0.3]  # lambda, mu, sigma
        bounds = [(0, 0.25), (-0.5, 0.5), (0.01, 3.0)]  # lambda, mu, sigma
    
    # Try multiple starting points
    best_result = None
    best_nll = np.inf
    
    sigma_starts = [0.1, 0.3, 0.5, 0.8, 1.0]
    mu_starts = [-0.1, 0.0, 0.1] if not fixed_mu else [None]
    
    for sigma_init in sigma_starts:
        for mu_init in mu_starts:
            if fixed_mu:
                init = [0.05, sigma_init]
            else:
                init = [0.05, mu_init, sigma_init]
            
            try:
                result = minimize(
                    negative_log_likelihood,
                    x0=init,
                    args=(deltas, chose_test, total_responses, fixed_mu),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except:
                continue
    
    if best_result is None:
        raise ValueError("Fitting failed")
    
    if fixed_mu:
        return {'lambda': best_result.x[0], 'mu': 0.0, 'sigma': best_result.x[1]}
    else:
        return {'lambda': best_result.x[0], 'mu': best_result.x[1], 'sigma': best_result.x[2]}


def main():
    """Fit all participants and save results."""
    
    participant_ids = ['as', 'dt', 'hh', 'ip', 'ln2', 'mh', 'ml', 'mt', 'oy', 'qs', 'sx']
    
    print("="*70)
    print("FITTING INDIVIDUAL PARTICIPANTS - CUMULATIVE NORMAL MODEL")
    print(f"Mu is {'FIXED at 0' if FIXED_MU else 'FREE'}")
    print("="*70)
    
    all_results = {}
    
    for pid in participant_ids:
        print(f"\n--- Fitting {pid} ---")
        
        try:
            # Fit auditory conditions
            aud_data = load_participant_data(pid, 'auditory')
            noise_levels = sorted(aud_data['audNoise'].unique())
            
            # Fit each noise level separately
            aud_results = {}
            for noise in noise_levels:
                result = fit_psychometric_single_condition(aud_data, noise_level=noise, fixed_mu=FIXED_MU)
                aud_results[noise] = result
                print(f"  Auditory noise={noise}: λ={result['lambda']:.3f}, μ={result['mu']:.3f}, σ={result['sigma']:.3f}")
            
            # Save auditory fits
            # Format: [lambda, sigma_high_noise, sigma_low_noise, mu_high_noise, mu_low_noise]
            # Assuming noise_levels = [0.1, 1.2] (sorted)
            low_noise = min(noise_levels)  # 0.1 = high reliability
            high_noise = max(noise_levels)  # 1.2 = low reliability
            
            aud_params = np.array([
                aud_results[high_noise]['lambda'],  # lambda (use one of them)
                aud_results[high_noise]['sigma'],   # sigma for HIGH noise (low reliability)
                aud_results[low_noise]['sigma'],    # sigma for LOW noise (high reliability)
                aud_results[high_noise]['mu'],      # mu for high noise
                aud_results[low_noise]['mu']        # mu for low noise
            ])
            
            output_file = f"data/{pid}_auditory_fits_cumNormal.mat"
            scipy.io.savemat(output_file, {'fittedParams': aud_params})
            print(f"  Saved to {output_file}")
            
        except Exception as e:
            print(f"  ERROR fitting auditory: {e}")
        
        try:
            # Fit visual condition
            vis_data = load_participant_data(pid, 'visual')
            vis_result = fit_psychometric_single_condition(vis_data, noise_level=None, fixed_mu=FIXED_MU)
            print(f"  Visual: λ={vis_result['lambda']:.3f}, μ={vis_result['mu']:.3f}, σ={vis_result['sigma']:.3f}")
            
            # Save visual fits
            vis_params = np.array([
                vis_result['lambda'],
                vis_result['sigma'],
                vis_result['mu']
            ])
            
            output_file = f"data/{pid}_visual_fits_cumNormal.mat"
            scipy.io.savemat(output_file, {'fittedParams': vis_params})
            print(f"  Saved to {output_file}")
            
        except Exception as e:
            print(f"  ERROR fitting visual: {e}")
        
        all_results[pid] = {'auditory': aud_results if 'aud_results' in dir() else None, 
                           'visual': vis_result if 'vis_result' in dir() else None}
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY (sigma values in proportion units, multiply by 500 for ms)")
    print("="*70)
    
    print("\nAuditory conditions:")
    print(f"{'PID':<6} {'σ_high_rel':>12} {'σ_low_rel':>12} {'μ_high_rel':>12} {'μ_low_rel':>12}")
    
    for pid in participant_ids:
        try:
            mat = scipy.io.loadmat(f"data/{pid}_auditory_fits_cumNormal.mat")
            p = mat['fittedParams'].flatten()
            # [lambda, sigma_high_noise, sigma_low_noise, mu_high_noise, mu_low_noise]
            print(f"{pid:<6} {p[2]:>12.4f} {p[1]:>12.4f} {p[4]:>12.4f} {p[3]:>12.4f}")
        except:
            print(f"{pid:<6} ERROR")
    
    print("\nVisual condition:")
    print(f"{'PID':<6} {'σ':>12} {'μ':>12}")
    
    for pid in participant_ids:
        try:
            mat = scipy.io.loadmat(f"data/{pid}_visual_fits_cumNormal.mat")
            p = mat['fittedParams'].flatten()
            print(f"{pid:<6} {p[1]:>12.4f} {p[2]:>12.4f}")
        except:
            print(f"{pid:<6} ERROR")


if __name__ == "__main__":
    main()
