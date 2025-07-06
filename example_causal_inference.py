#!/usr/bin/env python3
"""
Example script demonstrating how to use the causal inference model
for auditory-visual duration discrimination analysis.

This script shows how to:
1. Load your experimental data
2. Fit both standard psychometric and causal inference models
3. Compare model performance
4. Visualize results
"""

import sys
import os

# Add the current directory to path to import fitMain
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fitMain import *

def example_causal_inference_analysis():
    """
    Example analysis using the causal inference model.
    
    Replace 'your_data_file.csv' with your actual data file name.
    The data file should be in the 'data/' directory.
    """
    
    print("="*60)
    print("CAUSAL INFERENCE MODEL ANALYSIS EXAMPLE")
    print("="*60)
    
    # Step 1: Specify your data file
    # Replace this with your actual data file name
    data_file = "HH_all.csv"   #"your_data_file.csv"  # Update this!
    print(f"Loading data from: {data_file}")
    
    # Step 2: Set model parameters
    shared_sigma = True      # Whether to share sigma across noise conditions
    all_independent = False  # Whether each condition has independent parameters
    
    try:
        # Step 3: Run the complete analysis
        results = run_causal_inference_analysis(
            data_file=data_file,
            shared_sigma=shared_sigma,
            all_independent=all_independent
        )
        
        # Step 4: Extract key results
        print("\n" + "="*40)
        print("KEY FINDINGS")
        print("="*40)
        
        if results["causal_inference_fit"] is not None:
            ci_params = results["causal_inference_fit"].x
            lambda_, mu, sigma_av_a, sigma_av_v, p_common = ci_params
            
            print(f"Causal Inference Model Parameters:")
            print(f"  Lapse Rate (λ): {lambda_:.3f}")
            print(f"  PSE (μ): {mu:.3f}")
            print(f"  Auditory Noise (σ_av_a): {sigma_av_a:.3f}")
            print(f"  Visual Noise (σ_av_v): {sigma_av_v:.3f}")
            print(f"  Prior P(Common Cause): {p_common:.3f}")
            
            # Interpret the results
            print(f"\nInterpretation:")
            if p_common > 0.5:
                print(f"  - Observers tend to assume common cause (P={p_common:.3f})")
                print(f"  - Strong audiovisual integration expected")
            else:
                print(f"  - Observers tend to assume independent causes (P={p_common:.3f})")
                print(f"  - Weak audiovisual integration expected")
                
            if sigma_av_a > sigma_av_v:
                print(f"  - Visual modality more reliable (σ_av_v={sigma_av_v:.3f} < σ_av_a={sigma_av_a:.3f})")
            else:
                print(f"  - Auditory modality more reliable (σ_av_a={sigma_av_a:.3f} < σ_av_v={sigma_av_v:.3f})")
                
            # Model comparison
            if results["delta_aic"] < -2:
                print(f"  - Causal inference model strongly preferred (ΔAIC={results['delta_aic']:.1f})")
            elif results["delta_aic"] < 0:
                print(f"  - Causal inference model moderately preferred (ΔAIC={results['delta_aic']:.1f})")
            else:
                print(f"  - Standard model preferred (ΔAIC={results['delta_aic']:.1f})")
        
        return results
        
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found in 'data/' directory.")
        print("Please update the 'data_file' variable with your actual data file name.")
        return None
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

def example_manual_fitting():
    """
    Example of manually fitting the causal inference model to specific data.
    """
    print("\n" + "="*60)
    print("MANUAL FITTING EXAMPLE")
    print("="*60)
    
    # This example shows how to fit the model step by step
    # Replace with your actual data loading
    try:
        # Load data (replace with your file)
        data_file = "HH_all.csv"  # Update this!
        data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(
            data_file, isShared=True, isAllIndependent=False
        )
        
        # Group data
        grouped_data = groupByChooseTest(data)
        
        # Set initial guesses for causal inference model
        # [lambda_, mu, sigma_av_a, sigma_av_v, p_common_prior]
        initial_guesses = [0.05, 0.0, 0.2, 0.2, 0.5]
        
        print("Fitting causal inference model with custom parameters...")
        ci_fit = fit_causal_inference_model(grouped_data, initial_guesses)
        
        if ci_fit.success:
            print("Fit successful!")
            print(f"Parameters: {ci_fit.x}")
            
            # Plot results
            plot_causal_inference_psychometric(data, ci_fit.x, "Manual Fit")
        else:
            print("Fit failed:", ci_fit.message)
            
    except FileNotFoundError:
        print(f"Error: Data file not found. Please update the data_file variable.")

def example_parameter_exploration():
    """
    Example showing how different parameter values affect the model predictions.
    """
    print("\n" + "="*60)
    print("PARAMETER EXPLORATION EXAMPLE")
    print("="*60)
    
    # Create some example data points
    delta_dur = np.linspace(-0.5, 0.5, 11)
    conflicts = [0.0, 0.1, 0.2, 0.3]  # Different conflict levels
    
    # Model parameters to explore
    params_low_integration = [0.02, 0.0, 0.3, 0.3, 0.2]   # Low integration
    params_high_integration = [0.02, 0.0, 0.2, 0.2, 0.8]  # High integration
    
    plt.figure(figsize=(12, 5))
    
    for i, (params, label) in enumerate([
        (params_low_integration, "Low Integration (P(common)=0.2)"),
        (params_high_integration, "High Integration (P(common)=0.8)")
    ]):
        plt.subplot(1, 2, i+1)
        
        lambda_, mu, sigma_av_a, sigma_av_v, p_common = params
        
        for conflict in conflicts:
            y = np.zeros_like(delta_dur)
            for j, delta in enumerate(delta_dur):
                y[j] = causal_inference_psychometric_function(
                    delta, lambda_, mu, sigma_av_a, sigma_av_v, p_common, conflict
                )
            plt.plot(delta_dur, y, label=f'Conflict: {conflict}', linewidth=2)
        
        plt.xlabel('Duration Difference')
        plt.ylabel('P(choose test)')
        plt.title(label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("This example shows how:")
    print("- Higher P(common cause) leads to more similar curves across conflicts")
    print("- Lower P(common cause) leads to more separated curves")
    print("- This reflects the degree of audiovisual integration")

if __name__ == "__main__":
    print("Causal Inference Model for Audiovisual Duration Discrimination")
    print("=" * 60)
    
    # Run the main example
    results = example_causal_inference_analysis()
    
    # Run parameter exploration
    example_parameter_exploration()
    
    # If the main analysis worked, also show manual fitting
    if results is not None:
        example_manual_fitting()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("To use with your own data:")
    print("1. Update 'data_file' variable with your CSV file name")
    print("2. Ensure your data has the required columns:")
    print("   - delta_dur_percents: duration difference (%)")
    print("   - audNoise: auditory noise level")
    print("   - standardDur: standard duration")
    print("   - conflictDur: visual conflict level")
    print("   - responses: participant responses")
    print("   - chose_test: binary choice (test vs standard)")
    print("3. Run this script to compare standard vs causal inference models")


#----

# ===============================
# CAUSAL INFERENCE MODEL IMPLEMENTATION
# ===============================

def fusionAV(sigma_av_a, sigma_av_v, S_a, conflict):
    """
    Compute reliability-weighted fusion of auditory and visual estimates.
    
    Parameters:
    -----------
    sigma_av_a : float
        Auditory noise (standard deviation)
    sigma_av_v : float  
        Visual noise (standard deviation)
    S_a : float
        True auditory duration
    conflict : float
        Visual conflict (difference between visual and auditory durations)
        
    Returns:
    --------
    fused_mean : float
        Fused estimate of duration
    fused_sigma : float
        Uncertainty of fused estimate
    """
    S_v = S_a + conflict
    J_a = 1 / sigma_av_a**2  # Auditory precision
    J_v = 1 / sigma_av_v**2  # Visual precision
    w_a = J_a / (J_a + J_v)  # Auditory weight
    w_v = 1 - w_a  # Visual weight
    fused_mean = w_a * S_a + w_v * S_v
    fused_sigma = np.sqrt(1 / (J_a + J_v))
    return fused_mean, fused_sigma

def compute_likelihood_C1(m_a, m_v, sigma_av_a, sigma_av_v):
    """
    Compute likelihood of measurements under common cause (C=1).
    
    Parameters:
    -----------
    m_a : float or array
        Auditory measurement(s)
    m_v : float or array
        Visual measurement(s)
    sigma_av_a : float
        Auditory noise
    sigma_av_v : float
        Visual noise
        
    Returns:
    --------
    likelihood : float or array
        Likelihood under common cause
    """
    var_sum = sigma_av_a ** 2 + sigma_av_v ** 2
    return (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-((m_a - m_v) ** 2) / (2 * var_sum))

def compute_likelihood_C2(m_a, m_v, S_a, S_v, sigma_av_a, sigma_av_v):
    """
    Compute likelihood of measurements under independent causes (C=2).
    
    Parameters:
    -----------
    m_a : float or array
        Auditory measurement(s)
    m_v : float or array
        Visual measurement(s)
    S_a : float
        True auditory duration
    S_v : float
        True visual duration
    sigma_a : float
        Auditory noise
    sigma_v : float
        Visual noise
        
    Returns:
    --------
    likelihood : float or array
        Likelihood under independent causes
    """
    return norm.pdf(m_a, S_a, sigma_av_a) * norm.pdf(m_v, S_v, sigma_av_v)

def compute_posterior_common_cause(m_a, m_v, S_a, S_v, sigma_av_a, sigma_av_v, p_common_prior):
    """
    Compute posterior probability of common cause given measurements.
    
    Parameters:
    -----------
    m_a : float or array
        Auditory measurement(s)
    m_v : float or array
        Visual measurement(s)
    S_a : float
        True auditory duration
    S_v : float
        True visual duration  
    sigma_a : float
        Auditory noise
    sigma_v : float
        Visual noise
    p_common_prior : float
        Prior probability of common cause
        
    Returns:
    --------
    p_common_posterior : float or array
        Posterior probability of common cause
    """
    likeli_c1 = compute_likelihood_C1(m_a, m_v, sigma_a, sigma_v)
    likeli_c2 = compute_likelihood_C2(m_a, m_v, S_a, S_v, sigma_a, sigma_v)
    
    # Bayes' rule
    numerator = likeli_c1 * p_common_prior
    denominator = likeli_c1 * p_common_prior + likeli_c2 * (1 - p_common_prior)
    
    return numerator / denominator

def causal_inference_estimate(S_a, conflict, sigma_a, sigma_v, p_common_prior):
    """
    Compute causal inference estimate for auditory duration in AV condition.
    
    Parameters:
    -----------
    S_a : float
        True auditory duration
    conflict : float
        Visual conflict (S_v - S_a)
    sigma_a : float
        Auditory noise
    sigma_v : float
        Visual noise
    p_common_prior : float
        Prior probability of common cause
        
    Returns:
    --------
    hat_S_av_a : float
        Final auditory estimate after causal inference
    p_common_posterior : float
        Posterior probability of common cause
    """
    S_v = S_a + conflict
    
    # Simulate measurements
    m_a = np.random.normal(S_a, sigma_a)
    m_v = np.random.normal(S_v, sigma_v)
    
    # Compute posterior probability of common cause
    p_common_posterior = compute_posterior_common_cause(
        m_a, m_v, S_a, S_v, sigma_a, sigma_v, p_common_prior
    )
    
    # Estimates under each causal structure
    fused_mean, _ = fusionAV(sigma_a, sigma_v, S_a, conflict)  # C=1: fusion
    auditory_only = m_a  # C=2: auditory only
    
    # Model averaging
    hat_S_av_a = fused_mean * p_common_posterior + auditory_only * (1 - p_common_posterior)
    
    return hat_S_av_a, p_common_posterior

def causal_inference_psychometric_function(delta_dur, lambda_, mu, sigma_a, sigma_v, p_common_prior, conflict):
    """
    Psychometric function incorporating causal inference model.
    
    Parameters:
    -----------
    delta_dur : float or array
        Duration difference (test - standard)
    lambda_ : float
        Lapse rate
    mu : float
        PSE (point of subjective equality)
    sigma_a : float
        Auditory noise
    sigma_v : float
        Visual noise
    p_common_prior : float
        Prior probability of common cause
    conflict : float
        Visual conflict level
        
    Returns:
    --------
    p_choose_test : float or array
        Probability of choosing test interval
    """
    # Convert delta_dur to test and standard durations
    # Assuming standard duration is at PSE
    S_standard = 0  # Assuming normalized durations
    S_test = S_standard + delta_dur + mu
    
    # Generate estimates for both intervals using causal inference
    hat_S_test, _ = causal_inference_estimate(S_test, conflict, sigma_a, sigma_v, p_common_prior)
    hat_S_standard, _ = causal_inference_estimate(S_standard, conflict, sigma_a, sigma_v, p_common_prior)
    
    # Decision based on difference
    decision_variable = hat_S_test - hat_S_standard
    
    # Apply lapse rate and psychometric transformation
    p_choose_test = lambda_/2 + (1 - lambda_) * norm.cdf(decision_variable, loc=0, scale=sigma_a)
    
    return p_choose_test

# ===============================
# CAUSAL INFERENCE FITTING FUNCTIONS
# ===============================

def causal_inference_negative_log_likelihood(params, delta_dur, chose_test, total_responses, conflicts):
    """
    Negative log-likelihood for causal inference model.
    
    Parameters:
    -----------
    params : array
        [lambda_, mu, sigma_a, sigma_v, p_common_prior]
    delta_dur : array
        Duration differences
    chose_test : array
        Number of times test was chosen
    total_responses : array
        Total responses per condition
    conflicts : array
        Visual conflict levels
        
    Returns:
    --------
    nll : float
        Negative log-likelihood
    """
    lambda_, mu, sigma_a, sigma_v, p_common_prior = params
    
    # Compute probabilities for each data point
    p_choose_test = np.zeros_like(delta_dur, dtype=float)
    
    for i in range(len(delta_dur)):
        p_choose_test[i] = causal_inference_psychometric_function(
            delta_dur[i], lambda_, mu, sigma_a, sigma_v, p_common_prior, conflicts[i]
        )
    
    # Clip probabilities to avoid log(0)
    epsilon = 1e-9
    p_choose_test = np.clip(p_choose_test, epsilon, 1 - epsilon)
    
    # Compute negative log-likelihood
    log_likelihood = np.sum(
        chose_test * np.log(p_choose_test) + 
        (total_responses - chose_test) * np.log(1 - p_choose_test)
    )
    
    return -log_likelihood

def fit_causal_inference_model(grouped_data, init_guesses=None):
    """
    Fit the causal inference model to grouped data.
    
    Parameters:
    -----------
    grouped_data : DataFrame
        Grouped experimental data
    init_guesses : list, optional
        Initial parameter guesses [lambda_, mu, sigma_a, sigma_v, p_common_prior]
        
    Returns:
    --------
    result : OptimizeResult
        Fitted parameters and optimization result
    """
    if init_guesses is None:
        init_guesses = [0.05, 0.0, 0.2, 0.2, 0.5]  # Default initial guesses
    
    # Extract data
    delta_dur = grouped_data[intensityVariable].values
    chose_test = grouped_data['num_of_chose_test'].values
    total_responses = grouped_data['total_responses'].values
    conflicts = grouped_data[conflictVar].values
    
    # Set bounds: [lambda_, mu, sigma_a, sigma_v, p_common_prior]
    bounds = [
        (0, 0.25),    # lambda_: lapse rate
        (-2, 2),      # mu: PSE
        (0.01, 2),    # sigma_a: auditory noise
        (0.01, 2),    # sigma_v: visual noise
        (0, 1)        # p_common_prior: prior probability of common cause
    ]
    
    # Fit the model
    result = minimize(
        causal_inference_negative_log_likelihood,
        x0=init_guesses,
        args=(delta_dur, chose_test, total_responses, conflicts),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    return result

# ===============================
# PLOTTING FUNCTIONS FOR CAUSAL INFERENCE MODEL
# ===============================

def plot_causal_inference_psychometric(data, fitted_params, title_suffix=""):
    """
    Plot fitted causal inference psychometric functions.
    
    Parameters:
    -----------
    data : DataFrame
        Experimental data
    fitted_params : array
        Fitted parameters [lambda_, mu, sigma_a, sigma_v, p_common_prior]
    title_suffix : str
        Additional text for plot title
    """
    lambda_, mu, sigma_a, sigma_v, p_common_prior = fitted_params
    
    plt.figure(figsize=(12, 6))
    
    # Get unique conflict levels
    unique_conflicts = sorted(data[conflictVar].unique())
    colors = sns.color_palette("viridis", n_colors=len(unique_conflicts))
    
    for i, conflict_level in enumerate(unique_conflicts):
        # Filter data for this conflict level
        df_conflict = data[data[conflictVar] == conflict_level]
        grouped_conflict = groupByChooseTest(df_conflict)
        
        if len(grouped_conflict) == 0:
            continue
            
        # Plot data points
        bin_and_plot(grouped_conflict, bin_method='cut', bins=10, plot=True, color=colors[i])
        
        # Plot fitted curve
        x = np.linspace(-0.9, 0.9, 100)
        y = np.zeros_like(x)
        
        for j, delta in enumerate(x):
            y[j] = causal_inference_psychometric_function(
                delta, lambda_, mu, sigma_a, sigma_v, p_common_prior, conflict_level
            )
        
        plt.plot(x, y, color=colors[i], linewidth=3, 
                label=f'Conflict: {conflict_level:.3f}')
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Duration Difference (%)')
    plt.ylabel('P(choose test)')
    plt.title(f'Causal Inference Model {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = (f'λ={lambda_:.3f}, μ={mu:.3f}\n'
                 f'σ_a={sigma_a:.3f}, σ_v={sigma_v:.3f}\n'
                 f'P(common)={p_common_prior:.3f}')
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# ===============================
# MODEL COMPARISON FUNCTIONS
# ===============================

def fit_and_compare_models(data, n_starts=1):
    """
    Fit both standard psychometric and causal inference models and compare them.
    
    Parameters:
    -----------
    data : DataFrame
        Experimental data
    n_starts : int
        Number of random starting points for optimization
        
    Returns:
    --------
    results : dict
        Dictionary containing fitted models and comparison metrics
    """
    print("Fitting standard psychometric model...")
    standard_fit = fitMultipleStartingPoints(data, nStart=n_starts)
    
    print("\nFitting causal inference model...")
    grouped_data = groupByChooseTest(data)
    
    # Try multiple starting points for causal inference model
    best_ci_fit = None
    best_ci_nll = float('inf')
    
    for i in range(n_starts):
        # Random initial guesses
        init_guesses = [
            np.random.uniform(0.01, 0.1),    # lambda_
            np.random.uniform(-0.5, 0.5),   # mu
            np.random.uniform(0.1, 0.5),    # sigma_a
            np.random.uniform(0.1, 0.5),    # sigma_v
            np.random.uniform(0.2, 0.8)     # p_common_prior
        ]
        
        try:
            ci_fit = fit_causal_inference_model(grouped_data, init_guesses)
            
            # Compute negative log-likelihood for comparison
            delta_dur = grouped_data[intensityVariable].values
            chose_test = grouped_data['num_of_chose_test'].values
            total_responses = grouped_data['total_responses'].values
            conflicts = grouped_data[conflictVar].values
            
            ci_nll = causal_inference_negative_log_likelihood(
                ci_fit.x, delta_dur, chose_test, total_responses, conflicts
            )
            
            if ci_nll < best_ci_nll:
                best_ci_nll = ci_nll
                best_ci_fit = ci_fit
                
        except Exception as e:
            print(f"Fit {i+1} failed: {e}")
            continue
    
    if best_ci_fit is None:
        print("Warning: Causal inference model fitting failed!")
        return {"standard_fit": standard_fit, "causal_inference_fit": None}
    
    # Compute standard model NLL for comparison
    grouped_data = groupByChooseTest(data)
    delta_dur = grouped_data[intensityVariable].values
    chose_test = grouped_data['num_of_chose_test'].values
    total_responses = grouped_data['total_responses'].values
    conflicts = grouped_data[conflictVar].values
    noise_levels = grouped_data[sensoryVar].values
    
    standard_nll = nLLJoint(
        standard_fit.x, delta_dur, chose_test, total_responses, conflicts, noise_levels
    )
    
    # Compute AIC and BIC for model comparison
    n_data = len(delta_dur)
    
    # Standard model parameters
    if allIndependent:
        n_params_standard = len(standard_fit.x)
    else:
        n_params_standard = len(standard_fit.x)
    
    # Causal inference model has 5 parameters
    n_params_ci = 5
    
    # AIC = 2k - 2ln(L) = 2k + 2NLL
    aic_standard = 2 * n_params_standard + 2 * standard_nll
    aic_ci = 2 * n_params_ci + 2 * best_ci_nll
    
    # BIC = k*ln(n) - 2ln(L) = k*ln(n) + 2NLL
    bic_standard = n_params_standard * np.log(n_data) + 2 * standard_nll
    bic_ci = n_params_ci * np.log(n_data) + 2 * best_ci_nll
    
    # Prepare results
    results = {
        "standard_fit": standard_fit,
        "causal_inference_fit": best_ci_fit,
        "standard_nll": standard_nll,
        "causal_inference_nll": best_ci_nll,
        "standard_aic": aic_standard,
        "causal_inference_aic": aic_ci,
        "standard_bic": bic_standard,
        "causal_inference_bic": bic_ci,
        "n_params_standard": n_params_standard,
        "n_params_ci": n_params_ci,
        "delta_aic": aic_ci - aic_standard,
        "delta_bic": bic_ci - bic_standard
    }
    
    # Print comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(f"Standard Model:")
    print(f"  Parameters: {n_params_standard}")
    print(f"  NLL: {standard_nll:.2f}")
    print(f"  AIC: {aic_standard:.2f}")
    print(f"  BIC: {bic_standard:.2f}")
    print(f"\nCausal Inference Model:")
    print(f"  Parameters: {n_params_ci}")
    print(f"  NLL: {best_ci_nll:.2f}")
    print(f"  AIC: {aic_ci:.2f}")
    print(f"  BIC: {bic_ci:.2f}")
    print(f"\nModel Comparison:")
    print(f"  ΔAIC: {aic_ci - aic_standard:.2f} (negative favors CI model)")
    print(f"  ΔBIC: {bic_ci - bic_standard:.2f} (negative favors CI model)")
    
    if aic_ci < aic_standard:
        print(f"  → Causal Inference model preferred by AIC")
    else:
        print(f"  → Standard model preferred by AIC")
        
    if bic_ci < bic_standard:
        print(f"  → Causal Inference model preferred by BIC")
    else:
        print(f"  → Standard model preferred by BIC")
    
    print("="*50)
    
    return results

def plot_model_comparison(data, results):
    """
    Plot both models for visual comparison.
    
    Parameters:
    -----------
    data : DataFrame
        Experimental data
    results : dict
        Results from fit_and_compare_models
    """
    if results["causal_inference_fit"] is None:
        print("Cannot plot comparison: Causal inference model fit failed")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot standard model
    plt.sca(ax1)
    plot_fitted_psychometric(
        data, results["standard_fit"], nLambda, nSigma, 
        uniqueSensory, uniqueStandard, uniqueConflict,
        standardVar, sensoryVar, conflictVar, intensityVariable
    )
    ax1.set_title(f'Standard Model (AIC: {results["standard_aic"]:.1f})')
    
    # Plot causal inference model
    plt.sca(ax2)
    plot_causal_inference_psychometric(
        data, results["causal_inference_fit"].x, 
        title_suffix=f'(AIC: {results["causal_inference_aic"]:.1f})'
    )
    
    plt.tight_layout()
    plt.show()

# ===============================
# EXAMPLE USAGE FUNCTIONS
# ===============================

def run_causal_inference_analysis(data_file, shared_sigma=True, all_independent=False):
    """
    Complete analysis pipeline with causal inference model.
    
    Parameters:
    -----------
    data_file : str
        Name of the data file to analyze
    shared_sigma : bool
        Whether to use shared sigma across conditions
    all_independent : bool
        Whether to use independent parameters for each condition
        
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    print(f"Loading and preprocessing data: {data_file}")
    data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(
        data_file, shared_sigma, all_independent
    )
    
    print(f"Fitting and comparing models...")
    results = fit_and_compare_models(data, n_starts=1)
    
    print(f"Plotting comparison...")
    plot_model_comparison(data, results)
    
    return results

# ===============================

