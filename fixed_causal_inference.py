import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm import tqdm

def load_and_clean_data(filename):
    """Load data and handle common issues"""
    print("Loading and cleaning data...")
    
    # Load data
    data = pd.read_csv(f"data/{filename}")
    print(f"Original data shape: {data.shape}")
    
    # Remove rows with NaN values in critical columns
    critical_cols = ['testDurS', 'standardDur', 'conflictDur', 'audNoise', 'responses', 'order']
    data = data.dropna(subset=critical_cols)
    print(f"After removing NaN: {data.shape}")
    
    # Create response variables
    data['chose_test'] = (data['responses'] == data['order']).astype(int)
    data['deltaDurS'] = data['testDurS'] - data['standardDur']
    
    # Remove zero audNoise (if any)
    data = data[data['audNoise'] > 0]
    print(f"After removing zero audNoise: {data.shape}")
    
    # Work in log space for stimulus representation
    data['logDeltaDur'] = np.log(data['testDurS'] / data['standardDur'])
    data['logTestDur'] = np.log(data['testDurS'])
    data['logStandardDur'] = np.log(data['standardDur'])
    
    # Handle conflicts carefully for log space
    # For negative conflicts, we need to handle them properly
    # Visual conflict = visual_duration - auditory_duration
    # If conflict is negative, visual is shorter than auditory
    
    # Create visual durations based on conflicts
    data['visualTestDur'] = data['testDurS']  # Test has no conflict
    data['visualStandardDur'] = data['standardDur'] + data['conflictDur']
    
    # Only keep data where visual durations are positive (for log space)
    valid_visual = data['visualStandardDur'] > 0
    print(f"Rows with positive visual durations: {valid_visual.sum()}/{len(data)}")
    data = data[valid_visual]
    
    # Now we can safely take logs
    data['logVisualTestDur'] = np.log(data['visualTestDur'])
    data['logVisualStandardDur'] = np.log(data['visualStandardDur'])
    data['logConflict'] = data['logVisualStandardDur'] - data['logStandardDur']
    
    print(f"Final data shape: {data.shape}")
    print(f"Unique audNoise: {sorted(data['audNoise'].unique())}")
    print(f"Unique conflicts: {sorted(data['conflictDur'].unique())}")
    print(f"Conflict range: {data['conflictDur'].min():.3f} to {data['conflictDur'].max():.3f}")
    
    return data

def group_data_for_fitting(data):
    """Group data for psychometric fitting"""
    grouped = data.groupby(['logDeltaDur', 'audNoise', 'logConflict']).agg(
        num_chose_test=('chose_test', 'sum'),
        total_responses=('chose_test', 'count'),
        deltaDurS=('deltaDurS', 'first'),  # Keep original values for reference
        conflictDur=('conflictDur', 'first')
    ).reset_index()
    
    grouped['p_choose_test'] = grouped['num_chose_test'] / grouped['total_responses']
    
    print(f"Grouped data shape: {grouped.shape}")
    print(f"Total conditions: {len(grouped)}")
    print(f"Log conflict range: {grouped['logConflict'].min():.3f} to {grouped['logConflict'].max():.3f}")
    
    return grouped

def log_space_causal_inference_model(log_delta_dur, log_conflict, lambda_, sigma_a, sigma_v, p_c):
    """
    Causal inference model working in log duration space
    
    Parameters:
    - log_delta_dur: log(test/standard) duration ratio
    - log_conflict: log(visual/auditory) conflict for standard
    - lambda_: lapse rate
    - sigma_a, sigma_v: sensory noise in log space (std dev)
    - p_c: prior probability of common cause
    """
    
    # Log durations (standard = 0 in log ratio space)
    log_S_std = 0  # log(standard/standard) = 0
    log_S_test = log_delta_dur  # log(test/standard)
    
    # Visual log durations (with conflict for standard, no conflict for test)
    log_S_visual_std = log_conflict  # log(visual_standard/auditory_standard)
    log_S_visual_test = log_S_test  # No conflict for test: log(visual_test/auditory_test) = log(test/standard)
    
    # Measurements in log space (add noise later if needed)
    m_a_std = log_S_std  # Auditory measurement of standard
    m_v_std = log_S_visual_std  # Visual measurement of standard
    m_a_test = log_S_test  # Auditory measurement of test
    m_v_test = log_S_visual_test  # Visual measurement of test
    
    # Common cause likelihood for standard pair
    var_sum = sigma_a**2 + sigma_v**2
    if var_sum <= 0:
        return 0.5  # Return neutral response for invalid parameters
        
    likelihood_c1_std = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a_std - m_v_std)**2 / (2 * var_sum))
    
    # Independent causes likelihood for standard pair
    likelihood_c2_std = norm.pdf(m_a_std, loc=log_S_std, scale=sigma_a) * norm.pdf(m_v_std, loc=log_S_visual_std, scale=sigma_v)
    
    # Check for numerical issues
    if not (np.isfinite(likelihood_c1_std) and np.isfinite(likelihood_c2_std)):
        return 0.5
    
    # Posterior probability of common cause for standard
    denominator = likelihood_c1_std * p_c + likelihood_c2_std * (1 - p_c)
    if denominator <= 0:
        posterior_c1_std = 0.5  # Default to neutral if calculation fails
    else:
        posterior_c1_std = (likelihood_c1_std * p_c) / denominator
        
    # Clamp posterior to valid range
    posterior_c1_std = np.clip(posterior_c1_std, 0.001, 0.999)
    
    # Reliability-based weights for fusion
    w_a = (1/sigma_a**2) / (1/sigma_a**2 + 1/sigma_v**2)
    w_v = 1 - w_a
    
    # Estimates for standard in log space
    # Fusion estimate (optimal combination)
    fused_estimate_std = w_a * m_a_std + w_v * m_v_std
    
    # Segregation estimate (auditory only)
    segregated_estimate_std = m_a_std
    
    # Final estimate for standard (model averaging)
    final_estimate_std = posterior_c1_std * fused_estimate_std + (1 - posterior_c1_std) * segregated_estimate_std
    
    # For test, assume fusion (no conflict typically means high confidence in common cause)
    final_estimate_test = w_a * m_a_test + w_v * m_v_test
    
    # Decision in log space
    log_delta_estimate = final_estimate_test - final_estimate_std
    
    # Decision noise in log space
    sigma_decision = np.sqrt(sigma_a**2 + sigma_v**2) / 2  # Simplified decision noise
    
    if sigma_decision <= 0:
        return lambda_/2 + (1 - lambda_) * 0.5  # Neutral response
    
    # Psychometric function
    p_choose_test = lambda_/2 + (1 - lambda_) * norm.cdf(log_delta_estimate / sigma_decision)
    
    # Final clamp to valid probability range
    p_choose_test = np.clip(p_choose_test, 1e-9, 1 - 1e-9)
    
    return p_choose_test

def negative_log_likelihood_causal(params, data):
    """Negative log likelihood for causal inference model"""
    
    # Extract parameters for each SNR condition
    lambda_ = params[0]  # Shared lapse rate
    
    # High SNR (audNoise = 0.1) parameters
    sigma_a_high = params[1]
    sigma_v_high = params[2] 
    p_c_high = params[3]
    
    # Low SNR (audNoise = 1.2) parameters
    sigma_a_low = params[4]
    sigma_v_low = params[5]
    p_c_low = params[6]
    
    # Add soft constraints: noise should generally increase with lower SNR
    # P(common cause) relationship with noise is complex, so we'll be more flexible
    constraint_penalty = 0
    
    # Soft penalty for unrealistic noise ordering (not hard constraint)
    if sigma_a_low < sigma_a_high * 0.8:  # Allow some flexibility
        constraint_penalty += (sigma_a_high * 0.8 - sigma_a_low) * 100
    
    if sigma_v_low < sigma_v_high * 0.8:  # Allow some flexibility
        constraint_penalty += (sigma_v_high * 0.8 - sigma_v_low) * 100
    
    total_nll = constraint_penalty
    
    for _, row in data.iterrows():
        log_delta_dur = row['logDeltaDur']
        log_conflict = row['logConflict']
        snr = row['audNoise']
        n_chose_test = row['num_chose_test']
        n_total = row['total_responses']
        
        # Select parameters based on SNR
        if np.isclose(snr, 0.1):
            sigma_a, sigma_v, p_c = sigma_a_high, sigma_v_high, p_c_high
        elif np.isclose(snr, 1.2):
            sigma_a, sigma_v, p_c = sigma_a_low, sigma_v_low, p_c_low
        else:
            raise ValueError(f"Unknown SNR value: {snr}")
        
        # Compute probability using log space model
        try:
            p = log_space_causal_inference_model(log_delta_dur, log_conflict, lambda_, sigma_a, sigma_v, p_c)
            
            # Check for numerical issues
            if not np.isfinite(p):
                return 1e6
                
        except (ValueError, FloatingPointError, OverflowError):
            return 1e6
        
        # Clamp probability
        p = np.clip(p, 1e-9, 1 - 1e-9)
        
        # Add to negative log likelihood
        total_nll -= n_chose_test * np.log(p) + (n_total - n_chose_test) * np.log(1 - p)
    
    return total_nll

def fit_causal_inference_model(grouped_data, n_starts=5):
    """Fit causal inference model with multiple starting points"""
    
    print(f"Fitting causal inference model with {n_starts} starting points...")
    
    # Parameter bounds for log space: [lambda, sigma_a_high, sigma_v_high, p_c_high, sigma_a_low, sigma_v_low, p_c_low]
    bounds = [
        (0.001, 0.1),   # lambda (lapse rate) - tighter bound
        (0.05, 1.0),    # sigma_a_high (high SNR auditory noise in log space)
        (0.05, 1.0),    # sigma_v_high (high SNR visual noise in log space)  
        (0.2, 0.95),    # p_c_high (high SNR common cause prior)
        (0.05, 1.5),    # sigma_a_low (low SNR auditory noise in log space)
        (0.05, 1.5),    # sigma_v_low (low SNR visual noise in log space)
        (0.2, 0.95)     # p_c_low (low SNR common cause prior)
    ]
    
    best_nll = float('inf')
    best_params = None
    
    for start in tqdm(range(n_starts), desc="Fitting starts"):
        # Random initial guesses within bounds
        init_params = []
        for (lower, upper) in bounds:
            init_params.append(np.random.uniform(lower, upper))
        
        try:
            result = minimize(
                negative_log_likelihood_causal,
                x0=init_params,
                args=(grouped_data,),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success and result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
                print(f"Start {start+1}: NLL = {result.fun:.2f} ✓")
            else:
                print(f"Start {start+1}: Failed or worse fit")
                
        except Exception as e:
            print(f"Start {start+1}: Error - {e}")
            continue
    
    if best_params is None:
        raise ValueError("All fitting attempts failed!")
    
    print(f"\nBest fit: NLL = {best_nll:.2f}")
    print(f"Parameters: {best_params}")
    
    return best_params, best_nll

def plot_results(data, grouped_data, fitted_params):
    """Plot the fitted psychometric curves"""
    
    lambda_ = fitted_params[0]
    
    # Get unique values
    unique_snr = sorted([x for x in data['audNoise'].unique() if not np.isnan(x)])
    unique_conflicts = sorted([x for x in data['conflictDur'].unique() if not np.isnan(x)])
    
    fig, axes = plt.subplots(1, len(unique_snr), figsize=(6*len(unique_snr), 6))
    if len(unique_snr) == 1:
        axes = [axes]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_conflicts)))
    
    for snr_idx, snr in enumerate(unique_snr):
        ax = axes[snr_idx]
        
        # Get parameters for this SNR
        if np.isclose(snr, 0.1):
            sigma_a, sigma_v, p_c = fitted_params[1], fitted_params[2], fitted_params[3]
            snr_label = "High SNR (0.1)"
        else:
            sigma_a, sigma_v, p_c = fitted_params[4], fitted_params[5], fitted_params[6]
            snr_label = "Low SNR (1.2)"
        
        for conflict_idx, conflict in enumerate(unique_conflicts):
            color = colors[conflict_idx]
            
            # Filter data for this condition
            condition_data = grouped_data[
                (np.isclose(grouped_data['audNoise'], snr)) & 
                (np.isclose(grouped_data['conflictDur'], conflict, atol=1e-3))
            ]
            
            if len(condition_data) == 0:
                continue
            
            # Plot data points
            x_data = condition_data['logDeltaDur'].values
            y_data = condition_data['p_choose_test'].values
            sizes = condition_data['total_responses'].values * 20  # Scale point sizes
            
            ax.scatter(x_data, y_data, s=sizes, alpha=0.7, color=color, 
                      label=f'Conflict: {conflict*1000:.0f}ms')
            
            # Plot fitted curve
            # Get log conflict for this condition
            log_conflict = np.log((0.5 + conflict) / 0.5) if conflict != 0 else 0
            
            x_smooth = np.linspace(-0.4, 0.4, 100)
            y_smooth = log_space_causal_inference_model(
                x_smooth, log_conflict, lambda_, sigma_a, sigma_v, p_c
            )
            ax.plot(x_smooth, y_smooth, color=color, linewidth=2)
        
        ax.set_xlabel('Log Delta Duration')
        ax.set_ylabel('P(Choose Test)')
        ax.set_title(f'{snr_label}\\n$\\sigma_a$={sigma_a:.3f}, $\\sigma_v$={sigma_v:.3f}, P(C=1)={p_c:.3f}')
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def evaluate_model_fit(data, grouped_data, fitted_params):
    """Evaluate the quality of the model fit"""
    
    # Calculate R-squared and other metrics
    y_observed = []
    y_predicted = []
    
    lambda_ = fitted_params[0]
    
    for _, row in grouped_data.iterrows():
        log_delta_dur = row['logDeltaDur']
        log_conflict = row['logConflict']
        snr = row['audNoise']
        p_obs = row['p_choose_test']
        
        # Get parameters for this SNR
        if np.isclose(snr, 0.1):
            sigma_a, sigma_v, p_c = fitted_params[1], fitted_params[2], fitted_params[3]
        else:
            sigma_a, sigma_v, p_c = fitted_params[4], fitted_params[5], fitted_params[6]
        
        # Predict probability
        p_pred = log_space_causal_inference_model(log_delta_dur, log_conflict, lambda_, sigma_a, sigma_v, p_c)
        
        y_observed.append(p_obs)
        y_predicted.append(p_pred)
    
    y_observed = np.array(y_observed)
    y_predicted = np.array(y_predicted)
    
    # Calculate metrics
    ss_res = np.sum((y_observed - y_predicted) ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    rmse = np.sqrt(np.mean((y_observed - y_predicted) ** 2))
    
    print(f"\nMODEL FIT QUALITY:")
    print(f"R² = {r_squared:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"Mean absolute error = {np.mean(np.abs(y_observed - y_predicted)):.4f}")
    
    return r_squared, rmse

def main():
    """Main execution function"""
    
    # Load and clean data
    data = load_and_clean_data("dt_all.csv")
    
    # Group data for fitting
    grouped_data = group_data_for_fitting(data)
    
    # Fit the model
    try:
        fitted_params, nll = fit_causal_inference_model(grouped_data, n_starts=10)
        
        print("\n" + "="*50)
        print("FITTED PARAMETERS")
        print("="*50)
        print(f"Lapse rate (λ): {fitted_params[0]:.4f}")
        print(f"High SNR - σ_a: {fitted_params[1]:.4f}, σ_v: {fitted_params[2]:.4f}, P(C=1): {fitted_params[3]:.4f}")
        print(f"Low SNR  - σ_a: {fitted_params[4]:.4f}, σ_v: {fitted_params[5]:.4f}, P(C=1): {fitted_params[6]:.4f}")
        print(f"Negative Log Likelihood: {nll:.2f}")
        
        # Interpret parameters
        print(f"\nPARAMETER INTERPRETATION (Log Space):")
        print(f"• Auditory noise increases from {fitted_params[1]:.3f} to {fitted_params[4]:.3f} log units")
        print(f"• Visual noise increases from {fitted_params[2]:.3f} to {fitted_params[5]:.3f} log units") 
        print(f"• P(Common cause) increases from {fitted_params[3]:.3f} to {fitted_params[6]:.3f}")
        
        # Validate results
        if fitted_params[6] > fitted_params[3]:
            print("\n✓ VALID: P(C=1) increases with noise (as expected)")
        else:
            print("\n⚠ WARNING: P(C=1) decreases with noise (unexpected)")
        
        if fitted_params[4] >= fitted_params[1] and fitted_params[5] >= fitted_params[2]:
            print("✓ VALID: Sensory noise increases with lower SNR (as expected)")
        else:
            print("⚠ WARNING: Sensory noise decreases with lower SNR (unexpected)")
        
        # Evaluate model fit
        r2, rmse = evaluate_model_fit(data, grouped_data, fitted_params)
        
        # Plot results
        plot_results(data, grouped_data, fitted_params)
        
        # Evaluate model fit
        evaluate_model_fit(data, grouped_data, fitted_params)
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
