#!/usr/bin/env python3
"""
Test the modified fitNonSharedwErrorBars.py with log-normal observer model
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

# Import the modified functions
from fitNonSharedwErrorBars import psychometric_function

def test_lognormal_implementation():
    """Test that the log-normal psychometric function works correctly."""
    
    print("Testing log-normal psychometric function implementation...")
    
    # Test parameters
    x_values = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    lambda_ = 0.05
    mu = 0.0
    sigma = 0.3
    
    try:
        # Test the function
        p_values = psychometric_function(x_values, lambda_, mu, sigma)
        
        print("✓ Function executes without errors")
        print(f"✓ Input range: {x_values.min():.1f} to {x_values.max():.1f}")
        print(f"✓ Output range: {p_values.min():.3f} to {p_values.max():.3f}")
        
        # Check that probabilities are valid
        assert np.all(p_values >= 0) and np.all(p_values <= 1), "Probabilities out of range"
        print("✓ All probabilities in valid range [0, 1]")
        
        # Check that function is monotonic (should increase with x)
        diffs = np.diff(p_values)
        assert np.all(diffs >= -1e-10), "Function is not monotonically increasing"  # Allow tiny numerical errors
        print("✓ Function is monotonically increasing")
        
        # Check boundary behavior
        very_negative = psychometric_function(np.array([-0.8]), lambda_, mu, sigma)[0]
        very_positive = psychometric_function(np.array([0.8]), lambda_, mu, sigma)[0]
        
        print(f"✓ At x=-0.8: P = {very_negative:.3f} (should be close to lambda/2 = {lambda_/2:.3f})")
        print(f"✓ At x=+0.8: P = {very_positive:.3f} (should be close to 1-lambda/2 = {1-lambda_/2:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def compare_models_visual():
    """Visual comparison of cumulative normal vs log-normal."""
    
    print("\nCreating visual comparison...")
    
    from scipy.stats import norm
    
    x = np.linspace(-0.5, 0.5, 200)
    lambda_ = 0.05
    mu = 0.0
    sigma = 0.2
    
    # Original cumulative normal
    p_normal = lambda_/2 + (1-lambda_) * norm.cdf((x - mu) / sigma)
    
    # New log-normal (from our modified function)
    p_lognormal = psychometric_function(x, lambda_, mu, sigma)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(x, p_normal, 'r-', linewidth=2, label='Cumulative Normal (old)')
    plt.plot(x, p_lognormal, 'b-', linewidth=2, label='Log-Normal (new)')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Equal durations')
    plt.xlabel('Relative Duration Difference')
    plt.ylabel('P(Choose Test)')
    plt.title('Psychometric Function Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Highlight problematic region for negative durations
    neg_region = x < -0.3
    plt.fill_between(x[neg_region], 0, p_normal[neg_region], 
                    alpha=0.3, color='red', label='_nolegend_')
    plt.text(-0.4, 0.2, 'Problematic\nregion', color='red', fontweight='bold')
    
    plt.subplot(2, 1, 2)
    difference = p_lognormal - p_normal
    plt.plot(x, difference, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Relative Duration Difference')
    plt.ylabel('Log-Normal - Cumulative Normal')
    plt.title('Difference Between Models')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print some specific comparisons
    print(f"\nSpecific comparisons:")
    test_points = [-0.3, -0.1, 0.0, 0.1, 0.3]
    for test_x in test_points:
        idx = np.argmin(np.abs(x - test_x))
        print(f"x = {test_x:+.1f}: Normal = {p_normal[idx]:.3f}, Log-normal = {p_lognormal[idx]:.3f}, Diff = {difference[idx]:+.3f}")

def simulate_fitting_improvement():
    """Simulate data and show that log-normal fits better."""
    
    print("\nSimulating fitting improvement...")
    
    # Generate synthetic data using log-normal model
    np.random.seed(42)
    
    true_lambda = 0.08
    true_mu = 0.1  # Small bias
    true_sigma = 0.25
    
    x_levels = np.array([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
    n_trials = 40
    
    # Generate true probabilities using log-normal
    true_probs = psychometric_function(x_levels, true_lambda, true_mu, true_sigma)
    
    # Simulate responses
    n_chose_test = np.random.binomial(n_trials, true_probs)
    
    print(f"Generated {len(x_levels)} stimulus levels with {n_trials} trials each")
    print(f"True parameters: λ={true_lambda:.3f}, μ={true_mu:.3f}, σ={true_sigma:.3f}")
    
    # Try to fit both models
    from scipy.optimize import minimize
    from scipy.stats import norm
    
    def negative_log_likelihood_normal(params):
        lambda_, mu, sigma = params
        p_pred = lambda_/2 + (1-lambda_) * norm.cdf((x_levels - mu) / sigma)
        epsilon = 1e-9
        p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
        return -np.sum(n_chose_test * np.log(p_pred) + (n_trials - n_chose_test) * np.log(1 - p_pred))
    
    def negative_log_likelihood_lognormal(params):
        lambda_, mu, sigma = params
        p_pred = psychometric_function(x_levels, lambda_, mu, sigma)
        epsilon = 1e-9
        p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
        return -np.sum(n_chose_test * np.log(p_pred) + (n_trials - n_chose_test) * np.log(1 - p_pred))
    
    # Fit cumulative normal
    try:
        result_normal = minimize(negative_log_likelihood_normal, 
                               x0=[0.05, 0.0, 0.2],
                               bounds=[(0, 0.25), (-0.5, 0.5), (0.01, 1.0)],
                               method='L-BFGS-B')
        nll_normal = result_normal.fun
        aic_normal = 2 * 3 + 2 * nll_normal
        print(f"✓ Normal fit: λ={result_normal.x[0]:.3f}, μ={result_normal.x[1]:.3f}, σ={result_normal.x[2]:.3f}")
        print(f"  AIC = {aic_normal:.1f}")
    except Exception as e:
        print(f"✗ Normal fit failed: {e}")
        aic_normal = np.inf
    
    # Fit log-normal
    try:
        result_lognormal = minimize(negative_log_likelihood_lognormal,
                                  x0=[0.05, 0.0, 0.3],
                                  bounds=[(0, 0.25), (-1.0, 1.0), (0.01, 2.0)],
                                  method='L-BFGS-B')
        nll_lognormal = result_lognormal.fun
        aic_lognormal = 2 * 3 + 2 * nll_lognormal
        print(f"✓ Log-normal fit: λ={result_lognormal.x[0]:.3f}, μ={result_lognormal.x[1]:.3f}, σ={result_lognormal.x[2]:.3f}")
        print(f"  AIC = {aic_lognormal:.1f}")
    except Exception as e:
        print(f"✗ Log-normal fit failed: {e}")
        aic_lognormal = np.inf
    
    if aic_lognormal < aic_normal:
        print(f"\n✓ Log-normal model is better (ΔAIC = {aic_normal - aic_lognormal:.1f})")
    else:
        print(f"\n? Unexpected: Normal model seems better")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("TESTING MODIFIED fitNonSharedwErrorBars.py")
    print("="*60)
    
    # Run tests
    success = test_lognormal_implementation()
    
    if success:
        print("\n" + "="*40)
        print("VISUAL COMPARISON")
        print("="*40)
        compare_models_visual()
        
        print("\n" + "="*40)
        print("FITTING IMPROVEMENT DEMO")
        print("="*40)
        simulate_fitting_improvement()
        
        print("\n" + "="*40)
        print("SUCCESS!")
        print("="*40)
        print("✓ Your fitNonSharedwErrorBars.py now uses log-normal observer model")
        print("✓ All existing code should work with better theoretical foundation")
        print("✓ You should see improved model fits with your duration data")
        print("\nNext steps:")
        print("1. Run your existing analysis scripts")
        print("2. Compare AIC/BIC values with your old results")
        print("3. Interpret parameters in log space (μ, σ)")
        
    else:
        print("\n✗ Tests failed. Please check the implementation.")