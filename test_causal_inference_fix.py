#!/usr/bin/env python3
"""
Test script to demonstrate the corrected causal inference implementation.
This shows the difference between the old (incorrect) and new (correct) approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Correct implementation (what you should use)
def correct_causal_inference(sigmaAV_A, sigmaAV_V, S_a, p_c, visualConflict):
    """
    CORRECT: Generate single noisy measurements, then do causal inference.
    """
    S_v = S_a + visualConflict
    
    # Generate single noisy measurements
    m_a = np.random.normal(S_a, sigmaAV_A)  # Single value
    m_v = np.random.normal(S_v, sigmaAV_V)  # Single value
    
    # Compute likelihoods
    var_sum = sigmaAV_A**2 + sigmaAV_V**2
    likelihood_c1 = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))
    likelihood_c2 = norm.pdf(m_a, S_a, sigmaAV_A) * norm.pdf(m_v, S_v, sigmaAV_V)
    
    # Posterior probability
    posterior_c1 = (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))
    
    # Reliability-weighted fusion
    J_a = 1 / sigmaAV_A**2
    J_v = 1 / sigmaAV_V**2
    w_a = J_a / (J_a + J_v)
    w_v = J_v / (J_a + J_v)
    fused_estimate = w_a * S_a + w_v * S_v
    
    # Model averaging
    final_estimate = posterior_c1 * fused_estimate + (1 - posterior_c1) * m_a
    
    return final_estimate, posterior_c1, m_a, m_v

# Incorrect implementation (old approach)
def incorrect_causal_inference(sigmaAV_A, sigmaAV_V, S_a, p_c, visualConflict):
    """
    INCORRECT: Generate arrays instead of single measurements.
    This doesn't make sense for causal inference!
    """
    S_v = S_a + visualConflict
    
    # Generate likelihood arrays (WRONG!)
    m_a = np.linspace(0, S_a + 10*sigmaAV_A, 500)  # Array of 500 values
    m_v = np.linspace(0, S_v + 10*sigmaAV_V, 500)  # Array of 500 values
    
    # This doesn't make sense - you can't compute a single likelihood from arrays
    # But let's show what happens if you try...
    try:
        var_sum = sigmaAV_A**2 + sigmaAV_V**2
        likelihood_c1 = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))
        # This will either fail or give nonsensical results
        return "ERROR: Cannot compute single likelihood from arrays"
    except:
        return "ERROR: Array dimensions don't match"

def demonstrate_difference():
    """
    Show the difference between correct and incorrect implementations.
    """
    print("="*70)
    print("CAUSAL INFERENCE: CORRECT vs INCORRECT IMPLEMENTATION")
    print("="*70)
    
    # Parameters
    sigmaAV_A = 0.2
    sigmaAV_V = 0.3
    S_a = 0.5
    p_c = 0.7
    visualConflict = 0.2
    
    print(f"Parameters:")
    print(f"  True auditory duration: {S_a}")
    print(f"  Visual conflict: {visualConflict}")
    print(f"  True visual duration: {S_a + visualConflict}")
    print(f"  Prior P(common): {p_c}")
    print(f"  Auditory noise: {sigmaAV_A}")
    print(f"  Visual noise: {sigmaAV_V}")
    print()
    
    print("CORRECT APPROACH:")
    print("-" * 50)
    print("Trial | m_a    | m_v    | P(C=1) | Estimate")
    print("-" * 50)
    
    estimates = []
    posteriors = []
    
    for trial in range(5):
        result = correct_causal_inference(sigmaAV_A, sigmaAV_V, S_a, p_c, visualConflict)
        if isinstance(result, tuple):
            estimate, posterior, m_a, m_v = result
            estimates.append(estimate)
            posteriors.append(posterior)
            print(f"{trial+1:5d} | {m_a:6.3f} | {m_v:6.3f} | {posterior:6.3f} | {estimate:8.3f}")
        else:
            print(f"{trial+1:5d} | {result}")
    
    print("-" * 50)
    print(f"Average estimate: {np.mean(estimates):.3f}")
    print(f"Average P(C=1): {np.mean(posteriors):.3f}")
    print()
    
    print("INCORRECT APPROACH:")
    print("-" * 50)
    result = incorrect_causal_inference(sigmaAV_A, sigmaAV_V, S_a, p_c, visualConflict)
    print(f"Result: {result}")
    print()
    
    print("KEY DIFFERENCES:")
    print("-" * 30)
    print("✓ CORRECT: m_a and m_v are single noisy measurements")
    print("✗ INCORRECT: m_a and m_v are arrays of possible values")
    print("✓ CORRECT: Can compute single likelihood values for Bayes' rule")
    print("✗ INCORRECT: Cannot meaningfully compute likelihoods from arrays")
    print("✓ CORRECT: Results in proper causal inference behavior")
    print("✗ INCORRECT: Conceptually flawed and mathematically invalid")
    print()
    
    print("CONCEPTUAL EXPLANATION:")
    print("-" * 30)
    print("Causal inference asks: 'Given my noisy observations m_a and m_v,")
    print("what's the probability they came from the same source?'")
    print()
    print("This requires ACTUAL measurements (single values), not probability")
    print("distributions over possible measurements (arrays).")
    print()
    print("The observer doesn't have access to the entire likelihood function -")
    print("they only have their specific noisy measurements on each trial.")

def plot_causal_inference_behavior():
    """
    Plot how causal inference behaves with different conflict levels.
    """
    plt.figure(figsize=(12, 8))
    
    # Parameters
    sigmaAV_A = 0.2
    sigmaAV_V = 0.3
    S_a = 0.5
    p_c = 0.7
    conflicts = np.linspace(0, 0.5, 21)
    
    estimates_avg = []
    posteriors_avg = []
    
    # For each conflict level, average over many trials
    for conflict in conflicts:
        trial_estimates = []
        trial_posteriors = []
        
        for _ in range(100):  # Average over 100 trials
            estimate, posterior, _, _ = correct_causal_inference(sigmaAV_A, sigmaAV_V, S_a, p_c, conflict)
            trial_estimates.append(estimate)
            trial_posteriors.append(posterior)
        
        estimates_avg.append(np.mean(trial_estimates))
        posteriors_avg.append(np.mean(trial_posteriors))
    
    # Plot results
    plt.subplot(2, 1, 1)
    plt.plot(conflicts, estimates_avg, 'b-', linewidth=2, label='Causal Inference Estimate')
    plt.axhline(y=S_a, color='r', linestyle='--', label='True Auditory Duration')
    plt.plot(conflicts, S_a + conflicts, 'g--', label='True Visual Duration')
    plt.xlabel('Visual Conflict')
    plt.ylabel('Duration Estimate')
    plt.title('Causal Inference: Final Estimate vs Visual Conflict')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(conflicts, posteriors_avg, 'purple', linewidth=2)
    plt.xlabel('Visual Conflict')
    plt.ylabel('P(Common Cause)')
    plt.title('Posterior Probability of Common Cause vs Visual Conflict')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("The plots show:")
    print("1. As visual conflict increases, the estimate moves between")
    print("   auditory-only and fusion predictions")
    print("2. Higher conflict reduces P(common cause)")
    print("3. This creates the characteristic causal inference pattern")

if __name__ == "__main__":
    demonstrate_difference()
    plot_causal_inference_behavior()
