#!/usr/bin/env python3
"""
Demonstration of the corrected decision noise calculation in causal inference model.

This script shows:
1. The difference between the old (incorrect) and new (correct) decision noise formulas
2. How the decision noise depends on model parameters
3. The theoretical justification for the correction
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_decision_noise_old(sigma_av_a, sigma_av_v):
    """Old (incorrect) formula."""
    return np.sqrt(sigma_av_a**2 + sigma_av_v**2) / 2

def calculate_decision_noise_new(sigma_av_a, sigma_av_v, p_common):
    """New (theoretically correct) formula."""
    # Variance under common cause (optimal fusion)
    var_fusion = 1 / (1/sigma_av_a**2 + 1/sigma_av_v**2)
    
    # Variance under separate causes (auditory only)
    var_segregated = sigma_av_a**2
    
    # Expected variance of causal inference estimate
    var_estimate = p_common * var_fusion + (1 - p_common) * var_segregated
    
    # Decision noise for difference of two independent estimates
    sigma_decision = np.sqrt(2 * var_estimate)
    
    return sigma_decision

def demo_decision_noise_comparison():
    """Compare old vs new decision noise calculations."""
    
    # Example parameters
    sigma_av_a = 0.1  # Auditory noise in AV condition
    sigma_av_v = 0.15  # Visual noise in AV condition (higher than auditory)
    
    # Range of p_common values
    p_common_values = np.linspace(0, 1, 100)
    
    # Calculate decision noise with both methods
    noise_old = [calculate_decision_noise_old(sigma_av_a, sigma_av_v) for _ in p_common_values]
    noise_new = [calculate_decision_noise_new(sigma_av_a, sigma_av_v, p_c) for p_c in p_common_values]
    
    # Theoretical limits
    # When p_common = 1 (always integrate): use fusion variance
    var_fusion = 1 / (1/sigma_av_a**2 + 1/sigma_av_v**2)
    noise_fusion = np.sqrt(2 * var_fusion)
    
    # When p_common = 0 (never integrate): use auditory variance
    noise_segregated = np.sqrt(2) * sigma_av_a
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(p_common_values, noise_old, 'r--', linewidth=2, label='Old formula (constant)')
    plt.plot(p_common_values, noise_new, 'b-', linewidth=2, label='New formula (correct)')
    plt.axhline(y=noise_fusion, color='g', linestyle=':', alpha=0.7, label='Fusion limit')
    plt.axhline(y=noise_segregated, color='orange', linestyle=':', alpha=0.7, label='Segregation limit')
    plt.xlabel('p_common')
    plt.ylabel('Decision noise')
    plt.title('Decision Noise vs Integration Tendency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show how decision noise varies with modality reliabilities
    plt.subplot(2, 2, 2)
    sigma_A_values = np.linspace(0.05, 0.3, 50)
    p_common_fixed = 0.7
    
    noise_vary_A = [calculate_decision_noise_new(sigma_A, sigma_av_v, p_common_fixed) 
                    for sigma_A in sigma_A_values]
    
    plt.plot(sigma_A_values, noise_vary_A, 'b-', linewidth=2)
    plt.xlabel('σ_auditory')
    plt.ylabel('Decision noise')
    plt.title(f'Decision Noise vs Auditory Reliability\n(p_common = {p_common_fixed})')
    plt.grid(True, alpha=0.3)
    
    # Show the effect of visual reliability
    plt.subplot(2, 2, 3)
    sigma_V_values = np.linspace(0.05, 0.3, 50)
    
    noise_vary_V = [calculate_decision_noise_new(sigma_av_a, sigma_V, p_common_fixed) 
                    for sigma_V in sigma_V_values]
    
    plt.plot(sigma_V_values, noise_vary_V, 'b-', linewidth=2)
    plt.xlabel('σ_visual')
    plt.ylabel('Decision noise')
    plt.title(f'Decision Noise vs Visual Reliability\n(p_common = {p_common_fixed})')
    plt.grid(True, alpha=0.3)
    
    # Numerical comparison for specific values
    plt.subplot(2, 2, 4)
    p_test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    old_values = [calculate_decision_noise_old(sigma_av_a, sigma_av_v) for _ in p_test_values]
    new_values = [calculate_decision_noise_new(sigma_av_a, sigma_av_v, p_c) for p_c in p_test_values]
    
    x_pos = np.arange(len(p_test_values))
    width = 0.35
    
    plt.bar(x_pos - width/2, old_values, width, label='Old formula', color='red', alpha=0.7)
    plt.bar(x_pos + width/2, new_values, width, label='New formula', color='blue', alpha=0.7)
    
    plt.xlabel('p_common')
    plt.ylabel('Decision noise')
    plt.title('Numerical Comparison')
    plt.xticks(x_pos, [f'{p:.1f}' for p in p_test_values])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print theoretical analysis
    print("=== Decision Noise Analysis ===")
    print(f"Parameters: σ_av_a = {sigma_av_a:.3f}, σ_av_v = {sigma_av_v:.3f}")
    print()
    print("Old formula (constant):")
    print(f"  σ_decision = √(σ_av_a² + σ_av_v²) / 2 = {calculate_decision_noise_old(sigma_av_a, sigma_av_v):.4f}")
    print()
    print("New formula (depends on p_common):")
    print(f"  p_common = 0.0: σ_decision = {calculate_decision_noise_new(sigma_av_a, sigma_av_v, 0.0):.4f} (segregation)")
    print(f"  p_common = 0.5: σ_decision = {calculate_decision_noise_new(sigma_av_a, sigma_av_v, 0.5):.4f} (mixed)")
    print(f"  p_common = 1.0: σ_decision = {calculate_decision_noise_new(sigma_av_a, sigma_av_v, 1.0):.4f} (fusion)")
    print()
    print("Theoretical limits:")
    print(f"  Fusion limit: √(2 / (1/σ_av_a² + 1/σ_av_v²)) = {noise_fusion:.4f}")
    print(f"  Segregation limit: √2 × σ_av_a = {noise_segregated:.4f}")
    
    print("\n=== Key Insights ===")
    print("1. The old formula gives a constant decision noise regardless of integration tendency")
    print("2. The new formula correctly varies with p_common:")
    print("   - Low p_common → high decision noise (using unreliable auditory-only estimates)")
    print("   - High p_common → low decision noise (using reliable fused estimates)")
    print("3. The new formula has proper theoretical limits at p_common = 0 and p_common = 1")

if __name__ == "__main__":
    demo_decision_noise_comparison()
