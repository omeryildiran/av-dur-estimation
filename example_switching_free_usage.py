#!/usr/bin/env python3
"""
Example usage of the SwitchingFree model with SNR-dependent p_switch parameters.
"""

import numpy as np
import pandas as pd
from monteCarloClass import OmerMonteCarlo

def example_usage():
    """Demonstrate how to use the SwitchingFree model"""
    
    print("=== SwitchingFree Model Usage Example ===\n")
    
    # Create some example data (you would use your real data here)
    np.random.seed(123)
    n_trials = 200
    chose_test = np.random.choice([0, 1], n_trials)
    
    example_data = {
        'deltaDurS': np.random.uniform(-0.3, 0.3, n_trials),
        'audNoise': np.random.choice([0.1, 1.2], n_trials),  # SNR conditions
        'standardDur': np.full(n_trials, 0.5),
        'conflictDur': np.random.choice([-0.2, -0.1, 0, 0.1, 0.2], n_trials),
        'unbiasedVisualStandardDur': np.full(n_trials, 0.5) + np.random.choice([-0.2, 0, 0.2], n_trials),
        'unbiasedVisualTestDur': np.full(n_trials, 0.5) + np.random.uniform(-0.3, 0.3, n_trials),
        'testDurS': np.full(n_trials, 0.5) + np.random.uniform(-0.3, 0.3, n_trials),
        'chose_test': chose_test,
        'chose_standard': 1 - chose_test,
        'responses': chose_test,
        'order': np.ones(n_trials)
    }
    
    data = pd.DataFrame(example_data)
    
    print("1. Initialize the SwitchingFree model:")
    print("   mc_model = OmerMonteCarlo(data=data, ...)")
    print("   mc_model.modelName = 'switchingFree'")
    
    # Initialize the model
    mc_model = OmerMonteCarlo(
        data=data,
        intensityVar='deltaDurS',
        sensoryVar='audNoise',
        standardVar='standardDur',
        conflictVar='conflictDur'
    )
    
    mc_model.modelName = "switchingFree"
    mc_model.sharedLambda = True  # Use shared lambda for simplicity
    
    print("   ✓ Model initialized")
    
    print("\n2. Parameter structure for SwitchingFree model:")
    print("   With sharedLambda=True: [λ, σa1, σv, p_switch1, σa2, p_switch2] (6 parameters)")
    print("   - p_switch1: switching probability for SNR=0.1 (high SNR, less noisy)")
    print("   - p_switch2: switching probability for SNR=1.2 (low SNR, more noisy)")
    print("   - Values range from 0 (always auditory) to 1 (always visual)")
    print("   - No t_min/t_max needed since model doesn't use causal inference bounds")
    
    print("\n3. Example parameter values:")
    example_params = np.array([
        0.05,   # λ (lapse rate)
        0.3,    # σa1 (auditory noise for SNR=0.1)
        0.2,    # σv (visual noise)
        0.3,    # p_switch1 (for SNR=0.1, prefer auditory since it's more reliable)
        0.8,    # σa2 (auditory noise for SNR=1.2)
        0.7,    # p_switch2 (for SNR=1.2, prefer visual since auditory is very noisy)
    ])
    
    print(f"   Parameters: {example_params}")
    
    print("\n4. Parameter interpretation for each SNR condition:")
    for snr in [0.1, 1.2]:
        for conflict in [0.0, -0.1, 0.1]:
            λ, σa, σv, p_switch, t_min, t_max = mc_model.getParamsCausal(example_params, snr, conflict)
            print(f"   SNR={snr}, conflict={conflict:.1f}: λ={λ:.3f}, σa={σa:.3f}, σv={σv:.3f}, p_switch={p_switch:.3f}")
            
            if snr == 0.1:
                noise_desc = "low auditory noise"
                switch_desc = "prefer auditory" if p_switch < 0.5 else "prefer visual"
            else:
                noise_desc = "high auditory noise"  
                switch_desc = "prefer auditory" if p_switch < 0.5 else "prefer visual"
                
            print(f"      → {noise_desc}, {switch_desc} (p_visual={p_switch:.1f})")
    
    print("\n5. Calculate likelihood for these parameters:")
    try:
        likelihood = mc_model.nLLMonteCarloCausal(example_params, mc_model.groupedData)
        print(f"   Negative log-likelihood: {likelihood:.2f}")
        if likelihood < 1e10:
            print("   ✓ Likelihood calculation successful")
        else:
            print("   ✗ Likelihood calculation failed (returned error value)")
    except Exception as e:
        print(f"   ✗ Error in likelihood calculation: {e}")
    
    print("\n6. Model behavior explanation:")
    print("   The SwitchingFree model switches between auditory and visual estimates")
    print("   based on free switching probabilities that can differ by SNR condition:")
    print("   - When p_switch is low (< 0.5): tends to use auditory estimates")
    print("   - When p_switch is high (> 0.5): tends to use visual estimates") 
    print("   - This allows modeling condition-dependent modality preferences")
    print("   - Unlike reliability-based switching, these probabilities are free parameters")
    
    print("\n7. Key differences from other switching models:")
    print("   - switching: uses reliability-based switching (σa²/(σa²+σv²))")
    print("   - switchingWithConflict: modulates switching by conflict level")  
    print("   - switchingFree: uses free p_switch parameters independent of noise levels")
    print("   - This model has separate p_switch for each SNR condition")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    example_usage()