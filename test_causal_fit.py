#!/usr/bin/env python3
"""
Test script for the fixed causal inference fitting
"""

import pandas as pd
import numpy as np
from fitCausalInferenceMonteCarlo import *
import time

def create_test_data():
    """Create synthetic test data for validation"""
    np.random.seed(42)
    
    # Define conditions
    snr_levels = [0.1, 1.2]
    conflicts = [0.0, 0.1, 0.2]
    delta_durs = [-0.1, -0.05, 0.0, 0.05, 0.1]
    
    data_rows = []
    
    for snr in snr_levels:
        for conflict in conflicts:
            for delta in delta_durs:
                # Standard parameters
                standard_dur = 1.0
                test_dur = standard_dur + delta
                visual_standard = standard_dur + conflict
                visual_test = test_dur + conflict
                
                # Simulate responses (higher delta_dur -> more "test longer" responses)
                n_trials = 20
                p_test_longer = 0.5 + 0.3 * np.tanh(delta / 0.05)  # Sigmoid-like
                n_chose_test = np.random.binomial(n_trials, p_test_longer)
                
                data_rows.append({
                    'audNoise': snr,
                    'conflictDur': conflict,
                    'standardDur': standard_dur,
                    'unbiasedVisualStandardDur': visual_standard,
                    'testDurS': test_dur,
                    'unbiasedVisualTestDur': visual_test,
                    'deltaDurS': delta,
                    'num_of_chose_test': n_chose_test,
                    'total_responses': n_trials
                })
    
    return pd.DataFrame(data_rows)

def test_fitting():
    """Test the causal inference fitting"""
    print("Creating synthetic test data...")
    test_data = create_test_data()
    print(f"Created {len(test_data)} conditions")
    
    print("\nTesting parameter bounds and initialization...")
    # Test with a simple set of parameters
    test_params = [0.05, 0.2, 0.15, 0.6, 0.25, 0.18, 0.4]
    
    try:
        ll = nLLMonteCarloCausal(test_params, test_data)
        print(f"Initial negative log-likelihood: {ll:.6f}")
        
        print("\nStarting fitting process (reduced iterations for testing)...")
        
        # Quick test with fewer starts and iterations
        start_time = time.time()
        fitted_params = fitCausalInferenceMonteCarlo(test_data, nStart=2)
        end_time = time.time()
        
        print(f"\nFitting completed in {end_time - start_time:.2f} seconds")
        print(f"Fitted parameters: {fitted_params}")
        
        # Validate fitted parameters
        final_ll = nLLMonteCarloCausal(fitted_params, test_data)
        print(f"Final negative log-likelihood: {final_ll:.6f}")
        print(f"Improvement: {ll - final_ll:.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error during fitting: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Fixed Causal Inference Model")
    print("=" * 50)
    
    success = test_fitting()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("\nKey fixes applied:")
        print("1. Added lambda (lapse rate) parameter to the model")
        print("2. Fixed vectorization in causalInfDecision function") 
        print("3. Applied causal inference to both standard and test intervals consistently")
        print("4. Increased Monte Carlo samples for better approximation")
        print("5. Added multiple random starts for better optimization")
        print("6. Improved error handling and parameter bounds")
    else:
        print("\n❌ Test failed!")
