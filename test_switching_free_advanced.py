#!/usr/bin/env python3
"""
Additional test for SwitchingFree model with non-shared lambda configuration.
"""

import numpy as np
import pandas as pd
from monteCarloClass import OmerMonteCarlo

def create_test_data():
    """Create a simple test dataset"""
    np.random.seed(42)
    n_trials = 100
    
    chose_test = np.random.choice([0, 1], n_trials)
    
    data = {
        'deltaDurS': np.random.uniform(-0.2, 0.2, n_trials),
        'audNoise': np.random.choice([0.1, 1.2], n_trials),
        'standardDur': np.full(n_trials, 0.5),
        'conflictDur': np.random.choice([-0.2, 0, 0.2], n_trials),
        'unbiasedVisualStandardDur': np.full(n_trials, 0.5) + np.random.choice([-0.2, 0, 0.2], n_trials),
        'unbiasedVisualTestDur': np.full(n_trials, 0.5) + np.random.uniform(-0.2, 0.2, n_trials),
        'testDurS': np.full(n_trials, 0.5) + np.random.uniform(-0.2, 0.2, n_trials),
        'chose_test': chose_test,
        'chose_standard': 1 - chose_test,
        'responses': chose_test,
        'order': np.ones(n_trials)
    }
    
    return pd.DataFrame(data)

def test_switching_free_non_shared_lambda():
    """Test the SwitchingFree model with non-shared lambda"""
    print("Testing SwitchingFree model with non-shared lambda...")
    
    data = create_test_data()
    
    try:
        mc_model = OmerMonteCarlo(
            data=data,
            intensityVar='deltaDurS',
            sensoryVar='audNoise',
            standardVar='standardDur',
            conflictVar='conflictDur'
        )
        
        # Configure for non-shared lambda
        mc_model.modelName = "switchingFree"
        mc_model.sharedLambda = False
        mc_model.freeP_c = False
        
        print("✓ Model initialized with sharedLambda=False")
        
        # For switchingFree with sharedLambda=False: [λ, σa1, σv, p_switch1, σa2, λ2, λ3, p_switch2] (8 params)
        test_params = np.array([0.1, 0.5, 0.3, 0.4, 0.8, 0.15, 0.12, 0.6])
        
        print(f"✓ Test parameters: {test_params}")
        print(f"  Parameter length: {len(test_params)} (expected: 8)")
        
        # Test parameter extraction for both SNR conditions and different conflicts
        for snr in [0.1, 1.2]:
            for conflict in [0.0, -0.17, 0.25, -0.08, 0.17, -0.25, 0.08]:  # Different lambda groups
                result = mc_model.getParamsCausal(test_params, snr, conflict)
                λ, σa, σv, p_switch, t_min, t_max = result
                
                print(f"  SNR={snr}, conflict={conflict}: λ={λ:.3f}, σa={σa:.3f}, σv={σv:.3f}, p_switch={p_switch:.3f}")
                
                # Validate parameters (ignore t_min/t_max for switchingFree)
                assert 0 <= λ <= 1, f"Invalid lambda: {λ}"
                assert σa > 0, f"Invalid sigma_a: {σa}"
                assert σv > 0, f"Invalid sigma_v: {σv}"
                assert 0 <= p_switch <= 1, f"Invalid p_switch: {p_switch}"
        
        print("✓ Parameter extraction tests passed for non-shared lambda")
        
        # Test likelihood evaluation
        try:
            ll = mc_model.nLLMonteCarloCausal(test_params, mc_model.groupedData)
            print(f"✓ Test likelihood (non-shared λ): {ll}")
            if ll >= 1e10:
                print("✗ Likelihood evaluation returned error value")
                return False
        except Exception as e:
            print(f"✗ Error in likelihood evaluation: {e}")
            return False
        
        print("\n🎉 SwitchingFree with non-shared lambda tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_switching_free_non_shared_lambda()
    if success:
        print("\nSwitchingFree model (all configurations) is ready to use!")
        print("Parameter configurations:")
        print("1. sharedLambda=True:  [λ, σa1, σv, p_switch1, σa2, p_switch2] (6 params)")
        print("2. sharedLambda=False: [λ, σa1, σv, p_switch1, σa2, λ2, λ3, p_switch2] (8 params)")
        print("\nWhere:")
        print("- p_switch1 is used for SNR=0.1 (high SNR)")
        print("- p_switch2 is used for SNR=1.2 (low SNR)")
        print("- Different lambda values (λ, λ2, λ3) are used for different conflict groups when sharedLambda=False")
    else:
        print("\n❌ Issues found in SwitchingFree implementation.")