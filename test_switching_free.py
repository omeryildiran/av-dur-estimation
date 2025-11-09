#!/usr/bin/env python3
"""
Test script for the SwitchingFree model with SNR-dependent p_switch parameters.
"""

import numpy as np
import pandas as pd
from monteCarloClass import OmerMonteCarlo

def create_test_data():
    """Create a simple test dataset"""
    # Create a minimal dataset for testing
    np.random.seed(42)
    n_trials = 100
    
    # Create choice data
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
        'responses': chose_test,  # Simplified: responses same as chose_test
        'order': np.ones(n_trials)  # Assuming test always in position 1
    }
    
    return pd.DataFrame(data)

def test_switching_free_model():
    """Test the SwitchingFree model implementation"""
    print("Testing SwitchingFree model...")
    
    # Create test data
    data = create_test_data()
    
    try:
        # Initialize SwitchingFree model
        mc_model = OmerMonteCarlo(
            data=data,
            intensityVar='deltaDurS',
            sensoryVar='audNoise',
            standardVar='standardDur',
            conflictVar='conflictDur'
        )
        
        # Set model to SwitchingFree
        mc_model.modelName = "switchingFree"
        mc_model.sharedLambda = True  # Simplified configuration
        mc_model.freeP_c = False  # Not used in SwitchingFree
        
        print("✓ Model initialized successfully")
        
        # Test parameter extraction
        # For switchingFree with sharedLambda=True: [λ, σa1, σv, p_switch1, σa2, p_switch2] (6 params)
        test_params = np.array([0.1, 0.5, 0.3, 0.4, 0.8, 0.6])
        
        print(f"✓ Test parameters: {test_params}")
        print(f"  Parameter length: {len(test_params)} (expected: 6)")
        
        # Test parameter extraction for both SNR conditions
        for snr in [0.1, 1.2]:
            for conflict in [0.0, -0.2, 0.2]:
                result = mc_model.getParamsCausal(test_params, snr, conflict)
                λ, σa, σv, p_switch, t_min, t_max = result
                
                print(f"  SNR={snr}, conflict={conflict}: λ={λ:.3f}, σa={σa:.3f}, σv={σv:.3f}, p_switch={p_switch:.3f}")
                
                # Validate parameters (ignore t_min/t_max for switchingFree)
                assert 0 <= λ <= 1, f"Invalid lambda: {λ}"
                assert σa > 0, f"Invalid sigma_a: {σa}"
                assert σv > 0, f"Invalid sigma_v: {σv}"
                assert 0 <= p_switch <= 1, f"Invalid p_switch: {p_switch}"
        
        print("✓ Parameter extraction tests passed")
        
        # Test switching_free_vectorized function
        m_a = np.array([0.5, 0.6, 0.4])
        m_v = np.array([0.7, 0.5, 0.3])
        p_switch = 0.3
        
        estimates = mc_model.switching_free_vectorized(m_a, m_v, p_switch)
        print(f"✓ switching_free_vectorized test: estimates = {estimates}")
        assert len(estimates) == len(m_a), "Output length mismatch"
        
        # Test component fitting (simplified)
        print("Testing fitting components...")
        success = mc_model.test_fitting_components(mc_model.groupedData)
        if success:
            print("✓ Fitting components test passed")
        else:
            print("✗ Fitting components test failed")
            return False
            
        print("\n🎉 All SwitchingFree model tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing SwitchingFree model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_switching_free_model()
    if success:
        print("\nSwitchingFree model is ready to use!")
        print("\nExample usage:")
        print("```python")
        print("mc_model = OmerMonteCarlo(data=your_data, ...)")
        print("mc_model.modelName = 'switchingFree'")
        print("mc_model.sharedLambda = True")
        print("# Parameters: [λ, σa1, σv, p_switch1, σa2, p_switch2]")
        print("# p_switch1 is used for SNR=0.1, p_switch2 for SNR=1.2")
        print("```")
    else:
        print("\n❌ SwitchingFree model implementation has issues that need to be fixed.")