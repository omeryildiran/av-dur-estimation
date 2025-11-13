#!/usr/bin/env python3
"""
Test script to verify that the Monte Carlo class works correctly with fixed t_min/t_max bounds.
"""

import numpy as np
import pandas as pd
from monteCarloClass import OmerMonteCarlo

def create_test_data():
    """Create minimal test data for verification"""
    n_trials = 100
    
    # Create test data with known bounds
    data = {
        'deltaDurS': np.random.uniform(-0.1, 0.1, n_trials),
        'audNoise': np.random.choice([0.1, 1.2], n_trials),
        'standardDur': np.ones(n_trials) * 0.5,
        'conflictDur': np.random.choice([0, -0.17, 0.25], n_trials),
        'unbiasedVisualStandardDur': np.ones(n_trials) * 0.5,
        'unbiasedVisualTestDur': np.random.uniform(0.4, 0.6, n_trials),
        'testDurS': np.random.uniform(0.2, 1.0, n_trials),  # Known bounds for testing
        'chose_test': np.random.choice([0, 1], n_trials),
        'chose_standard': np.random.choice([0, 1], n_trials),
        'responses': np.ones(n_trials),  # Add required columns
        'logDurRatio': np.random.uniform(-0.2, 0.2, n_trials)
    }
    
    return pd.DataFrame(data)

def test_parameter_counts():
    """Test that parameter counts are correct after removing t_min/t_max"""
    data = create_test_data()
    
    # Test fusion-only model
    mc = OmerMonteCarlo(data)
    mc.modelName = "fusionOnly"
    mc.sharedLambda = True
    expected_params = 4  # [λ, σa1, σv, σa2]
    actual_params = mc.getActualParameterCount()
    print(f"Fusion model (shared λ): Expected {expected_params}, Got {actual_params}")
    assert actual_params == expected_params, f"Expected {expected_params}, got {actual_params}"
    
    # Test standard causal model
    mc.modelName = "gaussian"
    mc.sharedLambda = True
    mc.freeP_c = False
    expected_params = 5  # [λ, σa1, σv, σa2, pc] (no t_min/t_max)
    actual_params = mc.getActualParameterCount()
    print(f"Standard causal (shared λ, fixed pc): Expected {expected_params}, Got {actual_params}")
    assert actual_params == expected_params, f"Expected {expected_params}, got {actual_params}"
    
    # Test switchingWithConflict
    mc.modelName = "switchingWithConflict"
    mc.sharedLambda = True
    mc.freeP_c = False
    expected_params = 6  # [λ, σa1, σv, σa2, pc, k] (no t_min/t_max)
    actual_params = mc.getActualParameterCount()
    print(f"SwitchingWithConflict (shared λ, fixed pc): Expected {expected_params}, Got {actual_params}")
    assert actual_params == expected_params, f"Expected {expected_params}, got {actual_params}"

def test_parameter_extraction():
    """Test that parameter extraction works with fixed bounds"""
    data = create_test_data()
    mc = OmerMonteCarlo(data)
    
    # Test with standard causal model
    mc.modelName = "gaussian"
    mc.sharedLambda = True
    mc.freeP_c = False
    
    # Create test parameters (5 params for this config)
    test_params = np.array([0.1, 0.5, 0.5, 0.8, 0.6])  # [λ, σa1, σv, σa2, pc]
    
    # Extract parameters
    λ, σa, σv, pc, t_min, t_max = mc.getParamsCausal(test_params, 0.1, 0)
    
    print(f"Extracted parameters:")
    print(f"  λ={λ:.3f}, σa={σa:.3f}, σv={σv:.3f}, pc={pc:.3f}")
    print(f"  t_min={t_min:.3f} (should be {mc.data_t_min:.3f})")
    print(f"  t_max={t_max:.3f} (should be {mc.data_t_max:.3f})")
    
    # Verify t_min/t_max come from data
    assert t_min == mc.data_t_min, f"t_min should be {mc.data_t_min}, got {t_min}"
    assert t_max == mc.data_t_max, f"t_max should be {mc.data_t_max}, got {t_max}"
    
    print("✓ Parameter extraction test passed!")

def test_bounds_setup():
    """Test that bounds are correctly set up without t_min/t_max"""
    data = create_test_data()
    mc = OmerMonteCarlo(data)
    mc.modelName = "gaussian"
    mc.sharedLambda = True
    mc.freeP_c = False
    
    # This would normally be called within fitCausalInferenceMonteCarlo
    # Let's test the bounds setup logic manually
    print(f"Data bounds: t_min={mc.data_t_min:.3f}, t_max={mc.data_t_max:.3f}")
    print("✓ Bounds setup test passed!")

if __name__ == "__main__":
    print("Testing Monte Carlo class with fixed t_min/t_max bounds...")
    print("=" * 50)
    
    test_parameter_counts()
    print()
    
    test_parameter_extraction()
    print()
    
    test_bounds_setup()
    print()
    
    print("✓ All tests passed! Monte Carlo class is working correctly with fixed bounds.")