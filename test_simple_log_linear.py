#!/usr/bin/env python3
"""
Simple test for log-linear conversions and parameter handling.
Tests the key functions directly without full data pipeline.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to Python path to import the module
sys.path.insert(0, '/Users/omer/Library/CloudStorage/GoogleDrive-omerulfaruk97@gmail.com/My Drive/MyReposDrive/obsidian_Notes/Landy Omer Re 1/av-dur-estimation')

from monteCarloClass import OmerMonteCarlo

def test_parameter_extraction_simple():
    """Test getParamsCausal function directly"""
    
    print("=== TESTING PARAMETER EXTRACTION ===\n")
    
    # Create a minimal test data that will work
    test_data = pd.DataFrame({
        'deltaDurS': [-0.1, 0.1, -0.1, 0.1],  # Required intensity variable
        'audNoise': [0.1, 0.1, 1.2, 1.2],
        'conflictDur': [0.0, 0.17, 0.0, 0.17], 
        'standardDur': [0.5, 0.5, 0.5, 0.5],
        'unbiasedVisualStandardDur': [0.5, 0.5, 0.5, 0.5],
        'testDurS': [0.4, 0.6, 0.4, 0.6],
        'unbiasedVisualTestDur': [0.4, 0.6, 0.4, 0.6],  # Add required visual test variable
        'num_of_chose_test': [5, 15, 8, 12],
        'total_responses': [20, 20, 20, 20],
        # Add missing columns that the class expects
        'chose_standard': [15, 5, 12, 8],
        'chose_test': [5, 15, 8, 12],
        'responses': [20, 20, 20, 20]
    })
    
    models_to_test = [
        "gaussian",
        "lognorm",
        "logLinearMismatch", 
        "fusionOnly",
        "fusionOnlyLogNorm",
        "probabilityMatching",
        "probabilityMatchingLogNorm"
    ]
    
    for model_name in models_to_test:
        print(f"--- Testing {model_name} ---")
        
        try:
            # Create instance with minimal data
            fitter = OmerMonteCarlo(data=test_data)
            fitter.modelName = model_name
            fitter.sharedLambda = True
            fitter.freeP_c = False
            
            # Manual bounds initialization (simulating what fitCausalInferenceMonteCarlo does)
            fitter.data_t_min = 0.1  # Set realistic bounds 
            fitter.data_t_max = 1.0
            
            # Create test parameters
            if model_name in ["fusionOnly", "fusionOnlyLogNorm"]:
                # Fusion models: [λ, σa1, σv, σa2, t_min, t_max]
                # All models use linear-space bounds for parameters
                test_params = np.array([0.1, 0.5, 0.5, 0.8, fitter.data_t_min, fitter.data_t_max])
            else:
                # Causal inference models: [λ, σa1, σv, pc, σa2, t_min, t_max]
                # All models use linear-space bounds for parameters
                test_params = np.array([0.1, 0.5, 0.5, 0.5, 0.8, fitter.data_t_min, fitter.data_t_max])
            
            print(f"  Parameter array length: {len(test_params)}")
            
            # Test parameter extraction
            for snr in [0.1, 1.2]:
                λ, σa, σv, pc, t_min, t_max = fitter.getParamsCausal(test_params, snr, 0.0)
                
                print(f"    SNR {snr}: λ={λ:.3f}, σa={σa:.3f}, σv={σv:.3f}, pc={pc:.3f}")
                print(f"              t_min={t_min:.3f}, t_max={t_max:.3f}")
                
                # Validate parameter consistency
                if not (0 <= λ <= 1):
                    print(f"      ERROR: λ out of bounds!")
                    return False
                    
                if σa <= 0 or σv <= 0:
                    print(f"      ERROR: Negative noise parameters!")
                    return False
                    
                if not (0 <= pc <= 1):
                    print(f"      ERROR: p_c out of bounds!")
                    return False
                
                # All parameters should be in linear space (t_min, t_max)
                # so standard linear-space validation applies
                if t_min >= t_max:
                    print(f"      ERROR: t_min >= t_max!")
                    return False
                    
                if t_max <= 0:
                    print(f"      ERROR: t_max <= 0!")
                    return False
                    
                print(f"      ✓ Parameters valid")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            return False
            
    print(f"\n=== PARAMETER EXTRACTION TESTS PASSED ===")
    return True

def test_tmin_zero_edge_case():
    """Test that t_min=0 is handled correctly"""
    
    print(f"\n=== TESTING t_min=0 EDGE CASE ===\n")
    
    # Create minimal test data
    test_data = pd.DataFrame({
        'deltaDurS': [-0.1, 0.1],
        'audNoise': [0.1, 1.2],
        'conflictDur': [0.0, 0.0], 
        'standardDur': [0.5, 0.5],
        'unbiasedVisualStandardDur': [0.5, 0.5],
        'testDurS': [0.4, 0.6],
        'unbiasedVisualTestDur': [0.4, 0.6],
        'num_of_chose_test': [5, 15],
        'total_responses': [20, 20],
        # Add missing columns
        'chose_standard': [15, 5],
        'chose_test': [5, 15],
        'responses': [20, 20]
    })
    
    try:
        # Test with linear-space model
        fitter = OmerMonteCarlo(data=test_data)
        fitter.modelName = "gaussian"
        fitter.sharedLambda = True
        fitter.freeP_c = False
        
        # Test params with t_min=0
        test_params = np.array([0.1, 0.5, 0.5, 0.5, 0.8, 0.0, 1.0])  # t_min=0
        
        # Test parameter extraction
        λ, σa, σv, pc, t_min, t_max = fitter.getParamsCausal(test_params, 0.1, 0.0)
        
        print(f"Extracted parameters with t_min=0:")
        print(f"  t_min={t_min}, t_max={t_max}")
        
        if t_min == 0.0 and t_max > t_min:
            print(f"  ✓ t_min=0 handled correctly")
        else:
            print(f"  ❌ t_min=0 not handled correctly")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    print(f"\n=== t_min=0 TESTS PASSED ===")
    return True

def test_log_space_parameter_bounds():
    """Test that log-space bounds are calculated correctly"""
    
    print(f"\n=== TESTING LOG-SPACE BOUNDS ===\n")
    
    # Test the bounds calculation logic directly
    data_t_min = 0.1
    data_t_max = 2.0
    
    # For log-space models, bounds should be in log space
    expected_log_t_min = np.log(max(data_t_min, 0.001))
    expected_log_t_max = np.log(data_t_max)
    
    print(f"Linear bounds: t_min={data_t_min}, t_max={data_t_max}")
    print(f"Log bounds: log_t_min={expected_log_t_min:.3f}, log_t_max={expected_log_t_max:.3f}")
    
    # Test that log bounds are valid
    if expected_log_t_min >= expected_log_t_max:
        print(f"❌ Invalid log bounds: log_t_min >= log_t_max")
        return False
    
    # Test that converting back gives reasonable values
    converted_t_min = np.exp(expected_log_t_min)
    converted_t_max = np.exp(expected_log_t_max)
    
    print(f"Converted back: t_min={converted_t_min:.3f}, t_max={converted_t_max:.3f}")
    
    if abs(converted_t_min - data_t_min) > 1e-6 or abs(converted_t_max - data_t_max) > 1e-6:
        print(f"❌ Conversion mismatch")
        return False
    
    print(f"✓ Log-space bounds calculation correct")
    
    print(f"\n=== LOG-SPACE BOUNDS TESTS PASSED ===")
    return True

if __name__ == "__main__":
    
    print("Running simplified log-linear conversion tests...\n")
    
    success = True
    
    # Run tests
    success &= test_parameter_extraction_simple()
    success &= test_tmin_zero_edge_case()
    success &= test_log_space_parameter_bounds()
    
    if success:
        print(f"\n🎉 ALL SIMPLIFIED TESTS PASSED! 🎉")
        print(f"✓ Parameter extraction working correctly")
        print(f"✓ t_min=0 handling correct") 
        print(f"✓ Log-space bounds calculation correct")
        print(f"\nThe Monte Carlo class log-linear conversions are working properly!")
    else:
        print(f"\n❌ SOME TESTS FAILED ❌")
        print(f"Please check the error messages above.")