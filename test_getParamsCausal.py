#!/usr/bin/env python3
"""
Test script to identify parameter retrieval issues in getParamsCausal
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.append('/Users/omer/Library/CloudStorage/GoogleDrive-omerulfaruk97@gmail.com/My Drive/MyReposDrive/obsidian_Notes/Landy Omer Re 1/av-dur-estimation')

from monteCarloClass import OmerMonteCarlo

def test_getParamsCausal():
    """Test getParamsCausal function for parameter retrieval issues."""
    
    print("🔍 Testing getParamsCausal Function...")
    print("=" * 50)
    
    # Create dummy data
    dummy_data = pd.DataFrame({
        'deltaDurS': [0.0, 0.1],
        'audNoise': [0.1, 1.2],
        'standardDur': [0.5, 0.5],
        'conflictDur': [0.0, 0.08],
        'unbiasedVisualStandardDur': [0.5, 0.5],
        'unbiasedVisualTestDur': [0.5, 0.5],
        'testDurS': [0.5, 0.6],
        'chose_test': [1, 0],
        'chose_standard': [0, 1],
        'responses': [1, 1],
        'order': [1, 1]
    })
    
    # Initialize the model
    mc_model = OmerMonteCarlo(dummy_data)
    
    # Test different configurations
    print("\n1️⃣ Testing Missing freeP_c Attribute...")
    try:
        # This should fail because freeP_c is not defined
        test_params = [0.1, 0.2, 0.3, 0.7, 0.4, 0.05, 0.05]
        result = mc_model.getParamsCausal(test_params, 0.1, 0.0)
        print(f"   ❌ Expected error but got result: {result}")
    except AttributeError as e:
        print(f"   ✅ Caught expected AttributeError: {e}")
        print(f"   🔧 Need to define self.freeP_c in __init__")
    except Exception as e:
        print(f"   ⚠️ Unexpected error: {e}")
    
    # Fix: Add missing attributes
    print(f"\n2️⃣ Adding missing attributes...")
    mc_model.freeP_c = False  # Default to shared p_c
    print(f"   ✅ Added self.freeP_c = {mc_model.freeP_c}")
    
    # Test parameter structure for shared lambda, shared p_c
    print(f"\n3️⃣ Testing Parameter Structure (sharedLambda={mc_model.sharedLambda}, freeP_c={mc_model.freeP_c})...")
    
    # Expected structure: [lambda, sigma_a1, sigma_v1, p_c, sigma_a2, lambda_2, lambda_3]
    test_params = [0.1, 0.2, 0.3, 0.7, 0.4, 0.05, 0.05]
    
    # Test SNR conditions
    snr_values = [0.1, 1.2]
    conflict_values = [0.0, -0.08, 0.17, -0.25]
    
    for snr in snr_values:
        for conflict in conflict_values:
            try:
                lambda_, sigma_a, sigma_v, p_c = mc_model.getParamsCausal(test_params, snr, conflict)
                print(f"   ✅ SNR={snr}, conflict={conflict}: λ={lambda_:.3f}, σa={sigma_a:.3f}, σv={sigma_v:.3f}, pc={p_c:.3f}")
            except Exception as e:
                print(f"   ❌ SNR={snr}, conflict={conflict}: {e}")
    
    # Test with sharedLambda=True (removes lambda_2, lambda_3)
    print(f"\n4️⃣ Testing with sharedLambda=True...")
    mc_model.sharedLambda = True
    test_params_shared = [0.1, 0.2, 0.3, 0.7, 0.4]  # Only 5 parameters
    
    for snr in snr_values:
        for conflict in [0.0, -0.08]:  # Test fewer conflicts
            try:
                lambda_, sigma_a, sigma_v, p_c = mc_model.getParamsCausal(test_params_shared, snr, conflict)
                print(f"   ✅ SNR={snr}, conflict={conflict}: λ={lambda_:.3f}, σa={sigma_a:.3f}, σv={sigma_v:.3f}, pc={p_c:.3f}")
            except Exception as e:
                print(f"   ❌ SNR={snr}, conflict={conflict}: {e}")
    
    # Test with freeP_c=True
    print(f"\n5️⃣ Testing with freeP_c=True...")
    mc_model.freeP_c = True
    mc_model.sharedLambda = False
    test_params_free_pc = [0.1, 0.2, 0.3, 0.7, 0.4, 0.05, 0.05, 0.8]  # 8 parameters
    
    for snr in snr_values:
        try:
            lambda_, sigma_a, sigma_v, p_c = mc_model.getParamsCausal(test_params_free_pc, snr, 0.0)
            print(f"   ✅ SNR={snr}: λ={lambda_:.3f}, σa={sigma_a:.3f}, σv={sigma_v:.3f}, pc={p_c:.3f}")
        except Exception as e:
            print(f"   ❌ SNR={snr}: {e}")
    
    # Test edge cases
    print(f"\n6️⃣ Testing Edge Cases...")
    
    # Test invalid SNR
    try:
        result = mc_model.getParamsCausal(test_params_free_pc, 0.5, 0.0)  # Invalid SNR
        print(f"   ❌ Should have failed with invalid SNR")
    except ValueError as e:
        print(f"   ✅ Correctly caught invalid SNR: {e}")
    
    # Test sharedSigma_v logic
    print(f"\n7️⃣ Testing sharedSigma_v Logic...")
    mc_model.sharedSigma_v = True
    mc_model.freeP_c = False
    test_params = [0.1, 0.2, 0.3, 0.7, 0.4, 0.05, 0.05]
    
    # Both SNR conditions should use the same visual sigma when sharedSigma_v=True
    lambda1, sigma_a1, sigma_v1, p_c1 = mc_model.getParamsCausal(test_params, 0.1, 0.0)
    lambda2, sigma_a2, sigma_v2, p_c2 = mc_model.getParamsCausal(test_params, 1.2, 0.0)
    
    if sigma_v1 == sigma_v2:
        print(f"   ✅ Shared visual sigma working: σv1={sigma_v1:.3f} == σv2={sigma_v2:.3f}")
    else:
        print(f"   ❌ Shared visual sigma not working: σv1={sigma_v1:.3f} != σv2={sigma_v2:.3f}")
    
    return mc_model

def identify_parameter_indexing_issues():
    """Identify specific indexing issues in parameter retrieval."""
    
    print(f"\n🎯 Parameter Indexing Analysis")
    print("=" * 40)
    
    configurations = [
        {"sharedLambda": False, "freeP_c": False, "expected_params": 7},
        {"sharedLambda": True, "freeP_c": False, "expected_params": 5},
        {"sharedLambda": False, "freeP_c": True, "expected_params": 8},
        {"sharedLambda": True, "freeP_c": True, "expected_params": 6},
    ]
    
    for config in configurations:
        print(f"\nConfiguration: sharedLambda={config['sharedLambda']}, freeP_c={config['freeP_c']}")
        print(f"Expected parameters: {config['expected_params']}")
        
        # Create parameter layout
        param_layout = []
        
        # Base parameters
        param_layout.append("0: lambda")
        param_layout.append("1: sigma_av_a_1 (SNR=0.1)")
        param_layout.append("2: sigma_av_v_1")
        param_layout.append("3: p_c_1" if config['freeP_c'] else "3: p_c (shared)")
        param_layout.append("4: sigma_av_a_2 (SNR=1.2)")
        
        # Additional lambdas if not shared
        if not config['sharedLambda']:
            param_layout.append("5: lambda_2 (conflict -0.08, 0.17)")
            param_layout.append("6: lambda_3 (conflict -0.25, 0.08)")
            
        # Additional p_c if free
        if config['freeP_c']:
            if config['sharedLambda']:
                param_layout.append("5: p_c_2 (SNR=1.2)")
            else:
                param_layout.append("7: p_c_2 (SNR=1.2)")
        
        print("Parameter layout:")
        for i, param in enumerate(param_layout):
            print(f"  {param}")
        
        if len(param_layout) != config['expected_params']:
            print(f"❌ Mismatch: Expected {config['expected_params']}, got {len(param_layout)}")
        else:
            print(f"✅ Parameter count matches")

if __name__ == "__main__":
    model = test_getParamsCausal()
    identify_parameter_indexing_issues()
    
    print(f"\n📋 SUMMARY OF ISSUES FOUND:")
    print(f"1. ❌ Missing self.freeP_c attribute in __init__")
    print(f"2. ⚠️ sigma_av_v assignment issue when sharedSigma_v=True")
    print(f"3. ⚠️ Complex parameter indexing that changes based on configuration")
    print(f"4. ⚠️ No validation of parameter array length vs expected length")
    
    print(f"\n🔧 RECOMMENDED FIXES:")
    print(f"1. Add self.freeP_c = False in __init__")
    print(f"2. Fix sigma_av_v assignment logic")
    print(f"3. Add parameter length validation")
    print(f"4. Add clear parameter indexing documentation")
