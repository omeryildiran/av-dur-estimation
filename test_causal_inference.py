#!/usr/bin/env python3
"""
Test script to check for errors in causal inference model calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
import os

# Add the current directory to Python path
sys.path.append('/Users/omer/Library/CloudStorage/GoogleDrive-omerulfaruk97@gmail.com/My Drive/MyReposDrive/obsidian_Notes/Landy Omer Re 1/av-dur-estimation')

# Import the class
from monteCarloClass import OmerMonteCarlo

def test_causal_inference_functions():
    """Test the causal inference functions for mathematical correctness"""
    
    print("🔍 Testing Causal Inference Model Functions...")
    print("=" * 60)
    
    # Create dummy data for testing
    dummy_data = pd.DataFrame({
        'deltaDurS': np.tile(np.linspace(-0.3, 0.3, 10), 12),
        'audNoise': np.repeat([0.1, 1.2], 60),
        'standardDur': np.full(120, 0.5),
        'conflictDur': np.tile(np.repeat([-0.25, -0.17, -0.08, 0, 0.08, 0.17], 10), 2),
        'unbiasedVisualStandardDur': np.full(120, 0.5),
        'unbiasedVisualTestDur': np.full(120, 0.5),
        'testDurS': np.full(120, 0.5),
        'chose_test': np.random.binomial(1, 0.5, 120),
        'chose_standard': np.random.binomial(1, 0.5, 120),
        'responses': np.full(120, 1),
        'order': np.full(120, 1)
    })
    
    # Initialize the model
    mc_model = OmerMonteCarlo(dummy_data)
    mc_model.nSimul = 100  # Small for testing
    mc_model.modelName = "gaussian"
    
    # Test parameters
    test_params = np.array([0.1, 0.3, 0.4, 0.7, 0.5, 0.6])  # lambda, σa1, σv1, pc, σa2, σv2
    
    print("Test Parameters:")
    print(f"  λ={test_params[0]}, σa1={test_params[1]}, σv1={test_params[2]}")
    print(f"  pc={test_params[3]}, σa2={test_params[4]}, σv2={test_params[5]}")
    print()
    
    # Test 1: Parameter extraction
    print("1️⃣ Testing Parameter Extraction...")
    try:
        lambda_, sigma_a, sigma_v, p_c = mc_model.getParamsCausal(test_params, 0.1, 0.0)
        print(f"   ✅ SNR=0.1, conflict=0.0: λ={lambda_:.3f}, σa={sigma_a:.3f}, σv={sigma_v:.3f}, pc={p_c:.3f}")
        
        lambda_, sigma_a, sigma_v, p_c = mc_model.getParamsCausal(test_params, 1.2, 0.0)
        print(f"   ✅ SNR=1.2, conflict=0.0: λ={lambda_:.3f}, σa={sigma_a:.3f}, σv={sigma_v:.3f}, pc={p_c:.3f}")
    except Exception as e:
        print(f"   ❌ Parameter extraction failed: {e}")
        return False
    
    # Test 2: Fusion function
    print("\n2️⃣ Testing Fusion Function...")
    try:
        m_a, m_v = 0.5, 0.6
        sigma_a, sigma_v = 0.1, 0.2
        
        fused = mc_model.fusionAV_vectorized(m_a, m_v, sigma_a, sigma_v)
        
        # Manual calculation for verification
        J_a = 1 / sigma_a**2
        J_v = 1 / sigma_v**2
        w_a = J_a / (J_a + J_v)
        expected_fusion = w_a * m_a + (1 - w_a) * m_v
        
        if np.isclose(fused, expected_fusion):
            print(f"   ✅ Fusion correct: {fused:.4f} (expected: {expected_fusion:.4f})")
        else:
            print(f"   ❌ Fusion error: {fused:.4f} vs expected: {expected_fusion:.4f}")
            return False
    except Exception as e:
        print(f"   ❌ Fusion function failed: {e}")
        return False
    
    # Test 3: Single source likelihood
    print("\n3️⃣ Testing Single Source Likelihood...")
    try:
        m, sigma = 0.5, 0.1
        y_min, y_max = 0.1, 1.0
        
        p_single = mc_model.p_single(m, sigma, y_min, y_max)
        
        # Manual calculation
        hi_cdf = norm.cdf((y_max - m) / sigma)
        lo_cdf = norm.cdf((y_min - m) / sigma)
        expected_p_single = (hi_cdf - lo_cdf) / (y_max - y_min)
        
        if np.isclose(p_single, expected_p_single):
            print(f"   ✅ Single source likelihood correct: {p_single:.6f}")
        else:
            print(f"   ❌ Single source likelihood error: {p_single:.6f} vs {expected_p_single:.6f}")
            return False
    except Exception as e:
        print(f"   ❌ Single source likelihood failed: {e}")
        return False
    
    # Test 4: Common vs separate cause likelihoods
    print("\n4️⃣ Testing Common vs Separate Cause Likelihoods...")
    try:
        m_a, m_v = 0.5, 0.55
        sigma_a, sigma_v = 0.1, 0.15
        y_min, y_max = 0.1, 1.0
        
        L_C1 = mc_model.p_C1(m_a, m_v, sigma_a, sigma_v, y_max, y_min)
        L_C2 = mc_model.p_C2(m_a, m_v, sigma_a, sigma_v, y_max, y_min)
        
        print(f"   📊 L(C=1): {L_C1:.6f}")
        print(f"   📊 L(C=2): {L_C2:.6f}")
        
        # Check that likelihoods are positive
        if L_C1 > 0 and L_C2 > 0:
            print("   ✅ Both likelihoods are positive")
        else:
            print(f"   ❌ Invalid likelihoods: L_C1={L_C1}, L_C2={L_C2}")
            return False
            
        # Test posterior calculation
        p_c = 0.7
        post_C1 = mc_model.posterior_C1(m_a, m_v, sigma_a, sigma_v, p_c, mc_model.t_min, mc_model.t_max)
        
        # Manual posterior calculation
        expected_post = (L_C1 * p_c) / (L_C1 * p_c + L_C2 * (1 - p_c))
        
        if np.isclose(post_C1, expected_post):
            print(f"   ✅ Posterior correct: {post_C1:.4f}")
        else:
            print(f"   ❌ Posterior error: {post_C1:.4f} vs expected: {expected_post:.4f}")
            return False
            
    except Exception as e:
        print(f"   ❌ Likelihood calculation failed: {e}")
        return False
    
    # Test 5: Causal inference estimate
    print("\n5️⃣ Testing Causal Inference Estimate...")
    try:
        m_a, m_v = 0.5, 0.55
        sigma_a, sigma_v = 0.1, 0.15
        p_c = 0.7
        
        estimate = mc_model.causalInference_vectorized(m_a, m_v, sigma_a, sigma_v, p_c)
        
        # Manual calculation
        fused = mc_model.fusionAV_vectorized(m_a, m_v, sigma_a, sigma_v)
        post_C1 = mc_model.posterior_C1(m_a, m_v, sigma_a, sigma_v, p_c, mc_model.t_min, mc_model.t_max)
        expected_estimate = post_C1 * fused + (1 - post_C1) * m_a
        
        if np.isclose(estimate, expected_estimate):
            print(f"   ✅ Causal inference estimate correct: {estimate:.4f}")
        else:
            print(f"   ❌ Causal inference estimate error: {estimate:.4f} vs {expected_estimate:.4f}")
            return False
            
    except Exception as e:
        print(f"   ❌ Causal inference failed: {e}")
        return False
    
    # Test 6: Probability calculation
    print("\n6️⃣ Testing Probability Test Longer...")
    try:
        trueStims = (0.5, 0.6, 0.52, 0.62)  # S_a_s, S_a_t, S_v_s, S_v_t
        sigma_a, sigma_v = 0.1, 0.15
        p_c = 0.7
        lambda_ = 0.05
        
        p_test_longer = mc_model.probTestLonger_vectorized_mc(trueStims, sigma_a, sigma_v, p_c, lambda_)
        
        # Check that probability is in valid range
        if 0 <= p_test_longer <= 1:
            print(f"   ✅ P(test longer) = {p_test_longer:.4f} (valid range)")
        else:
            print(f"   ❌ Invalid probability: {p_test_longer}")
            return False
            
    except Exception as e:
        print(f"   ❌ Probability calculation failed: {e}")
        return False
    
    # Test 7: Check for different model types
    print("\n7️⃣ Testing Different Model Types...")
    for model_name in ["gaussian", "lognorm", "logLinearMismatch"]:
        try:
            mc_model.modelName = model_name
            p_test = mc_model.probTestLonger_vectorized_mc(trueStims, sigma_a, sigma_v, p_c, lambda_)
            
            if 0 <= p_test <= 1:
                print(f"   ✅ {model_name}: P(test longer) = {p_test:.4f}")
            else:
                print(f"   ❌ {model_name}: Invalid probability {p_test}")
                return False
                
        except Exception as e:
            print(f"   ❌ {model_name} failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Causal inference functions appear correct.")
    return True

def test_edge_cases():
    """Test edge cases that might cause numerical issues"""
    
    print("\n🔬 Testing Edge Cases...")
    print("=" * 60)
    
    # Create dummy data
    dummy_data = pd.DataFrame({
        'deltaDurS': [0.0],
        'audNoise': [0.1],
        'standardDur': [0.5],
        'conflictDur': [0.0],
        'unbiasedVisualStandardDur': [0.5],
        'unbiasedVisualTestDur': [0.5],
        'testDurS': [0.5],
        'chose_test': [1],
        'chose_standard': [0],
        'responses': [1],
        'order': [1]
    })
    
    mc_model = OmerMonteCarlo(dummy_data)
    mc_model.nSimul = 100
    mc_model.modelName = "gaussian"
    
    # Test 1: Very small noise values
    print("1️⃣ Testing very small noise values...")
    try:
        small_sigma = 1e-6
        result = mc_model.fusionAV_vectorized(0.5, 0.5, small_sigma, 0.1)
        print(f"   ✅ Small σa={small_sigma}: fusion = {result:.6f}")
    except Exception as e:
        print(f"   ⚠️ Small noise warning: {e}")
    
    # Test 2: Very large noise values
    print("\n2️⃣ Testing very large noise values...")
    try:
        large_sigma = 10.0
        result = mc_model.fusionAV_vectorized(0.5, 0.5, large_sigma, 0.1)
        print(f"   ✅ Large σa={large_sigma}: fusion = {result:.6f}")
    except Exception as e:
        print(f"   ⚠️ Large noise warning: {e}")
    
    # Test 3: Extreme p_c values
    print("\n3️⃣ Testing extreme p_c values...")
    for p_c in [0.001, 0.999]:
        try:
            post = mc_model.posterior_C1(0.5, 0.5, 0.1, 0.1, p_c, 0.1, 1.0)
            print(f"   ✅ p_c={p_c}: posterior = {post:.6f}")
        except Exception as e:
            print(f"   ⚠️ Extreme p_c={p_c} warning: {e}")
    
    print("\n✅ Edge case testing completed.")

if __name__ == "__main__":
    success = test_causal_inference_functions()
    test_edge_cases()
    
    if success:
        print("\n🎯 CONCLUSION: No mathematical errors detected in causal inference functions!")
    else:
        print("\n❌ CONCLUSION: Mathematical errors found - see details above.")
