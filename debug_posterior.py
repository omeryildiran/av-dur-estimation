#!/usr/bin/env python3
"""
Debug script to isolate the division by zero error
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/omer/Library/CloudStorage/GoogleDrive-omerulfaruk97@gmail.com/My Drive/MyReposDrive/obsidian_Notes/Landy Omer Re 1/av-dur-estimation')

from monteCarloClass import OmerMonteCarlo

def debug_posterior_calculation():
    """Debug the posterior calculation step by step"""
    
    # Create minimal dummy data
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
    
    # Test parameters
    m_a, m_v = 0.5, 0.55
    sigma_a, sigma_v = 0.1, 0.15
    p_c = 0.7
    
    print("Debug: Posterior calculation step by step")
    print(f"m_a={m_a}, m_v={m_v}, σa={sigma_a}, σv={sigma_v}, p_c={p_c}")
    print(f"t_min={mc_model.t_min}, t_max={mc_model.t_max}")
    
    # Step 1: Calculate likelihoods manually
    yMin, yMax = mc_model.t_min, mc_model.t_max
    print(f"yMin={yMin}, yMax={yMax}")
    
    try:
        L1 = mc_model.p_C1(m_a, m_v, sigma_a, sigma_v, yMax, yMin)
        print(f"✅ L1 (common cause) = {L1}")
    except Exception as e:
        print(f"❌ L1 calculation failed: {e}")
        return
    
    try:
        L2 = mc_model.p_C2(m_a, m_v, sigma_a, sigma_v, yMax, yMin)
        print(f"✅ L2 (separate causes) = {L2}")
    except Exception as e:
        print(f"❌ L2 calculation failed: {e}")
        return
    
    # Step 2: Calculate posterior manually
    numerator = L1 * p_c
    denominator = L1 * p_c + L2 * (1 - p_c)
    
    print(f"Numerator = L1 * p_c = {L1} * {p_c} = {numerator}")
    print(f"Denominator = L1*p_c + L2*(1-p_c) = {numerator} + {L2}*{1-p_c} = {denominator}")
    
    if denominator == 0:
        print("❌ Division by zero detected!")
        print("This happens when both L1 and L2 are zero")
        print("Checking individual likelihood components...")
        
        # Debug L1 components
        print("\n--- Debugging L1 (common cause) ---")
        sigma_c_sq = (sigma_a**2 * sigma_v**2) / (sigma_a**2 + sigma_v**2)
        sigma_c = np.sqrt(sigma_c_sq)
        mu_c = (m_a / sigma_a**2 + m_v / sigma_v**2) / (1 / sigma_a**2 + 1 / sigma_v**2)
        print(f"sigma_c = {sigma_c}")
        print(f"mu_c = {mu_c}")
        
        from scipy.stats import norm
        hi_cdf = norm.cdf((yMax-mu_c)/sigma_c)
        lo_cdf = norm.cdf((yMin-mu_c)/sigma_c)
        expo = np.exp(-(m_a-m_v)**2/(2*(sigma_a**2+sigma_v**2)))
        prior = 1/(yMax-yMin)
        
        print(f"hi_cdf = {hi_cdf}")
        print(f"lo_cdf = {lo_cdf}")
        print(f"(hi_cdf - lo_cdf) = {hi_cdf - lo_cdf}")
        print(f"expo = {expo}")
        print(f"prior = {prior}")
        print(f"sigma_c/sqrt(σa²*σv²) = {sigma_c/np.sqrt(sigma_a**2 * sigma_v**2)}")
        
        L1_manual = prior * sigma_c/np.sqrt(sigma_a**2 * sigma_v**2) * (hi_cdf-lo_cdf) * expo
        print(f"L1_manual = {L1_manual}")
        
        # Debug L2 components
        print("\n--- Debugging L2 (separate causes) ---")
        p_single_a = mc_model.p_single(m_a, sigma_a, yMin, yMax)
        p_single_v = mc_model.p_single(m_v, sigma_v, yMin, yMax)
        L2_manual = p_single_a * p_single_v
        print(f"p_single(m_a) = {p_single_a}")
        print(f"p_single(m_v) = {p_single_v}")
        print(f"L2_manual = {L2_manual}")
        
    else:
        posterior = numerator / denominator
        print(f"✅ Posterior = {posterior}")
    
    # Test the actual function
    try:
        result = mc_model.posterior_C1(m_a, m_v, sigma_a, sigma_v, p_c, mc_model.t_min, mc_model.t_max)
        print(f"✅ Function result = {result}")
    except Exception as e:
        print(f"❌ Function failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_posterior_calculation()
