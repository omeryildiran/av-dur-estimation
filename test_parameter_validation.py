#!/usr/bin/env python3
"""
Test script to validate the fixed getParamsCausal function
"""

import numpy as np
import pandas as pd
from monteCarloClass import OmerMonteCarlo

# Create dummy data for testing
dummy_data = pd.DataFrame({
    'deltaDurS': [0.1, 0.2, 0.3, -0.1, -0.2, 0.0],
    'audNoise': [0.1, 0.1, 1.2, 0.1, 1.2, 0.1],
    'standardDur': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'conflictDur': [0.0, -0.08, 0.17, -0.25, 0.08, 0.0],
    'longer': [0, 1, 1, 0, 1, 0],
    'unbiasedVisualStandardDur': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'unbiasedVisualTestDur': [1.1, 1.2, 1.3, 0.9, 0.8, 1.0],
    'testDurS': [1.1, 1.2, 1.3, 0.9, 0.8, 1.0]
})

print("🧪 Testing Fixed getParamsCausal Function")
print("=" * 50)

# Test different configurations
configs = [
    {"sharedLambda": False, "freeP_c": False, "expected_len": 7},
    {"sharedLambda": True, "freeP_c": False, "expected_len": 5},
    {"sharedLambda": False, "freeP_c": True, "expected_len": 8},
    {"sharedLambda": True, "freeP_c": True, "expected_len": 6}
]

for i, config in enumerate(configs, 1):
    print(f"\n{i}️⃣ Testing Configuration: sharedLambda={config['sharedLambda']}, freeP_c={config['freeP_c']}")
    
    # Create model instance
    model = OmerMonteCarlo(dummy_data)
    model.sharedLambda = config["sharedLambda"]
    model.freeP_c = config["freeP_c"]
    
    # Create parameter array of correct length
    params = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.05, 0.05, 0.8])[:config["expected_len"]]
    
    print(f"   📏 Parameter array length: {len(params)} (expected: {config['expected_len']})")
    
    # Test parameter extraction
    test_conditions = [
        (0.1, 0.0),
        (0.1, -0.08),
        (1.2, 0.0),
        (1.2, 0.17)
    ]
    
    try:
        for snr, conflict in test_conditions:
            lambda_, sigma_a, sigma_v, p_c = model.getParamsCausal(params, snr, conflict)
            print(f"   ✅ SNR={snr}, conflict={conflict}: λ={lambda_:.3f}, σa={sigma_a:.3f}, σv={sigma_v:.3f}, pc={p_c:.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n🎯 Testing Parameter Length Validation")
print("-" * 40)

# Test invalid parameter lengths
model = OmerMonteCarlo(dummy_data)
model.sharedLambda = False
model.freeP_c = False

try:
    # Try with wrong length
    wrong_params = np.array([0.1, 0.2, 0.3, 0.7, 0.4])  # 5 params instead of 7
    model.getParamsCausal(wrong_params, 0.1, 0.0)
    print("❌ Should have caught parameter length error")
except ValueError as e:
    print(f"✅ Correctly caught parameter length error: {e}")

print("\n🎉 All tests completed!")
