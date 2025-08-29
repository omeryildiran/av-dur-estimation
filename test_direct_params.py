#!/usr/bin/env python3
"""
Direct test of getParamsCausal function without full model initialization
"""

import numpy as np
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 Direct Testing of getParamsCausal Function")
print("=" * 50)

# Import the class
from monteCarloClass import OmerMonteCarlo

# Create a minimal instance by bypassing normal initialization
class TestOmerMonteCarlo(OmerMonteCarlo):
    def __init__(self):
        # Skip the parent __init__ and just set required attributes
        self.sharedSigma_v = True
        self.freeP_c = False
        self.sharedLambda = False

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
    model = TestOmerMonteCarlo()
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
model = TestOmerMonteCarlo()
model.sharedLambda = False
model.freeP_c = False

try:
    # Try with wrong length
    wrong_params = np.array([0.1, 0.2, 0.3, 0.7, 0.4])  # 5 params instead of 7
    model.getParamsCausal(wrong_params, 0.1, 0.0)
    print("❌ Should have caught parameter length error")
except ValueError as e:
    print(f"✅ Correctly caught parameter length error: {e}")

print("\n🎯 Testing freeP_c Index Logic")
print("-" * 40)

# Test the specific issue with freeP_c indexing
model = TestOmerMonteCarlo()

# Test sharedLambda=True, freeP_c=True case
model.sharedLambda = True
model.freeP_c = True
# For this configuration: [λ, σa1, σv, pc1, σa2, pc2]
params = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.85])  # pc2 should be 0.85

try:
    # Test SNR=1.2 case which had indexing issues
    lambda_, sigma_a, sigma_v, p_c = model.getParamsCausal(params, 1.2, 0.0)
    print(f"✅ sharedLambda=True, freeP_c=True, SNR=1.2: pc={p_c:.3f} (should be 0.85)")
    if np.isclose(p_c, 0.85):
        print("   🎯 Correct indexing for freeP_c!")
    else:
        print(f"   ❌ Wrong value: expected 0.85, got {p_c}")
except Exception as e:
    print(f"❌ Error in freeP_c indexing: {e}")

print("\n🔧 Parameter Layout Summary:")
print("sharedLambda=False, freeP_c=False: [λ, σa1, σv, pc, σa2, λ2, λ3] (7 params)")
print("sharedLambda=True,  freeP_c=False: [λ, σa1, σv, pc, σa2] (5 params)")  
print("sharedLambda=False, freeP_c=True:  [λ, σa1, σv, pc1, σa2, λ2, λ3, pc2] (8 params)")
print("sharedLambda=True,  freeP_c=True:  [λ, σa1, σv, pc1, σa2, pc2] (6 params)")

print("\n🎉 All tests completed!")
