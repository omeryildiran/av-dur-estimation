#!/usr/bin/env python3
"""
Test script to verify getActualParameterCount is accurate for all models.
"""

import numpy as np
import pandas as pd
from monteCarloClass import OmerMonteCarlo

def create_test_data():
    """Create test dataset"""
    np.random.seed(42)
    n_trials = 50
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

def test_parameter_counts():
    """Test parameter counts for all models"""
    print("=== Testing getActualParameterCount for all models ===\n")
    
    data = create_test_data()
    
    # All model names from the valid_models list
    all_models = [
        "gaussian", "lognorm", "logLinearMismatch", "fusionOnly", 
        "fusionOnlyLogNorm", "probabilityMatching", "probabilityMatchingLogNorm",
        "selection", "switching", "switchingWithConflict", "switchingFree"
    ]
    
    # Different configuration combinations to test
    configs = [
        {"sharedLambda": True, "freeP_c": False},
        {"sharedLambda": False, "freeP_c": False},
        {"sharedLambda": True, "freeP_c": True},
        {"sharedLambda": False, "freeP_c": True},
    ]
    
    results = {}
    
    for model_name in all_models:
        print(f"Testing model: {model_name}")
        results[model_name] = {}
        
        for config in configs:
            try:
                # Initialize model
                mc_model = OmerMonteCarlo(
                    data=data,
                    intensityVar='deltaDurS',
                    sensoryVar='audNoise',
                    standardVar='standardDur',
                    conflictVar='conflictDur'
                )
                
                # Set configuration
                mc_model.modelName = model_name
                mc_model.sharedLambda = config["sharedLambda"]
                mc_model.freeP_c = config["freeP_c"]
                
                # Get actual parameter count from function
                actual_count = mc_model.getActualParameterCount()
                
                # Get expected parameter count from the parameter length validation
                if model_name in ["fusionOnly", "fusionOnlyLogNorm"]:
                    expected_length = 6  # These don't vary with config
                elif model_name == "switchingWithConflict":
                    if config["freeP_c"]:
                        expected_length = 11 if not config["sharedLambda"] else 9
                    else:
                        expected_length = 10 if not config["sharedLambda"] else 8
                elif model_name == "switchingFree":
                    expected_length = 8 if not config["sharedLambda"] else 6
                elif config["freeP_c"]:
                    expected_length = 10 if not config["sharedLambda"] else 8
                else:
                    expected_length = 9 if not config["sharedLambda"] else 7
                
                # Store results
                config_key = f"sharedλ={config['sharedLambda']}, freeP_c={config['freeP_c']}"
                results[model_name][config_key] = {
                    'actual_count': actual_count,
                    'expected_length': expected_length,
                    'match': actual_count == expected_length
                }
                
                status = "✅" if actual_count == expected_length else "❌"
                print(f"  {config_key}: actual={actual_count}, expected={expected_length} {status}")
                
                if actual_count != expected_length:
                    print(f"    ⚠️  MISMATCH! getActualParameterCount() returned {actual_count}, but expected {expected_length}")
                
            except Exception as e:
                print(f"  {config_key}: ERROR - {e}")
                results[model_name][config_key] = {'error': str(e)}
        
        print()  # Blank line between models
    
    # Summary
    print("=== SUMMARY ===")
    all_good = True
    for model_name, configs in results.items():
        model_ok = True
        for config_key, result in configs.items():
            if 'error' in result:
                print(f"❌ {model_name} ({config_key}): ERROR - {result['error']}")
                model_ok = False
                all_good = False
            elif not result['match']:
                print(f"❌ {model_name} ({config_key}): Count mismatch")
                model_ok = False
                all_good = False
        
        if model_ok:
            print(f"✅ {model_name}: All configurations OK")
    
    if all_good:
        print("\n🎉 All parameter counts are accurate!")
    else:
        print("\n⚠️  Some parameter counts need fixing!")
    
    return results

if __name__ == "__main__":
    test_parameter_counts()