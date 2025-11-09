#!/usr/bin/env python3
"""
Summary of parameter counts for all models in the SwitchingFree implementation.
"""

import numpy as np
import pandas as pd
from monteCarloClass import OmerMonteCarlo

def create_test_data():
    """Create minimal test dataset"""
    np.random.seed(42)
    n_trials = 10
    chose_test = np.random.choice([0, 1], n_trials)
    
    data = {
        'deltaDurS': np.random.uniform(-0.1, 0.1, n_trials),
        'audNoise': np.random.choice([0.1, 1.2], n_trials),
        'standardDur': np.full(n_trials, 0.5),
        'conflictDur': np.random.choice([0, 0.1], n_trials),
        'unbiasedVisualStandardDur': np.full(n_trials, 0.5),
        'unbiasedVisualTestDur': np.full(n_trials, 0.5) + np.random.uniform(-0.1, 0.1, n_trials),
        'testDurS': np.full(n_trials, 0.5) + np.random.uniform(-0.1, 0.1, n_trials),
        'chose_test': chose_test,
        'chose_standard': 1 - chose_test,
        'responses': chose_test,
        'order': np.ones(n_trials)
    }
    return pd.DataFrame(data)

def show_parameter_summary():
    """Show parameter counts for all models"""
    print("=== Parameter Count Summary for All Models ===\n")
    
    data = create_test_data()
    
    all_models = [
        "gaussian", "lognorm", "logLinearMismatch", 
        "fusionOnly", "fusionOnlyLogNorm",
        "probabilityMatching", "probabilityMatchingLogNorm",
        "selection", "switching", "switchingWithConflict", "switchingFree"
    ]
    
    print(f"{'Model':<25} {'SharedLambda=T':<15} {'SharedLambda=F':<15} {'FreeP_c Effect':<15}")
    print("="*70)
    
    for model_name in all_models:
        try:
            # Create model instance
            mc_model = OmerMonteCarlo(
                data=data,
                intensityVar='deltaDurS',
                sensoryVar='audNoise',
                standardVar='standardDur',
                conflictVar='conflictDur'
            )
            mc_model.modelName = model_name
            
            # Test different configurations
            configs = [
                (True, False),   # sharedLambda=True, freeP_c=False
                (False, False),  # sharedLambda=False, freeP_c=False
                (True, True),    # sharedLambda=True, freeP_c=True
                (False, True),   # sharedLambda=False, freeP_c=True
            ]
            
            counts = []
            for shared_lambda, free_pc in configs:
                mc_model.sharedLambda = shared_lambda
                mc_model.freeP_c = free_pc
                counts.append(mc_model.getActualParameterCount())
            
            # Determine if freeP_c has an effect
            free_pc_effect = "Yes" if (counts[2] != counts[0] or counts[3] != counts[1]) else "No"
            
            print(f"{model_name:<25} {counts[0]:<15} {counts[1]:<15} {free_pc_effect:<15}")
            
        except Exception as e:
            print(f"{model_name:<25} ERROR: {str(e)[:30]}")
    
    print("\n=== Special Cases ===")
    print("• fusionOnly/fusionOnlyLogNorm: p_c fixed at 1.0 (always common cause)")
    print("• switchingFree: Uses p_switch parameters instead of p_c")
    print("• switchingWithConflict: Adds k parameter for conflict sensitivity")
    print("• switching: Uses reliability-based switching but still has standard parameter structure")
    
    print("\n=== Parameter Structure Examples ===")
    print("Standard models (sharedLambda=True, freeP_c=False):")
    print("  [λ, σa1, σv, p_c, σa2, t_min, t_max] (7 params)")
    print("\nSwitchingFree (sharedLambda=True):")  
    print("  [λ, σa1, σv, p_switch1, σa2, p_switch2] (6 params)")
    print("\nFusion models:")
    print("  [λ, σa1, σv, σa2] (4 params - no p_c, no bounds needed)")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # Suppress BADS warning for cleaner output
    show_parameter_summary()