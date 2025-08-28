#!/usr/bin/env python3
"""
Simple Fusion Model - Usage Example
===================================

This script demonstrates how to use the SimpleFusionModel for audiovisual 
duration estimation experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simpleFusionModel import SimpleFusionModel

def demo_simple_fusion_model():
    """Demonstrate the simple fusion model with real-looking data."""
    
    print("🎯 Simple Fusion Model Demo")
    print("=" * 40)
    
    # Create realistic synthetic data
    np.random.seed(42)  # For reproducible results
    
    # Experimental conditions
    delta_durs = np.linspace(-0.15, 0.15, 15)  # Duration differences
    snr_levels = [0.1, 1.2]  # Audio noise levels
    conflicts = [-0.08, -0.04, 0, 0.04, 0.08]  # AV conflict levels
    n_reps = 20  # Repetitions per condition
    
    # Generate data
    data_list = []
    for snr in snr_levels:
        for conflict in conflicts:
            for delta in delta_durs:
                for rep in range(n_reps):
                    # Calculate test duration
                    standard_dur = 0.5
                    test_dur = standard_dur + delta
                    
                    # Visual durations (with conflict)
                    visual_standard = standard_dur + conflict
                    visual_test = test_dur  # No conflict for test
                    
                    # Simulate response (higher probability for longer test)
                    # Add some noise and bias based on conflict and SNR
                    base_prob = 0.5 + delta * 2  # Basic psychometric function
                    
                    # Conflict effect: larger conflict = more bias toward audio
                    conflict_effect = -conflict * 0.5
                    
                    # Noise effect: higher noise = more variability
                    noise_effect = np.random.normal(0, snr * 0.1)
                    
                    prob = np.clip(base_prob + conflict_effect + noise_effect, 0.05, 0.95)
                    chose_test = np.random.binomial(1, prob)
                    
                    data_list.append({
                        'deltaDurS': delta,
                        'audNoise': snr,
                        'standardDur': standard_dur,
                        'conflictDur': conflict,
                        'testDurS': test_dur,
                        'unbiasedVisualStandardDur': visual_standard,
                        'unbiasedVisualTestDur': visual_test,
                        'chose_test': chose_test,
                        'chose_standard': 1 - chose_test,
                        'responses': 1,
                        'order': 1
                    })
    
    # Convert to DataFrame
    data = pd.DataFrame(data_list)
    print(f"📊 Generated {len(data)} trials")
    print(f"   SNR levels: {snr_levels}")
    print(f"   Conflict levels: {conflicts}")
    print(f"   Delta durations: {len(delta_durs)} levels")
    
    # Initialize the simple fusion model
    print(f"\n🔧 Initializing Simple Fusion Model...")
    model = SimpleFusionModel(data, dataName="demo_synthetic")
    
    # Fit the model
    print(f"\n⚡ Fitting model...")
    try:
        result = model.fitSimpleFusion(nStart=3, method='Powell')
        
        print(f"\n✅ Model fitting completed!")
        print(f"📈 Final parameters:")
        params = result.x
        print(f"   Lapse rate (λ): {params[0]:.3f}")
        print(f"   Audio noise (low SNR): {params[1]:.3f}")
        print(f"   Visual noise (low SNR): {params[2]:.3f}")
        print(f"   Fusion weight: {params[3]:.3f}")
        print(f"   Audio noise (high SNR): {params[4]:.3f}")
        print(f"   Visual noise (high SNR): {params[5]:.3f}")
        
        # Interpret fusion weight
        if params[3] > 0.7:
            strategy = "Strong fusion - integrates audio/visual even with conflict"
        elif params[3] > 0.3:
            strategy = "Balanced - moderates between fusion and segregation"
        else:
            strategy = "Segregation-biased - prefers single modality with conflict"
        
        print(f"\n🧠 Fusion strategy: {strategy}")
        
        # Plot results
        print(f"\n📊 Generating plots...")
        
        # 1. Fusion strategy plot
        model.plot_fusion_strategy(params)
        
        # 2. Data comparison plot
        model.compare_with_data(params)
        
        # 3. Decision boundary analysis
        plot_decision_boundaries(model, params)
        
        return model, result
        
    except Exception as e:
        print(f"❌ Fitting failed: {e}")
        return model, None

def plot_decision_boundaries(model, fitted_params):
    """Plot decision boundaries for fusion vs segregation."""
    
    plt.figure(figsize=(12, 8))
    
    # Create meshgrid of conflict and reliability ratio
    conflicts = np.linspace(0, 0.2, 50)
    reliability_ratios = np.logspace(-1, 1, 50)  # 0.1 to 10
    
    Conflict, ReliabilityRatio = np.meshgrid(conflicts, reliability_ratios)
    
    # Calculate decisions for each point
    decisions = np.zeros_like(Conflict)
    
    for i in range(len(conflicts)):
        for j in range(len(reliability_ratios)):
            conflict = conflicts[i]
            rel_ratio = reliability_ratios[j]
            
            # Simulate the decision logic
            if conflict < model.conflict_threshold:
                decision = 2  # Always fuse
            elif rel_ratio > model.reliability_threshold:
                decision = 0  # Use audio
            elif rel_ratio < (1 / model.reliability_threshold):
                decision = 1  # Use visual
            else:
                decision = 3  # Weighted combination
            
            decisions[j, i] = decision
    
    # Plot decision regions
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Audio only', 'Visual only', 'Always fuse', 'Weighted mix']
    
    contour = plt.contourf(Conflict, ReliabilityRatio, decisions, 
                          levels=[0, 1, 2, 3, 4], colors=colors, alpha=0.7)
    
    # Add threshold lines
    plt.axvline(x=model.conflict_threshold, color='black', linestyle='--', 
               label=f'Conflict threshold ({model.conflict_threshold})')
    plt.axhline(y=model.reliability_threshold, color='gray', linestyle='--',
               label=f'Reliability threshold ({model.reliability_threshold})')
    plt.axhline(y=1/model.reliability_threshold, color='gray', linestyle='--')
    
    plt.xlabel('Conflict Level')
    plt.ylabel('Reliability Ratio (Audio/Visual)')
    plt.title('Decision Boundaries')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add color legend
    import matplotlib.patches as patches
    for i, (color, label) in enumerate(zip(colors, labels)):
        plt.gca().add_patch(patches.Rectangle((0.15, 0.1 + i*0.05), 0.02, 0.03, 
                                            facecolor=color, alpha=0.7, transform=plt.gca().transAxes))
        plt.text(0.18, 0.115 + i*0.05, label, transform=plt.gca().transAxes, fontsize=10)
    
    # Plot example trajectories
    plt.subplot(1, 2, 2)
    
    # Get fitted parameters for both SNR conditions
    lambda1, sigma_a1, sigma_v1, fusion_weight = model.getParamsSimpleFusion(fitted_params, 0.1, 0)
    lambda2, sigma_a2, sigma_v2, _ = model.getParamsSimpleFusion(fitted_params, 1.2, 0)
    
    # Calculate reliability ratios for both conditions
    rel_ratio_1 = (1/sigma_a1**2) / (1/sigma_v1**2)
    rel_ratio_2 = (1/sigma_a2**2) / (1/sigma_v2**2)
    
    conflicts_example = np.linspace(0, 0.15, 20)
    
    plt.plot(conflicts_example, [rel_ratio_1] * len(conflicts_example), 
             'ro-', linewidth=2, markersize=8, label=f'Low noise (SNR=0.1)')
    plt.plot(conflicts_example, [rel_ratio_2] * len(conflicts_example), 
             'bo-', linewidth=2, markersize=8, label=f'High noise (SNR=1.2)')
    
    # Add threshold lines
    plt.axvline(x=model.conflict_threshold, color='black', linestyle='--', 
               label=f'Conflict threshold')
    plt.axhline(y=model.reliability_threshold, color='gray', linestyle='--',
               label=f'Reliability thresholds')
    plt.axhline(y=1/model.reliability_threshold, color='gray', linestyle='--')
    
    plt.xlabel('Conflict Level')
    plt.ylabel('Reliability Ratio (Audio/Visual)')
    plt.title('Model Conditions in Decision Space')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_models_demo():
    """Compare simple fusion with other strategies."""
    
    print(f"\n🔬 Model Comparison Demo")
    print("=" * 30)
    
    # Test conditions
    conflicts = np.array([0.02, 0.05, 0.1, 0.15])  # Increasing conflict
    sigma_a, sigma_v = 0.15, 0.20  # Noise levels
    
    print(f"Test conditions:")
    print(f"  Audio noise: {sigma_a}")
    print(f"  Visual noise: {sigma_v}")
    print(f"  Conflicts: {conflicts}")
    
    # Calculate reliability ratio
    rel_ratio = (1/sigma_a**2) / (1/sigma_v**2)
    print(f"  Reliability ratio (A/V): {rel_ratio:.2f}")
    
    plt.figure(figsize=(15, 5))
    
    strategies = {
        'Audio Only': lambda m_a, m_v, conflict: m_a,
        'Visual Only': lambda m_a, m_v, conflict: m_v,
        'Optimal Fusion': lambda m_a, m_v, conflict: optimal_fusion(m_a, m_v, sigma_a, sigma_v),
        'Simple Fusion Model': lambda m_a, m_v, conflict: simple_fusion_decision_standalone(m_a, m_v, sigma_a, sigma_v, conflict)
    }
    
    for idx, (name, strategy_func) in enumerate(strategies.items()):
        plt.subplot(1, 4, idx + 1)
        
        estimates = []
        for conflict in conflicts:
            m_a, m_v = 0.5, 0.5 + conflict
            estimate = strategy_func(m_a, m_v, conflict)
            estimates.append(estimate)
        
        plt.plot(conflicts * 1000, estimates, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Conflict (ms)')
        plt.ylabel('Duration Estimate (s)')
        plt.title(name)
        plt.grid(True, alpha=0.3)
        
        # Add reference lines
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Audio truth')
        plt.axhline(y=0.5 + conflicts[-1], color='blue', linestyle='--', alpha=0.5, label='Visual (max conflict)')
        
        if idx == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

def optimal_fusion(m_a, m_v, sigma_a, sigma_v):
    """Optimal reliability-weighted fusion."""
    J_a = 1 / sigma_a**2
    J_v = 1 / sigma_v**2
    w_a = J_a / (J_a + J_v)
    return w_a * m_a + (1 - w_a) * m_v

def simple_fusion_decision_standalone(m_a, m_v, sigma_a, sigma_v, conflict, 
                                    conflict_threshold=0.1, reliability_threshold=2.0, fusion_weight=0.6):
    """Standalone version of simple fusion decision for demo."""
    
    # Calculate reliability ratio
    reliability_ratio = (1/sigma_a**2) / (1/sigma_v**2)
    
    # Decision logic
    if conflict < conflict_threshold:
        # Low conflict -> optimal fusion
        J_a = 1 / sigma_a**2
        J_v = 1 / sigma_v**2
        w_a = J_a / (J_a + J_v)
        return w_a * m_a + (1 - w_a) * m_v
    
    elif reliability_ratio > reliability_threshold:
        # High conflict + audio dominance -> audio only
        return m_a
    
    elif reliability_ratio < (1 / reliability_threshold):
        # High conflict + visual dominance -> visual only
        return m_v
    
    else:
        # High conflict + similar reliability -> weighted combination
        J_a = 1 / sigma_a**2
        J_v = 1 / sigma_v**2
        w_a = J_a / (J_a + J_v)
        fused_estimate = w_a * m_a + (1 - w_a) * m_v
        
        # Segregated estimate (use more reliable)
        segregated_estimate = m_a if reliability_ratio > 1 else m_v
        
        return fusion_weight * fused_estimate + (1 - fusion_weight) * segregated_estimate

if __name__ == "__main__":
    # Run the main demo
    model, result = demo_simple_fusion_model()
    
    if result is not None:
        # Run comparison demo
        compare_models_demo()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📝 The Simple Fusion Model provides:")
        print(f"   ✅ Clear if-else decision logic")
        print(f"   ✅ Easy parameter interpretation")
        print(f"   ✅ Flexible fusion strategies")
        print(f"   ✅ Good visualization tools")
    else:
        print(f"\n⚠️ Demo had issues, but model structure is ready to use.")
