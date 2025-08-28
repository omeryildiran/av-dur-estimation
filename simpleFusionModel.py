#!/usr/bin/env python3
"""
Simple Fusion Model with If-Else Logic
======================================

A straightforward audiovisual fusion model that uses simple if-else statements
to decide between different integration strategies based on reliability and conflict.

This model is easier to understand and debug compared to the full causal inference model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm import tqdm
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fitMainClass import fitPychometric


class SimpleFusionModel(fitPychometric):
    """
    Simple fusion model with if-else logic for audiovisual integration.
    
    The model uses simple rules to decide when to:
    - Use audio only
    - Use visual only  
    - Fuse audio and visual
    - Use segregated estimates
    """
    
    def __init__(self, data, intensityVar='deltaDurS', allIndependent=True, sharedSigma=False, 
                 sensoryVar='audNoise', standardVar='standardDur', conflictVar='conflictDur', dataName=None):
        super().__init__(data, intensityVar, allIndependent, sharedSigma, sensoryVar, standardVar, conflictVar, dataName)
        
        self.fitType = 'Simple Fusion'
        self.modelName = "simple_fusion"
        self.dataName = dataName if dataName else "default_data"
        
        # Fusion strategy parameters
        self.conflict_threshold = 0.1  # Above this, consider signals as conflicting
        self.reliability_threshold = 2.0  # Reliability ratio threshold for dominance
        
        # Initialize bounds for duration estimates
        self.t_min = max(data["testDurS"].min() - 0.1, 0.1)
        self.t_max = data["testDurS"].max() + 0.1
        
        # Grouped data for fitting
        self.groupedData = self.groupByChooseTest(
            x=data,
            groupArgs=[
                self.intensityVar, sensoryVar, standardVar, conflictVar,
                "unbiasedVisualStandardDur", "unbiasedVisualTestDur", "testDurS"
            ]
        )
        
        print(f"✅ Simple Fusion Model initialized with {len(self.groupedData)} conditions")
        print(f"   Conflict threshold: {self.conflict_threshold}")
        print(f"   Reliability threshold: {self.reliability_threshold}")
        print(f"   Duration bounds: [{self.t_min:.2f}, {self.t_max:.2f}]")
    
    def getParamsSimpleFusion(self, params, SNR, conflict):
        """
        Extract parameters for simple fusion model.
        
        Parameters:
        -----------
        params : array-like
            [lambda, sigma_a1, sigma_v1, fusion_weight, sigma_a2, sigma_v2]
        SNR : float
            Signal-to-noise ratio (audio noise level)
        conflict : float
            Conflict level between audio and visual
            
        Returns:
        --------
        tuple : (lambda, sigma_a, sigma_v, fusion_weight)
        """
        lambda_ = params[0]  # Lapse rate
        fusion_weight = params[3]  # Weight for fusion vs segregation
        
        # Get noise parameters based on SNR condition
        if np.isclose(SNR, 0.1):  # Low noise condition
            sigma_a = params[1]
            sigma_v = params[2]
        elif np.isclose(SNR, 1.2):  # High noise condition
            sigma_a = params[4]
            sigma_v = params[5] if len(params) > 5 else params[2]  # Share visual noise if needed
        else:
            # Interpolate for other SNR values
            sigma_a = params[1] + (SNR - 0.1) * (params[4] - params[1]) / (1.2 - 0.1)
            sigma_v = params[2]
        
        return lambda_, sigma_a, sigma_v, fusion_weight
    
    def simple_fusion_decision(self, m_a, m_v, sigma_a, sigma_v, fusion_weight):
        """
        Simple fusion decision with if-else logic.
        
        Parameters:
        -----------
        m_a, m_v : float or array
            Audio and visual measurements
        sigma_a, sigma_v : float
            Audio and visual noise levels
        fusion_weight : float
            Weight parameter for fusion vs segregation
            
        Returns:
        --------
        estimate : float or array
            Final duration estimate
        """
        # Calculate conflict level
        conflict = np.abs(m_a - m_v)
        
        # Calculate reliability ratio
        reliability_a = 1 / (sigma_a**2)  # Higher reliability = lower noise
        reliability_v = 1 / (sigma_v**2)
        reliability_ratio = reliability_a / reliability_v
        
        # Decision logic with if-else statements
        if np.isscalar(conflict):
            # Scalar case
            estimate = self._single_fusion_decision(m_a, m_v, sigma_a, sigma_v, 
                                                   conflict, reliability_ratio, fusion_weight)
        else:
            # Array case - apply decision element-wise
            estimate = np.zeros_like(conflict)
            for i in range(len(conflict)):
                estimate[i] = self._single_fusion_decision(
                    m_a[i], m_v[i], sigma_a, sigma_v, 
                    conflict[i], reliability_ratio, fusion_weight
                )
        
        return estimate
    
    def _single_fusion_decision(self, m_a, m_v, sigma_a, sigma_v, conflict, reliability_ratio, fusion_weight):
        """Single decision for one measurement pair."""
        
        # Rule 1: Low conflict -> Always fuse optimally
        if conflict < self.conflict_threshold:
            # Optimal fusion (reliability-weighted average)
            J_a = 1 / sigma_a**2
            J_v = 1 / sigma_v**2
            w_a = J_a / (J_a + J_v)
            estimate = w_a * m_a + (1 - w_a) * m_v
            return estimate
        
        # Rule 2: High conflict + strong audio dominance -> Use audio
        elif reliability_ratio > self.reliability_threshold:
            estimate = m_a
            return estimate
        
        # Rule 3: High conflict + strong visual dominance -> Use visual  
        elif reliability_ratio < (1 / self.reliability_threshold):
            estimate = m_v
            return estimate
        
        # Rule 4: High conflict + similar reliability -> Weighted combination
        else:
            # Use fusion_weight to interpolate between fusion and segregation
            # fusion_weight = 0 -> full segregation (use audio)
            # fusion_weight = 1 -> full fusion
            
            # Optimal fusion estimate
            J_a = 1 / sigma_a**2
            J_v = 1 / sigma_v**2
            w_a = J_a / (J_a + J_v)
            fused_estimate = w_a * m_a + (1 - w_a) * m_v
            
            # Segregated estimate (use more reliable modality)
            if reliability_ratio > 1:
                segregated_estimate = m_a
            else:
                segregated_estimate = m_v
            
            # Weighted combination
            estimate = fusion_weight * fused_estimate + (1 - fusion_weight) * segregated_estimate
            return estimate
    
    def probTestLonger_simpleFusion(self, trueStims, sigma_a, sigma_v, fusion_weight, lambda_, nSimul=1000):
        """
        Calculate probability that test is judged longer than standard.
        
        Parameters:
        -----------
        trueStims : tuple
            (S_a_s, S_a_t, S_v_s, S_v_t) - true audio/visual standard/test durations
        sigma_a, sigma_v : float
            Audio and visual noise levels
        fusion_weight : float
            Fusion weight parameter
        lambda_ : float
            Lapse rate
        nSimul : int
            Number of Monte Carlo simulations
            
        Returns:
        --------
        float : Probability of choosing test as longer
        """
        S_a_s, S_a_t, S_v_s, S_v_t = trueStims
        
        # Generate noisy measurements
        m_a_s = np.random.normal(S_a_s, sigma_a, nSimul)
        m_v_s = np.random.normal(S_v_s, sigma_v, nSimul)
        m_a_t = np.random.normal(S_a_t, sigma_a, nSimul)
        m_v_t = np.random.normal(S_v_t, sigma_v, nSimul)
        
        # Get fusion estimates
        est_standard = self.simple_fusion_decision(m_a_s, m_v_s, sigma_a, sigma_v, fusion_weight)
        est_test = self.simple_fusion_decision(m_a_t, m_v_t, sigma_a, sigma_v, fusion_weight)
        
        # Calculate base probability
        p_base = np.mean(est_test > est_standard)
        
        # Add lapse rate
        p_final = (1 - lambda_) * p_base + lambda_ / 2
        
        return p_final
    
    def nLLSimpleFusion(self, params, groupedData):
        """
        Negative log-likelihood for simple fusion model.
        
        Parameters:
        -----------
        params : array-like
            Model parameters [lambda, sigma_a1, sigma_v1, fusion_weight, sigma_a2, sigma_v2]
        groupedData : DataFrame
            Grouped experimental data
            
        Returns:
        --------
        float : Negative log-likelihood
        """
        ll = 0
        
        # Check for invalid parameters
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return 1e10
        
        # Check parameter bounds
        if params[0] < 0 or params[0] > 0.5:  # lambda
            return 1e10
        if any(p <= 0 for p in params[1:3]):  # sigma values
            return 1e10
        if params[3] < 0 or params[3] > 1:  # fusion_weight
            return 1e10
        
        for i in range(len(groupedData)):
            # Get condition information
            currSNR = groupedData["audNoise"].iloc[i]
            currConflict = groupedData["conflictDur"].iloc[i]
            currResp = groupedData['num_of_chose_test'].iloc[i]
            totalResponses = groupedData['total_responses'].iloc[i]
            
            # Get parameters for this condition
            lambda_, sigma_a, sigma_v, fusion_weight = self.getParamsSimpleFusion(params, currSNR, currConflict)
            
            # Get true stimulus durations
            S_a_s = groupedData["standardDur"].iloc[i]
            S_v_s = groupedData["unbiasedVisualStandardDur"].iloc[i]
            S_a_t = groupedData["testDurS"].iloc[i]
            S_v_t = groupedData["unbiasedVisualTestDur"].iloc[i]
            trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
            
            # Calculate probability
            try:
                p_test_longer = self.probTestLonger_simpleFusion(trueStims, sigma_a, sigma_v, 
                                                               fusion_weight, lambda_, nSimul=100)
            except:
                return 1e10
            
            # Add to log-likelihood with numerical stability
            epsilon = 1e-10
            P = np.clip(p_test_longer, epsilon, 1 - epsilon)
            ll += np.log(P) * currResp + np.log(1 - P) * (totalResponses - currResp)
        
        # Check for invalid likelihood
        if np.isnan(ll) or np.isinf(ll):
            return 1e10
        
        return -ll
    
    def fitSimpleFusion(self, nStart=3, method='Powell'):
        """
        Fit the simple fusion model to data.
        
        Parameters:
        -----------
        nStart : int
            Number of random starting points
        method : str
            Optimization method ('Powell', 'L-BFGS-B', etc.)
            
        Returns:
        --------
        OptimizeResult : Best fitting result
        """
        # Parameter bounds: [lambda, sigma_a1, sigma_v1, fusion_weight, sigma_a2, sigma_v2]
        bounds = [
            (0, 0.3),      # lambda (lapse rate)
            (0.05, 1.0),   # sigma_a1 (audio noise, low SNR)
            (0.05, 1.0),   # sigma_v1 (visual noise, low SNR)  
            (0, 1),        # fusion_weight (0=segregate, 1=fuse)
            (0.05, 1.5),   # sigma_a2 (audio noise, high SNR)
            (0.05, 1.0),   # sigma_v2 (visual noise, high SNR)
        ]
        
        best_result = None
        best_ll = np.inf
        
        print(f"\n🔄 Fitting Simple Fusion Model with {nStart} random starts...")
        
        for attempt in tqdm(range(nStart), desc="Optimization"):
            # Random initialization
            x0 = np.array([
                np.random.uniform(0.01, 0.2),   # lambda
                np.random.uniform(0.1, 0.8),    # sigma_a1
                np.random.uniform(0.1, 0.8),    # sigma_v1
                np.random.uniform(0.2, 0.8),    # fusion_weight
                np.random.uniform(0.2, 1.2),    # sigma_a2
                np.random.uniform(0.1, 0.8),    # sigma_v2
            ])
            
            try:
                result = minimize(
                    self.nLLSimpleFusion,
                    x0=x0,
                    args=(self.groupedData,),
                    method=method,
                    bounds=bounds
                )
                
                if result.fun < best_ll:
                    best_ll = result.fun
                    best_result = result
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All optimization attempts failed!")
        
        print(f"\n✅ Best fit found:")
        print(f"   Final negative log-likelihood: {best_result.fun:.2f}")
        print(f"   Parameters: {best_result.x}")
        print(f"   λ={best_result.x[0]:.3f}, fusion_weight={best_result.x[3]:.3f}")
        
        return best_result
    
    def plot_fusion_strategy(self, fitted_params, conflict_range=np.linspace(0, 0.3, 50)):
        """
        Plot the fusion strategy as a function of conflict level.
        
        Parameters:
        -----------
        fitted_params : array-like
            Fitted model parameters
        conflict_range : array-like
            Range of conflict values to plot
        """
        plt.figure(figsize=(12, 8))
        
        # Test both SNR conditions
        snr_values = [0.1, 1.2]
        snr_labels = ['Low Noise (0.1)', 'High Noise (1.2)']
        
        for idx, (snr, label) in enumerate(zip(snr_values, snr_labels)):
            plt.subplot(2, 2, idx + 1)
            
            # Get parameters for this SNR
            lambda_, sigma_a, sigma_v, fusion_weight = self.getParamsSimpleFusion(fitted_params, snr, 0)
            
            # Test different conflict levels
            strategies = []
            estimates_a_only = []
            estimates_v_only = []
            estimates_fused = []
            estimates_actual = []
            
            # Fixed test measurements
            m_a, m_v_base = 0.5, 0.5
            
            for conflict in conflict_range:
                m_v = m_v_base + conflict
                
                # Calculate what each strategy would give
                est_a_only = m_a
                est_v_only = m_v
                
                # Optimal fusion
                J_a = 1 / sigma_a**2
                J_v = 1 / sigma_v**2
                w_a = J_a / (J_a + J_v)
                est_fused = w_a * m_a + (1 - w_a) * m_v
                
                # Actual model decision
                est_actual = self.simple_fusion_decision(m_a, m_v, sigma_a, sigma_v, fusion_weight)
                
                estimates_a_only.append(est_a_only)
                estimates_v_only.append(est_v_only)
                estimates_fused.append(est_fused)
                estimates_actual.append(est_actual)
            
            # Plot strategies
            plt.plot(conflict_range, estimates_a_only, 'r--', label='Audio only', alpha=0.7)
            plt.plot(conflict_range, estimates_v_only, 'b--', label='Visual only', alpha=0.7)  
            plt.plot(conflict_range, estimates_fused, 'g--', label='Optimal fusion', alpha=0.7)
            plt.plot(conflict_range, estimates_actual, 'k-', linewidth=3, label='Model decision')
            
            plt.axvline(x=self.conflict_threshold, color='gray', linestyle=':', alpha=0.7, 
                       label=f'Conflict threshold ({self.conflict_threshold})')
            
            plt.xlabel('Conflict Level')
            plt.ylabel('Duration Estimate')
            plt.title(f'{label} (σa={sigma_a:.2f}, σv={sigma_v:.2f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Add parameter summary
        plt.subplot(2, 2, 3)
        plt.axis('off')
        plt.text(0.1, 0.8, 'Model Parameters:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, f'Lapse rate (λ): {fitted_params[0]:.3f}', fontsize=12)
        plt.text(0.1, 0.6, f'Fusion weight: {fitted_params[3]:.3f}', fontsize=12)
        plt.text(0.1, 0.5, f'Conflict threshold: {self.conflict_threshold}', fontsize=12)
        plt.text(0.1, 0.4, f'Reliability threshold: {self.reliability_threshold}', fontsize=12)
        
        plt.text(0.1, 0.2, 'Noise Parameters:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.1, f'Low noise: σa={fitted_params[1]:.3f}, σv={fitted_params[2]:.3f}', fontsize=12)
        plt.text(0.1, 0.0, f'High noise: σa={fitted_params[4]:.3f}, σv={fitted_params[5]:.3f}', fontsize=12)
        
        # Add decision logic flowchart
        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.text(0.1, 0.9, 'Decision Logic:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.8, '1. Conflict < threshold → Optimal fusion', fontsize=10)
        plt.text(0.1, 0.7, '2. High conflict + Audio reliable → Audio only', fontsize=10)
        plt.text(0.1, 0.6, '3. High conflict + Visual reliable → Visual only', fontsize=10)
        plt.text(0.1, 0.5, '4. High conflict + Similar reliability → Weighted mix', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_data(self, fitted_params):
        """
        Compare model predictions with actual data.
        
        Parameters:
        -----------
        fitted_params : array-like
            Fitted model parameters
        """
        plt.figure(figsize=(15, 10))
        
        # Get unique conditions
        unique_snr = sorted(self.groupedData['audNoise'].unique())
        unique_conflicts = sorted(self.groupedData['conflictDur'].unique())
        
        plot_idx = 1
        for snr in unique_snr:
            for conflict in unique_conflicts:
                if plot_idx > 6:  # Limit number of subplots
                    break
                    
                plt.subplot(2, 3, plot_idx)
                
                # Get data for this condition
                condition_data = self.groupedData[
                    (self.groupedData['audNoise'] == snr) & 
                    (self.groupedData['conflictDur'] == conflict)
                ]
                
                if len(condition_data) == 0:
                    continue
                
                # Plot data points
                x_data = condition_data['deltaDurS']
                y_data = condition_data['num_of_chose_test'] / condition_data['total_responses']
                
                plt.scatter(x_data, y_data, color='red', s=50, alpha=0.7, label='Data')
                
                # Generate model predictions
                x_model = np.linspace(x_data.min(), x_data.max(), 50)
                y_model = []
                
                lambda_, sigma_a, sigma_v, fusion_weight = self.getParamsSimpleFusion(fitted_params, snr, conflict)
                
                for x_val in x_model:
                    # Calculate corresponding stimulus values
                    S_a_s = 0.5  # Standard duration
                    S_a_t = S_a_s + x_val  # Test duration
                    S_v_s = S_a_s + conflict  # Visual standard
                    S_v_t = S_a_t  # Visual test (assuming no conflict for test)
                    
                    trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
                    p_longer = self.probTestLonger_simpleFusion(trueStims, sigma_a, sigma_v, 
                                                              fusion_weight, lambda_, nSimul=200)
                    y_model.append(p_longer)
                
                plt.plot(x_model, y_model, 'blue', linewidth=2, label='Model')
                
                plt.xlabel('Duration Difference (s)')
                plt.ylabel('P(choose test)')
                plt.title(f'SNR={snr}, Conflict={conflict*1000:.0f}ms')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Simple Fusion Model - Example Usage")
    print("=" * 50)
    
    # Create dummy data for demonstration
    n_trials = 500
    dummy_data = pd.DataFrame({
        'deltaDurS': np.tile(np.linspace(-0.2, 0.2, 25), 20),
        'audNoise': np.repeat([0.1, 1.2], 250),
        'standardDur': np.full(n_trials, 0.5),
        'conflictDur': np.tile(np.repeat([-0.1, -0.05, 0, 0.05, 0.1], 25), 4),
        'unbiasedVisualStandardDur': np.full(n_trials, 0.5),
        'unbiasedVisualTestDur': np.full(n_trials, 0.5),
        'testDurS': np.full(n_trials, 0.5) + np.tile(np.linspace(-0.2, 0.2, 25), 20),
        'chose_test': np.random.binomial(1, 0.5, n_trials),
        'chose_standard': np.random.binomial(1, 0.5, n_trials),
        'responses': np.full(n_trials, 1),
        'order': np.full(n_trials, 1)
    })
    
    # Initialize model
    model = SimpleFusionModel(dummy_data, dataName="demo_data")
    
    # Test parameter extraction
    test_params = [0.1, 0.2, 0.3, 0.6, 0.4, 0.3]
    lambda_, sigma_a, sigma_v, fusion_weight = model.getParamsSimpleFusion(test_params, 0.1, 0.0)
    print(f"\n📊 Parameter extraction test:")
    print(f"   λ={lambda_:.3f}, σa={sigma_a:.3f}, σv={sigma_v:.3f}, fusion_weight={fusion_weight:.3f}")
    
    # Test fusion decision
    m_a, m_v = 0.5, 0.55  # Small conflict
    estimate1 = model.simple_fusion_decision(m_a, m_v, sigma_a, sigma_v, fusion_weight)
    
    m_a, m_v = 0.5, 0.7   # Large conflict
    estimate2 = model.simple_fusion_decision(m_a, m_v, sigma_a, sigma_v, fusion_weight)
    
    print(f"\n🧠 Fusion decision test:")
    print(f"   Small conflict (0.5 vs 0.55): estimate = {estimate1:.3f}")
    print(f"   Large conflict (0.5 vs 0.7): estimate = {estimate2:.3f}")
    
    print(f"\n✅ Simple Fusion Model ready for use!")
    print(f"   To fit: result = model.fitSimpleFusion()")
    print(f"   To plot: model.plot_fusion_strategy(result.x)")
