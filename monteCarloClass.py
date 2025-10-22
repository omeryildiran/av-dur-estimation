# A class for monte carlo inherited from fitMainClasss.py

# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.optimize as opt
import fitMainClass as fitMain

from fitMainClass import fitPychometric

from tqdm import tqdm

from scipy.optimize import minimize
from scipy.stats import norm
from pybads import BADS  # Only if installed

class TqdmMinimizeCallback:
    def __init__(self, total=100, show_progress=1):
        self.show_progress = show_progress
        if self.show_progress:
            self.pbar = tqdm(total=total, desc="Fitting Causal Inference", leave=True)
        self.last_nfev = 0

    def __call__(self, xk):
        # This callback is called after each iteration
        if self.show_progress:
            self.pbar.update(1)

    def close(self):
        if self.show_progress:
            self.pbar.close()



class OmerMonteCarlo(fitPychometric):
    """
    A class for fitting psychometric functions using Monte Carlo methods.
    Inherits from fitPychometric.
    """
    
    def __init__(self, data, intensityVar='deltaDurS', allIndependent=True, sharedSigma=0, sensoryVar='audNoise', 
                 standardVar='standardDur', conflictVar='conflictDur',dataName=None):
        super().__init__(data, intensityVar, allIndependent, sharedSigma, sensoryVar, standardVar, conflictVar,dataName) 
        
        self.fitType = 'Monte Carlo'
        self.nStart = 3  # Number of random starts for optimization (increased for debugging)
        self.nSimul = 10  # Number of simulations for Monte Carlo approximation
        self.optimizationMethod = 'scipy'  # Use scipy for better debugging (BADS can be harder to debug)
        self.modelFit = None  # Placeholder for fitted model
        self.simulatedData = None  # Placeholder for simulated data
        self.dataFit = None  # Placeholder for fitted data
        self.simDataFit = None  # Placeholder for simulated data fit
        self.groupedData = None  # Placeholder for grouped data
        self.modelName = "gaussian"  # Distribution of measurements, can be 'gaussian' or 'lognormal'
        
    
        
        self.visualStandardVar = "unbiasedVisualStandardDur"
        self.visualTestVar = "unbiasedVisualTestDur"
        self.audioTestVar = "testDurS"
        self.dataName = dataName if dataName else "default_data"
        self.data = data
        self.sharedSigma_v=True
        self.freeP_c = False  # Fix: Add missing attribute for parameter configuration
        self.logLikelihood= None  # Placeholder for log-likelihood
        self.sharedLambda=False

        self.groupedData= self.groupByChooseTest(x=data,
                groupArgs=[
                self.intensityVar, sensoryVar, standardVar, conflictVar,
                self.visualStandardVar, self.visualTestVar, self.audioTestVar
                    ]
                    )
        
        # t_min and t_max are now fitted parameters instead of fixed values from data
        # Store data bounds for initial parameter estimation and bounds constraints
        self.data_t_min = data["testDurS"].min()
        self.data_t_max = data["testDurS"].max()
        
        # Ensure we have reasonable data bounds for parameter initialization
        if self.data_t_max == self.data_t_min:
            duration_center = self.data_t_min
            self.data_t_min = max(0.1, duration_center - 0.5)  # At least 100ms range
            self.data_t_max = duration_center + 0.5
            print(f"Warning: All test durations identical ({duration_center}s). Using initial bounds [{self.data_t_min:.2f}, {self.data_t_max:.2f}]")



    def getParamsCausal(self,params,SNR,conflict):
        """Extract causal inference parameters for a specific condition (conflict, noise).
        
        Parameters:
        -----------
        params : array-like
            Parameter array with layout depending on configuration:
            - sharedLambda=False, freeP_c=False: [λ, σa1, σv, pc, σa2, λ2, λ3, t_min, t_max] (9 params)
            - sharedLambda=True, freeP_c=False: [λ, σa1, σv, pc, σa2, t_min, t_max] (7 params)
            - sharedLambda=False, freeP_c=True: [λ, σa1, σv, pc1, σa2, λ2, λ3, pc2, t_min, t_max] (10 params)
            - sharedLambda=True, freeP_c=True: [λ, σa1, σv, pc1, σa2, pc2, t_min, t_max] (8 params)
        SNR : float
            Signal-to-noise ratio (0.1 or 1.2)
        conflict : float
            Conflict level
            
        Returns:
        --------
        tuple : (lambda_, sigma_av_a, sigma_av_v, p_c, t_min, t_max)
        """
        
        # Validate parameter array length
        expected_lengths = {
            (True, True): 8,    # sharedLambda=True, freeP_c=True
            (True, False): 7,   # sharedLambda=True, freeP_c=False
            (False, True): 10,  # sharedLambda=False, freeP_c=True
            (False, False): 9   # sharedLambda=False, freeP_c=False
        }
        
        config_key = (self.sharedLambda, self.freeP_c)
        expected_length = expected_lengths[config_key]
        
        if len(params) != expected_length:
            raise ValueError(f"Parameter array length {len(params)} doesn't match expected length {expected_length} for configuration sharedLambda={self.sharedLambda}, freeP_c={self.freeP_c}")

        # Extract t_min and t_max (always last two parameters)
        t_min = params[-2]
        t_max = params[-1]

        # Extract lambda parameter
        lambda_=params[0]
        if not self.sharedLambda:
            if conflict in [ 0, -0.17,  0.25]:
                lambda_=params[0]
            elif conflict in [-0.08, 0.17]:
                lambda_=params[5]
            elif conflict in [ -0.25,  0.08]:
                lambda_=params[6]
        
        # Extract p_c parameter
        if self.freeP_c:
            if np.isclose(SNR, 0.1):
                p_c=params[3] # p_c is not shared across SNR conditions so we have two p_c parameters
            elif np.isclose(SNR,1.2):
                if self.sharedLambda:
                    p_c=params[5]  # Different index when sharedLambda=True
                else:
                    p_c=params[7]  # Different index when sharedLambda=False
            else:
                raise(ValueError(f"Unexpected SNR value: {SNR}. Expected 0.1 or 1.2."))
        else:
            p_c=params[3] # p_c is shared across SNR conditions no need to have two p_c parameters 

        # Extract sigma_av_v parameter  
        if self.sharedSigma_v:
            sigma_av_v = params[2]  # Use the shared sigma for visual noise
        else:
            # If not shared, would need separate logic here
            sigma_av_v = params[2]  # For now, still use index 2

        # Extract sigma_av_a parameter
        if np.isclose(SNR, 0.1):
            sigma_av_a=params[1]
        elif np.isclose(SNR,1.2):
            sigma_av_a=params[4]
        else:
            raise(ValueError(f"Unexpected SNR value: {SNR}. Expected 0.1 or 1.2."))

        return lambda_,sigma_av_a,sigma_av_v,p_c,t_min,t_max
    
    def test_fitting_components(self, groupedData):
        """Test individual components of the fitting process for debugging"""
        print("=== DEBUGGING FITTING COMPONENTS ===")
        
        # Test data bounds
        print(f"Data bounds: t_min={self.data_t_min:.3f}, t_max={self.data_t_max:.3f}")
        print(f"Grouped data shape: {groupedData.shape}")
        print(f"Configuration: sharedLambda={self.sharedLambda}, freeP_c={self.freeP_c}")
        
        # Test parameter array creation
        if self.freeP_c:
            expected_length = 10 if not self.sharedLambda else 8
        else:
            expected_length = 9 if not self.sharedLambda else 7
        print(f"Expected parameter length: {expected_length}")
        
        # Create test parameters
        test_params = [0.1, 0.5, 0.5, 0.5, 0.8]  # Basic 5 params
        if not self.sharedLambda:
            test_params.extend([0.1, 0.1])  # Add lambda2, lambda3
        if self.freeP_c:
            test_params.append(0.5)  # Add second p_c
        test_params.extend([0.2, 1.0])  # Add t_min, t_max
        
        test_params = np.array(test_params)
        print(f"Test params length: {len(test_params)}, values: {test_params}")
        
        # Test parameter extraction
        try:
            first_snr = groupedData["audNoise"].iloc[0]
            first_conflict = groupedData["conflictDur"].iloc[0]
            λ, σa, σv, pc, t_min, t_max = self.getParamsCausal(test_params, first_snr, first_conflict)
            print(f"Extracted params: λ={λ:.3f}, σa={σa:.3f}, σv={σv:.3f}, pc={pc:.3f}, t_min={t_min:.3f}, t_max={t_max:.3f}")
        except Exception as e:
            print(f"ERROR in parameter extraction: {e}")
            return False
            
        # Test likelihood evaluation
        try:
            ll = self.nLLMonteCarloCausal(test_params, groupedData)
            print(f"Test likelihood: {ll}")
            if ll >= 1e10:
                print("ERROR: Likelihood evaluation returned error value")
                return False
        except Exception as e:
            print(f"ERROR in likelihood evaluation: {e}")
            return False
            
        print("=== ALL COMPONENTS TESTED SUCCESSFULLY ===")
        return True
    
    # likelihood function
    def unimodalLikelihood(self, S, sigma):
        m = np.linspace(0, S + 10*sigma, 500)
        p_m = norm.pdf(m, loc=S, scale=sigma)
        return m, p_m

    # probability density function of a Gaussian distribution
    def gaussianPDF(self, x, S, sigma):
        return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-((x-S)**2)/(2*(sigma**2)))

    # Fusion function
    def fusionAV(self, m_a, m_v, sigma_av_a, sigma_av_v):
        J_AV_A = 1 / sigma_av_a**2
        J_AV_V = 1 / sigma_av_v**2
        w_a = J_AV_A / (J_AV_A + J_AV_V)
        w_v = 1 - w_a
        fused_S_av = w_a * m_a + w_v * m_v
        sigma_S_AV_hat = np.sqrt(1 / (J_AV_A + J_AV_V))
        return fused_S_av

    def fusionAV_vectorized(self, m_a, m_v, sigma_av_a, sigma_av_v):
        J_AV_A = 1 / sigma_av_a**2
        J_AV_V = 1 / sigma_av_v**2
        fused_S_av = (J_AV_A * m_a + J_AV_V * m_v) / (J_AV_A + J_AV_V)
        return fused_S_av



   # Vectorized causal inference functions 
    def p_single(self,m,sigma,t_min,t_max):
        """p(m | C=2)     and Gaussian measurement noise N(m; y, sigma^2). 
        and Gaussian measurement noise N(m; y, sigma^2)."""
        hi_cdf= norm.cdf((t_max - m) /sigma)
        lo_cdf=norm.cdf((t_min-m)/sigma)
        return (hi_cdf-lo_cdf)/(t_max-t_min)
        
    def L_C2(self, m_a,m_v,sigma_a,sigma_v,t_min,t_max):
        """ Likelihood of separate sources: product of two marginal likelihoods 
        two integral over two hidden duration y_a y_v"""

        return self.p_single(m_a,sigma_a,t_min,t_max) * self.p_single(m_v,sigma_v,t_min,t_max)


    def L_C1(self,m_a,m_v,sigma_a,sigma_v,t_min,t_max):

        sigma_c_sq = (sigma_a**2 * sigma_v**2) / (sigma_a**2 + sigma_v**2)
        sigma_c = np.sqrt(sigma_c_sq)
        mu_c = (m_a / sigma_a**2 + m_v / sigma_v**2) / (1 / sigma_a**2 + 1 / sigma_v**2)

        hi_cdf = norm.cdf((t_max-mu_c)/sigma_c)
        lo_cdf = norm.cdf((t_min-mu_c)/sigma_c)
        
        expo = np.exp(-(m_a-m_v)**2/(2*(sigma_a**2+sigma_v**2)))
        
        prior = 1/(t_max-t_min)

        if self.modelName == "logLinearMismatch":
            # If the model is log mis, we need to adjust the prior and it should be numerically integrated
            # over the log-transformed durations
            
            #.if m_a and m_v are in linear scale y_vals should be in linear scale as well
            y_vals = np.linspace(t_min, t_max, self.nSimul) 
            dy=y_vals[1] - y_vals[0]
            log_norm_const=np.log(t_max / t_min)

            # likelihoods under common cause    
            L_m_a=  norm.pdf(m_a, loc=y_vals, scale=sigma_a)  # shape: (n_points,)
            L_m_v = norm.pdf(m_v, loc=y_vals, scale=sigma_v)

            prior = 1/ ((y_vals +1e-10)*log_norm_const)  # Adjusted prior for log mis model
            # Calculate the integral over the log-transformed durations
            integrand=L_m_a * L_m_v * prior
            integral = np.sum(integrand*dy)
            integral = max(integral, 1e-10)
            return integral
                
        return prior * sigma_c/np.sqrt(sigma_a**2 * sigma_v**2) * (hi_cdf-lo_cdf) * expo


    def posterior_C1(self,m_a,m_v,sigma_a,
                                    sigma_v, p_c,
                                    t_min,t_max):

        #likelihoods
        L1 = self.L_C1(m_a, m_v, sigma_a, sigma_v, t_min, t_max)  
        L2 = self.L_C2(m_a, m_v, sigma_a, sigma_v, t_min, t_max)  # Fixed: consistent parameter order
        
        # posterior with numerical stability
        denominator = L1*p_c + L2*(1-p_c)
        
        # Handle both scalar and array cases
        if np.isscalar(denominator):
            if denominator == 0:
                postC1 = p_c
            else:
                postC1 = L1*p_c / denominator
        else:
            # Array case - use np.where to handle element-wise
            postC1 = np.where(denominator == 0, p_c, L1*p_c / denominator)
        
        return postC1
            
            


    def causalInference_vectorized(self, m_a, m_v, sigma_a, sigma_v, p_c, t_min, t_max):
        fused_S_av = self.fusionAV_vectorized(m_a, m_v, sigma_a, sigma_v)
        # Calculate likelihoods
        post_C1 = self.posterior_C1(m_a, m_v, sigma_a, sigma_v, p_c, t_min, t_max)

        # convert back to linear scale if measurements are in log scale
        if self.modelName == "lognorm" or self.modelName == "logLinearMismatch":
            m_a = np.exp(m_a)
            fused_S_av = np.exp(fused_S_av)
        

        final_estimate = post_C1 * fused_S_av + (1 - post_C1) * m_a
        return final_estimate
    
    
    def probTestLonger_vectorized_mc(self, trueStims, sigma_av_a, sigma_av_v, p_c, lambda_, t_min, t_max):
        if self.modelName == "gaussian":
            #print("Using Gaussian distribution for measurements")
            nSimul = self.nSimul
            S_a_s, S_a_t, S_v_s, S_v_t = trueStims # S is a single value m is random sampled array
            m_a_s = np.random.normal(S_a_s, sigma_av_a, nSimul)
            m_v_s = np.random.normal(S_v_s, sigma_av_v, nSimul)
            m_a_t = np.random.normal(S_a_t, sigma_av_a, nSimul)
            m_v_t = np.random.normal(S_v_t, sigma_av_v, nSimul)

            est_standard = self.causalInference_vectorized(m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c, t_min, t_max)
            # est_test = self.fusionAV_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v)
            ## or est_test computed using causalInference_vectorized
            est_test = self.causalInference_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v, p_c, t_min, t_max)

        # === Lognormal (Log-Space Gaussian Measurement) Model === #
        elif self.modelName == "lognorm":
            #print("Using lognormal distribution for measurements")
            nSimul = self.nSimul
            S_a_s, S_a_t, S_v_s, S_v_t = trueStims
            m_a_s = np.random.normal(loc=np.log(S_a_s), scale=sigma_av_a, size=nSimul)
            m_v_s = np.random.normal(loc=np.log(S_v_s), scale=sigma_av_v, size=nSimul)
            m_a_t = np.random.normal(loc=np.log(S_a_t), scale=sigma_av_a, size=nSimul)
            m_v_t = np.random.normal(loc=np.log(S_v_t), scale=sigma_av_v, size=nSimul)
            est_standard = self.causalInference_vectorized(m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c, np.log(t_min), np.log(t_max))
            est_test = self.causalInference_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v, p_c, np.log(t_min), np.log(t_max))

        #  === Log-Linear Mismatch Model === #
        elif self.modelName =="logLinearMismatch":
            """
            Main: Observer’s true measurements follow log-normal noise  but the observer assumes additive (normal) noise in linear time.
            Q: If the brain is fundamentally encoding durations in a nonlinear (log) fashion, 
            but using incorrect inference mechanisms — how will behavior deviate from optimality?
            """
            nSimul = self.nSimul
            S_a_s, S_a_t, S_v_s, S_v_t = trueStims
            # measurements are log-normal but the observer assumes normal noise
            m_a_s = np.exp(np.random.normal(loc=np.log(S_a_s), scale=sigma_av_a, size=nSimul)) # so that measurements are log-normal
            m_v_s = np.exp(np.random.normal(loc=np.log(S_v_s), scale=sigma_av_v, size=nSimul))
            m_a_t = np.exp(np.random.normal(loc=np.log(S_a_t), scale=sigma_av_a, size=nSimul))
            m_v_t = np.exp(np.random.normal(loc=np.log(S_v_t), scale=sigma_av_v, size=nSimul))
            est_standard = self.causalInference_vectorized(m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c, t_min, t_max)
            est_test = self.causalInference_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v, p_c, t_min, t_max)

        elif self.modelName == "fusionOnly":# Linear-Gaussian observer (classic model)
            nSimul = self.nSimul
            S_a_s, S_a_t, S_v_s, S_v_t = trueStims
            m_a_s = np.random.normal(S_a_s, sigma_av_a, nSimul)
            m_v_s = np.random.normal(S_v_s, sigma_av_v, nSimul)
            m_a_t = np.random.normal(S_a_t, sigma_av_a, nSimul)
            m_v_t = np.random.normal(S_v_t, sigma_av_v, nSimul)
            est_standard = self.fusionAV_vectorized(m_a_s, m_v_s, sigma_av_a, sigma_av_v)
            est_test = self.fusionAV_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v)

        elif self.modelName ==  "fusionOnlyLogNorm":## FusionOnlyLogNorm: Log-space observer with Gaussian noise in log space
            nSimul = self.nSimul
            S_a_s, S_a_t, S_v_s, S_v_t = trueStims
            m_a_s = np.random.normal(loc=np.log(S_a_s), scale=sigma_av_a, size=nSimul)
            m_v_s = np.random.normal(loc=np.log(S_v_s), scale=sigma_av_v, size=nSimul)
            m_a_t = np.random.normal(loc=np.log(S_a_t), scale=sigma_av_a, size=nSimul)
            m_v_t = np.random.normal(loc=np.log(S_v_t), scale=sigma_av_v, size=nSimul)
            est_standard = self.fusionAV_vectorized(m_a_s, m_v_s, sigma_av_a, sigma_av_v)
            est_test = self.fusionAV_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v)




        else:
            #break and raise error
            raise ValueError("Invalid modelName. Choose 'gaussian', 'lognorm', or 'logLinearMismatch'.")


        p_base = np.mean(est_test > est_standard)
        p_final = (1 - lambda_) * p_base + lambda_ / 2
        return p_final
    
    
    def nLLMonteCarloCausal(self, params, groupedData):

        """Negative log-likelihood for causal inference model"""
        ll = 0
        lenData = len(groupedData)
        
        # Check for invalid parameters
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            print(f"Invalid params: {params}")
            return 1e10
        
        # Additional validation for t_min and t_max
        try:
            # Get first condition to extract t_min and t_max
            first_snr = groupedData["audNoise"].iloc[0]
            first_conflict = groupedData["conflictDur"].iloc[0]
            λ, σa, σv, pc, t_min, t_max = self.getParamsCausal(params, first_snr, first_conflict)
            
            if t_min >= t_max:
                print(f"Invalid bounds: t_min={t_min:.3f} >= t_max={t_max:.3f}")
                return 1e10
            if t_min <= 0 or t_max <= 0:
                print(f"Non-positive bounds: t_min={t_min:.3f}, t_max={t_max:.3f}")
                return 1e10
                
        except Exception as e:
            print(f"Error extracting parameters: {e}")
            return 1e10
        
        for i in range(lenData):
            currSNR = groupedData["audNoise"].iloc[i]
            currConflict = groupedData["conflictDur"].iloc[i]
            currResp = groupedData['num_of_chose_test'].iloc[i]
            totalResponses = groupedData['total_responses'].iloc[i]

            # Get the parameters for the current condition
            try:
                lambda_, sigma_av_a, sigma_av_v, p_c, t_min, t_max = self.getParamsCausal(params, currSNR, currConflict)
            except Exception as e:
                print(f"Error in getParamsCausal for condition {i}: {e}")
                return 1e10

            # Get the true standard and test durations
            S_a_s = groupedData["standardDur"].iloc[i]
            S_v_s = groupedData["unbiasedVisualStandardDur"].iloc[i]
            S_a_t = groupedData["testDurS"].iloc[i]
            S_v_t = groupedData["unbiasedVisualTestDur"].iloc[i]
            trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
            
            # Calculate the probability of choosing the test duration being longer than the standard duration
            try:
                p_test_longer = self.probTestLonger_vectorized_mc(trueStims, sigma_av_a, sigma_av_v, p_c, lambda_, t_min, t_max)
                if np.isnan(p_test_longer) or np.isinf(p_test_longer):
                    print(f"Invalid probability at condition {i}: {p_test_longer}")
                    return 1e10
            except Exception as e:
                print(f"Error in probTestLonger_vectorized_mc for condition {i}: {e}")
                return 1e10
                
            # Calculate the likelihood for the current condition
            # add epsilon to avoid log(0)
            epsilon = 1e-10
            P = np.clip(p_test_longer, epsilon, 1 - epsilon)
            ll += np.log(P) * currResp + np.log(1 - P) * (totalResponses - currResp)
            
        # Check for invalid likelihood
        if np.isnan(ll) or np.isinf(ll):
            print(f"Invalid final log-likelihood: {ll}")
            return 1e10
            
        # Return the negative log-likelihood
        return -ll


    def fitCausalInferenceMonteCarlo(self, groupedData):
        """
        Fit causal inference model using Monte Carlo simulation with multiple random starts.
        Supports 'scipy' (default) or 'bads' optimization (if installed).
        """
        # Debug: Print data bounds information
        print(f"Data bounds: t_min={self.data_t_min:.3f}, t_max={self.data_t_max:.3f}")
        
        # Test components before starting optimization
        if not self.test_fitting_components(groupedData):
            print("ERROR: Component test failed. Aborting optimization.")
            return None
        
        # ---- PARAMETER BOUNDS ----
        if self.freeP_c:
            print("Fitting with free p_c parameters for each SNR condition.")
            bounds = np.array([
                (0, 0.4),      # 0 lambda_
                (0.1, 1.2),     # 1 sigma_av_a_1
                (0.1, 1.2),     # 2 sigma_av_v_1
                (0.001, 0.95), # 3 p_c_1
                (0.1, 1.7),     # 4 sigma_av_a_2
                (0, 0.4),      # 5 lambda_2
                (0, 0.4),      # 6 lambda_3
                (0.001, 0.95), # 7 p_c_2
                (0, 1), # 8 t_min (reasonable lower bound, must be less than data min)
                (max(0.05, self.data_t_max + 1), 10.0),  # 9 t_max (must be greater than data max)
            ])

        elif self.freeP_c==False:
            print("Fitting with shared p_c parameter across SNR conditions.")   
            bounds = np.array([
                (0, 0.25),      # 0 lambda_
                (0.1, 1.2),     # 1 sigma_av_a_1
                (0.1, 1.2),     # 2 sigma_av_v_1
                (0.001, 0.95),  # 3 p_c_1
                (0.1, 1.7),     # 4 sigma_av_a_2
                (0, 0.4),      # 5 lambda_2
                (0, 0.4),      # 6 lambda_3
                (0, 1), # 7 t_min (reasonable lower bound, must be less than data min)
                (0.05,max(self.data_t_max+1, 10.0)),  # 8 t_max (must be greater than data max)
            ])

        if self.sharedLambda:
            bounds = np.delete(bounds, [5,6], axis=0)
        
        # Debug: Print bounds information
        print(f"Bounds shape: {bounds.shape}")
        print(f"t_min bounds: {bounds[-2]}")
        print(f"t_max bounds: {bounds[-1]}")
        
        # Validate bounds
        if bounds[-1][0] <= bounds[-2][1]:
            print("WARNING: t_max lower bound <= t_min upper bound, adjusting...")
            bounds[-1] = (bounds[-2][1] + 0.1, bounds[-1][1])  # Ensure t_max_min > t_min_max
            
        # Test the likelihood function with reasonable parameters before optimization
        print("Testing likelihood function with reasonable parameters...")
        test_params = np.array([0.1, 0.5, 0.5, 0.5, 0.8])  # Basic parameters
        if not self.sharedLambda:
            test_params = np.append(test_params, [0.1, 0.1])  # Add lambda2, lambda3
        if self.freeP_c:
            test_params = np.append(test_params, [0.5])  # Add second p_c
        # Add t_min and t_max
        test_params = np.append(test_params, [0.2, 1.0])
        
        try:
            test_ll = self.nLLMonteCarloCausal(test_params, groupedData)
            print(f"Test likelihood evaluation: {test_ll}")
            if test_ll >= 1e10:
                print("ERROR: Test likelihood evaluation failed!")
                return None
        except Exception as e:
            print(f"ERROR: Test likelihood evaluation threw exception: {e}")
            return None
        
   

        # Initial best results
        best_result = None
        best_ll = np.inf
        nStart = self.nStart if hasattr(self, 'nStart') else 1
        optimizer = self.optimizationMethod if hasattr(self, 'optimizationMethod') else 'scipy'
        print(f"\nStarting {nStart} optimization attempts using '{optimizer}'...")
        print("Model is " + self.modelName)
        for attempt in tqdm(range(nStart), desc="Optimization Attempts"):
            # Random x0 initialization within bounds
            if self.freeP_c==False:
                x0 = np.array([
                    np.random.uniform(0.01, 0.25),  # 0 lambda_
                    np.random.uniform(0.1, 1.2),    # 1 sigma_av_a_1
                    np.random.uniform(0.1, 1.2),    # 2 sigma_av_v_1
                    np.random.uniform(0.1, 0.8),   # 3 p_c general
                    np.random.uniform(0.1, 1.7),    # 4 sigma_av_a_2
                    np.random.uniform(0.01, 0.25),  # 5 lambda_2
                    np.random.uniform(0.01, 0.25),  # 6 lambda_3
                    np.random.uniform(bounds[-2][0], bounds[-2][1]),  # 7 t_min (from bounds)
                    np.random.uniform(bounds[-1][0], bounds[-1][1]),  # 8 t_max (from bounds)
                ])
            elif self.freeP_c:
                x0 = np.array([
                    np.random.uniform(0.01, 0.25),  # 0 lambda_
                    np.random.uniform(0.1, 1.2),    # 1 sigma_av_a_1
                    np.random.uniform(0.1, 1.2),    # 2 sigma_av_v_1
                    np.random.uniform(0.1, 0.8),   # 3 p_c_1
                    np.random.uniform(0.1, 1.7),    # 4 sigma_av_a_2
                    np.random.uniform(0.01, 0.25),  # 5 lambda_2
                    np.random.uniform(0.01, 0.25),  # 6 lambda_3
                    np.random.uniform(0.1, 0.8),   # 7 p_c_2
                    np.random.uniform(bounds[-2][0], bounds[-2][1]),  # 8 t_min (from bounds)
                    np.random.uniform(bounds[-1][0], bounds[-1][1]),  # 9 t_max (from bounds)
                ])

            # if lambda is shared across conditions, remove lambda_2 and lambda_3 from x0 and bounds
            if self.sharedLambda:
                x0= np.delete(x0, [5,6])  # remove lambda_2 and lambda_3 if sharedLambda is True
            
            # Debug: Print initial parameters
            print(f"Attempt {attempt + 1}: x0 shape={x0.shape}, bounds shape={bounds.shape}")
            print(f"x0 t_min={x0[-2]:.3f}, t_max={x0[-1]:.3f}")
            
            # Validate x0 vs bounds
            for i, (x_val, (lb, ub)) in enumerate(zip(x0, bounds)):
                if not (lb <= x_val <= ub):
                    print(f"WARNING: x0[{i}]={x_val:.3f} outside bounds [{lb:.3f}, {ub:.3f}]")
                    x0[i] = np.clip(x_val, lb, ub)  # Fix out-of-bounds values

            try:
                if self.optimizationMethod == "bads":
                    # Prepare bounds
                    lb = bounds[:, 0]
                    ub = bounds[:, 1]
                    plb = bounds[:, 0] + 0.1 * (bounds[:, 1] - bounds[:, 0])
                    pub = bounds[:, 1] - 0.1 * (bounds[:, 1] - bounds[:, 0])

                    obj = lambda x: self.nLLMonteCarloCausal(x, groupedData)
                    bads = BADS(obj, x0, lb, ub, plb, pub, options={"display": "off"})
                    result = bads.optimize()  # returns OptimizeResult object

                else:
                    # Default to scipy
                    result = minimize(
                        self.nLLMonteCarloCausal,
                        x0=x0,
                        args=(groupedData),
                        method='Powell',
                        bounds=bounds
                    )
                # Check result attribute for best LL
                fval = getattr(result, 'fval', result.fun)
                xres = getattr(result, 'x', result.x)
                if fval < best_ll:
                    best_ll = fval
                    best_result = result

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue

        if best_result is None:
            raise RuntimeError("All optimization attempts failed!")

        fval = getattr(best_result, 'fval', best_result.fun)
        xres = getattr(best_result, 'x', best_result.x)
        print(f"\n✅ Best result from {nStart} attem           spts:")
        print(f"  → Final parameters: {xres}")
        print(f"  → Final log-likelihood: {fval:.6f}")

        return xres


    def simulateMonteCarloData(self, fittedParams, data, nSamples=10000):        
        simData = []
        # Extract unique combinations of trials from the dataset
    
        for _, trial in self.groupedData.iterrows():
            deltaDurS = trial["deltaDurS"]
            audioNoiseLevel = trial["audNoise"]
            conflictLevel = trial["conflictDur"]
            totalResponses = trial["total_responses"]
            #print(f"Simulating for audNoise: {audioNoiseLevel}, conflict: {conflictLevel}, deltaDurS: {deltaDurS}, totalResponses: {totalResponses}")

  
            # Unpack fitted parameters for the current audio noise level
            lambda_, sigma_av_a, sigma_av_v, p_c, t_min, t_max = self.getParamsCausal(fittedParams, audioNoiseLevel, conflictLevel)

            nSamples = 30 * int(totalResponses) #10* int(totalResponses)  # Scale number of samples by total responses for better simulation
            # Simulate responses for the current trial
            for _ in range(nSamples):
                S_a_s = 0.5
                S_v_s = S_a_s + conflictLevel
                S_a_t = S_a_s + deltaDurS
                S_v_t = S_a_t
                trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
                p_test_longer = self.probTestLonger_vectorized_mc(trueStims, sigma_av_a, sigma_av_v, p_c, lambda_, t_min, t_max)
                chose_test = np.random.binomial(1, p_test_longer)
                simData.append({
                    'standardDur': S_a_s,
                    'testDurS': S_a_t,
                    'deltaDurS': deltaDurS,
                    'logDeltaDur': np.log(S_a_t) - np.log(S_a_s),
                    'logDeltaDurMs': np.log(S_a_t * 1000) - np.log(S_a_s * 1000),
                    'unbiasedVisualStandardDur': S_v_s,
                    'unbiasedVisualTestDur': S_v_t,
                    'audNoise': audioNoiseLevel,
                    'conflictDur': conflictLevel,
                    'chose_test': chose_test,
                    'chose_standard': 1 - chose_test,  # Assuming binary choice
                    'responses': 1  # Each sample is a response
                })

        simData = pd.DataFrame(simData)
        self.simulatedData = simData
        return self.simulatedData


    def plot_posterior_vs_conflict(self, data, fittedParams, snr_list=[1.2, 0.1]):
        """
        Plot posterior probability vs conflict for given SNR values.
        """
        best_params = fittedParams
        posterior_by_condition = []

        # Get unique conflict and SNR combinations
        unique_conflicts = np.sort(data["conflictDur"].unique())
        for snr in snr_list:
            for conflict in unique_conflicts:
                λ, σa, σv, pc, t_min, t_max = self.getParamsCausal(best_params, snr, conflict)

                S_std = 0.5
                S_v = S_std + conflict
                m_a_samples, m_v_samples = None, None

                if self.modelName in ["lognorm"]:
                    m_a_samples = np.random.normal(loc=np.log(S_std), scale=σa, size=1000)
                    m_v_samples = np.random.normal(loc=np.log(S_v), scale=σv, size=1000)
                elif self.modelName in ["gaussian"]:
                    m_a_samples = np.random.normal(loc=S_std, scale=σa, size=1000)
                    m_v_samples = np.random.normal(loc=S_v, scale=σv, size=1000)
                elif self.modelName in ["logLinearMismatch"]:
                    m_a_samples = np.exp(np.random.normal(loc=np.log(S_std), scale=σa, size=1000))
                    m_v_samples = np.exp(np.random.normal(loc=np.log(S_v), scale=σv, size=1000))
                else:
                    raise ValueError("Invalid modelName. Choose 'gaussian', 'lognorm', or 'logLinearMismatch'.")

                posteriors = np.array([
                    self.posterior_C1(m_a, m_v, σa, σv, pc, t_min, t_max)
                    for m_a, m_v in zip(m_a_samples, m_v_samples)
                ])

                avg_posterior = np.mean(posteriors)
                posterior_by_condition.append({
                    "conflict_ms": conflict * 1000,
                    "posterior": avg_posterior,
                    "SNR": snr,
                    "prior_pc": pc
                })

        # Convert to DataFrame for easy plotting
        posterior_df = pd.DataFrame(posterior_by_condition)

        # Plotting
        plt.figure(figsize=(8, 5))
        for idx, snr in enumerate(snr_list):
            df_snr = posterior_df[posterior_df["SNR"] == snr]
            plt.subplot(1, 2, idx + 1)
            plt.scatter(df_snr["conflict_ms"], df_snr["posterior"], label=f"Posterior P(C=1) (SNR={snr})")
            plt.axhline(y=df_snr["prior_pc"].iloc[0], color='gray', linestyle='--',
                        label=f"P(C=1)={df_snr['prior_pc'].iloc[0]:.2f}")
            plt.xlabel('Conflict (ms)')
            plt.ylabel('Posterior Probability of Common Cause')
            plt.title(f'Posterior P(C=1) vs Conflict (SNR={snr})')
            plt.grid()
            plt.legend()

        plt.tight_layout()
        plt.show()



    def plot_mu_vs_conflict_MC_vs_Data(self):
        plt.figure(figsize=(16, 6))
        for i, standardLevel in enumerate(self.uniqueStandard):
            for j, audioNoiseLevel in enumerate(sorted(self.uniqueSensory)):
                for k, conflictLevel in enumerate(self.uniqueConflict):
                    plt.subplot(1, 2, j + 1)
                    color = sns.color_palette("viridis", as_cmap=True)(k / len(self.uniqueConflict))
                    
                    paramsSimDf = self.getParams(self.simDataFit.x, conflictLevel, audioNoiseLevel)  # lambda mu sigma
                    muModel = paramsSimDf[1]
                    muData = self.getParams(self.dataFit.x, conflictLevel, audioNoiseLevel)[1]
                    plt.scatter(conflictLevel, muData, color="red", s=40, alpha=0.7, label="Data Fit" if k == 0 else "")
                    plt.scatter(conflictLevel, muModel, color="blue", s=40,  alpha=0.7, label="Model Fit" if k == 0 else "")
                    plt.xlabel(f"Conflict (ms)")
                    plt.ylabel("Mu (sigma_av_a)")
                    plt.axhline(y=0, color='gray', linestyle='--')
                    plt.axvline(x=0, color='gray', linestyle='--')
                    plt.ylim(-0.3, 0.3)
                    plt.title(f"{self.dataName} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
                    plt.legend(fontsize=14, title_fontsize=14)
                    plt.grid()
        plt.show()


    def plot_mu_summary_with_errorbars(self, fittedParams, error_type='sem'):
        """
        Plots the average μ (PSE) across participants for each conflict and noise level,
        including error bars based on standard error or confidence intervals.
        """
        #ensure groupedData has participantID column
        if 'participantID' not in self.groupedData.columns:
            groupedData = self.groupByChooseTest(self.data, [self.intensityVar, self.sensoryVar, self.standardVar, self.conflictVar, self.visualStandardVar, self.visualTestVar, self.audioTestVar, 'participantID'])
            self.groupedData = groupedData  
            
        
        conflictLevels = np.sort(self.groupedData[self.conflictVar].unique())
        noiseLevels = np.sort(self.groupedData[self.sensoryVar].unique())

        plt.figure(figsize=(12, 5))

        for j, noiseLevel in enumerate(noiseLevels):
            mus_by_conflict = []

            for conflict in conflictLevels:
                mus = []

                for participant in self.groupedData['participantID'].unique():
                    subset = self.groupedData[
                        (self.groupedData[self.conflictVar] == conflict) &
                        (self.groupedData[self.sensoryVar] == noiseLevel) &
                        (self.groupedData['participantID'] == participant)
                    ]
                    if len(subset) == 0:
                        continue

                    # Extract PSE (mu) from psychometric fit of simulated or real data
                    # You may replace this with model prediction if simDataFit is None
                    try:
                        mu = self.getParams(self.simDataFit.x, conflict, noiseLevel)[1]
                        mus.append(mu)
                    except:
                        continue

                # Compute mean and error bars
                mu_mean = np.mean(mus)
                if error_type == 'sem':
                    mu_err = np.std(mus) / np.sqrt(len(mus))
                elif error_type == 'ci95':
                    mu_err = 1.96 * np.std(mus) / np.sqrt(len(mus))
                else:
                    mu_err = 0

                mus_by_conflict.append((conflict * 1000, mu_mean, mu_err))

            # Unpack values
            conflicts_ms, mu_means, mu_errs = zip(*mus_by_conflict)
            plt.subplot(1, 2, j + 1)
            plt.errorbar(conflicts_ms, mu_means, yerr=mu_errs, fmt='-o', capsize=4, label=f"SNR={noiseLevel}")
            plt.axhline(0, color='gray', linestyle='--')
            plt.xlabel("Conflict (ms)")
            plt.ylabel("Estimated μ (PSE)")
            plt.title(f"Mean μ vs Conflict (Noise={noiseLevel})")
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()

    def plotPsychometrics_MC_Data(self):
        "use self to get the  required stuff"
        print("Plotting psychometric curves for Monte Carlo model and data...")
        pltTitle = self.dataName + " "+ self.modelName+" Model Fit"
        plt.figure(figsize=(16, 8))
        #plt.title(f"{pltTitle} AV,A duration comparison", fontsize=26)
        plt.xlabel("Test Duration (ms)", fontsize=24)
        plt.ylabel("P(chose test)",fontsize=24)

        for i, standardLevel in enumerate(self.uniqueStandard):
            for j, audioNoiseLevel in enumerate(sorted(self.uniqueSensory)):
                for k, conflictLevel in enumerate(self.uniqueConflict):
                    plt.subplot(1, 2, j + 1)
                    x = np.linspace(-0.6, 0.6, 1000)
                    color = sns.color_palette("RdBu", as_cmap=True)(k / len(self.uniqueConflict))
                    

                    paramsSimFit=self.getParams(self.simDataFit.x, conflictLevel,audioNoiseLevel)
                    # Plot data fit: plot psychometric curve from fitted data
                    ySimSigmoid=self.psychometric_function(x, paramsSimFit[0],paramsSimFit[1],paramsSimFit[2])

                    
                    xPlot=np.linspace(0,1000,1000)
                    if k==0:
                        plt.plot(xPlot, ySimSigmoid, color=color, linestyle='-', label=f"conf: {int(conflictLevel*1000)}", linewidth=3, alpha=0.8)
                    else:
                        plt.plot(xPlot, ySimSigmoid, color=color, linestyle='-', linewidth=3, alpha=0.8,label=f"{int(conflictLevel*1000)}")
                    plt.axvline(500, linestyle='--')


                    "plot the monte carlo"
                    lambda_, sigma_av_a, sigma_av_v, p_c ,tmin,tmax= self.getParamsCausal(self.modelFit, audioNoiseLevel, conflictLevel)

            
                    #plt.axvline(x=0, color='gray', linestyle='--')
                    #plt.axhline(y=0.5, color='gray', linestyle='--')
                    #plt.xlabel(f"({self.intensityVar}) Test(stair-a)-Standard(a) Duration Difference Ratio(%)")
                    #plt.xlabel("Test Duration (ms)", fontsize=24)
                    #plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=26)
                    if j==0:
                        plt.legend(fontsize=14, title_fontsize=20)
                    plt.grid()


                    groupedDataSub = self.groupByChooseTest(
                        self.data[(self.data[self.standardVar] == standardLevel) & (self.data[self.sensoryVar] == audioNoiseLevel) & (self.data[self.conflictVar] == conflictLevel)],
                        [self.intensityVar, self.sensoryVar, self.standardVar, self.conflictVar, self.visualStandardVar, self.visualTestVar, self.audioTestVar, "testDurMs"]
                    )
                    bin_and_plot(groupedDataSub, bin_method='cut', bins=10, plot=True, color=color,binVar='testDurMs')
                    #plt.text(0.05, 0.8, fr"$\sigma_a$: {sigma_av_a:.2f}, $\sigma_v$: {sigma_av_v:.2f},", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
                    plt.tight_layout()
                    #plt.grid(True)
                    print(f"Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, Lambda: {lambda_:.3f}, Sigma_a: {sigma_av_a:.3f}, Sigma_v: {sigma_av_v:.3f}, p_c: {p_c:.3f}")
                    #plt.text(0.15, 0.9, f"P(C=1): {p_c:.2f}", fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
                    
        plt.show()



"TEST CODE"
##############################

# ----------------------------------------------
# This code is part of the Causal Inference Monte Carlo fitting process.
# --- <Main Function> ---
# It loads data, groups it by conditions, and fits the model using Monte Carlo simulation.
# The fitted parameters are then used to plot the results.
# ----------------------------------------------

##############################

from loadData import loadData
import time

if __name__ == "__main__":
    # Example usage
    data, dataName = loadData("mt_all.csv")

    intensityVariable = "deltaDurS"
    sensoryVar = "audNoise"
    standardVar = "standardDur"
    conflictVar = "conflictDur"

    visualStandardVar = "unbiasedVisualStandardDur"
    visualTestVar = "unbiasedVisualTestDur"
    audioStandardVar = "standardDur"
    audioTestVar = "testDurS"


    # Instantiate the Monte Carlo class
    mc_fitter = OmerMonteCarlo(
        data,
        intensityVar=intensityVariable
        # sensoryVar=sensoryVar,
        # standardVar=standardVar,
        # conflictVar=conflictVar
    )
    mc_fitter.dataName = dataName
    mc_fitter.nSimul = 100
    mc_fitter.optimizationMethod= "normal"  # Use BADS for optimization
    mc_fitter.nStart = 1  # Number of random starts for optimization


    groupedData = mc_fitter.groupByChooseTest(
        x=data,
        groupArgs=[
            intensityVariable, sensoryVar, standardVar, conflictVar,
            visualStandardVar, visualTestVar, audioTestVar
        ]
    )

    

    mc_fitter.modelName = "lognorm"  # Set measurement distribution to Gaussian
    timeStart = time.time()
    print(f"\nFitting Causal Inference Model for {dataName} with {len(groupedData)} unique conditions")
    fittedParams = mc_fitter.fitCausalInferenceMonteCarlo(groupedData)
    mc_fitter.modelFit = fittedParams  # Store the fitted parameters in the class instance
    print(f"\nFitted parameters for {dataName}: {fittedParams}")
    print(f"Time taken to fit: {time.time() - timeStart:.2f} seconds")

    # uniqueStandard = groupedData[standardVar].unique()
    # uniqueSensory = groupedData[sensoryVar].unique()
    # uniqueConflict = sorted(groupedData[conflictVar].unique())

    # simulate data for psychometric curve
    mc_fitter.simulatedData = mc_fitter.simulateMonteCarloData(fittedParams, mc_fitter.data ,nSamples=100)
    # Create a temporary fitter for simulated data
    temp_fitter = fitPychometric(mc_fitter.simulatedData)
    mc_fitter.simDataFit = temp_fitter.fitMultipleStartingPoints(nStart=1)

    "psychometric fit"
    mc_fitter.dataFit = mc_fitter.fitMultipleStartingPoints(nStart=1)


    # Plotting the results
    mc_fitter.plotPsychometrics_MC_Data()


                




    # Plot posterior vs conflict
    mc_fitter.plot_posterior_vs_conflict(data, fittedParams, snr_list=[1.2, 0.1])
    # Simulate and plot psychometric data
    uniqueSensory = np.unique(data[sensoryVar])
    uniqueConflict = np.unique(data[conflictVar])







