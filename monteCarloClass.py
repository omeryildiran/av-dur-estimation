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
    
    def __init__(self, data, intensityVar='deltaDurS', allIndependent=True, sharedSigma=False, sensoryVar='audNoise', 
                 standardVar='standardDur', conflictVar='conflictDur',dataName=None):
        super().__init__(data, intensityVar, allIndependent, sharedSigma, sensoryVar, standardVar, conflictVar,dataName) 
        
        self.fitType = 'Monte Carlo'
        self.nStart = 1  # Number of random starts for optimization
        self.nSimul = 10  # Number of simulations for Monte Carlo approximation
        self.optimizationMethod = 'BADS'  # Optimization method for fitting
        self.modelFit = None  # Placeholder for fitted model
        self.simulatedData = None  # Placeholder for simulated data
        self.dataFit = None  # Placeholder for fitted data
        self.simDataFit = None  # Placeholder for simulated data fit
        self.groupedData = None  # Placeholder for grouped data
        self.mDist = "gaussian"  # Distribution of measurements, can be 'gaussian' or 'lognormal'
        
    
        
        self.visualStandardVar = "unbiasedVisualStandardDur"
        self.visualTestVar = "unbiasedVisualTestDur"
        self.audioTestVar = "testDurS"
        self.dataName = dataName if dataName else "default_data"
        self.data = data
        self.sharedSigma_v=True

        self.groupedData= self.groupByChooseTest(x=data,
                groupArgs=[
                self.intensityVar, sensoryVar, standardVar, conflictVar,
                self.visualStandardVar, self.visualTestVar, self.audioTestVar
                    ]
                    )

    def getParamsCausal(self,params,SNR):
        """Extract causal inference parameters for a specific condition (conflict, noise)."""

        lambda_=params[0]
        if np.isclose(SNR, 0.1):
            sigma_av_a=params[1]
            sigma_av_v=params[2]
            p_c=params[3]
        elif np.isclose(SNR,1.2):
            sigma_av_a=params[4]
            sigma_av_v=params[5]
            p_c=params[6]

        if self.sharedSigma_v:
            sigma_av_v = params[2]  # Use the shared sigma for visual noise
            params[5] =params[2]  # Update the second sigma_av_v parameter for the second SNR condition 

        return lambda_,sigma_av_a,sigma_av_v,p_c
    
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
        w_a = J_AV_A / (J_AV_A + J_AV_V)
        w_v = 1 - w_a
        fused_S_av = w_a * m_a + w_v * m_v
        return fused_S_av

    # Likelihood under common cause
    def likelihood_C1(self, m_a, m_v, sigma_av_a, sigma_av_v):
        var_sum = sigma_av_a**2 + sigma_av_v**2
        return (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))

    # Likelihood under independent causes
    def likelihood_C2(self, m_a, m_v, S_a, S_v, sigma_av_a, sigma_av_v):
        return norm.pdf(m_a, loc=S_a, scale=sigma_av_a) * norm.pdf(m_v, loc=S_v, scale=sigma_av_v)

    # Posterior of common cause
    def posterior_C1(self, likelihood_c1, likelihood_c2, p_c):
        return (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))

    def causalInference(self, S_a, S_v, m_a, m_v, sigma_av_a, sigma_av_v, p_c):
        likelihood_c1 = self.likelihood_C1(m_a, m_v, sigma_av_a, sigma_av_v)
        likelihood_c2 = self.likelihood_C2(m_a, m_v, S_a, S_v, sigma_av_a, sigma_av_v)
        posterior_c1 = self.posterior_C1(likelihood_c1, likelihood_c2, p_c)
        posterior_c2 = 1 - posterior_c1
        fused_S_av = self.fusionAV(m_a, m_v, sigma_av_a, sigma_av_v)
        hat_S_AV_A_No_CC = m_a
        hat_S_AV_A_final = posterior_c1 * fused_S_av + posterior_c2 * hat_S_AV_A_No_CC
        return hat_S_AV_A_final

    def causalInference_vectorized(self, S_a, S_v, m_a, m_v, sigma_av_a, sigma_av_v, p_c):
        var_sum = sigma_av_a**2 + sigma_av_v**2
        likelihood_c1 = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))
        likelihood_c2 = norm.pdf(m_a, loc=S_a, scale=sigma_av_a) * norm.pdf(m_v, loc=S_v, scale=sigma_av_v)
        posterior_c1 = (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))
        fused_S_av = self.fusionAV_vectorized(m_a, m_v, sigma_av_a, sigma_av_v)

        final_estimate = posterior_c1 * fused_S_av + (1 - posterior_c1) * m_a
        return final_estimate


    def causalInfDecision(self, trueStims, measurements, sigma_av_a, sigma_av_v, p_c):
        S_a_s, S_a_t, S_v_s, S_v_t = trueStims
        if measurements[0] is None:
            m_a_s, m_a_t, m_v_s, m_v_t = S_a_s, S_a_t, S_v_s, S_v_t
        else:
            m_a_s, m_a_t, m_v_s, m_v_t = measurements
        standardShat = self.causalInference_vectorized(S_a_s, S_v_s, m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c)
        testShat = self.fusionAV_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v)
        decision = (testShat - standardShat) > 0
        return decision

    def probTestLonger(self, trueStims, sigma_av_a, sigma_av_v, p_c, lambda_=0):
        nSimul = self.nSimul
        S_a_s, S_a_t, S_v_s, S_v_t = trueStims
        m_a_s_arr = np.random.normal(S_a_s, sigma_av_a, nSimul)
        m_v_s_arr = np.random.normal(S_v_s, sigma_av_v, nSimul)
        m_a_t_arr = np.random.normal(S_a_t, sigma_av_a, nSimul)
        m_v_t_arr = np.random.normal(S_v_t, sigma_av_v, nSimul)
        if self.mDist == "lognorm":
            m_a_s_arr = np.random.lognormal(mean=np.log(S_a_s), sigma=sigma_av_a, size=nSimul)
            m_v_s_arr = np.random.lognormal(mean=np.log(S_v_s), sigma=sigma_av_v, size=nSimul)
            m_a_t_arr = np.random.lognormal(mean=np.log(S_a_t), sigma=sigma_av_a, size=nSimul)
            m_v_t_arr = np.random.lognormal(mean=np.log(S_v_t), sigma=sigma_av_v, size=nSimul)
        measurementsArr = np.array([m_a_s_arr, m_a_t_arr, m_v_s_arr, m_v_t_arr])
        stimArr = np.array([S_a_s, S_a_t, S_v_s, S_v_t])
        decisionArr = self.causalInfDecision(stimArr, measurementsArr, sigma_av_a, sigma_av_v, p_c)
        p_base = np.mean(decisionArr)
        p_final = (1 - lambda_) * p_base + lambda_ / 2
        return p_final
    
    
    def probTestLonger_vectorized_mc(self, trueStims, sigma_av_a, sigma_av_v, p_c, lambda_):
        if self.mDist == "gaussian":
            #print("Using Gaussian distribution for measurements")
            nSimul = self.nSimul
            S_a_s, S_a_t, S_v_s, S_v_t = trueStims
            m_a_s = np.random.normal(S_a_s, sigma_av_a, nSimul)
            m_v_s = np.random.normal(S_v_s, sigma_av_v, nSimul)
            m_a_t = np.random.normal(S_a_t, sigma_av_a, nSimul)
            m_v_t = np.random.normal(S_v_t, sigma_av_v, nSimul)
            est_standard = self.causalInference_vectorized(S_a_s, S_v_s, m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c)
            # est_test = self.fusionAV_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v)
            ## or est_test computed using causalInference_vectorized
            est_test = self.causalInference_vectorized(S_a_t, S_v_t, m_a_t, m_v_t, sigma_av_a, sigma_av_v, p_c)
            p_base = np.mean(est_test > est_standard)
            p_final = (1 - lambda_) * p_base + lambda_ / 2

        elif self.mDist == "lognorm":
            #print("Using lognormal distribution for measurements")
            nSimul = self.nSimul
            S_a_s, S_a_t, S_v_s, S_v_t = trueStims
            m_a_s = np.random.lognormal(mean=np.log(S_a_s), sigma=sigma_av_a, size=nSimul)
            m_v_s = np.random.lognormal(mean=np.log(S_v_s), sigma=sigma_av_v, size=nSimul)
            m_a_t = np.random.lognormal(mean=np.log(S_a_t), sigma=sigma_av_a, size=nSimul)
            m_v_t = np.random.lognormal(mean=np.log(S_v_t), sigma=sigma_av_v, size=nSimul)
            est_standard = self.causalInference_vectorized(np.log(S_a_s), np.log(S_v_s), m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c)
            #est_test = self.fusionAV_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v)
            ## or est_test computed using causalInference_vectorized 
            est_test = self.causalInference_vectorized(np.log(S_a_t), np.log(S_v_t), m_a_t, m_v_t, sigma_av_a, sigma_av_v, p_c)
            p_base = np.mean(est_test > est_standard)
            p_final = (1 - lambda_) * p_base + lambda_ / 2
        return p_final
    
    # def probTestLonger_vectorized_mc(self, trueStims, sigma_av_a, sigma_av_v, p_c, lambda_):
    #     if self.mDist == "gaussian":
    #         nSimul = self.nSimul
    #         S_a_s, S_a_t, S_v_s, S_v_t = trueStims
    #         m_a_s = np.random.normal(S_a_s, sigma_av_a, nSimul)
    #         m_v_s = np.random.normal(S_v_s, sigma_av_v, nSimul)
    #         m_a_t = np.random.normal(S_a_t, sigma_av_a, nSimul)
    #         m_v_t = np.random.normal(S_v_t, sigma_av_v, nSimul)
    #         est_standard = self.causalInference_vectorized(S_a_s, S_v_s, m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c)
    #         est_test = self.causalInference_vectorized(S_a_t, S_v_t, m_a_t, m_v_t, sigma_av_a, sigma_av_v, p_c)
    #         p_base = np.mean(est_test > est_standard)
    #         p_final = (1 - lambda_) * p_base + lambda_ / 2

    #     elif self.mDist == "lognorm":
    #         nSimul = self.nSimul
    #         S_a_s, S_a_t, S_v_s, S_v_t = trueStims
            
    #         # Work entirely in log space
    #         log_S_a_s = np.log(S_a_s)
    #         log_S_a_t = np.log(S_a_t)
    #         log_S_v_s = np.log(S_v_s)
    #         log_S_v_t = np.log(S_v_t)
            
    #         # Generate measurements in log space (normal distribution)
    #         log_m_a_s = np.random.normal(log_S_a_s, sigma_av_a, nSimul)
    #         log_m_v_s = np.random.normal(log_S_v_s, sigma_av_v, nSimul)
    #         log_m_a_t = np.random.normal(log_S_a_t, sigma_av_a, nSimul)
    #         log_m_v_t = np.random.normal(log_S_v_t, sigma_av_v, nSimul)
            
    #         # Causal inference in log space
    #         est_standard = self.causalInference_vectorized(log_S_a_s, log_S_v_s, log_m_a_s, log_m_v_s, sigma_av_a, sigma_av_v, p_c)
    #         est_test = self.causalInference_vectorized(log_S_a_t, log_S_v_t, log_m_a_t, log_m_v_t, sigma_av_a, sigma_av_v, p_c)
            
    #         p_base = np.mean(est_test > est_standard)
    #         p_final = (1 - lambda_) * p_base + lambda_ / 2
            
    #     return p_final
    
    
    def nLLMonteCarloCausal(self, params, groupedData):

        """Negative log-likelihood for causal inference model"""
        ll = 0
        lenData = len(groupedData)
        
        # Check for invalid parameters
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return 1e10
        
        for i in range(lenData):
            currSNR = groupedData["audNoise"].iloc[i]
            currConflict = groupedData["conflictDur"].iloc[i]
            currResp = groupedData['num_of_chose_test'].iloc[i]
            totalResponses = groupedData['total_responses'].iloc[i]
            
            # Get the parameters for the current condition
            lambda_, sigma_av_a, sigma_av_v, p_c = self.getParamsCausal(params, currSNR)

            # Get the true standard and test durations
            S_a_s = groupedData["standardDur"].iloc[i]
            S_v_s = groupedData["unbiasedVisualStandardDur"].iloc[i]
            S_a_t = groupedData["testDurS"].iloc[i]
            S_v_t = groupedData["unbiasedVisualTestDur"].iloc[i]
            trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
            
            # Calculate the probability of choosing the test duration being longer than the standard duration
            try:
                p_test_longer = self.probTestLonger_vectorized_mc(trueStims, sigma_av_a, sigma_av_v, p_c, lambda_)
            except:
                return 1e10
                
            # Calculate the likelihood for the current condition
            # add epsilon to avoid log(0)
            epsilon = 1e-10
            P = np.clip(p_test_longer, epsilon, 1 - epsilon)
            ll += np.log(P) * currResp + np.log(1 - P) * (totalResponses - currResp)
            
        # Check for invalid likelihood
        if np.isnan(ll) or np.isinf(ll):
            return 1e10
            
        # Return the negative log-likelihood
        return -ll


    def fitCausalInferenceMonteCarlo(self, groupedData):
        """
        Fit causal inference model using Monte Carlo simulation with multiple random starts.
        Supports 'scipy' (default) or 'bads' optimization (if installed).
        """
        # Parameter bounds
        bounds = np.array([
            (0, 0.3),     # lambda_
            (0.1, 1.5),    # sigma_av_a_1
            (0.1, 1.5),    # sigma_av_v_1
            (0.05, 0.99),  # p_c_1
            (0.1, 1.5),    # sigma_av_a_2
            (0.1, 1.5),    # sigma_av_v_2
            (0.05, 0.99)   # p_c_2
        ])

        # Initial best results
        best_result = None
        best_ll = np.inf
        nStart = self.nStart if hasattr(self, 'nStart') else 1
        optimizer = self.optimizationMethod if hasattr(self, 'optimizationMethod') else 'scipy'
        print(f"\nStarting {nStart} optimization attempts using '{optimizer}'...")

        for attempt in tqdm(range(nStart), desc="Optimization Attempts"):
            # Random x0 initialization within bounds
            x0 = np.array([
                np.random.uniform(0.01, 0.24),  # lambda_
                np.random.uniform(0.1, 1.5),    # sigma_av_a_1
                np.random.uniform(0.1, 1.5),    # sigma_av_v_1
                np.random.uniform(0.1, 0.98),   # p_c_1
                np.random.uniform(0.1, 1.5),    # sigma_av_a_2
                np.random.uniform(0.1, 1.5),    # sigma_av_v_2
                np.random.uniform(0.1, 0.98),   # p_c_2,
            ])

            try:
                if self.optimizationMethod == "bads":
                    # Prepare bounds
                    lb = bounds[:, 0]
                    ub = bounds[:, 1]
                    plb = lb * 1.2
                    pub = ub * 0.9

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
        print(f"\n✅ Best result from {nStart} attempts:")
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


            # Unpack fitted parameters for the current audio noise level
            lambda_, sigma_av_a, sigma_av_v, p_c = self.getParamsCausal(fittedParams, audioNoiseLevel)

            nSamples = 10* int(totalResponses)  # Scale number of samples by total responses for better simulation
            # Simulate responses for the current trial
            for _ in range(nSamples):
                S_a_s = 0.5
                S_v_s = S_a_s + conflictLevel
                S_a_t = S_a_s + deltaDurS
                S_v_t = S_a_t
                trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
                p_test_longer = self.probTestLonger_vectorized_mc(trueStims, sigma_av_a, sigma_av_v, p_c, lambda_)
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
        return simData


    def plot_posterior_vs_conflict(self, data, fittedParams, snr_list=[1.2, 0.1]):
        """
        Plot posterior probability vs conflict for given SNR values.
        snr_list: list of SNR values to plot (default: [1.2, 0.1])
        """
        delta_dur_values = data["deltaDurS"].values
        conflict_values = data["conflictDur"].values
        snr_values = data["audNoise"].values
        best_params = fittedParams  # Use the best fitted parameters from the previous fitting

        posterior_values = []
        for delta, conflict, snr in zip(delta_dur_values, conflict_values, snr_values):
            λ, σa, σv, pc = self.getParamsCausal(best_params, snr)
            S_std = 0.5
            S_test = S_std + delta
            S_v = S_std + conflict

            m_a = S_std
            m_v = S_v

            L1 = self.likelihood_C1(m_a, m_v, σa, σv)
            L2 = self.likelihood_C2(m_a, m_v, S_std, S_v, σa, σv)
            posterior = self.posterior_C1(L1, L2, pc)
            posterior_values.append(posterior)

        plt.figure(figsize=(8, 5))
        for idx, noisy_snr_value in enumerate(snr_list):
            mask_noisy = np.isclose(snr_values, noisy_snr_value)
            conflicts_noisy = conflict_values[mask_noisy]
            posteriors_noisy = np.array(posterior_values)[mask_noisy]
            plt.subplot(1, 2, idx + 1)
            plt.scatter(conflicts_noisy * 1000, posteriors_noisy, alpha=0.6, label=f'Posterior P(C=1) (SNR={noisy_snr_value})')
            plt.xlabel('Conflict (ms)')
            plt.ylabel('Posterior Probability of Common Cause')
            plt.title(f'Posterior P(C=1) vs Conflict (SNR={noisy_snr_value})')
            plt.axhline(y=self.getParamsCausal(fittedParams, noisy_snr_value)[3], color='gray', linestyle='--', label=f'P(C=1)={self.getParamsCausal(fittedParams, noisy_snr_value)[3]:.2f}')
            plt.legend()
            plt.ylim(0, 1)
            plt.grid()
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
                    plt.scatter(conflictLevel, muData, color="red", s=40, alpha=0.7)
                    plt.scatter(conflictLevel, muModel, color="blue", s=40,  alpha=0.7)
                    plt.xlabel(f"Conflict (ms)")
                    plt.ylabel("Mu (sigma_av_a)")
                    plt.axhline(y=0, color='gray', linestyle='--')
                    plt.axvline(x=0, color='gray', linestyle='--')
                    plt.ylim(-0.3, 0.3)
                    plt.title(f"{self.dataName} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
                    plt.legend(fontsize=14, title_fontsize=14)
                    plt.grid()
        plt.show()


    def plotPsychometrics_MC_Data(self):
        "use self to get the  required stuff"

        pltTitle = self.dataName + " Causal Inference Model Fit"
        plt.figure(figsize=(16, 6))
        for i, standardLevel in enumerate(self.uniqueStandard):
            for j, audioNoiseLevel in enumerate(sorted(self.uniqueSensory)):
                for k, conflictLevel in enumerate(self.uniqueConflict):
                    plt.subplot(1, 2, j + 1)
                    x = np.linspace(-0.5, 0.5, 1000)
                    color = sns.color_palette("viridis", as_cmap=True)(k / len(self.uniqueConflict))
                    paramsSimDf=self.getParams(self.simDataFit.x, conflictLevel, audioNoiseLevel)

                    # # Plot simulation: plot simulated data points (proportion chose test) for each deltaDurS
                    # simDf = self.simulatedData[
                    #     (self.simulatedData[sensoryVar] == audioNoiseLevel) &
                    #     (self.simulatedData[conflictVar] == conflictLevel)
                    # ]
                    # if not simDf.empty:
                    #     simDf = simDf.sort_values(by=self.intensityVariable)
                    #     x_sim = simDf[self.intensityVariable].values
                    #     y_sim = simDf['chose_test'] / simDf['responses']
                    #     #plt.scatter(x_sim, y_sim, color=color, s=40, marker='o', label=f"SimData c={int(conflictLevel*1000)}", alpha=0.7)
                    ySimSigmoid=self.psychometric_function(x, paramsSimDf[0],paramsSimDf[1],paramsSimDf[2])
                    plt.plot(x, ySimSigmoid, color=color)


                    "plot the monte carlo"
                    lambda_, sigma_av_a, sigma_av_v, p_c = self.getParamsCausal(self.modelFit, audioNoiseLevel)
                    S_a_s = 0.5
                    S_v_s = S_a_s + conflictLevel
                    # plot the psychometric curve for the monte carlo model simulations
                    # y = np.zeros_like(x)
                    # for idx in range(len(x)):
                    #     y[idx] = self.probTestLonger([S_a_s, S_a_s + x[idx], S_v_s, S_a_s + x[idx]], sigma_av_a, sigma_av_v, p_c, lambda_)
                    
                    # plt.plot(x, y, color=color, label=f"c: {int(conflictLevel*1000)}, $\sigma_a$: {sigma_av_a:.2f}, $\sigma_v$: {sigma_av_v:.2f}", linewidth=4,alpha=0.3)
            

            
                    plt.axvline(x=0, color='gray', linestyle='--')
                    plt.axhline(y=0.5, color='gray', linestyle='--')
                    plt.xlabel(f"({self.intensityVar}) Test(stair-a)-Standard(a) Duration Difference Ratio(%)")
                    plt.ylabel("P(chose test)")
                    plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
                    plt.legend(fontsize=14, title_fontsize=14)
                    plt.grid()


                    groupedDataSub = self.groupByChooseTest(
                        self.data[(self.data[self.standardVar] == standardLevel) & (self.data[self.sensoryVar] == audioNoiseLevel) & (self.data[self.conflictVar] == conflictLevel)],
                        [self.intensityVar, self.sensoryVar, self.standardVar, self.conflictVar, self.visualStandardVar, self.visualTestVar, self.audioTestVar]
                    )
                    self.bin_and_plot(groupedDataSub, bin_method='cut', bins=10, plot=True, color=color)
                    plt.text(0.05, 0.8, f"$\sigma_a$: {sigma_av_a:.2f}, $\sigma_v$: {sigma_av_v:.2f},", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
                    plt.tight_layout()
                    plt.grid(True)
                    print(f"Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, Lambda: {lambda_:.3f}, Sigma_a: {sigma_av_a:.3f}, Sigma_v: {sigma_av_v:.3f}, p_c: {p_c:.3f}")
                    plt.text(0.15, 0.9, f"P(C=1): {p_c:.2f}", fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
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
    data, dataName = loadData("dt_all.csv")

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
    mc_fitter.optimizationMethod= "bads"  # Use BADS for optimization
    mc_fitter.nStart = 1  # Number of random starts for optimization


    groupedData = mc_fitter.groupByChooseTest(
        x=data,
        groupArgs=[
            intensityVariable, sensoryVar, standardVar, conflictVar,
            visualStandardVar, visualTestVar, audioTestVar
        ]
    )

    

    mc_fitter.mDist = "lognorm"  # Set measurement distribution to Gaussian
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
    mc_fitter.simDataFit=mc_fitter.fitMultipleStartingPoints(mc_fitter.simulatedData,1)

    "psychometric fit"
    mc_fitter.dataFit= mc_fitter.fitMultipleStartingPoints(data,1)


    # Plotting the results
    mc_fitter.plotPsychometrics_MC_Data()


                




    # Plot posterior vs conflict
    mc_fitter.plot_posterior_vs_conflict(data, fittedParams, snr_list=[1.2, 0.1])
    # Simulate and plot psychometric data
    uniqueSensory = np.unique(data[sensoryVar])
    uniqueConflict = np.unique(data[conflictVar])







