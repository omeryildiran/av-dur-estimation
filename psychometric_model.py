"""
Psychometric Model Fitting Module
==================================

A refactored, modular version of the psychometric function fitting code
that can be easily imported and used in Jupyter notebooks without global variable issues.

Author: Refactored for modular usage
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, linregress
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PsychometricModel:
    """
    A class for fitting psychometric functions to duration discrimination data.
    
    This class encapsulates all the configuration and methods needed for fitting
    psychometric models with different parameter sharing configurations.
    """
    
    def __init__(self, all_independent=True, shared_sigma=False):
        """
        Initialize the psychometric model.
        
        Parameters:
        -----------
        all_independent : bool
            If True, each condition gets its own lambda, mu, sigma parameters
        shared_sigma : bool
            If True, sigma is shared across conditions (only used when all_independent=False)
        """
        self.all_independent = all_independent
        self.shared_sigma = shared_sigma
        
        # Data configuration
        self.intensity_var = "deltaDurS"
        self.sensory_var = "audNoise"
        self.standard_var = "standardDur"
        self.conflict_var = "conflictDur"
        
        # Will be set during data loading
        self.unique_sensory = None
        self.unique_conflict = None
        self.unique_standard = None
        self.n_lambda = None
        self.n_sigma = None
        self.n_mu = None
        
        # Fitted parameters
        self.fitted_params = None
        self.fit_result = None
        
    def load_data(self, data_path):
        """
        Load and preprocess duration discrimination data.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV data file
            
        Returns:
        --------
        data : pd.DataFrame
            Preprocessed data
        data_name : str
            Name derived from filename
        """
        if not data_path.startswith('data/'):
            data_path = f"data/{data_path}"
            
        data = pd.read_csv(data_path)
        data_name = data_path.split("/")[-1].split(".")[0]
        
        # Basic preprocessing
        data["testDurMs"] = data["testDurS"] * 1000
        data["standardDurMs"] = data["standardDur"] * 1000
        data["conflictDurMs"] = data["conflictDur"] * 1000
        data["DeltaDurMs"] = data["testDurMs"] - data["standardDurMs"]
        
        data = data.round({'standardDur': 2, 'conflictDur': 2})
        
        # Remove NaN values
        data = data[~data['conflictDur'].isna()]
        data = data[~data['audNoise'].isna()]
        
        # Visual PSE calculations
        if "VisualPSE" not in data.columns:
            data["VisualPSE"] = data['recordedDurVisualStandard'] - data["standardDur"] - data['conflictDur']
        
        data['visualPSEBias'] = data['recordedDurVisualStandard'] - data["standardDur"] - data['conflictDur']
        data['visualPSEBiasTest'] = data['recordedDurVisualTest'] - data["testDurS"]
        
        data["unbiasedVisualStandardDur"] = data["recordedDurVisualStandard"] - data["visualPSEBias"]
        data["unbiasedVisualTestDur"] = data["recordedDurVisualTest"] - data["visualPSEBiasTest"]
        
        # Filter data
        data = data[data['audNoise'] != 0]
        data = data[data['standardDur'] != 0]
        data[self.standard_var] = data[self.standard_var].round(2)
        data[self.conflict_var] = data[self.conflict_var].round(3)
        
        # Get unique values
        self.unique_sensory = data[self.sensory_var].unique()
        self.unique_standard = data[self.standard_var].unique()
        self.unique_conflict = sorted(data[self.conflict_var].unique())
        
        self.n_lambda = len(self.unique_standard)
        self.n_sigma = len(self.unique_sensory)
        self.n_mu = len(self.unique_conflict) * self.n_sigma
        
        # Define response columns
        data['chose_test'] = (data['responses'] == data['order']).astype(int)
        data['chose_standard'] = (data['responses'] != data['order']).astype(int)
        
        print(f"Loaded {len(data)} trials")
        print(f"Unique sensory levels: {self.unique_sensory}")
        print(f"Unique standard levels: {self.unique_standard}")
        print(f"Unique conflict levels: {self.unique_conflict}")
        
        return data, data_name
    
    def group_by_choose_test(self, data, group_args=None):
        """
        Group data by experimental conditions and count responses.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw trial data
        group_args : list, optional
            List of columns to group by
            
        Returns:
        --------
        pd.DataFrame
            Grouped data with response counts
        """
        if group_args is None:
            group_args = [self.intensity_var, self.sensory_var, 
                         self.standard_var, self.conflict_var]
        
        grouped = data.groupby(group_args).agg(
            num_of_chose_test=('chose_test', 'sum'),
            total_responses=('responses', 'count'),
            num_of_chose_standard=('chose_standard', 'sum'),
        ).reset_index()
        
        grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']
        return grouped
    
    def psychometric_function(self, x, lambda_, mu, sigma):
        """
        Psychometric function with lapse rate.
        
        Parameters:
        -----------
        x : array-like
            Stimulus intensities
        lambda_ : float
            Lapse rate
        mu : float
            Point of subjective equality (bias)
        sigma : float
            Standard deviation (sensitivity)
            
        Returns:
        --------
        array-like
            Probability of choosing test stimulus
        """
        return lambda_/2 + (1-lambda_) * norm.cdf((x - mu) / sigma)
    
    def get_params(self, params, conflict, audio_noise):
        """
        Extract parameters for a specific condition.
        
        Parameters:
        -----------
        params : array-like
            Full parameter vector
        conflict : float
            Conflict level
        audio_noise : float
            Audio noise level
            
        Returns:
        --------
        tuple
            (lambda, mu, sigma) for the condition
        """
        if self.all_independent:
            # Each condition has its own lambda, mu, sigma
            noise_idx = np.where(np.isclose(self.unique_sensory, audio_noise, atol=1e-6))[0][0]
            conflict_idx = np.where(np.isclose(self.unique_conflict, conflict, atol=1e-6))[0][0]
            cond_idx = noise_idx * len(self.unique_conflict) + conflict_idx
            
            lambda_ = params[cond_idx * 3 + 0]
            mu = params[cond_idx * 3 + 1]
            sigma = params[cond_idx * 3 + 2]
            
        elif self.shared_sigma:
            # Shared sigma across conditions
            lambda_ = params[0]
            noise_idx = np.where(np.isclose(self.unique_sensory, audio_noise, atol=1e-6))[0][0]
            sigma = params[noise_idx + 1]
            
            conflict_idx = np.where(np.isclose(self.unique_conflict, conflict, atol=1e-6))[0][0]
            noise_offset = noise_idx * len(self.unique_conflict)
            mu_idx = self.n_lambda + self.n_sigma + noise_offset + conflict_idx
            mu = params[mu_idx]
            
        else:
            # Independent mu and sigma per condition, shared lambda
            lambda_ = params[0]
            noise_idx = np.where(np.isclose(self.unique_sensory, audio_noise, atol=1e-6))[0][0]
            conflict_idx = np.where(np.isclose(self.unique_conflict, conflict, atol=1e-6))[0][0]
            
            n_conditions = len(self.unique_conflict) * len(self.unique_sensory)
            cond_idx = conflict_idx * len(self.unique_sensory) + noise_idx
            
            mu = params[self.n_lambda + cond_idx]
            sigma = params[self.n_lambda + n_conditions + cond_idx]
        
        return lambda_, mu, sigma
    
    def negative_log_likelihood_joint(self, params, delta_dur, responses, total_responses, 
                                    conflicts, noise_levels):
        """
        Compute negative log-likelihood for all conditions.
        
        Parameters:
        -----------
        params : array-like
            Parameter vector
        delta_dur : array-like
            Stimulus intensities
        responses : array-like
            Number of "test longer" responses
        total_responses : array-like
            Total number of responses per condition
        conflicts : array-like
            Conflict levels for each condition
        noise_levels : array-like
            Noise levels for each condition
            
        Returns:
        --------
        float
            Negative log-likelihood
        """
        # Precompute parameters for each condition
        lam = np.empty(len(delta_dur))
        mu = np.empty(len(delta_dur))
        sigma = np.empty(len(delta_dur))
        
        for i in range(len(delta_dur)):
            lam[i], mu[i], sigma[i] = self.get_params(params, conflicts[i], noise_levels[i])
        
        # Vectorized psychometric function
        p = lam / 2 + (1 - lam) * norm.cdf((delta_dur - mu) / sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        
        # Vectorized negative log-likelihood
        nll = -np.sum(responses * np.log(p) + (total_responses - responses) * np.log(1 - p))
        return nll
    
    def estimate_initial_guesses(self, levels, chose_test, total_resp):
        """
        Estimate initial parameter guesses from data.
        
        Parameters:
        -----------
        levels : array-like
            Stimulus intensities
        chose_test : array-like
            Number of "test longer" responses
        total_resp : array-like
            Total responses
            
        Returns:
        --------
        list
            [lambda, mu, sigma] initial guesses
        """
        proportions = chose_test / total_resp
        slope, intercept, _, _, _ = linregress(levels, proportions)
        
        mu_guess = (0.5 - intercept) / slope
        lapse_rate_guess = 0.03
        
        # Compute sigma from slope
        sigma_guess = (1 - lapse_rate_guess) / (np.sqrt(2 * np.pi) * abs(slope)) * np.exp(-0.5)
        sigma_guess = max(sigma_guess - 0.1, 0.01)  # Regularize
        
        return [lapse_rate_guess, mu_guess, sigma_guess]
    
    def generate_multiple_init_guesses(self, single_init_guesses, n_start):
        """
        Generate multiple initial guesses for optimization.
        
        Parameters:
        -----------
        single_init_guesses : list
            Single set of initial guesses
        n_start : int
            Number of different starting points
            
        Returns:
        --------
        list
            List of initial guess sets
        """
        if n_start == 1:
            return [single_init_guesses]
        
        init_lambdas = np.linspace(0.01, 0.1, n_start)
        init_mus = np.linspace(-0.8, 0.8, n_start)
        init_sigmas = np.linspace(0.01, 1.5, n_start)
        
        multiple_init_guesses = []
        for lambda_ in init_lambdas:
            for mu in init_mus:
                for sigma in init_sigmas:
                    multiple_init_guesses.append([lambda_, mu, sigma])
        
        return multiple_init_guesses[:n_start]  # Limit to n_start
    
    def fit_joint(self, grouped_data, init_guesses):
        """
        Fit the joint psychometric model.
        
        Parameters:
        -----------
        grouped_data : pd.DataFrame
            Grouped data with response counts
        init_guesses : list
            Initial parameter guesses [lambda, mu, sigma]
            
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization result
        """
        n_sensory = len(grouped_data[self.sensory_var].unique())
        n_conflict = len(grouped_data[self.conflict_var].unique())
        
        if self.all_independent:
            # Each condition gets its own parameters
            n_cond = n_sensory * n_conflict
            init_guesses_full = (init_guesses * n_cond)[:n_cond*3]
            bounds = [(0, 0.25), (-2, 2), (0.01, 2)] * n_cond
            
        elif self.shared_sigma:
            # Shared sigma configuration
            init_guesses_full = ([init_guesses[0]] * self.n_lambda + 
                               [init_guesses[2]] * n_sensory + 
                               [init_guesses[1]] * n_sensory * n_conflict)
            bounds = ([(0, 0.25)] * self.n_lambda + 
                     [(0.01, 1)] * n_sensory + 
                     [(-1, 1)] * n_sensory * n_conflict)
        else:
            # Independent mu and sigma per condition
            init_guesses_full = ([init_guesses[0]] * self.n_lambda + 
                               [init_guesses[1]] * n_sensory * n_conflict + 
                               [init_guesses[2]] * n_sensory * n_conflict)
            bounds = ([(0, 0.25)] * self.n_lambda + 
                     [(-2, 2)] * n_sensory * n_conflict + 
                     [(0.01, 2)] * n_sensory * n_conflict)
        
        # Prepare data
        intensities = grouped_data[self.intensity_var].values
        chose_tests = grouped_data['num_of_chose_test'].values
        total_responses = grouped_data['total_responses'].values
        conflicts = grouped_data[self.conflict_var].values
        noise_levels = grouped_data[self.sensory_var].values
        
        # Minimize negative log-likelihood
        result = minimize(
            self.negative_log_likelihood_joint,
            x0=init_guesses_full,
            args=(intensities, chose_tests, total_responses, conflicts, noise_levels),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return result
    
    def fit_multiple_starting_points(self, data, n_start=3):
        """
        Fit the model with multiple random starting points.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw trial data
        n_start : int
            Number of different starting points to try
            
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Best optimization result
        """
        # Group data
        grouped_data = self.group_by_choose_test(data)
        
        # Prepare for fitting
        levels = grouped_data[self.intensity_var].values
        responses = grouped_data['num_of_chose_test'].values
        total_resp = grouped_data['total_responses'].values
        
        # Get initial guesses
        single_init_guesses = self.estimate_initial_guesses(levels, responses, total_resp)
        multiple_init_guesses = self.generate_multiple_init_guesses(single_init_guesses, n_start)
        
        # Fit with multiple starting points
        best_fit = None
        best_nll = float('inf')
        
        disable_progress = len(multiple_init_guesses) == 1
        
        for i, init_guess in enumerate(tqdm(multiple_init_guesses, 
                                          desc="Fitting multiple starting points",
                                          disable=disable_progress)):
            fit = self.fit_joint(grouped_data, init_guess)
            
            if fit.success and fit.fun < best_nll:
                best_nll = fit.fun
                best_fit = fit
        
        if best_fit is None:
            raise ValueError("All optimization attempts failed!")
        
        self.fitted_params = best_fit.x
        self.fit_result = best_fit
        
        return best_fit
    
    def plot_fitted_psychometric(self, data, title_prefix=""):
        """
        Plot fitted psychometric functions.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw trial data
        title_prefix : str
            Prefix for plot titles
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before plotting. Call fit_multiple_starting_points first.")
        
        plt.figure(figsize=(15, 8))
        
        for j, audio_noise_level in enumerate(sorted(self.unique_sensory)):
            plt.subplot(1, len(self.unique_sensory), j+1)
            
            for k, conflict_level in enumerate(self.unique_conflict):
                lambda_, mu, sigma = self.get_params(self.fitted_params, conflict_level, audio_noise_level)
                
                # Filter data for current condition
                df = data[
                    (np.isclose(data[self.sensory_var], audio_noise_level, atol=1e-6)) &
                    (np.isclose(data[self.conflict_var], conflict_level, atol=1e-6))
                ]
                
                if len(df) == 0:
                    continue
                
                df_filtered = self.group_by_choose_test(df)
                
                # Plot fitted curve
                x = np.linspace(-0.9, 0.9, 500)
                y = self.psychometric_function(x, lambda_, mu, sigma)
                color = sns.color_palette("viridis", as_cmap=True)(k / len(self.unique_conflict))
                
                plt.plot(x, y, color=color, 
                        label=f"C: {int(conflict_level*1000)}, λ: {lambda_:.2f}, μ: {mu:.2f}, σ: {sigma:.2f}", 
                        linewidth=3)
                
                # Plot data points
                if len(df_filtered) > 0:
                    self.bin_and_plot(df_filtered, color=color)
                
                print(f"Noise: {audio_noise_level}, Conflict: {conflict_level}, "
                      f"Lambda: {lambda_:.3f}, Mu: {mu:.3f}, Sigma: {sigma:.3f}")
            
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel(f"Duration Difference ({self.intensity_var})")
            plt.ylabel("P(chose test)")
            plt.title(f"{title_prefix} Noise: {audio_noise_level}")
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def bin_and_plot(self, data, bins=8, color="blue"):
        """
        Bin data and plot as scatter points.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Grouped data
        bins : int
            Number of bins
        color : str
            Color for plotting
        """
        if len(data) == 0:
            return
        
        data_copy = data.copy()
        data_copy['bin'] = pd.cut(data_copy[self.intensity_var], bins=bins, 
                                 labels=False, include_lowest=True)
        
        grouped = data_copy.groupby('bin').agg(
            x_mean=(self.intensity_var, 'mean'),
            y_mean=('p_choose_test', 'mean'),
            total_resp=('total_responses', 'sum')
        ).dropna()
        
        if len(grouped) > 0:
            plt.scatter(grouped['x_mean'], grouped['y_mean'], 
                       s=grouped['total_resp']/data_copy['total_responses'].sum()*400, 
                       color=color, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    def get_parameter_summary(self):
        """
        Get a summary of fitted parameters.
        
        Returns:
        --------
        pd.DataFrame
            Summary table of parameters by condition
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted first.")
        
        summary_data = []
        
        for noise in self.unique_sensory:
            for conflict in self.unique_conflict:
                lambda_, mu, sigma = self.get_params(self.fitted_params, conflict, noise)
                summary_data.append({
                    'noise_level': noise,
                    'conflict_level': conflict,
                    'lambda': lambda_,
                    'mu': mu,
                    'sigma': sigma
                })
        
        return pd.DataFrame(summary_data)


def load_data_simple(data_path):
    """
    Simple data loading function for quick use.
    
    Parameters:
    -----------
    data_path : str
        Path to data file
        
    Returns:
    --------
    pd.DataFrame
        Processed data
    """
    model = PsychometricModel()
    data, _ = model.load_data(data_path)
    return data
