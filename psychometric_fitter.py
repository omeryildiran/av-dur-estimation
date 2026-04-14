"""
Psychometric Function Fitter for Duration Estimation Tasks

A class-based implementation for fitting log-normal psychometric functions
to duration estimation data with support for multiple conditions (noise levels,
conflicts, etc.) and participant-level error bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, linregress
from scipy.optimize import minimize
from tqdm import tqdm


class PsychometricFitter:
    """
    Fit and analyze psychometric functions for duration estimation tasks.
    
    This class uses a log-normal observer model which is appropriate for duration
    data because it:
    1. Gives zero probability for negative durations
    2. Naturally incorporates Weber's law-like behavior
    3. Is consistent with scalar timing theory
    """
    
    def __init__(self, data_path=None, data=None, fix_mu=False, intensity_variable="testDurS"):
        """
        Initialize the PsychometricFitter.
        
        Parameters
        ----------
        data_path : str, optional
            Path to CSV data file (relative to 'data/' directory)
        data : pd.DataFrame, optional
            Pre-loaded DataFrame (if data_path is None)
        fix_mu : bool, default=False
            Whether to fix mu (bias) to 0
        intensity_variable : str, default="testDurS"
            Column name for test duration variable
        """
        self.fix_mu = fix_mu
        self.intensity_variable = intensity_variable
        self.sensory_var = "audNoise"
        self.standard_var = "standardDur"
        self.conflict_var = "conflictDur"
        self.test_dur_var = "testDurS"
        
        # Load data
        if data_path is not None:
            self.data = self._load_data(data_path)
        elif data is not None:
            self.data = data
            self._prepare_data()
        else:
            raise ValueError("Either data_path or data must be provided")
        
        # Extract unique values and counts
        self._extract_data_properties()
        
        # Store fitted parameters
        self.fitted_params = None
        self.fit_result = None
    
    def _load_data(self, data_name):
        """Load and preprocess data from CSV file."""
        data = pd.read_csv(f"data/{data_name}")
        
        # Filter out invalid rows
        data = data[data['audNoise'] != 0]
        data = data[data['standardDur'] != 0]
        
        # Add time conversion columns
        data["testDurMs"] = data["testDurS"] * 1000
        data["standardDurMs"] = data["standardDur"] * 1000
        
        # Prepare data before returning
        self.data = data
        self._prepare_data()
        return self.data
    
    def _prepare_data(self, data=None):
        """Prepare data with necessary columns and filtering."""
        if data is not None:
            # If data is passed, use it and update self.data
            self.data = data
        
        # Work with self.data
        data = self.data
        
        # Round to 2 decimal places
        data = data.round({
            'standardDur': 2, 
            'audNoise': 2, 
            'conflictDur': 2, 
            'delta_dur_percents': 2
        })
        
        # Add conflict column if missing
        if self.conflict_var not in data.columns:
            data[self.conflict_var] = 0
        
        # Add participantID if missing (create a default one)
        if 'participantID' not in data.columns:
            data['participantID'] = 'default_participant'
        
        # Add visual standard duration if missing
        if 'recordedDurVisualStandard' not in data.columns:
            data['recordedDurVisualStandard'] = 1
        else:
            data['recordedDurVisualStandard'] = round(data['recordedDurVisualStandard'], 3)
        
        # Filter visual duration data if present
        if 'recordedDurVisualStandard' in data.columns and 'recordedDurVisualTest' in data.columns:
            data = data[data['recordedDurVisualStandard'] <= 998]
            data = data[data['recordedDurVisualStandard'] >= 0]
            data = data[data['recordedDurVisualTest'] <= 998]
            data = data[data['recordedDurVisualTest'] >= 0]
        
        # Define response columns
        data['chose_test'] = (data['responses'] == data['order']).astype(int)
        data['chose_standard'] = (data['responses'] != data['order']).astype(int)
        
        # Add visual bias if columns exist
        if 'recordedDurVisualStandard' in data.columns:
            data['visualPSEBias'] = (data['recordedDurVisualStandard'] - 
                                     data["standardDur"] - data['conflictDur'])
        
        # Add riseDur if missing
        if 'riseDur' not in data.columns:
            data['riseDur'] = 1
        
        # Round key columns
        data['standard_dur'] = round(data['standardDur'], 2)
        data["delta_dur_percents"] = round(data["delta_dur_percents"], 2)
        data['conflictDur'] = round(data['conflictDur'], 2)
        
        self.data = data
    
    def _extract_data_properties(self):
        """Extract unique values and counts from data."""
        self.unique_sensory = self.data[self.sensory_var].unique()
        self.unique_standard = self.data[self.standard_var].unique()
        self.unique_conflict = sorted(self.data[self.conflict_var].unique())
        
        self.n_lambda = len(self.unique_standard)
        self.n_sigma = len(self.unique_sensory)
        self.n_mu = len(self.unique_conflict) * self.n_sigma
        
        print(f"Unique sensory levels: {self.unique_sensory}")
        print(f"Unique standard levels: {self.unique_standard}")
        print(f"Unique conflict levels: {self.unique_conflict}")
    
    @staticmethod
    def psychometric_function(test_dur, standard_dur, lambda_, mu, sigma, fix_mu=False):
        """
        Log-normal psychometric function.
        
        Parameters
        ----------
        test_dur : float or array
            Test duration(s) in seconds
        standard_dur : float or array
            Standard duration in seconds
        lambda_ : float
            Lapse rate
        mu : float
            Bias parameter (in log space)
        sigma : float
            Discrimination parameter (in log space)
        fix_mu : bool
            Whether to fix mu to 0
        
        Returns
        -------
        float or array
            Probability of choosing test as longer
        """
        if fix_mu:
            mu = 0
        
        # Ensure positive durations
        test_dur = np.maximum(test_dur, 1e-10)
        standard_dur = np.maximum(standard_dur, 1e-10)
        
        # Calculate log ratio
        log_ratio = np.log(test_dur / standard_dur)
        
        # Apply bias and normalize
        z_score = (log_ratio - mu) / sigma
        
        # Use standard normal CDF
        p_longer = norm.cdf(z_score)
        
        # Apply lapse rate
        p = lambda_ / 2 + (1 - lambda_) * p_longer
        
        return p
    
    @staticmethod
    def compute_sigma_from_slope(slope, lapse_rate=0.02):
        """Compute sigma from psychometric function slope."""
        sigma = (1 - lapse_rate) / (np.sqrt(2 * np.pi) * slope) * np.exp(-0.5)
        return sigma
    
    @staticmethod
    def mu_to_pse_shift(mu, standard_dur):
        """
        Convert log-space bias (mu) to linear PSE shift.
        
        PSE shift = standard_dur * (exp(mu) - 1)
        """
        pse = standard_dur * np.exp(mu)
        return pse - standard_dur
    
    @staticmethod
    def pse_shift_to_mu(pse_shift, standard_dur):
        """Convert linear PSE shift to log-space bias (mu)."""
        pse = standard_dur + pse_shift
        return np.log(pse / standard_dur)
    
    @staticmethod
    def calculate_pse_stats(mu, sigma, lambda_, standard_dur):
        """
        Calculate comprehensive PSE statistics.
        
        Returns
        -------
        dict
            Dictionary with PSE statistics including pure and lapse-corrected values
        """
        # Pure PSE (ignoring lapse rate)
        pse_pure = standard_dur * np.exp(mu)
        pse_shift_pure = pse_pure - standard_dur
        
        # PSE accounting for lapse rate
        if lambda_ < 1.0:
            target_p = (0.5 - lambda_ / 2) / (1 - lambda_)
            target_p = np.clip(target_p, 1e-10, 1 - 1e-10)
            z_target = norm.ppf(target_p)
            log_pse_ratio = mu + sigma * z_target
            pse_lapse_corrected = standard_dur * np.exp(log_pse_ratio)
            pse_shift_lapse_corrected = pse_lapse_corrected - standard_dur
        else:
            pse_lapse_corrected = standard_dur
            pse_shift_lapse_corrected = 0
        
        return {
            'pse_pure': pse_pure,
            'pse_shift_pure': pse_shift_pure,
            'pse_lapse_corrected': pse_lapse_corrected,
            'pse_shift_lapse_corrected': pse_shift_lapse_corrected,
            'pse_shift_percent': (pse_shift_pure / standard_dur) * 100
        }
    
    def group_by_choose_test(self, data=None):
        """Group data by test choice for psychometric curve fitting."""
        if data is None:
            data = self.data
        
        grouped = data.groupby([
            self.intensity_variable, self.sensory_var, 
            self.standard_var, self.conflict_var, "testDurMs"
        ]).agg(
            num_of_chose_test=('chose_test', 'sum'),
            total_responses=('responses', 'count'),
            num_of_chose_standard=('chose_standard', 'sum'),
        ).reset_index()
        
        grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']
        return grouped
    
    def group_by_choose_test_with_participants(self, data=None):
        """Group data with participant information for error bar calculation."""
        if data is None:
            data = self.data
        
        participant_grouped = data.groupby([
            self.intensity_variable, self.sensory_var, self.standard_var, 
            self.conflict_var, 'testDurMs', 'participantID'
        ]).agg(
            num_of_chose_test=('chose_test', 'sum'),
            total_responses=('responses', 'count'),
            num_of_chose_standard=('chose_standard', 'sum'),
        ).reset_index()
        
        participant_grouped['p_choose_test'] = (
            participant_grouped['num_of_chose_test'] / participant_grouped['total_responses']
        )
        
        return participant_grouped
    
    def estimate_initial_guesses(self, levels, chose_test, total_resp):
        """
        Estimate initial parameter guesses from data.
        
        Returns
        -------
        list
            [lambda, mu, sigma] initial guesses
        """
        proportions = chose_test / total_resp
        slope, intercept, _, _, _ = linregress(levels, proportions)
        
        mu_guess = (0.5 - intercept) / slope if slope != 0 else 0
        lapse_rate_guess = 0.03
        sigma_guess = self.compute_sigma_from_slope(slope, lapse_rate_guess) - 0.1
        
        return [lapse_rate_guess, mu_guess, sigma_guess]
    
    def get_params(self, params, conflict, audio_noise):
        """
        Extract parameters for specific condition from full parameter vector.
        
        Parameters
        ----------
        params : array
            Full parameter vector
        conflict : float
            Conflict level
        audio_noise : float
            Audio noise level
        
        Returns
        -------
        tuple
            (lambda_, mu, sigma) for the specified condition
        """
        # Lambda (shared across conditions)
        lambda_ = params[0]
        
        # Find indices
        noise_idx = np.where(self.unique_sensory == audio_noise)[0][0]
        conflict_idx = np.where(self.unique_conflict == conflict)[0][0]
        
        # Calculate sigma index
        sigma_idx = self.n_lambda - 1 + ((conflict_idx + 1) * (noise_idx + 1))
        sigma = params[sigma_idx]
        
        # Calculate mu index
        mu_idx = self.n_lambda - 1 + ((len(params) - 1) // 2) + ((conflict_idx + 1) * (noise_idx + 1))
        mu = params[mu_idx]
        
        if self.fix_mu:
            mu = 0
        
        return lambda_, mu, sigma
    
    def negative_log_likelihood(self, params, test_dur, standard_dur, chose_test, total_responses):
        """Compute negative log-likelihood for a single condition."""
        lambda_, mu, sigma = params
        
        if self.fix_mu:
            mu = 0
        
        p = self.psychometric_function(test_dur, standard_dur, lambda_, mu, sigma, self.fix_mu)
        
        # Avoid numerical issues
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        
        # Compute log-likelihood
        log_likelihood = np.sum(
            chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p)
        )
        
        return -log_likelihood
    
    def nll_joint(self, params, test_durations, standard_durations, responses, 
                  total_responses, conflicts, noise_levels):
        """Compute negative log-likelihood for all conditions jointly."""
        nll = 0
        
        for i in range(len(test_durations)):
            test_dur = test_durations[i]
            standard_dur = standard_durations[i]
            conflict = conflicts[i]
            audio_noise = noise_levels[i]
            total_response = total_responses[i]
            chose_test = responses[i]
            
            # Get condition-specific parameters
            lambda_, mu, sigma = self.get_params(params, conflict, audio_noise)
            
            # Calculate probability
            p = self.psychometric_function(test_dur, standard_dur, lambda_, mu, sigma, self.fix_mu)
            
            # Avoid numerical issues
            epsilon = 1e-9
            p = np.clip(p, epsilon, 1 - epsilon)
            
            # Add to negative log-likelihood
            nll += -1 * (chose_test * np.log(p) + (total_response - chose_test) * np.log(1 - p))
        
        return nll
    
    def fit_joint(self, grouped_data, init_guesses):
        """
        Fit joint model across all conditions.
        
        Parameters
        ----------
        grouped_data : pd.DataFrame
            Grouped data from group_by_choose_test
        init_guesses : list
            Initial parameter guesses [lambda, mu, sigma]
        
        Returns
        -------
        OptimizeResult
            Scipy optimization result
        """
        n_sensory = len(self.unique_sensory)
        n_conflict = len(self.unique_conflict)
        
        # Build initial guess vector
        init_guesses_full = (
            [init_guesses[0]] * self.n_lambda +  # lambda
            [init_guesses[2]] * n_sensory * n_conflict +  # sigma
            [init_guesses[1]] * n_sensory * n_conflict  # mu
        )
        
        # Extract data
        test_durations = grouped_data[self.intensity_variable].values
        standard_durations = grouped_data[self.standard_var].values
        chose_tests = grouped_data['num_of_chose_test'].values
        total_responses = grouped_data['total_responses'].values
        conflicts = grouped_data[self.conflict_var].values
        noise_levels = grouped_data[self.sensory_var].values
        
        # Set bounds
        bounds = (
            [(0, 0.25)] * self.n_lambda +  # lambda bounds
            [(0.01, 2.0)] * n_sensory * n_conflict +  # sigma bounds
            [(-1, 1)] * n_sensory * n_conflict  # mu bounds
        )
        
        # Optimize
        result = minimize(
            self.nll_joint,
            x0=init_guesses_full,
            args=(test_durations, standard_durations, chose_tests, 
                  total_responses, conflicts, noise_levels),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return result
    
    def generate_multiple_init_guesses(self, single_init_guesses, n_start):
        """Generate multiple initial guesses for multi-start optimization."""
        if n_start == 1:
            return [single_init_guesses]
        
        init_lambdas = np.linspace(0.01, 0.1, n_start)
        init_mus = np.linspace(-0.73, 0.73, n_start)
        init_sigmas = np.linspace(0.01, 0.9, n_start)
        
        multiple_init_guesses = []
        for lambda_ in init_lambdas:
            for mu in init_mus:
                for sigma in init_sigmas:
                    multiple_init_guesses.append([lambda_, mu, sigma])
        
        return multiple_init_guesses
    
    def fit(self, n_start=3, verbose=True):
        """
        Fit psychometric model with multiple starting points.
        
        Parameters
        ----------
        n_start : int, default=3
            Number of different initial guesses to try
        verbose : bool, default=True
            Whether to print progress
        
        Returns
        -------
        OptimizeResult
            Best fit result
        """
        # Group data
        grouped_data = self.group_by_choose_test()
        
        # Extract data for initial guess estimation
        test_durations = grouped_data[self.intensity_variable].values
        standard_durations = grouped_data[self.standard_var].values
        responses = grouped_data['num_of_chose_test'].values
        total_resp = grouped_data['total_responses'].values
        
        # Estimate initial guesses
        percentage_diffs = (test_durations - standard_durations) / standard_durations
        single_init_guesses = self.estimate_initial_guesses(percentage_diffs, responses, total_resp)
        
        # Generate multiple starting points
        multiple_init_guesses = self.generate_multiple_init_guesses(single_init_guesses, n_start)
        
        # Fit with multiple starting points
        best_fit = None
        best_nll = float('inf')
        
        disable_progress = len(multiple_init_guesses) == 1 or not verbose
        
        for init_guess in tqdm(multiple_init_guesses, desc="Fitting", disable=disable_progress):
            fit = self.fit_joint(grouped_data, init_guesses=init_guess)
            
            # Extract data for NLL computation
            conflicts = grouped_data[self.conflict_var].values
            noise_levels = grouped_data[self.sensory_var].values
            
            nll = self.nll_joint(
                fit.x, test_durations, standard_durations, responses, 
                total_resp, conflicts, noise_levels
            )
            
            if nll < best_nll:
                best_nll = nll
                best_fit = fit
        
        self.fit_result = best_fit
        self.fitted_params = best_fit.x
        
        if verbose:
            print(f"\nBest fit NLL: {best_nll:.2f}")
            print(f"Fitted parameters: {self.fitted_params}")
        
        return best_fit
    
    def bin_and_plot_with_error_bars(self, data, bin_method='cut', bins=10, 
                                     bin_range=None, plot=True, color="blue", 
                                     bin_var="delta_dur_percents"):
        """
        Bin data and plot with error bars calculated across participants.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to bin and plot
        bin_method : str
            'cut' or 'manual'
        bins : int
            Number of bins (if bin_method='cut')
        bin_range : array
            Bin edges (if bin_method='manual')
        plot : bool
            Whether to plot
        color : str
            Plot color
        bin_var : str
            Variable to bin on
        
        Returns
        -------
        pd.DataFrame
            Binned summary statistics
        """
        # Check if we have multiple participants
        n_participants = data['participantID'].nunique()
        
        # Get participant-level data
        participant_data = self.group_by_choose_test_with_participants(data)
        
        # Bin the data
        if bin_method == 'cut':
            participant_data['bin'] = pd.cut(
                participant_data[bin_var], bins=bins, 
                labels=False, include_lowest=True
            )
        elif bin_method == 'manual':
            participant_data['bin'] = np.digitize(participant_data[bin_var], bins=bin_range) - 1
        
        # Calculate statistics across participants
        bin_summary = participant_data.groupby('bin').agg(
            x_mean=(bin_var, 'mean'),
            y_mean=('p_choose_test', 'mean'),
            y_sem=('p_choose_test', lambda x: np.std(x) / np.sqrt(len(x)) if len(x) > 1 else 0),
            y_std=('p_choose_test', 'std'),
            n_participants=('participantID', 'nunique'),
            total_resp=('total_responses', 'sum')
        ).reset_index()
        
        if plot and len(bin_summary) > 0:
            # Only show error bars if we have multiple participants
            if n_participants > 1:
                plt.errorbar(
                    bin_summary['x_mean'], bin_summary['y_mean'],
                    yerr=bin_summary['y_sem'],
                    fmt='o', color=color, capsize=5, capthick=2,
                    markersize=8, alpha=0.8, elinewidth=2
                )
            else:
                # Plot without error bars for single participant
                plt.scatter(
                    bin_summary['x_mean'], bin_summary['y_mean'],
                    s=bin_summary['total_resp'] / bin_summary['total_resp'].sum() * 900,
                    color=color, alpha=0.8
                )
            
            # Add participant count
            if not bin_summary['n_participants'].empty:
                n_participants = bin_summary['n_participants'].iloc[0]
                plt.text(
                    0.02, 0.95, f'N = {n_participants}',
                    transform=plt.gca().transAxes, fontsize=16
                )
        
        return bin_summary
    
    def plot_fitted_psychometric(self, show_error_bars=True, figsize=(10, 10)):
        """
        Plot fitted psychometric functions with data.
        
        Parameters
        ----------
        show_error_bars : bool, default=True
            Whether to show error bars across participants
        figsize : tuple, default=(10, 10)
            Figure size
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        # Determine if cross-modal or unimodal
        plotting_crossmodal = not self.fix_mu
        
        if plotting_crossmodal:
            standard_label = "Visual:"
            test_label = "Auditory"
            chose_label = "auditory stimulus perceived longer"
            colors = ["blue", "red"]
        else:
            standard_label = "Standard:"
            test_label = "Test"
            chose_label = "test stimulus perceived longer"
            colors = ["black", "blue", "red"]
        
        plt.figure(figsize=figsize)
        labeled_standard = False
        
        for i, standard_level in enumerate(self.unique_standard):
            for j, audio_noise_level in enumerate(self.unique_sensory):
                for k, conflict_level in enumerate(self.unique_conflict):
                    # Get parameters for this condition
                    lambda_, mu, sigma = self.get_params(
                        self.fitted_params, conflict_level, audio_noise_level
                    )
                    
                    # Calculate statistics
                    weber_fraction = sigma / np.sqrt(2) / standard_level
                    sigma_sensory = sigma / np.sqrt(2)
                    sigma_sensory_linear = standard_level * (np.exp(sigma_sensory) - 1)
                    weber_fraction_linear = sigma_sensory_linear / standard_level
                    
                    pse_stats = self.calculate_pse_stats(mu, sigma, lambda_, standard_level)
                    
                    # Print statistics
                    print(f"\n=== Audio Noise: {audio_noise_level:.2f}, Conflict: {conflict_level:.2f} ===")
                    print(f"Sigma Sensory: {sigma_sensory:.3f}, Sigma sensory (linear): {sigma_sensory_linear:.3f} s")
                    print(f"Weber fraction (linear): {weber_fraction_linear:.3f}")
                    print(f"Raw Parameters - Lambda: {lambda_:.3f}, Mu: {mu:.3f}, Sigma: {sigma:.3f}")
                    print(f"PSE shift (pure): {pse_stats['pse_shift_pure']*1000:+.1f} ms ({pse_stats['pse_shift_percent']:+.1f}%)")
                    
                    # Filter data for this condition
                    df = self.data[round(self.data[self.standard_var], 2) == round(standard_level, 2)]
                    df = df[df[self.sensory_var] == audio_noise_level]
                    df = df[df[self.conflict_var] == conflict_level]
                    
                    df_filtered = self.group_by_choose_test(df)
                    test_durations = df_filtered[self.intensity_variable].values
                    
                    if len(test_durations) == 0:
                        continue
                    
                    # Create smooth curve
                    min_x = min(test_durations) * 0.8
                    max_x = max(test_durations) * 1.2
                    x_smooth = np.linspace(min_x, max_x, 1000)
                    standard_dur_array = np.full_like(x_smooth, standard_level)
                    y = self.psychometric_function(
                        x_smooth, standard_dur_array, lambda_, mu, sigma, self.fix_mu
                    )
                    
                    # Plot curve
                    color = colors[j]
                    labels_dict = {
                        0.1: "Auditory (low noise)", 
                        1.2: "Auditory (high noise)",
                        99: "Visual", 
                        0.03: "High noise"
                    }
                    plt.plot(
                        x_smooth * 1000, y, color=color, linewidth=4,
                        label=f"{labels_dict.get(audio_noise_level, audio_noise_level)}"
                    )
                    
                    # Add reference lines
                    plt.axhline(y=0.5, color='gray', linestyle='--')
                    
                    if not labeled_standard:
                        plt.axvline(standard_level * 1000, linestyle='--')
                        labeled_standard = True
                    
                    # Plot data points
                    if show_error_bars:
                        self.bin_and_plot_with_error_bars(
                            df, bin_method='cut', bins=10, plot=True,
                            color=color, bin_var="testDurMs"
                        )
                    
                    # Labels and formatting
                    font_size = 16
                    plt.xlabel(f"{test_label} duration (ms)", fontsize=font_size)
                    plt.ylabel(f"Proportion {chose_label}", fontsize=font_size)
                    plt.xticks(fontsize=font_size - 2)
                    plt.yticks(fontsize=font_size - 2)
                    plt.legend(fontsize=14, title_fontsize=font_size)
                    plt.text(
                        0.05, 0.9, f"{standard_label} {standard_level*1000:.0f}ms",
                        fontsize=16, ha='left', va='top', transform=plt.gca().transAxes
                    )
                    
                    # Set axis limits
                    plt.xticks(np.arange(0, int(max_x * 1000) + 1, step=250))
                    plt.xlim(0, 1000)
        
        plt.tight_layout()
        plt.show()
    
    def get_condition_params(self, conflict=None, audio_noise=None):
        """
        Get fitted parameters for a specific condition.
        
        Parameters
        ----------
        conflict : float, optional
            Conflict level (uses first if None)
        audio_noise : float, optional
            Audio noise level (uses first if None)
        
        Returns
        -------
        dict
            Dictionary with lambda_, mu, sigma for the condition
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        if conflict is None:
            conflict = self.unique_conflict[0]
        if audio_noise is None:
            audio_noise = self.unique_sensory[0]
        
        lambda_, mu, sigma = self.get_params(self.fitted_params, conflict, audio_noise)
        
        return {
            'lambda': lambda_,
            'mu': mu,
            'sigma': sigma,
            'conflict': conflict,
            'audio_noise': audio_noise
        }
    
    def predict(self, test_dur, standard_dur, conflict=None, audio_noise=None):
        """
        Predict probability of choosing test as longer.
        
        Parameters
        ----------
        test_dur : float or array
            Test duration(s)
        standard_dur : float or array
            Standard duration(s)
        conflict : float, optional
            Conflict level (uses first if None)
        audio_noise : float, optional
            Audio noise level (uses first if None)
        
        Returns
        -------
        float or array
            Predicted probability
        """
        params = self.get_condition_params(conflict, audio_noise)
        return self.psychometric_function(
            test_dur, standard_dur, params['lambda'], 
            params['mu'], params['sigma'], self.fix_mu
        )
