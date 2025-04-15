import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, linregress
from scipy.optimize import minimize
import seaborn as sns
from typing import Union, Tuple, List, Optional, Dict

class PsychometricFitter:
    """
    A class for fitting psychometric functions to psychophysical data.
    
    Features:
    - Fitting cumulative Gaussian psychometric functions
    - Maximum likelihood estimation
    - Parameter estimation with multiple starting points
    - Confidence intervals via parametric bootstrapping
    - Visualization of fitted curves and binned data
    """
    
    def __init__(self, use_lapse_rate: bool = True):
        """
        Initialize the PsychometricFitter.
        
        Parameters:
        -----------
        use_lapse_rate : bool
            Whether to include a lapse rate parameter in the model
        """
        self.use_lapse_rate = use_lapse_rate
        self.fitted_params = None
        self.confidence_intervals = None
        self.bootstrap_results = None
        
    def psychometric_function(self, intensities: np.ndarray, 
                             lapse_rate: float, 
                             mu: float, 
                             sigma: float) -> np.ndarray:
        """
        Cumulative normal psychometric function with optional lapse rate.
        
        Parameters:
        -----------
        intensities : np.ndarray
            Stimulus intensities or deltas
        lapse_rate : float
            Lapse rate parameter (0-1)
        mu : float
            Mean of the cumulative Gaussian (PSE)
        sigma : float
            Standard deviation of the cumulative Gaussian (JND)
            
        Returns:
        --------
        np.ndarray
            Probability of choosing test stimulus
        """
        # Cumulative distribution function with mean mu and standard deviation sigma
        cdf = norm.cdf(intensities, loc=mu, scale=sigma)
        # Take into account lapse rate
        return lapse_rate * 0.5 + (1 - lapse_rate) * cdf
    
    def derivative_psychometric_function(self, intensities: np.ndarray,
                                        lapse_rate: float,
                                        mu: float,
                                        sigma: float) -> np.ndarray:
        """
        Derivative of the psychometric function.
        
        Parameters:
        -----------
        intensities : np.ndarray
            Stimulus intensities or deltas
        lapse_rate : float
            Lapse rate parameter (0-1)
        mu : float
            Mean of the cumulative Gaussian (PSE)
        sigma : float
            Standard deviation of the cumulative Gaussian (JND)
            
        Returns:
        --------
        np.ndarray
            Derivative values
        """
        return (1 - lapse_rate) * (1 / (np.sqrt(2 * np.pi) * sigma)) * \
               np.exp(-((intensities - mu) ** 2) / (2 * sigma ** 2))
    
    def _negative_log_likelihood(self, params: np.ndarray,
                               levels: np.ndarray,
                               responses: np.ndarray,
                               total_responses: np.ndarray,
                               fixed_lapse: Optional[float] = None,
                               fixed_sigma: Optional[float] = None) -> float:
        """
        Negative log-likelihood function for optimization.
        
        Parameters:
        -----------
        params : np.ndarray
            Parameters to optimize
        levels : np.ndarray
            Stimulus levels
        responses : np.ndarray
            Number of "test chosen" responses at each level
        total_responses : np.ndarray
            Total number of responses at each level
        fixed_lapse : float or None
            Fixed lapse rate value (if None, lapse rate is optimized)
        fixed_sigma : float or None
            Fixed sigma value (if None, sigma is optimized)
            
        Returns:
        --------
        float
            Negative log-likelihood value
        """
        if fixed_lapse is None and fixed_sigma is None:
            lambda_, mu, sigma = params  # Unpack parameters
        elif fixed_lapse is not None and fixed_sigma is None:
            lambda_ = fixed_lapse
            mu, sigma = params
        elif fixed_sigma is not None and fixed_lapse is not None:
            mu = params[0]
            sigma = fixed_sigma
            lambda_ = fixed_lapse
        else:  # fixed_sigma is not None and fixed_lapse is None
            mu, lambda_ = params
            sigma = fixed_sigma
            
        p = self.psychometric_function(levels, lambda_, mu, sigma)
        epsilon = 1e-9  # Add a small number to avoid log(0)
        p = np.clip(p, epsilon, 1 - epsilon)
        log_likelihood = np.sum(responses * np.log(p) + 
                              (total_responses - responses) * np.log(1 - p))
        return -log_likelihood
    
    def _estimate_initial_guesses(self, levels: np.ndarray,
                               responses: np.ndarray,
                               total_responses: np.ndarray,
                               max_sigma_ratio: float = 0.2) -> List[float]:
        """
        Estimate initial parameter values using linear regression.
        
        Parameters:
        -----------
        levels : np.ndarray
            Stimulus levels
        responses : np.ndarray
            Number of "test chosen" responses at each level
        total_responses : np.ndarray
            Total number of responses at each level
        max_sigma_ratio : float
            Maximum sigma as a proportion of the range of intensities
            
        Returns:
        --------
        List[float]
            Initial guesses for [lambda, mu, sigma]
        """
        proportions = responses / total_responses
        
        # Linear regression to estimate slope and intercept
        slope, intercept, _, _, _ = linregress(levels, proportions)
        mu_guess = (0.5 - intercept) / slope
        
        lapse_rate_guess = 0.03  # 3% as a reasonable guess
        sigma_guess = self._compute_sigma_from_slope(slope, lapse_rate_guess)
        
        # Regularize sigma to avoid overestimation
        intensity_range = np.abs(max(levels) - min(levels))
        max_sigma = intensity_range * max_sigma_ratio
        sigma_guess = min(sigma_guess, max_sigma)
        
        return [lapse_rate_guess, mu_guess, sigma_guess]
    
    def _compute_sigma_from_slope(self, slope: float, lapse_rate: float = 0.02) -> float:
        """
        Compute sigma value from the slope of a linear fit.
        
        Parameters:
        -----------
        slope : float
            Slope of the linear regression
        lapse_rate : float
            Lapse rate parameter
            
        Returns:
        --------
        float
            Sigma value
        """
        sigma = (1 - lapse_rate) / (np.sqrt(2 * np.pi) * slope) * np.exp(-0.5)
        return sigma
    
    def fit(self, levels: np.ndarray,
           responses: np.ndarray,
           total_responses: np.ndarray,
           init_guesses: Optional[List[float]] = None,
           fixed_lapse: Optional[float] = None,
           fixed_sigma: Optional[float] = None,
           use_multiple_starts: bool = True,
           n_starts: int = 5) -> np.ndarray:
        """
        Fit the psychometric function to the data.
        
        Parameters:
        -----------
        levels : np.ndarray
            Stimulus levels
        responses : np.ndarray
            Number of "test chosen" responses at each level
        total_responses : np.ndarray
            Total number of responses at each level
        init_guesses : List[float] or None
            Initial parameter values [lambda, mu, sigma]
        fixed_lapse : float or None
            Fixed lapse rate value (if None, lapse rate is optimized)
        fixed_sigma : float or None
            Fixed sigma value (if None, sigma is optimized)
        use_multiple_starts : bool
            Whether to use multiple starting points
        n_starts : int
            Number of starting points if use_multiple_starts is True
        
        Returns:
        --------
        np.ndarray
            Fitted parameters [lambda, mu, sigma]
        """
        if init_guesses is None:
            init_guesses = self._estimate_initial_guesses(levels, responses, total_responses)
        
        if use_multiple_starts:
            multi_start_guesses = self._generate_multiple_starts(n_starts)
            best_fit = self._fit_multiple_starting_points(
                levels, responses, total_responses, multi_start_guesses,
                fixed_lapse, fixed_sigma
            )
        else:
            bounds = self._get_bounds(fixed_lapse, fixed_sigma)
            params = self._get_params_to_optimize(init_guesses, fixed_lapse, fixed_sigma)
            
            result = minimize(
                self._negative_log_likelihood, 
                x0=params,
                args=(levels, responses, total_responses, fixed_lapse, fixed_sigma),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            best_fit = self._reconstruct_params(result.x, fixed_lapse, fixed_sigma)
        
        self.fitted_params = best_fit
        return best_fit
    
    def _generate_multiple_starts(self, n_starts: int = 5) -> List[List[float]]:
        """
        Generate multiple starting points for optimization.
        
        Parameters:
        -----------
        n_starts : int
            Number of points for each parameter
            
        Returns:
        --------
        List[List[float]]
            List of starting points [lambda, mu, sigma]
        """
        multi_start_guesses = []
        
        # Create grid of starting points
        init_lambdas = np.linspace(0.001, 0.1, n_starts)
        init_mus = np.linspace(-0.2, 0.2, n_starts)
        init_sigmas = np.linspace(0.05, 0.5, n_starts)
        
        for init_lambda in init_lambdas:
            for init_mu in init_mus:
                for init_sigma in init_sigmas:
                    multi_start_guesses.append([init_lambda, init_mu, init_sigma])
        
        return multi_start_guesses
    
    def _fit_multiple_starting_points(self, levels: np.ndarray,
                                    responses: np.ndarray,
                                    total_responses: np.ndarray,
                                    multi_start_guesses: List[List[float]],
                                    fixed_lapse: Optional[float] = None,
                                    fixed_sigma: Optional[float] = None) -> np.ndarray:
        """
        Fit using multiple starting points and select the best fit.
        
        Parameters:
        -----------
        levels : np.ndarray
            Stimulus levels
        responses : np.ndarray
            Number of "test chosen" responses at each level
        total_responses : np.ndarray
            Total number of responses at each level
        multi_start_guesses : List[List[float]]
            List of starting points [lambda, mu, sigma]
        fixed_lapse : float or None
            Fixed lapse rate value
        fixed_sigma : float or None
            Fixed sigma value
            
        Returns:
        --------
        np.ndarray
            Best fitted parameters [lambda, mu, sigma]
        """
        best_fit = None
        best_nll = float('inf')
        
        for init_guesses in multi_start_guesses:
            try:
                bounds = self._get_bounds(fixed_lapse, fixed_sigma)
                params = self._get_params_to_optimize(init_guesses, fixed_lapse, fixed_sigma)
                
                result = minimize(
                    self._negative_log_likelihood, 
                    x0=params,
                    args=(levels, responses, total_responses, fixed_lapse, fixed_sigma),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                fit = self._reconstruct_params(result.x, fixed_lapse, fixed_sigma)
                nll = self._negative_log_likelihood(
                    fit, levels, responses, total_responses,
                    fixed_lapse=None, fixed_sigma=None  # Always evaluate full model
                )
                
                if nll < best_nll:
                    best_nll = nll
                    best_fit = fit
            except Exception as e:
                continue
        
        return best_fit
    
    def _get_bounds(self, fixed_lapse: Optional[float] = None,
                  fixed_sigma: Optional[float] = None) -> List[Tuple[float, float]]:
        """
        Get optimization bounds based on fixed parameters.
        
        Parameters:
        -----------
        fixed_lapse : float or None
            Fixed lapse rate value
        fixed_sigma : float or None
            Fixed sigma value
            
        Returns:
        --------
        List[Tuple[float, float]]
            Bounds for optimization
        """
        if fixed_lapse is None and fixed_sigma is None:
            return [(0, 0.2), (-0.4, 0.4), (0.01, 1)]
        elif fixed_lapse is not None and fixed_sigma is None:
            return [(-0.4, 0.4), (0.01, 1)]
        elif fixed_sigma is not None and fixed_lapse is None:
            return [(0, 0.2), (-0.4, 0.4)]
        else:  # both fixed
            return [(-0.4, 0.4)]
    
    def _get_params_to_optimize(self, init_guesses: List[float],
                              fixed_lapse: Optional[float] = None,
                              fixed_sigma: Optional[float] = None) -> np.ndarray:
        """
        Get parameters to optimize based on which are fixed.
        
        Parameters:
        -----------
        init_guesses : List[float]
            Initial parameter values [lambda, mu, sigma]
        fixed_lapse : float or None
            Fixed lapse rate value
        fixed_sigma : float or None
            Fixed sigma value
            
        Returns:
        --------
        np.ndarray
            Parameters to optimize
        """
        if fixed_lapse is None and fixed_sigma is None:
            return init_guesses
        elif fixed_lapse is not None and fixed_sigma is None:
            return init_guesses[1:]
        elif fixed_sigma is not None and fixed_lapse is None:
            return [init_guesses[0], init_guesses[1]]
        else:  # both fixed
            return [init_guesses[1]]
    
    def _reconstruct_params(self, opt_params: np.ndarray,
                          fixed_lapse: Optional[float] = None,
                          fixed_sigma: Optional[float] = None) -> np.ndarray:
        """
        Reconstruct full parameter array from optimized parameters.
        
        Parameters:
        -----------
        opt_params : np.ndarray
            Optimized parameters
        fixed_lapse : float or None
            Fixed lapse rate value
        fixed_sigma : float or None
            Fixed sigma value
            
        Returns:
        --------
        np.ndarray
            Full parameter array [lambda, mu, sigma]
        """
        if fixed_lapse is None and fixed_sigma is None:
            return opt_params
        elif fixed_lapse is not None and fixed_sigma is None:
            return np.array([fixed_lapse, opt_params[0], opt_params[1]])
        elif fixed_sigma is not None and fixed_lapse is None:
            return np.array([opt_params[0], opt_params[1], fixed_sigma])
        else:  # both fixed
            return np.array([fixed_lapse, opt_params[0], fixed_sigma])
            
    def bootstrap(self, levels: np.ndarray,
                 responses: np.ndarray,
                 total_responses: np.ndarray,
                 n_bootstrap: int = 1000) -> np.ndarray:
        """
        Perform parametric bootstrapping to estimate confidence intervals.
        
        Parameters:
        -----------
        levels : np.ndarray
            Stimulus levels
        responses : np.ndarray
            Number of "test chosen" responses at each level
        total_responses : np.ndarray
            Total number of responses at each level
        n_bootstrap : int
            Number of bootstrap samples
            
        Returns:
        --------
        np.ndarray
            Bootstrap parameter estimates (n_bootstrap x 3)
        """
        if self.fitted_params is None:
            raise ValueError("Must fit the model before bootstrapping")
            
        bootstrap_params = []
        
        for _ in range(n_bootstrap):
            # Generate synthetic data based on fitted parameters
            p = self.psychometric_function(levels, *self.fitted_params)
            simulated_responses = np.random.binomial(n=total_responses, p=p)
            
            # Fit the psychometric function to synthetic data
            init_guesses = self._estimate_initial_guesses(
                levels, simulated_responses, total_responses
            )
            
            try:
                bootstrap_fit = self.fit(
                    levels, simulated_responses, total_responses, 
                    init_guesses=init_guesses, use_multiple_starts=False
                )
                bootstrap_params.append(bootstrap_fit)
            except:
                # If fitting fails, use the original parameters
                bootstrap_params.append(self.fitted_params)
        
        self.bootstrap_results = np.array(bootstrap_params)
        
        # Calculate confidence intervals
        self.confidence_intervals = np.percentile(
            self.bootstrap_results, [2.5, 97.5], axis=0
        )
        
        return self.bootstrap_results
    
    def plot_fit(self, levels: np.ndarray,
                responses: np.ndarray,
                total_responses: np.ndarray,
                bin_data: bool = True,
                n_bins: int = 8,
                plot_ci: bool = False,
                ax: Optional[plt.Axes] = None,
                color: str = 'blue',
                label: Optional[str] = None,
                show_params: bool = True) -> plt.Axes:
        """
        Plot the fitted psychometric function with data points.
        
        Parameters:
        -----------
        levels : np.ndarray
            Stimulus levels
        responses : np.ndarray
            Number of "test chosen" responses at each level
        total_responses : np.ndarray
            Total number of responses at each level
        bin_data : bool
            Whether to bin the data points
        n_bins : int
            Number of bins if bin_data is True
        plot_ci : bool
            Whether to plot confidence intervals
        ax : plt.Axes or None
            Matplotlib axes to plot on
        color : str
            Color for the fitted curve
        label : str or None
            Label for the plot legend
        show_params : bool
            Whether to show parameter values on plot
            
        Returns:
        --------
        plt.Axes
            Matplotlib axes with the plot
        """
        if self.fitted_params is None:
            raise ValueError("Must fit the model before plotting")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        # Prepare smooth curve for plotting
        x = np.linspace(min(levels) - 0.1, max(levels) + 0.1, 100)
        y = self.psychometric_function(x, *self.fitted_params)
        
        # Plot the fitted curve
        ax.plot(x, y, color=color, lw=2, label=label)
        
        # Plot confidence interval if requested
        if plot_ci and self.confidence_intervals is not None:
            ci_low = np.percentile(self.bootstrap_results, 2.5, axis=0)
            ci_high = np.percentile(self.bootstrap_results, 97.5, axis=0)
            
            y_low = self.psychometric_function(x, *ci_low)
            y_high = self.psychometric_function(x, *ci_high)
            
            ax.fill_between(x, y_low, y_high, alpha=0.2, color=color)
        
        # Plot raw or binned data
        if bin_data:
            binned_data = self._bin_data(levels, responses, total_responses, n_bins)
            ax.scatter(
                binned_data['x_mean'], 
                binned_data['p_choose_test'],
                s=binned_data['total_resp']/np.max(binned_data['total_resp'])*200,
                color=color, 
                alpha=0.6, 
                edgecolor='black'
            )
        else:
            proportions = responses / total_responses
            ax.scatter(levels, proportions, color=color, alpha=0.6, edgecolor='black')
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Show parameter values
        if show_params:
            param_text = (
                f"λ = {self.fitted_params[0]:.3f}\n"
                f"μ = {self.fitted_params[1]:.3f}\n"
                f"σ = {self.fitted_params[2]:.3f}"
            )
            
            ax.text(
                0.05, 0.05, param_text, 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
            )
        
        # Set labels and title
        ax.set_xlabel('Stimulus Level')
        ax.set_ylabel('Proportion "Test" Responses')
        ax.set_ylim(-0.05, 1.05)
        
        if label is not None:
            ax.legend()
            
        return ax
    
    def _bin_data(self, levels: np.ndarray,
                responses: np.ndarray,
                total_responses: np.ndarray,
                n_bins: int) -> Dict:
        """
        Bin data points for visualization.
        
        Parameters:
        -----------
        levels : np.ndarray
            Stimulus levels
        responses : np.ndarray
            Number of "test chosen" responses at each level
        total_responses : np.ndarray
            Total number of responses at each level
        n_bins : int
            Number of bins
            
        Returns:
        --------
        Dict
            Dictionary with binned data
        """
        # Create a DataFrame for binning
        df = pd.DataFrame({
            'level': levels,
            'responses': responses,
            'total_resp': total_responses
        })
        
        # Create bins
        df['bin'] = pd.cut(df['level'], bins=n_bins, labels=False)
        
        # Group by bin
        binned = df.groupby('bin').agg(
            x_mean=('level', 'mean'),
            sum_responses=('responses', 'sum'),
            total_resp=('total_resp', 'sum')
        ).reset_index()
        
        # Calculate proportions
        binned['p_choose_test'] = binned['sum_responses'] / binned['total_resp']
        
        return binned
    
    def plot_bootstrap_distributions(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot distributions of bootstrapped parameters.
        
        Returns:
        --------
        Tuple[plt.Figure, List[plt.Axes]]
            Figure and axes with the plots
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap before plotting distributions")
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        param_names = ['Lambda (Lapse Rate)', 'Mu (PSE)', 'Sigma (JND)']
        
        for i, (name, ax) in enumerate(zip(param_names, axes)):
            sns.histplot(self.bootstrap_results[:, i], ax=ax, kde=True)
            ax.axvline(x=self.fitted_params[i], color='red', linestyle='--')
            ax.set_title(name)
            ax.set_xlabel('Parameter Value')
            
            # Add confidence interval
            if self.confidence_intervals is not None:
                low, high = self.confidence_intervals[:, i]
                ax.axvline(x=low, color='gray', linestyle=':')
                ax.axvline(x=high, color='gray', linestyle=':')
                ax.text(
                    0.05, 0.95, 
                    f"95% CI: [{low:.3f}, {high:.3f}]",
                    transform=ax.transAxes,
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.7)
                )
        
        plt.tight_layout()
        return fig, axes
        
    def get_pse(self) -> float:
        """Get the Point of Subjective Equality (PSE)"""
        if self.fitted_params is None:
            raise ValueError("Must fit the model before getting PSE")
        return self.fitted_params[1]
    
    def get_jnd(self) -> float:
        """Get the Just Noticeable Difference (JND)"""
        if self.fitted_params is None:
            raise ValueError("Must fit the model before getting JND")
        return self.fitted_params[2]
    
    def get_lapse_rate(self) -> float:
        """Get the lapse rate"""
        if self.fitted_params is None:
            raise ValueError("Must fit the model before getting lapse rate")
        return self.fitted_params[0]
    
    def get_threshold(self, p: float = 0.75) -> float:
        """
        Get the threshold value for a specific performance level.
        
        Parameters:
        -----------
        p : float
            Performance level (0-1)
            
        Returns:
        --------
        float
            Threshold value
        """
        if self.fitted_params is None:
            raise ValueError("Must fit the model before getting threshold")
            
        lapse, mu, sigma = self.fitted_params
        
        # Adjust p for lapse rate
        p_adj = (p - lapse * 0.5) / (1 - lapse)
        
        # Inverse of CDF
        return norm.ppf(p_adj, loc=mu, scale=sigma)
    


# example usage data
dataName="ek_bimodalDurEst_2025-04-14_17h29.23.677.csv"
dataPath = f"bimodal_audioVisual_durEst/dataBimodal/{dataName}"
noiseVar='audNoise'
data = pd.read_csv(dataPath)
data = data[data['audNoise'] != 0]
# Define columns for chosing test or standard
data['chose_test'] = (data['responses'] == data['order']).astype(int)
data['chose_standard'] = (data['responses'] != data['order']).astype(int)
try:
    print(data[noiseVar]>1)
except:
    data[noiseVar]=1

data['standard_dur']=data['standardDur']


def groupByChooseTest(x):
    grouped = x.groupby(['delta_dur_percents', noiseVar, 'standard_dur']).agg(
        num_of_chose_test=('chose_test', 'sum'),
        total_responses=('responses', 'count'),
        num_of_chose_standard=('chose_standard', 'sum')
    ).reset_index()
    grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']

    return grouped

def groupByStandardDur(x):
    grouped = x.groupby(['delta_dur_percents', noiseVar, 'standard_dur']).agg(
        num_of_chose_test=('chose_test', 'sum'),
        total_responses=('responses', 'count'),
        num_of_chose_standard=('chose_standard', 'sum')
    ).reset_index()
    grouped['pChooseStandard'] = grouped['num_of_chose_standard'] / grouped['total_responses']

    return grouped

grouped=groupByChooseTest(data)
# p_choose_test
#sort the group
grouped = grouped.sort_values([ 'standard_dur'])

analyzer = PsychometricFitter(use_lapse_rate=True)
# Extract data from the DataFrame
levels = grouped['delta_dur_percents'].to_numpy()
responses = grouped['num_of_chose_test'].to_numpy()
total_responses = grouped['total_responses'].to_numpy()
# Fit the psychometric function
fitted_params = analyzer.fit(levels, responses, total_responses)
# Print fitted parameters
print(f"Fitted parameters: {fitted_params}")

