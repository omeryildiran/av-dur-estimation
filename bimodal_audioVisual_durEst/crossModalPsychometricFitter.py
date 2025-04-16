# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.optimize import minimize
from scipy.stats import norm
class PsychometricFitter:
    def __init__(self, condition_vars=None):
        """
        Initialize the fitter with condition variables.
        
        Parameters:
        -----------
        condition_vars : dict
            Dictionary mapping condition variable names to their unique values
            e.g., {'sensoryVar': [0.1, 0.5], 'conflictVar': [-0.5, 0, 0.5]}
        """
        self.condition_vars = condition_vars or {}
        self.param_structure = {}
        self.fitted_params = None
        self.param_indices = {}
        
    def set_conditions(self, data, condition_vars):
        """
        Set the condition variables and their unique values from data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data frame containing the condition variables
        condition_vars : list
            List of condition variable names
        """
        self.condition_vars = {var: sorted(data[var].unique()) for var in condition_vars}
        self._build_param_structure()
        return self
    
    def _build_param_structure(self):
        """Build parameter structure based on condition variables."""
        # Default structure with lapse rate(s)
        self.param_structure = {'lambda': 1}
        
        # Add sigma parameters (one per sensory noise level)
        if 'sensoryVar' in self.condition_vars:
            self.param_structure['sigma'] = len(self.condition_vars['sensoryVar'])
        else:
            self.param_structure['sigma'] = 1
            
        # Add mu parameters (one per combination of sensory noise and conflict)
        if 'sensoryVar' in self.condition_vars and 'conflictVar' in self.condition_vars:
            self.param_structure['mu'] = len(self.condition_vars['sensoryVar']) * len(self.condition_vars['conflictVar'])
        elif 'conflictVar' in self.condition_vars:
            self.param_structure['mu'] = len(self.condition_vars['conflictVar'])
        else:
            self.param_structure['mu'] = 1
        
        # Build parameter indices
        self._build_param_indices()
    
    def _build_param_indices(self):
        """Build mapping between condition values and parameter indices."""
        self.param_indices = {}
        
        # Lambda indices
        lambda_start = 0
        for i in range(self.param_structure['lambda']):
            self.param_indices[f'lambda_{i}'] = lambda_start + i
            
        # Sigma indices
        sigma_start = lambda_start + self.param_structure['lambda']
        if 'sensoryVar' in self.condition_vars:
            for i, sensory_val in enumerate(self.condition_vars['sensoryVar']):
                self.param_indices[f'sigma_{sensory_val}'] = sigma_start + i
        else:
            self.param_indices['sigma'] = sigma_start
            
        # Mu indices
        mu_start = sigma_start + self.param_structure['sigma']
        if 'sensoryVar' in self.condition_vars and 'conflictVar' in self.condition_vars:
            for i, sensory_val in enumerate(self.condition_vars['sensoryVar']):
                for j, conflict_val in enumerate(self.condition_vars['conflictVar']):
                    idx = i * len(self.condition_vars['conflictVar']) + j
                    self.param_indices[f'mu_{sensory_val}_{conflict_val}'] = mu_start + idx
        elif 'conflictVar' in self.condition_vars:
            for j, conflict_val in enumerate(self.condition_vars['conflictVar']):
                self.param_indices[f'mu_{conflict_val}'] = mu_start + j
        else:
            self.param_indices['mu'] = mu_start
    
    def get_param(self, params, param_type, *condition_values):
        """
        Get parameter value for specific conditions.
        
        Parameters:
        -----------
        params : array-like
            Parameter vector
        param_type : str
            Parameter type ('lambda', 'sigma', or 'mu')
        *condition_values : values
            Values for condition variables (order matters)
            
        Returns:
        --------
        float
            Parameter value
        """
        if param_type == 'lambda':
            # For now, assume single lapse rate
            return params[0]
            
        elif param_type == 'sigma':
            if not condition_values or 'sensoryVar' not in self.condition_vars:
                return params[self.param_indices['sigma']]
            else:
                sensory_val = condition_values[0]
                return params[self.param_indices[f'sigma_{sensory_val}']]
                
        elif param_type == 'mu':
            if 'sensoryVar' in self.condition_vars and 'conflictVar' in self.condition_vars:
                if len(condition_values) >= 2:
                    sensory_val, conflict_val = condition_values[:2]
                    return params[self.param_indices[f'mu_{sensory_val}_{conflict_val}']]
            elif 'conflictVar' in self.condition_vars:
                if condition_values:
                    conflict_val = condition_values[0]
                    return params[self.param_indices[f'mu_{conflict_val}']]
            return params[self.param_indices['mu']]
    
    def create_initial_params(self):
        """
        Create initial parameter vector.
        
        Returns:
        --------
        numpy.ndarray
            Initial parameter vector
        """
        total_params = sum(self.param_structure.values())
        params = np.zeros(total_params)
        
        # Set default initial values
        lambda_start = 0
        for i in range(self.param_structure['lambda']):
            params[lambda_start + i] = 0.03  # Default lapse rate
            
        sigma_start = lambda_start + self.param_structure['lambda']
        for i in range(self.param_structure['sigma']):
            params[sigma_start + i] = 0.1  # Default sigma
            
        mu_start = sigma_start + self.param_structure['sigma']
        for i in range(self.param_structure['mu']):
            params[mu_start + i] = 0.0  # Default mu
            
        return params
    
    def negative_log_likelihood_unified(self, params, delta_dur, chose_tests, total_responses, 
                                        conflicts=None, noise_levels=None):
        """
        Compute negative log-likelihood for unified model.
        
        Parameters:
        -----------
        params : array-like
            Parameter vector
        delta_dur : array-like
            Stimulus levels
        chose_tests : array-like
            Number of "test chosen" responses
        total_responses : array-like
            Total number of responses
        conflicts : array-like or None
            Conflict levels (if None, use default)
        noise_levels : array-like or None
            Noise levels (if None, use default)
            
        Returns:
        --------
        float
            Negative log-likelihood
        """
        nll = 0
        
        for i in range(len(delta_dur)):
            x = delta_dur[i]
            
            # Select appropriate parameters based on conditions
            lambda_ = self.get_param(params, 'lambda')
            
            if noise_levels is not None:
                sigma = self.get_param(params, 'sigma', noise_levels[i])
            else:
                sigma = self.get_param(params, 'sigma')
                
            if conflicts is not None and noise_levels is not None:
                mu = self.get_param(params, 'mu', noise_levels[i], conflicts[i])
            elif conflicts is not None:
                mu = self.get_param(params, 'mu', conflicts[i])
            else:
                mu = self.get_param(params, 'mu')
            
            # Calculate probability
            p = self.psychometric_function(x, lambda_, mu, sigma)
            
            # Handle numerical issues
            epsilon = 1e-9
            p = np.clip(p, epsilon, 1 - epsilon)
            
            # Add to negative log-likelihood
            nll += -1 * (chose_tests[i] * np.log(p) + (total_responses[i] - chose_tests[i]) * np.log(1 - p))
            
        return nll
    
    def fit(self, levels, responses, total_responses, conflicts=None, noise_levels=None, 
            init_params=None, use_multiple_starts=False):
        """
        Fit the psychometric function.
        
        Parameters:
        -----------
        levels : array-like
            Stimulus levels
        responses : array-like
            Number of "test chosen" responses
        total_responses : array-like
            Total number of responses
        conflicts : array-like or None
            Conflict levels
        noise_levels : array-like or None
            Noise levels
        init_params : array-like or None
            Initial parameter values
        use_multiple_starts : bool
            Whether to use multiple starting points
            
        Returns:
        --------
        scipy.optimize.OptimizeResult
            Optimization result
        """
        if init_params is None:
            init_params = self.create_initial_params()
            
        total_params = sum(self.param_structure.values())
        bounds = [(0, 0.2)] * self.param_structure['lambda']  # Lambda bounds
        bounds += [(0.01, 1)] * self.param_structure['sigma']  # Sigma bounds
        bounds += [(-0.4, 0.4)] * self.param_structure['mu']  # Mu bounds
        
        if use_multiple_starts:
            return self._fit_multiple_starts(levels, responses, total_responses, 
                                           conflicts, noise_levels, bounds)
        else:
            result = minimize(
                self.negative_log_likelihood_unified,
                x0=init_params,
                args=(levels, responses, total_responses, conflicts, noise_levels),
                bounds=bounds,
                method='L-BFGS-B'
            )
            self.fitted_params = result.x
            return result
    
    def _fit_multiple_starts(self, levels, responses, total_responses, 
                           conflicts, noise_levels, bounds, n_starts=5):
        """Fit with multiple starting points."""
        # Implementation depends on your specific needs
        # You could generate multiple starting points and select the best fit
        pass
    
    def plot_fit(self, ax=None, data=None, sensory_val=None, conflict_val=None, 
                color=None, label=None, show_params=True):
        """
        Plot fitted psychometric function for specific conditions.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on
        data : pandas.DataFrame or None
            Data to plot
        sensory_val : float or None
            Sensory noise level
        conflict_val : float or None
            Conflict level
        color : str or None
            Line color
        label : str or None
            Line label
        show_params : bool
            Whether to show parameters on the plot
            
        Returns:
        --------
        matplotlib.axes.Axes
            Axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        if self.fitted_params is None:
            raise ValueError("Must fit model before plotting")
            
        # Get parameters for these conditions
        lambda_ = self.get_param(self.fitted_params, 'lambda')
        sigma = self.get_param(self.fitted_params, 'sigma', sensory_val)
        mu = self.get_param(self.fitted_params, 'mu', sensory_val, conflict_val)
        
        # Plot curve
        x = np.linspace(-1, 1, 100)
        y = self.psychometric_function(x, lambda_, mu, sigma)
        line = ax.plot(x, y, color=color, label=label)
        
        # Show parameters if requested
        if show_params:
            param_text = f"λ = {lambda_:.3f}\nμ = {mu:.3f}\nσ = {sigma:.3f}"
            ax.text(0.05, 0.05, param_text, transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.5))
            
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        return ax
    
    def psychometric_function(self, x, lambda_, mu, sigma):
        """Psychometric function implementation."""
        p = lambda_/2 + (1-lambda_) * norm.cdf((x - mu) / sigma)
        return p
    


def bin_and_plot(data, bin_method='cut', bins=10, bin_range=None, plot=True,color="blue"):
    if bin_method == 'cut':
        data['bin'] = pd.cut(data[intensityVariable], bins=bins, labels=False, include_lowest=True, retbins=False)
    elif bin_method == 'manual':
        data['bin'] = np.digitize(data[intensityVariable], bins=bin_range) - 1
    
    grouped = data.groupby('bin').agg(
        x_mean=(intensityVariable, 'mean'),
        y_mean=('p_choose_test', 'mean'),
        total_resp=('total_responses', 'sum')
    )

    if plot:
        plt.scatter(grouped['x_mean'], grouped['y_mean'], s=grouped['total_resp']/data['total_responses'].sum()*900, color=color)




#load data
dataName="_mainExpAvDurEstimate_2025-04-15_10h10.47.483.csv"
#"_mainExpAvDurEstimate_2025-03-27_15h13.32.171.csv"
#"_visualDurEstimate_2025-03-12_20h35.26.573.csv"

data = pd.read_csv("mainExpAvDurEstimate/dataAvMain/"+dataName)
data['avgAVDeltaS'] = (data['deltaDurS'] + (data['recordedDurVisualTest'] - data['recordedDurVisualStandard'])) / 2
# Calculate deltaDurPercentVisual just as the difference between the test and standard visual durations over the standard visual duration
data['deltaDurPercentVisual'] = ((data['recordedDurVisualTest'] - data['recordedDurVisualStandard']) / data['recordedDurVisualStandard'] 
)
data['avgAVDeltaPercent'] = data[['delta_dur_percents', 'deltaDurPercentVisual']].mean(axis=1)
data

# Define columns for chosing test or standard
data['chose_test'] = (data['responses'] == data['order']).astype(int)
data['chose_standard'] = (data['responses'] != data['order']).astype(int)
try:
    data["riseDur"]>1
except:
    data["riseDur"]=1

data['standard_dur']=data['standardDur']
data[:3]
data = data[data['audNoise'] != 0]
data['visualPSEBias'] = data['recordedDurVisualStandard'] -data["standardDur"]-data['conflictDur']
data=data[data['recordedDurVisualStandard'] <=998]
# print column names
print(data.columns)
intensityVariable = "delta_dur_percents"
sensoryVar="audNoise"
standardVar="standardDur"
conflictVar="conflictDur"

uniqueSensory = data[sensoryVar].unique()
uniqueStandard = data[standardVar].unique()
uniqueConflict = data[conflictVar].unique()
print(f"uniqueSensory: {uniqueSensory} \n uniqueStandard: {uniqueStandard} \n uniqueConflict: {uniqueConflict}")

def groupByChooseTest(x):
    grouped = x.groupby([intensityVariable, sensoryVar, standardVar,conflictVar]).agg(
        num_of_chose_test=('chose_test', 'sum'),
        total_responses=('responses', 'count'),
        num_of_chose_standard=('chose_standard', 'sum'),
    ).reset_index()
    grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']

    return grouped

groupedData = groupByChooseTest(data)
groupedData['p_chose_test'] = groupedData['num_of_chose_test'] / groupedData['total_responses']
groupedData['p_chose_test'] = groupedData['num_of_chose_test'] / groupedData['total_responses']
groupedData['delta_dur_percents'] = groupedData['delta_dur_percents'].astype(float)
groupedData['delta_dur_percents'] = groupedData['delta_dur_percents'].astype(float)

# Define columns for chosing test or standard
data['chose_test'] = (data['responses'] == data['order']).astype(int)
data['chose_standard'] = (data['responses'] != data['order']).astype(int)
try:
    data["riseDur"]>1
except:
    data["riseDur"]=1

data['standard_dur']=data['standardDur']
data[:3]
data = data[data['audNoise'] != 0]
data['visualPSEBias'] = data['recordedDurVisualStandard'] -data["standardDur"]-data['conflictDur']
data=data[data['recordedDurVisualStandard'] <=998]


# 1. Initialize the fitter with your condition structure
fitter = PsychometricFitter()
fitter.set_conditions(data, [sensoryVar, conflictVar, standardVar])

# 2. Create initial parameter vector
init_params = fitter.create_initial_params()

# 3. Fit the model
result = fitter.fit(
    levels=groupedData[intensityVariable],
    responses=groupedData['num_of_chose_test'],
    total_responses=groupedData['total_responses'],
    conflicts=groupedData['conflictDur'],
    noise_levels=groupedData['audNoise'],
    init_params=init_params
)

# 4. Plot results for specific conditions
fig, axes = plt.subplots(len(uniqueSensory), len(uniqueConflict), figsize=(12, 8))

for i, sensory_val in enumerate(uniqueSensory):
    for j, conflict_val in enumerate(uniqueConflict):
        ax = axes[i, j]
        
        # Filter data for these conditions
        df = data[(data['audNoise'] == sensory_val) & (data['conflictDur'] == conflict_val)]
        filtered_df = groupByChooseTest(df)
        
        # Plot data points
        bin_and_plot(filtered_df)
        
        # Plot fitted curve
        fitter.plot_fit(
            ax=ax, 
            sensory_val=sensory_val, 
            conflict_val=conflict_val,
            color='blue', 
            label=f'Noise: {sensory_val}, Conflict: {conflict_val}'
        )
        
        # Set title and labels
        ax.set_title(f'Noise: {sensory_val}, Conflict: {conflict_val}')
        ax.set_xlabel('Delta Duration')
        ax.set_ylabel('P(Choose Test)')

plt.tight_layout()
plt.show()