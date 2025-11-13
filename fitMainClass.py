import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, linregress
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings('ignore')
from loadData import loadData

class fitPychometric:
    def __init__(self, data, intensityVar='logDurRatio', allIndependent=False, sharedSigma=True,sensoryVar = 'audNoise', 
                 standardVar = 'standardDur', conflictVar = 'conflictDur', dataName=None):
        # Create log ratio intensity variable for Weber's law compliance
        data = self._create_log_ratio_variable(data)
        
        self.data = data
        self.intensityVar = intensityVar
        self.allIndependent = allIndependent
        self.sharedSigma = sharedSigma
        self.sensoryVar = 'audNoise'
        self.standardVar = 'standardDur'
        self.conflictVar = 'conflictDur'
        self.uniqueSensory = data[self.sensoryVar].unique()
        self.uniqueStandard = data[self.standardVar].unique()
        self.uniqueConflict = sorted(data[self.conflictVar].unique())
        self.nLambda = 3  # Now fixed to 3 lambda groups based on conflict ranges
        self.nSigma = len(self.uniqueSensory)  # Sigma varies by sensory condition
        self.nMu = len(self.uniqueConflict) * len(self.uniqueSensory) 
        self.dataName= None # placeholder for data name if needed 
        
    def _create_log_ratio_variable(self, data):
        """
        Create log ratio intensity variable for Weber's law compliance.
        
        Creates logDurRatio = log(testDurS / standardDur) which accounts for 
        scalar variability and Weber's law. This makes the psychometric model
        work in log space where noise scales appropriately.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data containing testDurS and standardDur columns
            
        Returns:
        --------
        pandas.DataFrame
            Data with added logDurRatio column
        """
        # Create log ratio: log(test/standard)
        # Add small epsilon to avoid log(0) issues
        epsilon = 1e-10
        data = data.copy()
        
        # Ensure we have the required columns
        if 'testDurS' not in data.columns or 'standardDur' not in data.columns:
            raise ValueError("Data must contain 'testDurS' and 'standardDur' columns")
        
        # Calculate log ratio
        test_dur = np.maximum(data['testDurS'], epsilon)
        standard_dur = np.maximum(data['standardDur'], epsilon)
        
        # Log ratio = log(test/standard) = log(test) - log(standard)
        # This accounts for Weber's law: equal log differences represent equal proportional differences
        # e.g., log(0.6/0.5) = log(1.2) ≈ 0.182, log(1.2/1.0) = log(1.2) ≈ 0.182
        data['logDurRatio'] = np.log(test_dur) - np.log(standard_dur)
        
        # Also keep the original deltaDurS for backward compatibility
        if 'deltaDurS' not in data.columns:
            data['deltaDurS'] = data['testDurS'] - data['standardDur']
            
        print(f"Created logDurRatio variable: range [{data['logDurRatio'].min():.3f}, {data['logDurRatio'].max():.3f}]")
        print(f"  → This represents log(test/standard) for Weber's law compliance")
            
        return data


    def groupByChooseTest(self, x, groupArgs=None):
        """Group data by specified arguments and calculate test choice statistics."""
        if groupArgs is None:
            groupArgs = [self.intensityVar,self.sensoryVar, self.standardVar, self.conflictVar]
        grouped = x.groupby(groupArgs).agg(
            num_of_chose_test=('chose_test', 'sum'),
            total_responses=('responses', 'count'),
            num_of_chose_standard=('chose_standard', 'sum'),
        ).reset_index()
        grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']
        return grouped

    def compute_sigma_from_slope(self, slope, lapse_rate=0.02):
        sigma = (1 - lapse_rate) / (np.sqrt(2 * np.pi) * slope) * np.exp(-0.5)
        return sigma

    def bin_and_plot(self, data, bin_method='cut', bins=10, bin_range=None, plot=True, color="blue"):
        if len(data) == 0:
            return
            
        if bin_method == 'cut':
            data['bin'] = pd.cut(data[self.intensityVar], bins=bins, labels=False, include_lowest=True, retbins=False)
        elif bin_method == 'manual':
            data['bin'] = np.digitize(data[self.intensityVar], bins=bin_range) - 1
        
        grouped = data.groupby('bin').agg(
            x_mean=(self.intensityVar, 'mean'),
            y_mean=('p_choose_test', 'mean'),
            total_resp=('total_responses', 'sum')
        ).dropna()  # Remove any NaN values

        if plot and len(grouped) > 0:
            # Calculate point sizes: minimum 50, maximum 300, scaled by response count
            max_resp = grouped['total_resp'].max()
            min_resp = grouped['total_resp'].min()
            if max_resp > min_resp:
                normalized_sizes = 50 + (grouped['total_resp'] - min_resp) / (max_resp - min_resp) * 250
            else:
                normalized_sizes = 100  # Default size if all responses are equal
                
            plt.scatter(grouped['x_mean'], grouped['y_mean'], 
                       s=normalized_sizes, 
                       color=color, 
                       alpha=0.8, 
                       edgecolors='white', 
                       linewidth=1.5,
                       zorder=5)  # Put points on top

    def estimate_initial_guesses(self, levels, chooseTest, totalResp):
        """
        Estimate initial guesses for lambda, mu, and sigma with slope adjustment and sigma regularization.
        """
        intensities = levels
        chose_test = chooseTest
        total_resp = totalResp

        # Compute proportion of "chose test"
        proportions = chose_test / total_resp

        # Perform linear regression to estimate slope and intercept
        slope, intercept, _, _, _ = linregress(intensities, proportions)
        mu_guess = (0.5 - intercept) / slope

        lapse_rate_guess = 0.03  # 3% as a reasonable guess
        sigma_guess = self.compute_sigma_from_slope(slope, lapse_rate_guess) - 0.1

        # Regularize sigma to avoid overestimation
        intensity_range = np.abs(max(intensities)) - np.abs(min(intensities))

        return [lapse_rate_guess, mu_guess, sigma_guess]
    

    def _get_lambda_index(self, conflict):
        """
        Determine which lambda parameter to use based on conflict value.
        
        Parameters:
        -----------
        conflict : float
            The conflict value
            
        Returns:
        --------
        int
            Index of the lambda parameter to use (0, 1, or 2)
        """
        # Group 1: lambda[0] for conflicts [0, -0.17, 0.25]
        if np.isclose(conflict, 0.0, atol=1e-3) or np.isclose(conflict, -0.17, atol=1e-3) or np.isclose(conflict, 0.25, atol=1e-3):
            return 0
        # Group 2: lambda[1] for conflicts [-0.08, 0.17]
        elif np.isclose(conflict, -0.08, atol=1e-3) or np.isclose(conflict, 0.17, atol=1e-3):
            return 1
        # Group 3: lambda[2] for conflicts [-0.25, 0.08]
        elif np.isclose(conflict, -0.25, atol=1e-3) or np.isclose(conflict, 0.08, atol=1e-3):
            return 2
        else:
            raise ValueError(f"Conflict value {conflict} does not match any predefined group")

    def getParams(self, params, conflict, audio_noise):
        if self.allIndependent and not self.sharedSigma:
            noise_idx = np.where(self.uniqueSensory == round(audio_noise, 3))[0][0]
            conflict_idx = np.where(self.uniqueConflict == round(conflict, 3))[0][0]
            nCond = len(self.uniqueSensory) * len(self.uniqueConflict)
            cond_idx = noise_idx * len(self.uniqueConflict) + conflict_idx
            lambda_ = params[cond_idx * 3 + 0]
            mu     = params[cond_idx * 3 + 1]
            sigma  = params[cond_idx * 3 + 2]
            return lambda_, mu, sigma
        elif not self.allIndependent and self.sharedSigma:
            # Use conflict-based lambda grouping
            lambda_idx = self._get_lambda_index(conflict)
            lambda_ = params[lambda_idx]
            
            noise_idx_array = np.where(self.uniqueSensory == audio_noise)[0]
            if len(noise_idx_array) == 0:
                raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
            noise_idx = noise_idx_array[0]
            
            # Sigma starts after the 3 lambda parameters
            sigma = params[3 + noise_idx]
            
            conflict_idx_array = np.where(np.isclose(self.uniqueConflict, conflict, atol=1e-1))[0]
            if len(conflict_idx_array) == 0:
                raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
            conflict_idx = conflict_idx_array[0]
            noise_offset = noise_idx * len(self.uniqueConflict)
            
            # Mu starts after lambdas and sigmas
            mu_idx = 3 + len(self.uniqueSensory) + noise_offset + conflict_idx
            mu = params[mu_idx]
            return lambda_, mu, sigma
        elif not self.allIndependent and not self.sharedSigma:
            # Use conflict-based lambda grouping
            lambda_idx = self._get_lambda_index(conflict)
            lambda_ = params[lambda_idx]
            
            noise_idx_array = np.where(self.uniqueSensory == round(audio_noise, 3))[0]
            if len(noise_idx_array) == 0:
                raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
            conflict_idx_array = np.where(self.uniqueConflict == conflict)[0]
            if len(conflict_idx_array) == 0:
                raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
            noise_idx = noise_idx_array[0]
            conflict_idx = conflict_idx_array[0]
            nConditions = len(self.uniqueConflict) * len(self.uniqueSensory)
            cond_idx = conflict_idx * len(self.uniqueSensory) + noise_idx
            
            # Mu starts after the 3 lambda parameters
            mu = params[3 + cond_idx]
            # Sigma starts after lambdas and mus
            sigma = params[3 + nConditions + cond_idx]
            return lambda_, mu, sigma
        else:
            raise ValueError("Parameter configuration not supported.")

    def getParamIndexes(self, params, conflict, audio_noise):
        lambda_ = params[0]
        noise_idx_array = np.where(self.uniqueSensory == round(audio_noise,3))[0]
        if len(noise_idx_array) == 0:
            raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
        noise_idx = noise_idx_array[0]
        conflict_idx_array = np.where(self.uniqueConflict==round(conflict,3))[0]
        if len(conflict_idx_array) == 0:
            raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
        conflict_idx = conflict_idx_array[0]
        sigma_idx = noise_idx + 1
        noise_offset = noise_idx * len(self.uniqueConflict)
        mu_idx = self.nLambda + ((len(params)-1)//2) + (conflict_idx+noise_idx)
        return lambda_, mu_idx, sigma_idx

    def psychometric_function(self, x, lambda_, mu, sigma):
        """
        Psychometric function using log-ratio and z-score normalization.
        
        This implements Weber's law compliant psychometric function:
        1. Apply bias (mu) and normalize by discrimination parameter (sigma)
        2. Use standard normal CDF for monotonicity
        3. Add lapse rate (lambda)
        
        Parameters:
        -----------
        x : array-like
            Log ratio values (log(test/standard))
        lambda_ : float
            Lapse rate parameter
        mu : float  
            Bias parameter (PSE in log space)
            Represents log(test/standard) at Point of Subjective Equality
            To convert to PSE bias in time units: bias_ms = standard_ms * mu
        sigma : float
            Discrimination parameter (JND in log space)
            
        Returns:
        --------
        array-like
            Probability of choosing test as longer
        """
        # Apply bias and normalize by discrimination parameter
        z_score = (x - mu) / sigma
        
        # Use standard normal CDF (this ensures monotonicity)
        p_longer = norm.cdf(z_score)
        
        # Add lapse rate: final probability accounts for random responses
        p_final = lambda_/2 + (1-lambda_) * p_longer
        
        return p_final

    def negative_log_likelihood(self, params, delta_dur, chose_test, total_responses):
        lambda_, mu, sigma = params
        p = self.psychometric_function(delta_dur, lambda_, mu, sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        log_likelihood = np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
        return -log_likelihood



    # Fit psychometric function
    def fit_psychometric_function(self, levels, nResp, totalResp, init_guesses=[0, 0, 0]):
        bounds = [(0, 0.5), (-2, +2), (0.01, 1)]  # Reasonable bounds
        result = minimize(
            self.negative_log_likelihood,
            x0=init_guesses,
            args=(levels, nResp, totalResp),
            bounds=bounds,
            method='L-BFGS-B'
        )
        return result.x

    # Fitting function for joint model
    def fitJoint(self, grouped_data, initGuesses):
        if self.allIndependent:
            nCond = len(self.uniqueSensory) * len(self.uniqueConflict)
            initGuesses = (initGuesses * nCond)[:nCond * 3]
            intensities = grouped_data[self.intensityVar]
            chose_tests = grouped_data['num_of_chose_test']
            total_responses = grouped_data['total_responses']
            conflicts = grouped_data[self.conflictVar]
            noise_levels = grouped_data[self.sensoryVar]
            bounds = []
            for _ in range(nCond):
                bounds.extend([(0, 0.4), (-2, 2), (0.01, 2)])
        elif self.sharedSigma:
            nSensoryVar = len(self.uniqueSensory)
            nConflictVar = len(self.uniqueConflict)
            nLambda = 3  # Fixed to 3 lambda groups
            # Structure: [lambda1, lambda2, lambda3, sigma1, sigma2, ..., mu1, mu2, ...]
            initGuesses = [initGuesses[0]] * nLambda + [initGuesses[2]] * nSensoryVar + [initGuesses[1]] * nSensoryVar * nConflictVar
            intensities = grouped_data[self.intensityVar]
            chose_tests = grouped_data['num_of_chose_test']
            total_responses = grouped_data['total_responses']
            conflicts = grouped_data[self.conflictVar]
            noise_levels = grouped_data[self.sensoryVar]
            bounds = [(0, 0.4)] * nLambda + [(0.01, +1)] * nSensoryVar + [(-1, +1)] * nSensoryVar * nConflictVar
        else:
            nSensoryVar = len(self.uniqueSensory)
            nConflictVar = len(self.uniqueConflict)
            nLambda = 3  # Fixed to 3 lambda groups
            nConditions = nSensoryVar * nConflictVar
            # Structure: [lambda1, lambda2, lambda3, mu1, mu2, ..., sigma1, sigma2, ...]
            initGuesses = [initGuesses[0]] * nLambda + [initGuesses[1]] * nConditions + [initGuesses[2]] * nConditions
            intensities = grouped_data[self.intensityVar]
            chose_tests = grouped_data['num_of_chose_test']
            total_responses = grouped_data['total_responses']
            conflicts = grouped_data[self.conflictVar]
            noise_levels = grouped_data[self.sensoryVar]
            bounds = [(0, 0.4)] * nLambda + [(-2, +2)] * nConditions + [(0.01, +2)] * nConditions

        result = minimize(
            self.nLLJoint,
            x0=initGuesses,
            args=(intensities, chose_tests, total_responses, conflicts, noise_levels),
            bounds=bounds,
            method='L-BFGS-B'
        )
        return result

    def nLLJoint(self, params, delta_dur, responses, total_responses, conflicts, noise_levels):
        lam = np.empty(len(delta_dur))
        mu = np.empty(len(delta_dur))
        sigma = np.empty(len(delta_dur))
        for i in range(len(delta_dur)):
            lam[i], mu[i], sigma[i] = self.getParams(params, conflicts[i], noise_levels[i])
        p = lam / 2 + (1 - lam) * norm.cdf((delta_dur - mu) / sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        nll = -np.sum(responses * np.log(p) + (total_responses - responses) * np.log(1 - p))
        return nll

    # Fit the psychometric function to the grouped data with multiple initial guesses
    def multipleInitGuessesWEstimate(self, singleInitGuesses, nStart):
        initLambdas = np.linspace(0.01, 0.1, nStart)
        initMus = np.linspace(-0.95, 0.95, nStart)
        initSigmas = np.linspace(0.01, 1.5, nStart)
        multipleInitGuesses = []
        if nStart == 1:
            # estimate initial guesses
            initLambdas = [singleInitGuesses[0]]
            initMus = [singleInitGuesses[1]]
            initSigmas = [singleInitGuesses[2]]
            multipleInitGuesses.append([initLambdas[0], initMus[0], initSigmas[0]])
        else:
            for initLambda in initLambdas:
                for initMu in initMus:
                    for initSigma in initSigmas:
                        multipleInitGuesses.append([initLambda, initMu, initSigma])
        return multipleInitGuesses

    def fitMultipleStartingPoints(self, data=None, nStart=1):
        # Use self.data (which has logDurRatio) instead of passed data if not specified
        if data is None:
            data = self.data
        # Ensure the data has the required logDurRatio column
        if 'logDurRatio' not in data.columns:
            data = self._create_log_ratio_variable(data)
            
        # group data and prepare for fitting
        groupedData = self.groupByChooseTest(data)
        levels = groupedData[self.intensityVar].values
        responses = groupedData['num_of_chose_test'].values
        totalResp = groupedData['total_responses'].values
        conflictLevels = groupedData[self.conflictVar].values
        noiseLevels = groupedData[self.sensoryVar].values

        # Prepare multiple initial guesses
        singleInitGuesses = self.estimate_initial_guesses(levels, responses, totalResp)
        multipleInitGuesses = self.multipleInitGuessesWEstimate(singleInitGuesses, nStart)

        # Fit the model with multiple starting points
        best_fit = None
        best_nll = float('inf')
        disable = len(multipleInitGuesses) == 1

        for i in tqdm(range(len(multipleInitGuesses)), desc="Fitting multiple starting points", disable=disable):
            fit = self.fitJoint(groupedData, initGuesses=multipleInitGuesses[i])
            nll = self.nLLJoint(fit.x, levels, responses, totalResp, conflictLevels, noiseLevels)
            if nll < best_nll:
                best_nll = nll
                best_fit = fit

        return best_fit
    

    def plot_fitted_psychometric(self, best_fit, pltTitle="Psychometric Fit"):
        colors = sns.color_palette("viridis", n_colors=len(self.uniqueSensory))
        plt.figure(figsize=(12*2.5, 6*2))

        # Calculate data range for better plotting limits
        data_min = self.data[self.intensityVar].min()
        data_max = self.data[self.intensityVar].max()
        data_range = data_max - data_min
        
        # Use a more conservative range focusing on the central 90% of data plus some margin
        data_5th = self.data[self.intensityVar].quantile(0.05)
        data_95th = self.data[self.intensityVar].quantile(0.95)
        central_range = data_95th - data_5th
        plot_margin = max(central_range * 0.3, 0.2)  # At least 0.2 log units margin
        
        x_min = data_5th - plot_margin
        x_max = data_95th + plot_margin
        
        # Ensure we don't make the range too extreme
        x_min = max(x_min, data_min - 0.5)
        x_max = min(x_max, data_max + 0.5)
        
        print(f"Data range: [{data_min:.3f}, {data_max:.3f}]")
        print(f"Central range (5th-95th percentile): [{data_5th:.3f}, {data_95th:.3f}]")
        print(f"Plot range: [{x_min:.3f}, {x_max:.3f}]")

        for i, standardLevel in enumerate(self.uniqueStandard):
            for j, audioNoiseLevel in enumerate(sorted(self.uniqueSensory)):
                for k, conflictLevel in enumerate(self.uniqueConflict):
                    lambda_, mu, sigma = self.getParams(best_fit.x, conflictLevel, audioNoiseLevel)
                    # Filter the data for the current standard and audio noise levels
                    df = self.data[round(self.data[self.standardVar], 2) == round(standardLevel,2)]
                    df = df[df[self.sensoryVar] == audioNoiseLevel]
                    df = df[df[self.conflictVar] == conflictLevel]
                    dfFiltered = self.groupByChooseTest(df)
                    levels = dfFiltered[self.intensityVar].values
                    if len(levels) == 0:
                        continue
                    responses = dfFiltered['num_of_chose_test'].values
                    totalResponses = dfFiltered['total_responses'].values

                    plt.subplot(1, 2, j+1)
                    # Use adaptive range based on actual data
                    x = np.linspace(x_min, x_max, 500)
                    y = self.psychometric_function(x, lambda_, mu, sigma)
                    color = sns.color_palette("viridis", as_cmap=True)(k / len(self.uniqueConflict))
                    plt.plot(x, y, color=color, label=f"C: {int(conflictLevel*1000)}, $\\lambda$: {lambda_:.2f} $\\mu$: {mu:.2f}, $\\sigma$: {sigma:.2f}", linewidth=4)
                    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
                    plt.xlabel(f"({self.intensityVar}) Log Duration Ratio: log(Test/Standard)")
                    plt.ylabel("P(chose test)")
                    plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
                    plt.legend(fontsize=12, title_fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.grid(alpha=0.3)
                    self.bin_and_plot(dfFiltered, bin_method='cut', bins=10, plot=True, color=color)
                    
                    # Set reasonable axis limits
                    plt.xlim(x_min, x_max)
                    plt.ylim(-0.05, 1.05)
                    
                    plt.tight_layout()
                    plt.grid(True, alpha=0.3)
                    print(f"Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, lambda: {lambda_:.3f} Mu: {mu:.3f}, Sigma: {sigma:.3f}")
        plt.show()
        
    def simulate_dataset(self, params, gdf):
        sim_gdf = gdf.copy()
        lam   = np.empty(len(gdf))
        mu    = np.empty(len(gdf))
        sigma = np.empty(len(gdf))
        for idx, row in gdf.iterrows():
            lam_i, mu_i, sig_i = self.getParams(params,
                                                row[self.conflictVar],
                                                row[self.sensoryVar])
            lam[idx], mu[idx], sigma[idx] = lam_i, mu_i, sig_i
        p_choose_test = self.psychometric_function(sim_gdf[self.intensityVar].values, lam, mu, sigma)
        sim_gdf['num_of_chose_test'] = np.random.binomial(
            n = sim_gdf['total_responses'].values.astype(int),
            p = p_choose_test)
        return sim_gdf

    def paramBootstrap(self, fitParams, nBoots):
        groupedData = self.groupByChooseTest(self.data)
        nBootParams = []
        for _ in tqdm(range(nBoots), desc="Bootstrapping", unit="iteration"):
            simData = self.simulate_dataset(fitParams, groupedData)
            initGuessEstimate = self.estimate_initial_guesses(simData[self.intensityVar].values, 
                                                                simData['num_of_chose_test'], 
                                                                simData['total_responses'])
            bootFit = self.fitJoint(grouped_data=simData, initGuesses=initGuessEstimate)
            nBootParams.append(bootFit.x)
        return np.vstack(nBootParams)

    def plot_conflict_vs_pse(self, best_fit, allBootedFits):
        plt.figure(figsize=(12, 6))
        m = 0
        for i, standardLevel in enumerate(self.uniqueStandard):
            for j, audioNoiseLevel in enumerate(self.uniqueSensory):
                for k, conflictLevel in enumerate(self.uniqueConflict):
                    lambda_, mu, sigma = self.getParams(best_fit.x, conflictLevel, audioNoiseLevel)
                    m += 1        
                    plt.subplot(1, 2, j + 1)
                    color = sns.color_palette("viridis", as_cmap=True)(k / len(self.uniqueConflict))
                    
                    # Convert mu from log space to milliseconds
                    # mu is in log space: log(test/standard) = mu at PSE
                    # So test = standard * exp(mu) at PSE
                    # PSE bias in ms = standard * (exp(mu) - 1)
                    # For small mu, exp(mu) ≈ 1 + mu, so bias ≈ standard * mu
                    pse_bias_ms = standardLevel * 1000 * mu  # Convert to ms
                    
                    plt.scatter(conflictLevel * 1000, pse_bias_ms, color=color, s=100)
                    plt.xlabel("Visual Conflict(ms)")
                    plt.ylabel("PSE Bias (ms)")
                    plt.title(f"Standard: {standardLevel*1000:.0f}ms, Noise: {audioNoiseLevel}")
                    plt.grid()
                    plt.axhline(y=0, color='gray', linestyle='--')
                    plt.axvline(x=0, color='gray', linestyle='--')
                    mu_all = []
                    for fit in allBootedFits:
                        lambda_, muBooted, sigma = self.getParams(fit, conflictLevel, audioNoiseLevel)
                        mu_all.append(muBooted)
                    mu_ci = np.percentile(mu_all, [2.5, 97.5])
                    
                    # Convert confidence intervals to milliseconds
                    pse_ci_ms = standardLevel * 1000 * np.array(mu_ci)
                    lower_err = np.maximum(pse_bias_ms - pse_ci_ms[0], 0)
                    upper_err = np.maximum(pse_ci_ms[1] - pse_bias_ms, 0)
                    
                    plt.errorbar(conflictLevel*1000, pse_bias_ms, yerr=[[lower_err], [upper_err]], fmt='o', color=color, capsize=10
                                , label=f"95% CI: [{pse_ci_ms[0]:.1f}, {pse_ci_ms[1]:.1f}] ms", linewidth=2)
                    
                    # Adjust y-limits based on standard duration
                    max_bias = standardLevel * 1000 * 0.5  # Allow for up to 50% bias
                    plt.ylim(-max_bias, max_bias)
        plt.show()
    def plotStairCases(self, data):
        uniqueStairs = data['current_stair'].unique()
        uniqueStairs = sorted(uniqueStairs)[:-1]
        plt.figure(figsize=(20, 10))
        for idx, stair in enumerate(uniqueStairs):
            df = data[data['current_stair'] == stair].reset_index(drop=True)
            plt.subplot(2, 2, idx + 1)
            for trialN in range(df.shape[0]):
                color = 'green' if df['chose_test'][trialN] == 1 or df['chose_test'][trialN] == "True" else 'red'
                # Use delta_dur_percents if available, otherwise use the current intensity variable
                if 'delta_dur_percents' in df.columns:
                    y_val = df['delta_dur_percents'][trialN]
                    y_data = df['delta_dur_percents']
                    ylabel = "Delta Duration %"
                else:
                    y_val = df[self.intensityVar][trialN]
                    y_data = df[self.intensityVar]
                    ylabel = f"{self.intensityVar}"
                
                plt.scatter(trialN, y_val, color=color, s=60, alpha=0.5)
                plt.plot(y_data, color='blue')
                plt.title(f"Stair {stair}")
                plt.xlabel("Trial Number")
                plt.ylabel(ylabel)
                plt.axhline(y=0, color='gray', linestyle='--')
                if 'delta_dur_percents' in df.columns:
                    plt.ylim(-0.9, 0.9)
                plt.grid()
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit joint psychometric model.")
    parser.add_argument("--dataName", type=str, help="CSV file name in data/ directory")
    parser.add_argument("--sharedSigma", action="store_true", help="Use shared sigma across noise levels")
    args = parser.parse_args()

    dataName = args.dataName
    allIndependent = False
    sharedSigma = args.sharedSigma

    if not dataName:
        dataName = "mu_all.csv"
    pltTitle = dataName.split("_")[0] + " " + dataName.split("_")[1]

    # Load data (assuming loadData returns a DataFrame)
    # You may need to adjust this line to match your actual loadData implementation
    data, dataName = loadData("dt_all.csv")

    fit_model = fitPychometric(data, sharedSigma=1, allIndependent=0)
    fit_model.dataName = dataName

    # Print model configuration
    print(f"\nModel Configuration:")
    print(f"  - Lambda parameters: {fit_model.nLambda} (grouped by conflict ranges)")
    print(f"  - Sigma parameters: {fit_model.nSigma} (one per sensory noise level)")
    print(f"  - Mu parameters: {fit_model.nMu} (one per sensory×conflict combination)")
    print(f"  - Total parameters: {fit_model.nLambda + fit_model.nSigma + fit_model.nMu}")
    print(f"  - Unique conflicts: {fit_model.uniqueConflict}")
    print(f"  - Unique sensory levels: {fit_model.uniqueSensory}\n")

    best_fit = fit_model.fitMultipleStartingPoints(nStart=1)
    print(f"\nFitted parameters: {best_fit.x}")
    print(f"Number of parameters: {len(best_fit.x)}")
    print(f"  - Lambda (3): {best_fit.x[0:3]}")
    print(f"  - Sigma ({fit_model.nSigma}): {best_fit.x[3:3+fit_model.nSigma]}")
    print(f"  - Mu ({fit_model.nMu}): first 5 shown {best_fit.x[3+fit_model.nSigma:3+fit_model.nSigma+5]}...")

    fit_model.plot_fitted_psychometric(best_fit, pltTitle=pltTitle)
    #fit_model.plotStairCases(data)

    # # Plot the relation between conflict and PSE (mu) with confidence intervals
    "Uncomment to PSE plot"
    allBootedFits = fit_model.paramBootstrap(best_fit.x, nBoots=2)
    fit_model.plot_conflict_vs_pse(best_fit, allBootedFits)