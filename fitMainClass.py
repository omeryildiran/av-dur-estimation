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
    def __init__(self, data, intensityVar='testDurS', allIndependent=True, sharedSigma=False,sensoryVar = 'audNoise', 
                 standardVar = 'standardDur', conflictVar = 'conflictDur', dataName=None):
        self.data = data
        self.intensityVar = intensityVar  # Use raw test duration for log-normal model
        self.allIndependent = allIndependent
        self.sharedSigma = sharedSigma
        self.sensoryVar = 'audNoise'
        self.standardVar = 'standardDur'
        self.conflictVar = 'conflictDur'
        self.uniqueSensory = data[self.sensoryVar].unique()
        self.uniqueStandard = data[self.standardVar].unique()
        self.uniqueConflict = sorted(data[self.conflictVar].unique())
        self.nLambda = len(self.uniqueSensory)
        self.nSigma = len(self.uniqueStandard)
        self.nMu = len(self.uniqueConflict) * len(self.uniqueSensory) 
        self.dataName= None # placeholder for data name if needed 
        
        


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
        if bin_method == 'cut':
            data['bin'] = pd.cut(data[self.intensityVar], bins=bins, labels=False, include_lowest=True, retbins=False)
        elif bin_method == 'manual':
            data['bin'] = np.digitize(data[self.intensityVar], bins=bin_range) - 1
        
        grouped = data.groupby('bin').agg(
            x_mean=(self.intensityVar, 'mean'),
            y_mean=('p_choose_test', 'mean'),
            total_resp=('total_responses', 'sum')
        )

        if plot:
            # Convert seconds to milliseconds for plotting
            plt.scatter(grouped['x_mean']*1000, grouped['y_mean'], s=grouped['total_resp']/data['total_responses'].sum()*900, color=color)

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
            lambda_ = params[0]
            noise_idx_array = np.where(self.uniqueSensory == audio_noise)[0]
            if len(noise_idx_array) == 0:
                raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
            noise_idx = noise_idx_array[0]
            sigma = params[noise_idx + 1]
            conflict_idx_array = np.where(np.isclose(self.uniqueConflict, conflict, atol=1e-1))[0]
            if len(conflict_idx_array) == 0:
                raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
            conflict_idx = conflict_idx_array[0]
            noise_offset = noise_idx * len(self.uniqueConflict)
            mu_idx = self.nLambda + self.nSigma + noise_offset + conflict_idx
            mu = params[mu_idx]
            return lambda_, mu, sigma
        elif not self.allIndependent and not self.sharedSigma:
            lambda_ = params[0]
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
            mu = params[self.nLambda + cond_idx]
            sigma = params[self.nLambda + nConditions + cond_idx]
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

    def psychometric_function(self, test_dur, standard_dur, lambda_, mu, sigma):
        """
        Log-normal observer model for duration estimation.
        
        Parameters:
        - test_dur: test duration(s) in seconds
        - standard_dur: standard duration in seconds (can be scalar or array)
        - lambda_: lapse rate 
        - mu: bias parameter (in log space)
        - sigma: discrimination parameter (in log space)
        """
        # Ensure positive durations
        test_dur = np.maximum(test_dur, 1e-10)
        standard_dur = np.maximum(standard_dur, 1e-10)
        
        # Calculate the log ratio of durations
        log_ratio = np.log(test_dur / standard_dur)
        
        # Apply bias and normalize by discrimination parameter
        z_score = (log_ratio - mu) / sigma
        
        # Use standard normal CDF (this ensures monotonicity)
        p_longer = norm.cdf(z_score)
        
        # Apply lapse rate
        p = lambda_/2 + (1 - lambda_) * p_longer
        
        return p

    def negative_log_likelihood(self, params, test_dur, standard_dur, chose_test, total_responses):
        lambda_, mu, sigma = params
        p = self.psychometric_function(test_dur, standard_dur, lambda_, mu, sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        log_likelihood = np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
        return -log_likelihood



    # Fit psychometric function
    def fit_psychometric_function(self, test_durations, standard_durations, nResp, totalResp, init_guesses=[0, 0, 0]):
        bounds = [(0, 0.25), (-1.0, +1.0), (0.01, 2.0)]  # Bounds for log-normal model
        result = minimize(
            self.negative_log_likelihood,
            x0=init_guesses,
            args=(test_durations, standard_durations, nResp, totalResp),
            bounds=bounds,
            method='L-BFGS-B'
        )
        return result.x

    # Fitting function for joint model
    def fitJoint(self, grouped_data, initGuesses):
        if self.allIndependent:
            nCond = len(self.uniqueSensory) * len(self.uniqueConflict)
            initGuesses = (initGuesses * nCond)[:nCond * 3]
            test_durations = grouped_data[self.intensityVar].values  # Raw test durations
            standard_durations = grouped_data[self.standardVar].values  # Standard durations
            chose_tests = grouped_data['num_of_chose_test'].values
            total_responses = grouped_data['total_responses'].values
            conflicts = grouped_data[self.conflictVar].values
            noise_levels = grouped_data[self.sensoryVar].values
            bounds = []
            for _ in range(nCond):
                bounds.extend([(0, 0.25), (-1.0, 1.0), (0.01, 2)])
        elif self.sharedSigma:
            nSensoryVar = len(self.uniqueSensory)
            nConflictVar = len(self.uniqueConflict)
            nLambda = self.nLambda
            initGuesses = [initGuesses[0]] * nLambda + [initGuesses[1]] * nSensoryVar + [initGuesses[2]] * nSensoryVar * nConflictVar
            test_durations = grouped_data[self.intensityVar].values  # Raw test durations
            standard_durations = grouped_data[self.standardVar].values  # Standard durations
            chose_tests = grouped_data['num_of_chose_test'].values
            total_responses = grouped_data['total_responses'].values
            conflicts = grouped_data[self.conflictVar].values
            noise_levels = grouped_data[self.sensoryVar].values
            bounds = [(0, 0.25)] * nLambda + [(0.01, +2.0)] * nSensoryVar + [(-1.0, +1.0)] * nSensoryVar * nConflictVar
        else:
            nSensoryVar = len(self.uniqueSensory)
            nConflictVar = len(self.uniqueConflict)
            nLambda = self.nLambda
            initGuesses = [initGuesses[0]] * nLambda + [initGuesses[1]] * nSensoryVar * nConflictVar + [initGuesses[2]] * nSensoryVar * nConflictVar
            test_durations = grouped_data[self.intensityVar].values  # Raw test durations
            standard_durations = grouped_data[self.standardVar].values  # Standard durations
            chose_tests = grouped_data['num_of_chose_test'].values
            total_responses = grouped_data['total_responses'].values
            conflicts = grouped_data[self.conflictVar].values
            noise_levels = grouped_data[self.sensoryVar].values
            bounds = [(0, 0.25)] * nLambda + [(-1.0, +1.0)] * nSensoryVar * nConflictVar + [(0.01, +2.0)] * nSensoryVar * nConflictVar

        result = minimize(
            self.nLLJoint,
            x0=initGuesses,
            args=(test_durations, standard_durations, chose_tests, total_responses, conflicts, noise_levels),
            bounds=bounds,
            method='L-BFGS-B'
        )
        return result

    def nLLJoint(self, params, test_durations, standard_durations, responses, total_responses, conflicts, noise_levels):
        """
        Compute negative log likelihood for all conditions using log-space psychometric function.
        """
        nll = 0
        
        # Loop through each data point 
        for i in range(len(test_durations)):
            test_dur = test_durations[i]
            standard_dur = standard_durations[i]
            conflict = conflicts[i]
            audio_noise = noise_levels[i]
            total_response = total_responses[i]
            chose_test = responses[i]
            
            # Get appropriate parameters for this condition
            lambda_, mu, sigma = self.getParams(params, conflict, audio_noise)
            
            # Calculate probability of choosing test using log-space model
            p = self.psychometric_function(test_dur, standard_dur, lambda_, mu, sigma)
            
            # Avoid numerical issues
            epsilon = 1e-9
            p = np.clip(p, epsilon, 1 - epsilon)
            
            # Add to negative log-likelihood
            nll += -1 * (chose_test * np.log(p) + (total_response - chose_test) * np.log(1 - p))
        
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

    def fitMultipleStartingPoints(self, data, nStart=1):
        # group data and prepare for fitting
        groupedData = self.groupByChooseTest(data)
        test_durations = groupedData[self.intensityVar].values  # Raw test durations
        standard_durations = groupedData[self.standardVar].values  # Standard durations
        responses = groupedData['num_of_chose_test'].values
        totalResp = groupedData['total_responses'].values
        conflictLevels = groupedData[self.conflictVar].values
        noiseLevels = groupedData[self.sensoryVar].values

        # For initial guess estimation, convert to percentage differences for compatibility
        percentage_diffs = (test_durations - standard_durations) / standard_durations
        singleInitGuesses = self.estimate_initial_guesses(percentage_diffs, responses, totalResp)
        multipleInitGuesses = self.multipleInitGuessesWEstimate(singleInitGuesses, nStart)

        # Fit the model with multiple starting points
        best_fit = None
        best_nll = float('inf')
        disable = len(multipleInitGuesses) == 1

        for i in tqdm(range(len(multipleInitGuesses)), desc="Fitting multiple starting points", disable=disable):
            fit = self.fitJoint(groupedData, initGuesses=multipleInitGuesses[i])
            nll = self.nLLJoint(fit.x, test_durations, standard_durations, responses, totalResp, conflictLevels, noiseLevels)
            if nll < best_nll:
                best_nll = nll
                best_fit = fit

        return best_fit
    

    def plot_fitted_psychometric(self, best_fit, pltTitle="Psychometric Fit"):
        colors = sns.color_palette("viridis", n_colors=len(self.uniqueSensory))
        plt.figure(figsize=(12*2.5, 6*2))

        for i, standardLevel in enumerate(self.uniqueStandard):
            for j, audioNoiseLevel in enumerate(sorted(self.uniqueSensory)):
                for k, conflictLevel in enumerate(self.uniqueConflict):
                    lambda_, mu, sigma = self.getParams(best_fit.x, conflictLevel, audioNoiseLevel)
                    
                    # Calculate sensory noise and Weber fraction
                    sigmaSensory = sigma/np.sqrt(2)
                    sigmaSensoryLinear = standardLevel * (np.exp(sigmaSensory) - 1)
                    weberFractionLinear = sigmaSensoryLinear / standardLevel
                    
                    # Calculate PSE in linear space
                    pse_pure = standardLevel * np.exp(mu)
                    pse_shift_pure = pse_pure - standardLevel
                    
                    print(f"\n=== Audio Noise: {audioNoiseLevel:.2f}, Conflict: {conflictLevel:.2f} ===")
                    print(f"Raw Parameters - Lambda: {lambda_:.3f}, Mu: {mu:.3f}, Sigma: {sigma:.3f}")
                    print(f"Sensory noise sigma (σ/√2): {sigmaSensory:.3f}")
                    print(f"Sigma sensory (linear): {sigmaSensoryLinear:.3f} s")
                    print(f"Weber fraction (linear): {weberFractionLinear:.3f}")
                    print(f"PSE (pure): {pse_pure*1000:.1f} ms")
                    print(f"PSE shift (pure): {pse_shift_pure*1000:+.1f} ms ({(pse_shift_pure/standardLevel)*100:+.1f}%)")
                    
                    # Filter the data for the current standard and audio noise levels
                    df = self.data[round(self.data[self.standardVar], 2) == round(standardLevel,2)]
                    df = df[df[self.sensoryVar] == audioNoiseLevel]
                    df = df[df[self.conflictVar] == conflictLevel]
                    dfFiltered = self.groupByChooseTest(df)
                    test_durations = dfFiltered[self.intensityVar].values  # Raw test durations
                    if len(test_durations) == 0:
                        continue
                    responses = dfFiltered['num_of_chose_test'].values
                    totalResponses = dfFiltered['total_responses'].values

                    plt.subplot(1, 2, j+1)
                    
                    # Create range of test durations for smooth curve
                    minX = min(test_durations) * 0.8  # Use raw duration range
                    maxX = max(test_durations) * 1.2
                    x_smooth = np.linspace(minX, maxX, 1000)
                    
                    # For plotting, we need to provide both test and standard durations
                    standard_dur_array = np.full_like(x_smooth, standardLevel)  # Constant standard duration
                    y = self.psychometric_function(x_smooth, standard_dur_array, lambda_, mu, sigma)
                    
                    color = sns.color_palette("viridis", as_cmap=True)(k / len(self.uniqueConflict))
                    plt.plot(x_smooth*1000, y, color=color, label=f"C: {int(conflictLevel*1000)}, $\lambda$: {lambda_:.2f} $\mu$: {mu:.2f}, $\sigma$: {sigma:.2f}", linewidth=4)
                    plt.axvline(x=standardLevel*1000, color='gray', linestyle='--')  # Show standard duration line
                    plt.axhline(y=0.5, color='gray', linestyle='--')
                    plt.xlabel(f"Test duration (ms)")
                    plt.ylabel("P(chose test)")
                    plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
                    plt.legend(fontsize=14, title_fontsize=14)
                    plt.grid()
                    self.bin_and_plot(dfFiltered, bin_method='cut', bins=10, plot=True, color=color)
                    plt.text(0.05, 0.8, f"Shared $\lambda$: {lambda_:.2f}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
                    plt.tight_layout()
                    plt.grid(True)
        plt.show()
        
    def simulate_dataset(self, params, gdf):
        """
        Produce one synthetic data set that has the SAME predictors
        (test_dur, standard_dur, conflict, noise, total_responses) as 'gdf' but with
        binomially simulated response counts.
        """
        sim_gdf = gdf.copy()
        lam   = np.empty(len(gdf))
        mu    = np.empty(len(gdf))
        sigma = np.empty(len(gdf))
        for idx, row in gdf.iterrows():
            lam_i, mu_i, sig_i = self.getParams(params,
                                                row[self.conflictVar],
                                                row[self.sensoryVar])
            lam[idx], mu[idx], sigma[idx] = lam_i, mu_i, sig_i
        
        # Use log-space psychometric function
        test_durations = sim_gdf[self.intensityVar].values
        standard_durations = sim_gdf[self.standardVar].values
        p_choose_test = self.psychometric_function(test_durations, standard_durations, lam, mu, sigma)
        
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
        """
        Plot the relation between conflict and PSE (mu) with confidence intervals.
        In log-space model, PSE = standard * exp(mu), so PSE shift = standard * (exp(mu) - 1)
        """
        plt.figure(figsize=(12, 6))
        m = 0
        for i, standardLevel in enumerate(self.uniqueStandard):
            for j, audioNoiseLevel in enumerate(self.uniqueSensory):
                for k, conflictLevel in enumerate(self.uniqueConflict):
                    lambda_, mu, sigma = self.getParams(best_fit.x, conflictLevel, audioNoiseLevel)
                    
                    # Calculate PSE shift in linear space (ms)
                    pse_shift_ms = standardLevel * (np.exp(mu) - 1) * 1000
                    
                    m += 1        
                    plt.subplot(1, 2, j + 1)
                    color = sns.color_palette("viridis", as_cmap=True)(k / len(self.uniqueConflict))
                    plt.scatter(conflictLevel * 1000, pse_shift_ms, color=color, s=100)
                    plt.xlabel("Visual Conflict(ms)")
                    plt.ylabel("PSE shift (ms)")
                    plt.title(f"Standard: {standardLevel}, Noise: {audioNoiseLevel}")
                    plt.grid()
                    plt.axhline(y=0, color='gray', linestyle='--')
                    plt.axvline(x=0, color='gray', linestyle='--')
                    
                    mu_all = []
                    for fit in allBootedFits:
                        lambda_, muBooted, sigma = self.getParams(fit, conflictLevel, audioNoiseLevel)
                        mu_all.append(muBooted)
                    
                    # Convert mu confidence intervals to PSE shift in ms
                    mu_ci = np.percentile(mu_all, [2.5, 97.5])
                    pse_shift_lower = standardLevel * (np.exp(mu_ci[0]) - 1) * 1000
                    pse_shift_upper = standardLevel * (np.exp(mu_ci[1]) - 1) * 1000
                    
                    lower_err = np.maximum(pse_shift_ms - pse_shift_lower, 0)
                    upper_err = np.maximum(pse_shift_upper - pse_shift_ms, 0)
                    plt.errorbar(conflictLevel*1000, pse_shift_ms, yerr=[[lower_err], [upper_err]], fmt='o', color=color, capsize=10
                                , label=f"95% CI: {pse_shift_lower:.2f} - {pse_shift_upper:.2f}", linewidth=2)
                    plt.ylim(-350, 350)
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
                plt.scatter(trialN, df['delta_dur_percents'][trialN], color=color, s=60, alpha=0.5)
                plt.plot(df['delta_dur_percents'], color='blue')
                plt.title(f"Stair {stair}")
                plt.xlabel("Test(stair)-Standard(0.5s) Duration Difference Ratio")
                plt.ylabel("Delta Duration %")
                plt.axhline(y=0, color='gray', linestyle='--')
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
    allIndependent = True
    sharedSigma = args.sharedSigma

    if not dataName:
        dataName = "HH_all.csv"
    pltTitle = dataName.split("_")[0] + " " + dataName.split("_")[1]

    # Load data (assuming loadData returns a DataFrame)
    # You may need to adjust this line to match your actual loadData implementation
    data, dataName = loadData("HH_all.csv")

    fit_model = fitPychometric(data, sharedSigma=sharedSigma, allIndependent=allIndependent)
    fit_model.dataName = dataName

    best_fit = fit_model.fitMultipleStartingPoints(data, nStart=1)
    print(f"Fitted parameters: {best_fit.x}")

    fit_model.plot_fitted_psychometric(best_fit, pltTitle=pltTitle)
    #fit_model.plotStairCases(data)

    # # Plot the relation between conflict and PSE (mu) with confidence intervals
    "Uncomment to PSE plot"
    # allBootedFits = fit_model.paramBootstrap(best_fit.x, nBoots=10)
    # fit_model.plot_conflict_vs_pse(best_fit, allBootedFits)