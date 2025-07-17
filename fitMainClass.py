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
    def __init__(self, data, intensityVar='deltaDurS', allIndependent=True, sharedSigma=False,sensoryVar = 'audNoise', 
                 standardVar = 'standardDur', conflictVar = 'conflictDur'):
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
            plt.scatter(grouped['x_mean'], grouped['y_mean'], s=grouped['total_resp']/data['total_responses'].sum()*900, color=color)

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

    def psychometric_function(self, x, lambda_, mu, sigma):
        cdf = norm.cdf(x, loc=mu, scale=sigma)
        p = lambda_/2 + (1-lambda_) * norm.cdf((x - mu) / sigma)
        return p

    def negative_log_likelihood(self, params, delta_dur, chose_test, total_responses):
        lambda_, mu, sigma = params
        p = self.psychometric_function(delta_dur, lambda_, mu, sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        log_likelihood = np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
        return -log_likelihood



    # Fit psychometric function
    def fit_psychometric_function(self, levels, nResp, totalResp, init_guesses=[0, 0, 0]):
        bounds = [(0, 0.25), (-2, +2), (0.01, 1)]  # Reasonable bounds
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
                bounds.extend([(0, 0.25), (-2, 2), (0.01, 2)])
        elif self.sharedSigma:
            nSensoryVar = len(self.uniqueSensory)
            nConflictVar = len(self.uniqueConflict)
            nLambda = self.nLambda
            initGuesses = [initGuesses[0]] * nLambda + [initGuesses[1]] * nSensoryVar + [initGuesses[2]] * nSensoryVar * nConflictVar
            intensities = grouped_data[self.intensityVar]
            chose_tests = grouped_data['num_of_chose_test']
            total_responses = grouped_data['total_responses']
            conflicts = grouped_data[self.conflictVar]
            noise_levels = grouped_data[self.sensoryVar]
            bounds = [(0, 0.25)] * nLambda + [(0.01, +1)] * nSensoryVar + [(-1, +1)] * nSensoryVar * nConflictVar
        else:
            nSensoryVar = len(self.uniqueSensory)
            nConflictVar = len(self.uniqueConflict)
            nLambda = self.nLambda
            initGuesses = [initGuesses[0]] * nLambda + [initGuesses[1]] * nSensoryVar * nConflictVar + [initGuesses[2]] * nSensoryVar * nConflictVar
            intensities = grouped_data[self.intensityVar]
            chose_tests = grouped_data['num_of_chose_test']
            total_responses = grouped_data['total_responses']
            conflicts = grouped_data[self.conflictVar]
            noise_levels = grouped_data[self.sensoryVar]
            bounds = [(0, 0.25)] * nLambda + [(-2, +2)] * nSensoryVar * nConflictVar + [(0.01, +2)] * nSensoryVar * nConflictVar

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
        initMus = np.linspace(-0.83, 0.83, nStart)
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
                    x = np.linspace(-0.9, 0.9, 500)
                    y = self.psychometric_function(x, lambda_, mu, sigma)
                    color = sns.color_palette("viridis", as_cmap=True)(k / len(self.uniqueConflict))
                    plt.plot(x, y, color=color, label=f"C: {int(conflictLevel*1000)}, $\lambda$: {lambda_:.2f} $\mu$: {mu:.2f}, $\sigma$: {sigma:.2f}", linewidth=4)
                    plt.axvline(x=0, color='gray', linestyle='--')
                    plt.axhline(y=0.5, color='gray', linestyle='--')
                    plt.xlabel(f"({self.intensityVar}) Test(stair-a)-Standard(a) Duration Difference Ratio(%)")
                    plt.ylabel("P(chose test)")
                    plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
                    plt.legend(fontsize=14, title_fontsize=14)
                    plt.grid()
                    self.bin_and_plot(dfFiltered, bin_method='cut', bins=10, plot=True, color=color)
                    plt.text(0.05, 0.8, f"Shared $\lambda$: {lambda_:.2f}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
                    plt.tight_layout()
                    plt.grid(True)
                    print(f"Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, Mu: {mu:.3f}, Sigma: {sigma:.3f}")

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
                    plt.scatter(conflictLevel * 1000, mu * 1000 / 2, color=color, s=100)
                    plt.xlabel("Visual Conflict(ms)")
                    plt.ylabel("PSE (ms)")
                    plt.title(f"Standard: {standardLevel}, Noise: {audioNoiseLevel}")
                    plt.grid()
                    plt.axhline(y=0, color='gray', linestyle='--')
                    plt.axvline(x=0, color='gray', linestyle='--')
                    mu_all = []
                    for fit in allBootedFits:
                        lambda_, muBooted, sigma = self.getParams(fit, conflictLevel, audioNoiseLevel)
                        mu_all.append(muBooted)
                    mu_ci = np.percentile(mu_all, [2.5, 97.5])
                    lower_err = np.maximum(mu*1000/2-mu_ci[0]*1000/2, 0)
                    upper_err = np.maximum(mu_ci[1]*1000/2-mu*1000/2, 0)
                    plt.errorbar(conflictLevel*1000, mu*1000/2, yerr=[[lower_err], [upper_err]], fmt='o', color=color, capsize=10
                                , label=f"95% CI for mu: {mu_ci[0]*1000/2:.2f} - {mu_ci[1]*1000/2:.2f}", linewidth=2)
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