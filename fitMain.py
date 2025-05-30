# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# function for loading data
def loadData(dataName, isShared):
    global data, sharedSigma, intensityVariable, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu
    sharedSigma = isShared  # Set to True if you want to use shared sigma across noise levels
    intensityVariable="delta_dur_percents"

    sensoryVar="audNoise"
    standardVar="standardDur"
    conflictVar="conflictDur"
    


    data = pd.read_csv("data/"+dataName)
    data= data[data['audNoise'] != 0]
    data=data[data['standardDur'] != 0]
    data[standardVar] = data[standardVar].round(2)
    data[conflictVar] = data[conflictVar].round(3)
    uniqueSensory = data[sensoryVar].unique()
    uniqueStandard = data[standardVar].unique()
    uniqueConflict = sorted(data[conflictVar].unique())
    print(f"uniqueSensory: {uniqueSensory} \n uniqueStandard: {uniqueStandard} \n uniqueConflict: {uniqueConflict}")

    #data['avgAVDeltaS'] = (data['deltaDurS'] + (data['recordedDurVisualTest'] - data['recordedDurVisualStandard'])) / 2
    #data['deltaDurPercentVisual'] = ((data['recordedDurVisualTest'] - data['recordedDurVisualStandard']) / data['recordedDurVisualStandard'])
    #data['avgAVDeltaPercent'] = data[['delta_dur_percents', 'deltaDurPercentVisual']].mean(axis=1)

    # Define columns for chosing test or standard
    data['chose_test'] = (data['responses'] == data['order']).astype(int)
    data['chose_standard'] = (data['responses'] != data['order']).astype(int)
    data['visualPSEBias'] = data['recordedDurVisualStandard'] -data["standardDur"]-data['conflictDur']
    data['conflictDur'] = data['conflictDur'].round(3)
    data['standard_dur']=data['standardDur']

    try:
        data["riseDur"]>1
    except:
        data["riseDur"]=1
    
    data[standardVar] = round(data[standardVar], 2)

    data['standard_dur']=round(data['standardDur'],2)
    data["delta_dur_percents"]=round(data["delta_dur_percents"],2)

    #data=data[data['recordedDurVisualStandard'] <=998]
    #data=data[data['recordedDurVisualStandard'] >=0]
    nLambda=len(uniqueStandard)
    nSigma=len(uniqueSensory)
    nMu=len(uniqueConflict)*nSigma
    
    return data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda,nSigma, nMu

intensityVariable="delta_dur_percents"

sensoryVar="audNoise"
standardVar="standardDur"
conflictVar="conflictDur"

def groupByChooseTest(x):
    #print(f"Grouping by {intensityVariable}, {sensoryVar}, {standardVar}, {conflictVar}")
    grouped = x.groupby([intensityVariable, sensoryVar, standardVar,conflictVar]).agg(
        num_of_chose_test=('chose_test', 'sum'),
        total_responses=('responses', 'count'),
        num_of_chose_standard=('chose_standard', 'sum'),
    ).reset_index()
    grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']

    return grouped

# Compute sigma from slope
def compute_sigma_from_slope(slope, lapse_rate=0.02):
    sigma = (1 - lapse_rate) / (np.sqrt(2 * np.pi) * slope)*np.exp(-0.5)
    return sigma

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

from scipy.stats import linregress

def estimate_initial_guesses(levels,chooseTest,totalResp):
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

    #print(slope, intercept)
    lapse_rate_guess= 0.03  # 5% as a reasonable guess
    sigma_guess= compute_sigma_from_slope(slope,lapse_rate_guess)-0.1

    # Regularize sigma to avoid overestimation
    intensity_range = np.abs(max(intensities)) - np.abs(min(intensities))
    
    return [lapse_rate_guess, mu_guess, sigma_guess]


from tqdm import tqdm

def getParams(params, conflict, audio_noise, nLambda, nSigma):

    if sharedSigma==False:
        # Get lambda (lapse rate)
        lambda_ = params[0]    
        # Get sigma based on noise level
        
        # Get noise index safely
        noise_idx_array = np.where(uniqueSensory == round(audio_noise,3))[0]
        if len(noise_idx_array) == 0:
            raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
        
        # Get conflict index safely
        conflict_idx_array = np.where(uniqueConflict==conflict)[0]
        if len(conflict_idx_array) == 0:
            raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
        conflict_idx = conflict_idx_array[0]
        
        noise_idx = noise_idx_array[0]
    #    print(f"noise_idx: {noise_idx}, conflict_idx: {conflict_idx}, noise_offset: {noise_offset}, lenParams: {len(params)}")

        # sigma is after lambda, so we need to find its index
        sigma_idx = nLambda-1  + ((len(params)-1)//2) + ((conflict_idx+1)*(noise_idx+1))#+ nSigma + noise_offset + conflict_idx
        sigma = params[sigma_idx]  # +1 because lambda is first

        noise_offset = noise_idx * len(uniqueConflict)
        # mu is after lambda and sigma, so we need to find its index
        #mu_idx = nLambda+ nSigma + noise_offset + conflict_idx
        mu_idx = nLambda-1 +((conflict_idx+1)*(noise_idx+1))#+ nSigma + noise_offset + conflict_idx
        
        mu = params[mu_idx]

    else:
        # Get lambda (lapse rate)
        lambda_ = params[0]    
        # Get sigma based on noise level
        # Get noise index safely
        noise_idx_array = np.where(uniqueSensory == audio_noise)[0]
        if len(noise_idx_array) == 0:
            raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
        noise_idx = noise_idx_array[0]
        sigma = params[noise_idx + 1]  # +1 because lambda is first
        
        # Get conflict index safely
        conflict_idx_array = np.where(np.isclose(uniqueConflict, conflict, atol=1e-1))[0]
        if len(conflict_idx_array) == 0:
            raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
        conflict_idx = conflict_idx_array[0]
        noise_offset = noise_idx * len(uniqueConflict)
        mu_idx = nLambda + nSigma + noise_offset + conflict_idx
        mu = params[mu_idx]


    return lambda_, mu, sigma
    

def getParamIndexes(params, conflict, audio_noise, nLambda, nSigma):
    # Get lambda (lapse rate)
    lambda_ = params[0]    
    # Get sigma based on noise level
    # Get noise index safely
    noise_idx_array = np.where(uniqueSensory == round(audio_noise,3))[0]
    if len(noise_idx_array) == 0:
        raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
    noise_idx = noise_idx_array[0]
    
    # Get conflict index safely
    conflict_idx_array = np.where(uniqueConflict==round(conflict,3))[0]
    if len(conflict_idx_array) == 0:
        raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
    conflict_idx = conflict_idx_array[0]
    
    # sigma is after lambda, so we need to find its index
    sigma_idx = noise_idx + 1  # +1 because lambda is first 
    
    noise_offset = noise_idx * len(uniqueConflict)
    
    # mu is after lambda and sigma, so we need to find its index
    mu_idx = nLambda +((len(params)-1)//2) + (conflict_idx+noise_idx)#+ nSigma + noise_offset + conflict_idx
    
    return lambda_, mu_idx, sigma_idx

from scipy.stats import norm
from scipy.optimize import minimize
import argparse

def psychometric_function(x, lambda_, mu, sigma):
    # Cumulative distribution function with mean mu and standard deviation sigma
    cdf = norm.cdf(x, loc=mu, scale=sigma) 
    # take into account of lapse rate and return the probability of choosing test
    p = lambda_/2 + (1-lambda_) * norm.cdf((x - mu) / sigma)
    #return lapse_rate * 0.5 + (1 - lapse_rate) * cdf 
    return p

    #p = lambda_/2 + (1-lambda_) * norm.cdf((x - mu) / sigma)

# Negative log-likelihood
def negative_log_likelihood(params, delta_dur, chose_test, total_responses):
    lambda_, mu, sigma = params # Unpack parameters
    
    p = psychometric_function(delta_dur, lambda_, mu, sigma) # Compute probability of choosing test
    epsilon = 1e-9 # Add a small number to avoid log(0) when calculating thxe log-likelihood
    p = np.clip(p, epsilon, 1 - epsilon) # Clip p to avoid log(0) and log(1)
    # Compute the negative log-likelihood
    log_likelihood = np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
    return -log_likelihood


# Fit psychometric function
def fit_psychometric_function(levels,nResp, totalResp,init_guesses=[0,0,0]):
    # then fits the psychometric function
    # order is lambda mu sigma
    #initial_guess = [0, -0.2, 0.05]  # Initial guess for [lambda, mu, sigma]
    bounds = [(0, 0.25), (-2, +2), (0.01, 1)]  # Reasonable bounds
    # fitting is done here
    result = minimize(
        negative_log_likelihood, x0=init_guesses, 
        args=(levels, nResp, totalResp),  #       Pass the data and fixed parameters
        bounds=bounds,
        method='L-BFGS-B' 
    )
    # returns the fitted parameters lambda, mu, sigma
    return result.x


# Update nLLJoint to use getParams
def nLLJoint(params, delta_dur, responses, total_responses, conflicts, noise_levels):
    """
    Compute negative log likelihood for all conditions.
    """
    nll = 0
    
    # Loop through each data point 
    for i in range(len(delta_dur)):
        x = delta_dur[i]
        conflict = conflicts[i]
        audio_noise = noise_levels[i]
        total_response = total_responses[i]
        chose_test = responses[i]
        
        # Get appropriate parameters for this condition
        lambda_, mu, sigma = getParams(params, conflict, audio_noise, nLambda, nSigma)
        
        # Calculate probability of choosing test
        p = psychometric_function(x, lambda_, mu, sigma)
        
        # Avoid numerical issues
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        
        # Add to negative log-likelihood
        nll += -1 * (chose_test * np.log(p) + (total_response - chose_test) * np.log(1 - p))
    
    return nll

# fitting function for joint model
def fitJoint(grouped_data,  initGuesses):
    if sharedSigma==True:
        # Initialize guesses for parameters 
        # lambda, mu,sigma
        initGuesses= [initGuesses[0]]*nLambda +  [initGuesses[1]]*nSensoryVar+[initGuesses[2]]*nSensoryVar*nConflictVar
        
        intensities = grouped_data[intensityVariable]
        chose_tests = grouped_data['num_of_chose_test']
        total_responses = grouped_data['total_responses']
        conflicts = grouped_data[conflictVar]
        noise_levels = grouped_data[sensoryVar]
        
        
        # Set bounds for parameters
        # array of parameters in order of lambda, mu, sigma
        bounds = [(0,0.25)]*nLambda +  [(0.01, +1)]*nSensoryVar+[(-1, +1)]*nSensoryVar*nConflictVar 


        # Minimize negative log-likelihood
        result = minimize(
            nLLJoint,
            x0=initGuesses,
            args=(intensities, chose_tests, total_responses, conflicts, noise_levels),
            bounds=bounds,
            method='L-BFGS-B'  # Use L-BFGS-B for bounded optimization
        )
    else:
        # Initialize guesses for parameters 
        # lambda, mu, sigma
        initGuesses= [initGuesses[0]]*nLambda + [initGuesses[1]]*nSensoryVar*nConflictVar+ [initGuesses[2]]*nSensoryVar*nConflictVar
        
        intensities = grouped_data[intensityVariable]
        chose_tests = grouped_data['num_of_chose_test']
        total_responses = grouped_data['total_responses']
        conflicts = grouped_data[conflictVar]
        noise_levels = grouped_data[sensoryVar]
        
        
        # Set bounds for parameters
        # array of parameters in order of lambda, mu, sigma
        bounds = [(0, 0.25)]*nLambda + [(-2, +2)]*nSensoryVar*nConflictVar + [(0.01, +2)]*nSensoryVar*nConflictVar 

    #print(f" len initGuesses: {len(initGuesses)}")

    # Minimize negative log-likelihood
    result = minimize(
        nLLJoint,
        x0=initGuesses,
        args=(intensities, chose_tests, total_responses, conflicts, noise_levels),
        bounds=bounds,
        method='L-BFGS-B'  # Use L-BFGS-B for bounded optimization
    )
    
    return result

# Fit the psychometric function to the grouped data
def multipleInitGuessesWEstimate(singleInitGuesses, nStart):
    initLambdas=np.linspace(0.01, 0.1, nStart)
    initMus=np.linspace(-0.83, 0.83, nStart)
    initSigmas=np.linspace(0.01, 1.5, nStart)
    multipleInitGuesses = []
    if nStart == 1:
        # estimate initial guesses
        initLambdas = [singleInitGuesses[0]]
        initMus = [singleInitGuesses[1]]
        initSigmas = [singleInitGuesses[2]]
        multipleInitGuesses.append([initLambdas[0], initMus[0], initSigmas[0]])
    else:
        for i, initLambda in enumerate(initLambdas):
            for j, initMu in enumerate(initMus):
                for k, initSigma in enumerate(initSigmas):
                    multipleInitGuesses.append([initLambda, initMu, initSigma])
    return multipleInitGuesses


def fitMultipleStartingPoints(data,nStart=3):
    # group data and prepare for fitting
    groupedData = groupByChooseTest(data)
    global nLambda, nSensoryVar, nConflictVar, uniqueSensory, uniqueConflict
    nSensoryVar = len(uniqueSensory)  # Number of sensory variables
    nConflictVar = len(uniqueConflict)  # Number of conflict variables
    uniqueSensory = data['audNoise'].unique()
    uniqueConflict = sorted(data[conflictVar].unique())
    
    levels = groupedData[intensityVariable].values
    responses = groupedData['num_of_chose_test'].values
    totalResp = groupedData['total_responses'].values
    conflictLevels = groupedData[conflictVar].values
    noiseLevels = groupedData[sensoryVar].values

    # Prepare multiple initial guesses
    singleInitGuesses = estimate_initial_guesses(levels, responses, totalResp)

    multipleInitGuesses = multipleInitGuessesWEstimate(singleInitGuesses, nStart)

    # Fit the model with multiple starting points
    
    best_fit = None
    best_nll = float('inf')  # Initialize with infinity
    disable=False
    if len(multipleInitGuesses)==1:
        disable=True
    
    for i in tqdm(range(len(multipleInitGuesses)), desc="Fitting multiple starting points",disable=disable):
        
        fit = fitJoint(groupedData, initGuesses=multipleInitGuesses[i])
        nll = nLLJoint(fit.x, levels, responses, totalResp, conflictLevels, noiseLevels)

        if nll < best_nll:
            best_nll = nll
            best_fit = fit

    return best_fit

def plot_fitted_psychometric(data, best_fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict, standardVar, sensoryVar, conflictVar, intensityVariable):
    colors = sns.color_palette("viridis", n_colors=len(uniqueSensory))  # Use Set2 palette for different noise levels
    plt.figure(figsize=(12*2.5, 6*2))
    for i, standardLevel in enumerate(uniqueStandard):
        for j, audioNoiseLevel in enumerate(uniqueSensory):
            print(f"standardLevel: {standardLevel}, audioNoiseLevel: {audioNoiseLevel}, uniqueConflict: {uniqueConflict}")
            for k, conflictLevel in enumerate(uniqueConflict):
                lambda_, mu, sigma = getParams(best_fit.x, conflictLevel, audioNoiseLevel, nLambda, nSigma)
                # Filter the data for the current standard and audio noise levels
                df = data[round(data[standardVar], 2) == round(standardLevel,2)]
                df = df[df[sensoryVar] == audioNoiseLevel]
                df = df[df[conflictVar] == conflictLevel]
                dfFiltered = groupByChooseTest(df)
                levels = dfFiltered[intensityVariable].values
                if len(levels) == 0:
                    continue
                responses = dfFiltered['num_of_chose_test'].values
                totalResponses = dfFiltered['total_responses'].values
                
                # Fit the psychometric function
                plt.subplot(1, 2, j+1)
                maxX = max(levels) + 0.1
                minX = min(levels) - 0.1
                x = np.linspace(-0.9, 0.9, 500)
                y = psychometric_function(x, lambda_, mu, sigma)
                color=sns.color_palette("viridis", as_cmap=True)(k / len(uniqueConflict))  # Use a colormap for different conflict levels
                plt.plot(x, y, color=color, label=f"Conflict: {int(conflictLevel*1000)}, PSE: {int(mu*1000)}, $\sigma$: {sigma:.3f}", linewidth=4,)
                plt.axvline(x=0, color='gray', linestyle='--')
                plt.axhline(y=0.5, color='gray', linestyle='--')
                plt.xlabel(f"({intensityVariable}) Test(stair-a)-Standard(a) Duration Difference Ratio(%)")
                plt.ylabel("P(chose test)")
                plt.title(f"Multimodal auditory duration comparison", fontsize=16)
                plt.legend(fontsize=14, title_fontsize=14)
                plt.grid()
                bin_and_plot(dfFiltered, bin_method='cut', bins=10, plot=True, color=color)

                plt.text(0.05, 0.8, f"Shared $\lambda$: {lambda_:.2f}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
                plt.tight_layout()
                plt.grid(True)

                # print the fitted parameters
                print(f"Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, Mu: {mu:.3f}, Sigma: {sigma:.3f}")
    plt.show()


def simulate_dataset(params, gdf):
    """
    Produce one synthetic data set that has the SAME predictors
    (delta, conflict, noise, total_responses) as 'gdf' but with
    binomially simulated response counts.
    """
    sim_gdf = gdf.copy()

    # vectorised computation to avoid Python loop
    lam   = np.empty(len(gdf))
    mu    = np.empty(len(gdf))
    sigma = np.empty(len(gdf))

    # Map each row's (conflict, noise) to its λ, μ, σ
    for idx, row in gdf.iterrows():
        lam_i, mu_i, sig_i = getParams(params,
                                        row[conflictVar],
                                        row[sensoryVar],
                                        nLambda,
                                        nSigma)
        lam[idx], mu[idx], sigma[idx] = lam_i, mu_i, sig_i

    p_choose_test = psychometric_function(sim_gdf[intensityVariable].values,
                                          lam, mu, sigma)

    # Binomial draw
    sim_gdf['num_of_chose_test'] = np.random.binomial(
        n = sim_gdf['total_responses'].values.astype(int),
        p = p_choose_test)

    return sim_gdf

def paramBootstrap(fitParams, nBoots):
    
    groupedData = groupByChooseTest(data)
    nBootParams = []
    # Add tqdm progress bar
    for _ in tqdm(range(nBoots), desc="Bootstrapping", unit="iteration"):
        #1- do simulation
        simData = simulate_dataset(fitParams, groupedData)
        # simulate data by taking binomials at this
        initGuessEstimate = estimate_initial_guesses(simData[intensityVariable].values, 
                                                    simData['num_of_chose_test'], 
                                                    simData['total_responses'])
        # fit the data to boot
        bootFit = fitJoint(grouped_data=simData, initGuesses=initGuessEstimate)

        # save
        nBootParams.append(bootFit.x)

    return np.vstack(nBootParams)

def plot_conflict_vs_pse(best_fit, allBootedFits):
    """
    Plot the relation between conflict and PSE (mu) with confidence intervals.
    """
    plt.figure(figsize=(12, 6))
    m = 0
    for i, standardLevel in enumerate(uniqueStandard):
        for j, audioNoiseLevel in enumerate(uniqueSensory):
            for k, conflictLevel in enumerate(uniqueConflict):
                lambda_, mu, sigma = getParams(best_fit.x, conflictLevel, audioNoiseLevel, nLambda, nSigma)
                m += 1        
                plt.subplot(1, 2, j + 1)
                color = sns.color_palette("viridis", as_cmap=True)(k / len(uniqueConflict))
                plt.scatter(conflictLevel * 1000, mu * 1000 / 2, color=color, s=100)
                plt.xlabel("Visual Conflict(ms)")
                plt.ylabel("PSE (ms)")
                plt.title(f"Standard: {standardLevel}, Noise: {audioNoiseLevel}")
                plt.grid()
                plt.axhline(y=0, color='gray', linestyle='--')
                plt.axvline(x=0, color='gray', linestyle='--')

                mu_idx = nLambda + nSigma + ((j) * len(uniqueConflict)) + k
                mu_all = allBootedFits[:, mu_idx]
                mu_ci = np.percentile(mu_all, [2.5, 97.5])
                lower_err = np.maximum(mu*1000/2-mu_ci[0]*1000/2, 0)
                upper_err = np.maximum(mu_ci[1]*1000/2-mu*1000/2, 0)
                plt.errorbar(conflictLevel*1000, mu*1000/2, yerr=[[lower_err], [upper_err]], fmt='o', color=color, capsize=10
                            , label=f"95% CI for mu: {mu_ci[0]*1000/2:.2f} - {mu_ci[1]*1000/2:.2f}", linewidth=2)
                plt.ylim(-350, 350)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit joint psychometric model.")
    parser.add_argument("--dataName", type=str, help="CSV file name in data/ directory")
    parser.add_argument("--sharedSigma", action="store_true", help="Use shared sigma across noise levels")
    args = parser.parse_args()

    dataName = args.dataName
    global sharedSigma

    sharedSigma = args.sharedSigma

    if not dataName:
        dataName = "IP_mainExpAvDurEstimate_2025-05-30_11h21.09.224.csv"  # Default data file if not provided
        
    sharedSigma = False  # Set to True if you want to use shared sigma across noise levels
    # Example usage
    data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(dataName, sharedSigma)
    fit = fitMultipleStartingPoints(data, nStart=1)
    # print the fitted parameters
    print(f"Fitted parameters: {fit.x}")

    # Plot the fitted psychometric functions
    plot_fitted_psychometric(
        data, fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
        standardVar, sensoryVar, conflictVar, intensityVariable
    )

    # Plot the relation between conflict and PSE (mu) with confidence intervals
    #allBootedFits = paramBootstrap(fit.x, nBoots=10)
    #plot_conflict_vs_pse(fit, allBootedFits)

# example usage on how to run the code on terminal
# python jointSharedSigma.py --dataName "LN_main_all.csv" --sharedSigma True

# # run the code to test the functions
# if __name__ == "__main__":
#     # Example usage
#     data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda,nSigma,nMu = loadData("LC_auditoryDurEst_2025-05-23_16h37.43.184.csv")
#     grouped_data = groupByChooseTest(data)
    
#     # Fit the model with multiple starting points
#     fit = fitMultipleStartingPoints(data, nStart=3)
    
#     # Get fitted parameters
#     lambda_, mu, sigma = getFittedParams(fit)
    
#     print(f"Fitted parameters: lambda={lambda_}, mu={mu}, sigma={sigma}")

