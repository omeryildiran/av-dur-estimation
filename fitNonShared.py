# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# function for loading data
def loadData(dataName):
    intensityVariable="delta_dur_percents"

    sensoryVar="audNoise"
    standardVar="standardDur"
    conflictVar="conflictDur"
    


    data = pd.read_csv("data/"+dataName)
    # ignore firts 3 rows
    data= data[data['audNoise'] != 0]
    data=data[data['standardDur'] != 0]
    uniqueSensory = data[sensoryVar].unique()
    uniqueStandard = data[standardVar].unique()
 
    try:
        uniqueConflict = sorted(data[conflictVar].unique())
    except:
        data[conflictVar] = 0
        uniqueConflict = [0]
    print(f"uniqueSensory: {uniqueSensory} \n uniqueStandard: {uniqueStandard} \n uniqueConflict: {uniqueConflict}")
    try:
        data['recordedDurVisualStandard'] = round(data['recordedDurVisualStandard'], 3)
    except:
        data['recordedDurVisualStandard'] = 1
    #data['avgAVDeltaS'] = (data['deltaDurS'] + (data['recordedDurVisualTest'] - data['recordedDurVisualStandard'])) / 2
    #data['deltaDurPercentVisual'] = ((data['recordedDurVisualTest'] - data['recordedDurVisualStandard']) / data['recordedDurVisualStandard'])
    #data['avgAVDeltaPercent'] = data[['delta_dur_percents', 'deltaDurPercentVisual']].mean(axis=1)

    # Define columns for chosing test or standard
    data['chose_test'] = (data['responses'] == data['order']).astype(int)
    data['chose_standard'] = (data['responses'] != data['order']).astype(int)
    data['visualPSEBias'] = data['recordedDurVisualStandard'] -data["standardDur"]-data['conflictDur']
    data['conflictDur'] = data['conflictDur'].round(2)
    data['standard_dur']=data['standardDur']

    try:
        data["riseDur"]>1
    except:
        data["riseDur"]=1
    
    data[standardVar] = round(data[standardVar], 2)

    data['standard_dur']=round(data['standardDur'],2)
    data["delta_dur_percents"]=round(data["delta_dur_percents"],2)
    data['conflictDur']=round(data['conflictDur'],2)

    # delete data[stair]=1U1D
    #data = data[data['current_stair'] != '1U1D']
    try:
        print(len(data[data['recordedDurVisualStandard']<0]), " trials with negative visual standard duration")
        print(len(data[data['recordedDurVisualTest']<0]), " trials with negative visual test duration")


        data=data[data['recordedDurVisualStandard'] <=998]
        data=data[data['recordedDurVisualStandard'] >=0]
        data=data[data['recordedDurVisualTest'] <=998]
        data=data[data['recordedDurVisualTest'] >=0]
    except:
        pass

    #data = data[3:]  # Ignore first 3 rows
    nLambda=len(uniqueStandard)
    nSigma=len(uniqueSensory)
    nMu=len(uniqueConflict)*nSigma
    return data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda,nSigma, nMu

intensityVariable="delta_dur_percents"



def groupByChooseTest(x):
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

    # Get lambda (lapse rate)
    lambda_ = params[0]    
    # Get sigma based on noise level
    
    # Get noise index safely
    noise_idx_array = np.where(uniqueSensory == audio_noise)[0]
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
    sigma_idx = nLambda-1  + ((conflict_idx+1)*(noise_idx+1))#+ nSigma + noise_offset + conflict_idx
    sigma = params[sigma_idx]  # +1 because lambda is first

    noise_offset = noise_idx * len(uniqueConflict)
    # mu is after lambda and sigma, so we need to find its index
    #mu_idx = nLambda+ nSigma + noise_offset + conflict_idx
    mu_idx = nLambda-1 +((len(params)-1)//2) + ((conflict_idx+1)*(noise_idx+1))#+ nSigma + noise_offset + conflict_idx
    
    mu = params[mu_idx]
    if fixedMu:
        mu = 0
    return lambda_, mu, sigma

def getParamIndexes(params, conflict, audio_noise, nLambda, nSigma):
    # Get lambda (lapse rate)
    lambda_ = params[0]    
    # Get sigma based on noise level
    # Get noise index safely
    noise_idx_array = np.where(uniqueSensory == audio_noise)[0]
    if len(noise_idx_array) == 0:
        raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
    noise_idx = noise_idx_array[0]
    
    # Get conflict index safely
    conflict_idx_array = np.where(uniqueConflict==conflict)[0]
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

def psychometric_function(x, lambda_, mu, sigma):
    if fixedMu:
        mu = 0
        cdf = norm.cdf(x, scale=sigma)
        p= lambda_/2 + (1-lambda_) * norm.cdf(x / sigma)
    else:
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
    if fixedMu:
        mu = 0
    
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
    bounds = [(0, 0.25), (-0.73, +0.73), (0.01, 1.5)]  # Reasonable bounds
    if fixedMu:
        bounds = [(0, 0.25), (0.01, 1.5)]
        init_guesses = [init_guesses[0],  init_guesses[2]]  # Set mu to 0 if fixed
    # fitting is done here
    result = minimize(
        negative_log_likelihood, x0=init_guesses, 
        args=(levels, nResp, totalResp),  # Pass the data and fixed parameters
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
    
    # Initialize guesses for parameters 
    # lambda, sigma, mu
    initGuesses= [initGuesses[0]]*nLambda + [initGuesses[2]]*nSensoryVar*nConflictVar+ [initGuesses[1]]*nSensoryVar*nConflictVar
    
    intensities = grouped_data[intensityVariable]
    chose_tests = grouped_data['num_of_chose_test']
    total_responses = grouped_data['total_responses']
    conflicts = grouped_data[conflictVar]
    noise_levels = grouped_data[sensoryVar]
    
    
    # Set bounds for parameters
    bounds = [(0, 0.25)]*nLambda + [(0.01, +1.5)]*nSensoryVar*nConflictVar + [(-1, +1)]*nSensoryVar*nConflictVar


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
    initMus=np.linspace(-0.73, 0.73, nStart)
    initSigmas=np.linspace(0.01, 0.9, nStart)
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
    uniqueConflict = data['conflictDur'].unique()
    
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
    print(f"Fitted parameters: {best_fit.x}")
    colors = sns.color_palette("viridis", n_colors=len(uniqueSensory))  # Use Set2 palette for different noise levels
    plt.figure(figsize=(10, 10))
    for i, standardLevel in enumerate(uniqueStandard):
        for j, audioNoiseLevel in enumerate(uniqueSensory):
            for k, conflictLevel in enumerate(uniqueConflict):
                lambda_, mu, sigma = getParams(best_fit.x, conflictLevel, audioNoiseLevel, nLambda, nSigma)
                print(f"Standard: {standardLevel}, Audio Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, Lambda: {lambda_}, Mu: {mu}, Sigma: {sigma}")
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
                maxX = max(levels) + 0.1
                minX = min(levels) - 0.1
                x = np.linspace(minX, maxX, 500)
                y = psychometric_function(x, lambda_, mu, sigma)
                color = colors[j]
                plt.plot(x, y, color=color, label=f"Noise: {audioNoiseLevel}\n $\\mu$: {mu:.2f}, $\\sigma$: {sigma:.2f}", linewidth=4)
                plt.axvline(x=0, color='gray', linestyle='--')
                plt.axhline(y=0.5, color='gray', linestyle='--')
                plt.xlabel(f"({intensityVariable}) Test(stair)-Standard(0.5s) Duration Difference Ratio")
                plt.ylabel("P(chose test)")

                plt.title(f" {pltTitle} ", fontsize=16)
                plt.legend(fontsize=14, title_fontsize=14)
                plt.grid()
                bin_and_plot(dfFiltered, bin_method='cut', bins=10, plot=True, color=color)
                plt.text(0.05, 0.8, f"Shared $\lambda$: {lambda_:.2f}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
                plt.tight_layout()
                plt.grid(True)
    plt.show()


def plotStairCases(data):

    # select the current stair
    uniqueStairs = data['current_stair'].unique()
    uniqueStairs= sorted(uniqueStairs)[:-1]
    plt.figure(figsize=(20, 10))
    for idx, stair in enumerate(uniqueStairs):
        df= data[data['current_stair'] == stair]. reset_index(drop=True)
        plt.subplot(2, 2, idx+1)
        for trialN in range(df.shape[0]):
            color = 'green' if df['chose_test'][trialN] == 1 or df['chose_test'][trialN] == "True" else 'red'            
            #print(f"Trial {trialN}, delta_dur_percents: {df['delta_dur_percents'][trialN]}, is_correct: {df['is_correct'][trialN]}")
            plt.scatter(trialN,df['delta_dur_percents'][trialN], color=color, s=60, alpha=0.5)
            plt.plot(df['delta_dur_percents'], color='blue')

            plt.title(f"Stair {stair}")
            plt.xlabel("Test(stair)-Standard(0.5s) Duration Difference Ratio")
            plt.ylabel("Delta Duration %")
            plt.axhline(y=0, color='gray', linestyle='--')
            #plt.axvline(x=0, color='gray', linestyle='--')
            plt.ylim(-0.9, 0.9)
            plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fixedMu =0  # Set to True to ignore the bias in the model
    dataName = "IP_bimodalDurEst_2025-05-27_15h03.01.323.csv"
    # Example usage
    data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(dataName)
    pltTitle=dataName.split("_")[1]
    pltTitle=dataName.split("_")[0]+str(" ")+pltTitle
    grouped_data = groupByChooseTest(data)
    fit = fitMultipleStartingPoints(data, nStart=1)
    #print the fitted parameters
    print(f"Fitted parameters: {fit.x}")

    #Plot the fitted psychometric functions
    plot_fitted_psychometric(
        data, fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
        standardVar, sensoryVar, conflictVar, intensityVariable)
    plotStairCases(data)




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

