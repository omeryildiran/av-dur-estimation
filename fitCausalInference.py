# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import norm
from scipy.optimize import minimize

# function for loading data
def loadData(dataName, isShared, isAllIndependent):
    global data, sharedSigma, intensityVariable, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu, allIndependent
    sharedSigma = isShared  # Set to True if you want to use shared sigma across noise levels
    allIndependent = isAllIndependent  # Set to 1 if you want to use independent parameters for each condition
    intensityVariable="delta_dur_percents"

    sensoryVar="audNoise"
    standardVar="standardDur"
    conflictVar="conflictDur"
    global pltTitle
    pltTitle=dataName.split("_")[1]
    pltTitle=dataName.split("_")[0]+str(" ")+pltTitle    



    data = pd.read_csv("data/"+dataName)
    data["testDurMs"]= data["testDurS"]*1000
    data["standardDurMs"]= data["standardDur"]*1000
    data["conflictDurMs"]= data["conflictDur"]*1000
    data["DeltaDurMs"]= data["testDurMs"] - data["standardDurMs"]

    data = data.round({'standardDur': 2, 'conflictDur': 2})
    # if nan in conflictDur remove those rows
    data = data[~data['conflictDur'].isna()]

    # if nan in audNoise remove those rows
    data = data[~data['audNoise'].isna()]
    if "VisualPSE" not in data.columns:
        data["VisualPSE"]=data['recordedDurVisualStandard'] -data["standardDur"]-data['conflictDur']

    print(f"\n Total trials before cleaning\n: {len(data)}")
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
    data['visualPSEBiasTest'] = data['recordedDurVisualTest'] -data["testDurS"]

    try: 
        data['biasCheckTest'] = np.isclose(data['visualPSEBiasTest'], data['VisualPSE'], atol=0.012)
        data['biasCheckStandard'] = np.isclose(data['visualPSEBias'], data['VisualPSE'], atol=0.012)
        data["testDurSCheck"] = (abs(data['recordedDurVisualTest'] - data['testDurS']-data["VisualPSE"]) < 0.03)
        data["testDurSCheckBias"] = (abs(data['recordedDurVisualTest'] - data['testDurS']-data["VisualPSE"]) < 0.03)

        data["standardDurCheck"] = (abs(data['recordedDurVisualStandard'] - data['standardDur']-data["VisualPSE"]-data['conflictDur']) < 0.03)
        data["testDurSCompare"] = abs(data['recordedDurVisualTest'] - data['testDurS']-data["VisualPSE"])
        data["standardDurCompare"] = abs(data['recordedDurVisualStandard'] - data['standardDur']-data["VisualPSE"]-data['conflictDur'])

        #print len of testDurSCheck and standardDurCheck false
        print("")
        print(len(data[data['testDurSCheck'] == False]), " trials with testDurSCheck False")
        print(len(data[data['standardDurCheck'] == False]), " trials with standardDurCheck False\n")
        # print number of abs(testDurSCompare
        print(len(data[abs(data['testDurSCompare']) > 0.03]), " trials with abs(testDurSCompare) > 0.05")
        print(len(data[abs(data['standardDurCompare']) > 0.03]), " trials with abs(standardDurCompare) > 0.05")
        print("")
        print(len(data[data['testDurSCheckBias'] == False]), " trials with testDurSCheckBias False")

    except:
        print("Bias check failed!!!! No bias check columns found. Skipping bias check.")
        pass
    data['conflictDur'] = data['conflictDur'].round(3)
    data['standard_dur']=data['standardDur']

    try:
        data["riseDur"]>1
    except:
        data["riseDur"]=1
    
    data[standardVar] = round(data[standardVar], 2)

    data['standard_dur']=round(data['standardDur'],2)
    data["delta_dur_percents"]=round(data["delta_dur_percents"],2)
    try:
        print(len(data[data['recordedDurVisualTest']<0]), " trials with negative visual test duration")
        print(len(data[data['recordedDurVisualStandard']<0]), " trials with negative visual standard duration")
    except:
        print("No negative visual test or standard duration found.")




    try:
        print(f'testdurCompare > 0.05: {len(data[data["testDurSCompare"] > 0.05])} trials')

        print(len(data[data['recordedDurVisualStandard']<0]), " trials with negative visual standard duration")
        print(len(data[data['recordedDurVisualTest']<0]), " trials with negative visual test duration")


        data=data[data['recordedDurVisualStandard'] <=998]
        data=data[data['recordedDurVisualStandard'] >=0]
        data=data[data['recordedDurVisualTest'] <=998]
        data=data[data['recordedDurVisualTest'] >=0]
        #clean trials where standardDurCheck and testDurSCheck are false
        data=data[data['standardDurCompare'] < 0.03]
        data=data[data['testDurSCompare']< 0.03]
    except:
        pass

    print(f"total trials after cleaning: {len(data)}")
    nLambda=len(uniqueStandard)
    nSigma=len(uniqueSensory)
    nMu=len(uniqueConflict)*nSigma
    
    data["logStandardDur"] = np.log(data[standardVar])
    data["logConflictDur"] = np.log(data[conflictVar])
    data["logTestDur"] = np.log(data["testDurS"])
    data["logDeltaDur"] = data["logTestDur"] - data["logStandardDur"]

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
    if allIndependent and not sharedSigma:  # if all parameters are independent
        # Each (noise, conflict) pair has its own lambda, mu, sigma
        noise_idx = np.where(uniqueSensory == round(audio_noise, 3))[0][0]
        conflict_idx = np.where(uniqueConflict == round(conflict, 3))[0][0]
        nCond = len(uniqueSensory) * len(uniqueConflict)
        cond_idx = noise_idx * len(uniqueConflict) + conflict_idx
        lambda_ = params[cond_idx * 3 + 0]
        mu     = params[cond_idx * 3 + 1]
        sigma  = params[cond_idx * 3 + 2]
        return lambda_, mu, sigma
    elif allIndependent==False and sharedSigma: #  if sigma is shared
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
    elif allIndependent==False and sharedSigma==False:
        lambda_ = params[0]

        # Get indices
        noise_idx_array = np.where(uniqueSensory == round(audio_noise, 3))[0]
        if len(noise_idx_array) == 0:
            raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")

        conflict_idx_array = np.where(uniqueConflict == conflict)[0]
        if len(conflict_idx_array) == 0:
            raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")

        noise_idx = noise_idx_array[0]
        conflict_idx = conflict_idx_array[0]

        nConditions = len(uniqueConflict) * len(uniqueSensory)
        cond_idx = conflict_idx * len(uniqueSensory) + noise_idx

        mu = params[nLambda + cond_idx]
        sigma = params[nLambda + nConditions + cond_idx]

        return lambda_, mu, sigma

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
    Vectorized negative log likelihood for all conditions.
    """
    if allIndependent:
        lam = np.empty(len(delta_dur))
        mu = np.empty(len(delta_dur))
        sigma = np.empty(len(delta_dur))
        for i in range(len(delta_dur)):
            lam[i], mu[i], sigma[i] = getParams(params, conflicts[i], noise_levels[i], nLambda, nSigma)

        # Vectorized psychometric function
        p = lam / 2 + (1 - lam) * norm.cdf((delta_dur - mu) / sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)

        # Vectorized negative log-likelihood
        nll = -np.sum(responses * np.log(p) + (total_responses - responses) * np.log(1 - p))
        return nll
    else:
        # Precompute parameter arrays for each data point
        lam = np.empty(len(delta_dur))
        mu = np.empty(len(delta_dur))
        sigma = np.empty(len(delta_dur))
        for i in range(len(delta_dur)):
            lam[i], mu[i], sigma[i] = getParams(params, conflicts[i], noise_levels[i], nLambda, nSigma)

        # Vectorized psychometric function
        p = lam / 2 + (1 - lam) * norm.cdf((delta_dur - mu) / sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)

        # Vectorized negative log-likelihood
        nll = -np.sum(responses * np.log(p) + (total_responses - responses) * np.log(1 - p))
        return nll

# ===============================
# JOINT CAUSAL INFERENCE FITTING
# ===============================

def getCausalInferenceParams(params, conflict, audio_noise, nLambda, nSigma):
    """
    Extract causal inference parameters for a specific condition (conflict, noise).
    Similar to getParams but for causal inference model with 4 parameters.
    
    Parameters:
    -----------
    params : array
        Full parameter array 
    conflict : float
        Visual conflict level
    audio_noise : float
        Auditory noise level
    nLambda : int
        Number of lambda parameters
    nSigma : int
        Number of sigma parameters
        
    Returns:
    --------
    lambda_ : float
        Lapse rate for this condition
    sigma_av_a : float
        Auditory noise in AV condition
    sigma_av_v : float
        Visual noise in AV condition  
    p_common : float
        Prior probability of common cause
    """
    if allIndependent and not sharedSigma:
        # Each (noise, conflict) pair has its own set of causal inference parameters
        noise_idx = np.where(uniqueSensory == round(audio_noise, 3))[0][0]
        conflict_idx = np.where(uniqueConflict == round(conflict, 3))[0][0]
        nCond = len(uniqueSensory) * len(uniqueConflict)
        cond_idx = noise_idx * len(uniqueConflict) + conflict_idx
        
        # Each condition has 4 parameters: lambda, sigma_av_a, sigma_av_v, p_common
        lambda_ = params[cond_idx * 4 + 0]
        sigma_av_a = params[cond_idx * 4 + 1]
        sigma_av_v = params[cond_idx * 4 + 2]
        p_common = params[cond_idx * 4 + 3]
        
        return lambda_, sigma_av_a, sigma_av_v, p_common
        
    elif allIndependent == False and sharedSigma:
        # Shared structure: 
        # - Single lambda across all conditions
        # - sigma_av_a varies by noise level only
        # - sigma_av_v varies by noise level only  
        # - p_common varies by (noise, conflict) combination
        
        # Get lambda (shared across all conditions)
        lambda_ = params[0]
        
        # Get noise index
        noise_idx_array = np.where(uniqueSensory == audio_noise)[0]
        if len(noise_idx_array) == 0:
            raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
        noise_idx = noise_idx_array[0]
        
        # sigma_av_a and sigma_av_v depend on noise level
        sigma_av_a = params[1 + noise_idx]  # After lambda
        sigma_av_v = params[1 + nSigma + noise_idx]  # After lambda and all sigma_av_a
        
        # Get conflict index
        conflict_idx_array = np.where(np.isclose(uniqueConflict, conflict, atol=1e-1))[0]
        if len(conflict_idx_array) == 0:
            raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
        conflict_idx = conflict_idx_array[0]
        
        # p_common varies by (noise, conflict) combination
        noise_offset = noise_idx * len(uniqueConflict)
        p_common_idx = 1 + 2*nSigma + noise_offset + conflict_idx
        p_common = params[p_common_idx]
        
        return lambda_, sigma_av_a, sigma_av_v, p_common
        
    else:
        # Non-shared structure: each (noise, conflict) has its own parameters
        lambda_ = params[0]  # Shared lambda
        
        # Get indices
        noise_idx_array = np.where(uniqueSensory == round(audio_noise, 3))[0]
        if len(noise_idx_array) == 0:
            raise ValueError(f"audio_noise value {audio_noise} not found in uniqueSensory.")
            
        conflict_idx_array = np.where(uniqueConflict == conflict)[0]
        if len(conflict_idx_array) == 0:
            raise ValueError(f"conflict value {conflict} not found in uniqueConflict.")
            
        noise_idx = noise_idx_array[0]
        conflict_idx = conflict_idx_array[0]
        
        nConditions = len(uniqueConflict) * len(uniqueSensory)
        cond_idx = conflict_idx * len(uniqueSensory) + noise_idx
        
        # Parameters are organized as: lambda, then sigma_av_a for each condition, 
        # then sigma_av_v for each condition, then p_common for each condition
        sigma_av_a = params[nLambda + cond_idx]
        sigma_av_v = params[nLambda + nConditions + cond_idx]
        p_common = params[nLambda + 2*nConditions + cond_idx]
        
        return lambda_, sigma_av_a, sigma_av_v, p_common

def causalInferenceNLLJoint(params, delta_dur, responses, total_responses, conflicts, noise_levels):
    """
    Joint negative log-likelihood for causal inference model across all conditions.
    Similar to nLLJoint but for causal inference model.
    
    Parameters:
    -----------
    params : array
        Parameter array (structure depends on allIndependent and sharedSigma flags)
    delta_dur : array
        Duration differences for each data point
    responses : array
        Number of times test was chosen for each data point
    total_responses : array
        Total responses for each data point
    conflicts : array
        Visual conflict levels for each data point
    noise_levels : array
        Auditory noise levels for each data point
        
    Returns:
    --------
    nll : float
        Negative log-likelihood
    """
    # Precompute parameter arrays for each data point
    lambda_arr = np.empty(len(delta_dur))
    sigma_av_a_arr = np.empty(len(delta_dur))
    sigma_av_v_arr = np.empty(len(delta_dur))
    p_common_arr = np.empty(len(delta_dur))
    
    for i in range(len(delta_dur)):
        lambda_arr[i], sigma_av_a_arr[i], sigma_av_v_arr[i], p_common_arr[i] = getCausalInferenceParams(
            params, conflicts[i], noise_levels[i], nLambda, nSigma
        )
    
    # Compute probabilities using vectorized causal inference psychometric function
    p_choose_test = np.zeros_like(delta_dur, dtype=float)
    
    for i in range(len(delta_dur)):
        p_choose_test[i] = causalInferencePsychometric(
            delta_dur[i], lambda_arr[i], sigma_av_a_arr[i], 
            sigma_av_v_arr[i], p_common_arr[i], conflicts[i]
        )
    
    # Clip probabilities to avoid log(0)
    epsilon = 1e-9
    p_choose_test = np.clip(p_choose_test, epsilon, 1 - epsilon)
    
    # Vectorized negative log-likelihood
    nll = -np.sum(responses * np.log(p_choose_test) + 
                  (total_responses - responses) * np.log(1 - p_choose_test))
    
    return nll

def fitCausalInferenceJoint(grouped_data, initGuesses):
    """
    Fit causal inference model jointly across all conditions.
    Similar to fitJoint but for causal inference model.
    
    Parameters:
    -----------
    grouped_data : DataFrame
        Grouped experimental data
    initGuesses : list
        Initial parameter guesses [lambda_, sigma_av_a, sigma_av_v, p_common]
        
    Returns:
    --------
    result : OptimizeResult
        Fitted parameters and optimization result
    """
    if allIndependent:
        nCond = len(uniqueSensory) * len(uniqueConflict)
        # Each condition has 4 parameters: lambda, sigma_av_a, sigma_av_v, p_common
        initGuesses = (initGuesses * nCond)[:nCond*4]
        
        # Extract data
        intensities = grouped_data[intensityVariable]
        chose_tests = grouped_data['num_of_chose_test']
        total_responses = grouped_data['total_responses']
        conflicts = grouped_data[conflictVar]
        noise_levels = grouped_data[sensoryVar]
        
        # Bounds for each parameter set per condition
        bounds = []
        for _ in range(nCond):
            bounds.extend([
                (0, 0.25),    # lambda_: lapse rate
                (0.01, 2),    # sigma_av_a: auditory noise in AV
                (0.01, 2),    # sigma_av_v: visual noise in AV  
                (0, 1)        # p_common: prior probability of common cause
            ])
            
    elif sharedSigma == True:
        # Shared structure: lambda (1) + sigma_av_a per noise (nSigma) + 
        # sigma_av_v per noise (nSigma) + p_common per (noise,conflict) (nSigma*len(uniqueConflict))
        nParams = 1 + 2*nSigma + nSigma*len(uniqueConflict)
        initGuesses = ([initGuesses[0]] +  # lambda
                      [initGuesses[1]]*nSigma +  # sigma_av_a per noise level
                      [initGuesses[2]]*nSigma +  # sigma_av_v per noise level  
                      [initGuesses[3]]*nSigma*len(uniqueConflict))  # p_common per condition
        
        intensities = grouped_data[intensityVariable]
        chose_tests = grouped_data['num_of_chose_test']
        total_responses = grouped_data['total_responses']
        conflicts = grouped_data[conflictVar]
        noise_levels = grouped_data[sensoryVar]
        
        # Set bounds
        bounds = ([(0, 0.25)] +  # lambda
                 [(0.01, 2)]*nSigma +  # sigma_av_a bounds
                 [(0.01, 2)]*nSigma +  # sigma_av_v bounds
                 [(0, 1)]*nSigma*len(uniqueConflict))  # p_common bounds
                 
    else:
        # Each (noise, conflict) combination has its own sigma_av_a, sigma_av_v, p_common
        # but shared lambda
        nConditions = nSigma * len(uniqueConflict)
        initGuesses = ([initGuesses[0]] +  # shared lambda
                      [initGuesses[1]]*nConditions +  # sigma_av_a per condition
                      [initGuesses[2]]*nConditions +  # sigma_av_v per condition
                      [initGuesses[3]]*nConditions)   # p_common per condition
        
        intensities = grouped_data[intensityVariable]
        chose_tests = grouped_data['num_of_chose_test']
        total_responses = grouped_data['total_responses']
        conflicts = grouped_data[conflictVar]
        noise_levels = grouped_data[sensoryVar]
        
        # Set bounds
        bounds = ([(0, 0.25)] +  # lambda
                 [(0.01, 2)]*nConditions +  # sigma_av_a bounds
                 [(0.01, 2)]*nConditions +  # sigma_av_v bounds  
                 [(0, 1)]*nConditions)      # p_common bounds

    # Minimize negative log-likelihood
    result = minimize(
        causalInferenceNLLJoint,
        x0=initGuesses,
        args=(intensities, chose_tests, total_responses, conflicts, noise_levels),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    return result

def nLLJoint(params, delta_dur, responses, total_responses, conflicts, noise_levels):
    """
    Vectorized negative log likelihood for all conditions.
    """
    if allIndependent:
        lam = np.empty(len(delta_dur))
        mu = np.empty(len(delta_dur))
        sigma = np.empty(len(delta_dur))
        for i in range(len(delta_dur)):
            lam[i], mu[i], sigma[i] = getParams(params, conflicts[i], noise_levels[i], nLambda, nSigma)

        # Vectorized psychometric function
        p = lam / 2 + (1 - lam) * norm.cdf((delta_dur - mu) / sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)

        # Vectorized negative log-likelihood
        nll = -np.sum(responses * np.log(p) + (total_responses - responses) * np.log(1 - p))
        return nll
    else:
        # Precompute parameter arrays for each data point
        lam = np.empty(len(delta_dur))
        mu = np.empty(len(delta_dur))
        sigma = np.empty(len(delta_dur))
        for i in range(len(delta_dur)):
            lam[i], mu[i], sigma[i] = getParams(params, conflicts[i], noise_levels[i], nLambda, nSigma)

        # Vectorized psychometric function
        p = lam / 2 + (1 - lam) * norm.cdf((delta_dur - mu) / sigma)
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)

        # Vectorized negative log-likelihood
        nll = -np.sum(responses * np.log(p) + (total_responses - responses) * np.log(1 - p))
        return nll

# ===============================
# CAUSAL INFERENCE MODEL FUNCTIONS
# ===============================

def fusionAV(sigma_av_a, sigma_av_v, S_a, conflict):
    """
    Optimal fusion of auditory and visual estimates.
    
    Parameters:
    -----------
    sigma_av_a : float
        Auditory noise in AV condition
    sigma_av_v : float
        Visual noise in AV condition
    S_a : float
        Auditory estimate
    conflict : float
        Visual conflict (difference between visual and auditory)
        
    Returns:
    --------
    fused_estimate : float
        Optimally fused estimate
    fused_variance : float
        Variance of fused estimate
    """
    # Visual estimate includes conflict
    S_v = S_a + conflict
    
    # Optimal fusion weights (reliability-weighted average)
    w_a = (1/sigma_av_a**2) / (1/sigma_av_a**2 + 1/sigma_av_v**2)
    w_v = (1/sigma_av_v**2) / (1/sigma_av_a**2 + 1/sigma_av_v**2)
    
    # Fused estimate
    fused_estimate = w_a * S_a + w_v * S_v
    
    # Fused variance (inverse of sum of precisions)
    fused_variance = 1 / (1/sigma_av_a**2 + 1/sigma_av_v**2)
    
    return fused_estimate, fused_variance

def causalInference(S_a, conflict, sigma_av_a, sigma_av_v, p_common):
    """
    Causal inference model for audiovisual integration.
    
    Parameters:
    -----------
    S_a : float
        Auditory stimulus estimate
    conflict : float
        Visual conflict level (difference from auditory)
    sigma_av_a : float
        Auditory noise in AV condition
    sigma_av_v : float
        Visual noise in AV condition
    p_common : float
        Prior probability of common cause
        
    Returns:
    --------
    estimate : float
        Final causal inference estimate
    p_posterior : float
        Posterior probability of common cause
    """
    # Visual estimate includes conflict
    S_v = S_a + conflict
    
    # Likelihood under common cause (C=1): optimal fusion
    fused_estimate, fused_variance = fusionAV(sigma_av_a, sigma_av_v, S_a, conflict)
    
    # For likelihood calculation, we need to consider the probability of observing 
    # the discrepancy between auditory and visual estimates
    discrepancy = S_v - S_a  # This should equal conflict in our setup
    
    # Likelihood of discrepancy under common cause
    # The discrepancy should be small under common cause
    var_discrepancy_common = sigma_av_a**2 + sigma_av_v**2
    likelihood_common = norm.pdf(discrepancy, loc=0, scale=np.sqrt(var_discrepancy_common))
    
    # Likelihood under separate causes (C=2)
    # Under separate causes, discrepancy can be large (determined by prior on conflicts)
    # We assume a uniform prior over conflicts, so likelihood is constant
    # For simplicity, we use a wide Gaussian
    var_discrepancy_separate = var_discrepancy_common * 10  # Much larger variance
    likelihood_separate = norm.pdf(discrepancy, loc=0, scale=np.sqrt(var_discrepancy_separate))
    
    # Posterior probability of common cause using Bayes rule
    evidence_common = likelihood_common * p_common
    evidence_separate = likelihood_separate * (1 - p_common)
    evidence_total = evidence_common + evidence_separate
    
    if evidence_total > 0:
        p_posterior = evidence_common / evidence_total
    else:
        p_posterior = p_common  # Fallback to prior
    
    # Estimates under each causal structure
    estimate_common = fused_estimate  # Optimal fusion
    estimate_separate = S_a  # Auditory only (for duration task)
    
    # Model averaging
    final_estimate = p_posterior * estimate_common + (1 - p_posterior) * estimate_separate
    
    return final_estimate, p_posterior

def causalInferencePsychometric(delta_dur, lambda_, sigma_av_a, sigma_av_v, p_common, conflict):
    """
    Psychometric function for causal inference model.
    
    Parameters:
    -----------
    delta_dur : float
        Duration difference (test - standard)
    lambda_ : float
        Lapse rate
    sigma_av_a : float
        Auditory noise in AV condition
    sigma_av_v : float
        Visual noise in AV condition
    p_common : float
        Prior probability of common cause
    conflict : float
        Visual conflict level
        
    Returns:
    --------
    p_choose_test : float
        Probability of choosing test interval
    """
    # Generate estimates for both intervals using causal inference
    # Standard interval (reference)
    S_standard = 0.0  # Assume normalized to zero
    estimate_standard, _ = causalInference(S_standard, conflict, sigma_av_a, sigma_av_v, p_common)
    
    # Test interval
    S_test = S_standard + delta_dur
    estimate_test, _ = causalInference(S_test, conflict, sigma_av_a, sigma_av_v, p_common)
    
    # Decision variable
    decision_variable = estimate_test - estimate_standard
    
    # Calculate decision noise using the corrected formula
    var_fusion = 1 / (1/sigma_av_a**2 + 1/sigma_av_v**2)
    var_segregated = sigma_av_a**2
    var_estimate = p_common * var_fusion + (1 - p_common) * var_segregated
    sigma_decision = np.sqrt(2 * var_estimate)
    
    # Apply psychometric function with lapse rate
    p_choose_test = lambda_/2 + (1 - lambda_) * norm.cdf(decision_variable, loc=0, scale=sigma_decision)
    
    return p_choose_test

def causal_inference_negative_log_likelihood(params, delta_dur, chose_test, total_responses, conflicts):
    """
    Negative log-likelihood for causal inference model.
    
    Parameters:
    -----------
    params : array
        [lambda_, sigma_av_a, sigma_av_v, p_common]
    delta_dur : array
        Duration differences
    chose_test : array
        Number of times test was chosen
    total_responses : array
        Total responses per condition
    conflicts : array
        Visual conflict levels
        
    Returns:
    --------
    nll : float
        Negative log-likelihood
    """
    lambda_, sigma_av_a, sigma_av_v, p_common = params
    
    # Compute probabilities
    p = np.array([causalInferencePsychometric(dd, lambda_, sigma_av_a, sigma_av_v, p_common, conf) 
                  for dd, conf in zip(delta_dur, conflicts)])
    
    # Clip probabilities to avoid log(0)
    epsilon = 1e-9
    p = np.clip(p, epsilon, 1 - epsilon)
    
    # Compute negative log-likelihood
    nll = -np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
    
    return nll

def fit_causal_inference(delta_dur, chose_test, total_responses, conflicts, 
                        init_params=[0.05, 0.2, 0.3, 0.7]):
    """
    Fit causal inference model to data.
    
    Parameters:
    -----------
    delta_dur : array
        Duration differences
    chose_test : array
        Number of times test was chosen
    total_responses : array
        Total responses per condition
    conflicts : array
        Visual conflict levels
    init_params : list
        Initial parameter guesses [lambda_, sigma_av_a, sigma_av_v, p_common]
        
    Returns:
    --------
    result : OptimizeResult
        Fitting result
    """
    bounds = [
        (0, 0.25),    # lambda_: lapse rate
        (0.01, 2),    # sigma_av_a: auditory noise in AV
        (0.01, 2),    # sigma_av_v: visual noise in AV
        (0, 1)        # p_common: prior probability of common cause
    ]
    
    result = minimize(
        causal_inference_negative_log_likelihood,
        x0=init_params,
        args=(delta_dur, chose_test, total_responses, conflicts),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    return result

# ===============================
# ORIGINAL PSYCHOMETRIC FUNCTIONS (CONTINUED)
# ===============================

def plot_fitted_psychometric(data, fitted_params, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
                            standardVar, sensoryVar, conflictVar, intensityVariable, title="Fitted Psychometric Functions"):
    """
    Plot fitted psychometric functions for the given data and model parameters.
    
    Parameters:
    -----------
    data : DataFrame
        The input data containing trial information
    fitted_params : array
        The fitted parameters from the psychometric function
    nLambda : int
        Number of lambda parameters (for subsetting fitted_params)
    nSigma : int
        Number of sigma parameters (for subsetting fitted_params)
    uniqueSensory : array
        Unique values of the sensory variable (e.g., auditory noise levels)
    uniqueStandard : array
        Unique values of the standard duration
    uniqueConflict : array
        Unique values of the conflict duration
    standardVar : str
        The name of the standard duration variable in the data
    sensoryVar : str
        The name of the sensory variable in the data
    conflictVar : str
        The name of the conflict variable in the data
    intensityVariable : str
        The name of the intensity variable (e.g., "delta_dur_percents")
    title : str
        The title of the plot
    
    Returns:
    --------
    None
    """
    # Create a grid of subplots
    n_rows = len(uniqueSensory)
    n_cols = len(uniqueConflict)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(title, fontsize=16)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    for i, noise in enumerate(uniqueSensory):
        for j, conflict in enumerate(uniqueConflict):
            ax = axes[i * n_cols + j]
            
            # Subset data for this noise and conflict level
            subset = data[(data[sensoryVar] == noise) & (data[conflictVar] == conflict)]
            
            if len(subset) == 0:
                continue  # No data for this condition
            
            # Compute the x values (intensity levels) for the psychometric function
            x_values = np.linspace(-0.1, 0.1, 100)  # Adjusted range for better visualization
            
            # Compute the fitted y values using the psychometric function
            if allIndependent:
                # For independent parameters, subset the fitted_params accordingly
                param_idx = i * len(uniqueConflict) + j
                lambda_fit = fitted_params[param_idx * 3]
                mu_fit = fitted_params[param_idx * 3 + 1]
                sigma_fit = fitted_params[param_idx * 3 + 2]
            else:
                # For shared parameters, use the appropriate shared values
                lambda_fit = fitted_params[0]
                mu_fit = fitted_params[nLambda + i]
                sigma_fit = fitted_params[nLambda + nSigma + i]
            
            y_values = psychometric_function(x_values, lambda_fit, mu_fit, sigma_fit)
            
            # Plot the data and the fitted psychometric function
            ax.plot(x_values, y_values, label='Fitted Psychometric Function', color='red')
            ax.scatter(subset[intensityVariable], subset['p_choose_test'], s=10, color='blue', label='Data')
            
            # Set axis labels and title
            ax.set_xlabel('Intensity (Delta Dur %)')
            ax.set_ylabel('P(Choose Test)')
            ax.set_title(f'Noise: {noise}, Conflict: {conflict}')
            ax.legend()
    
    # Hide any empty subplots
    for k in range(len(uniqueSensory) * len(uniqueConflict), len(axes)):
        fig.delaxes(axes[k])
    
    plt.show()

def fitCausalInferenceMultipleStartingPoints(data, nStart=5):
    """
    Fit causal inference model with multiple starting points for robustness.
    Similar to fitMultipleStartingPoints but for causal inference model.
    
    Parameters:
    -----------
    data : DataFrame
        Experimental data
    nStart : int
        Number of starting points to try
        
    Returns:
    --------
    best_result : OptimizeResult
        Best fitting result across all starting points
    """
    # Group data
    grouped_data = groupByChooseTest(data)
    
    # Generate multiple starting points
    best_result = None
    best_nll = np.inf
    
    for i in range(nStart):
        # Random initial guesses within reasonable bounds
        lambda_init = np.random.uniform(0.01, 0.1)
        sigma_av_a_init = np.random.uniform(0.1, 0.8)
        sigma_av_v_init = np.random.uniform(0.1, 0.8)
        p_common_init = np.random.uniform(0.3, 0.9)
        
        initGuesses = [lambda_init, sigma_av_a_init, sigma_av_v_init, p_common_init]
        
        try:
            result = fitCausalInferenceJoint(grouped_data, initGuesses)
            
            if result.success and result.fun < best_nll:
                best_nll = result.fun
                best_result = result
                
        except Exception as e:
            print(f"Starting point {i+1} failed: {e}")
            continue
    
    if best_result is None:
        raise RuntimeError("All starting points failed to converge")
        
    print(f"Best fit found with NLL: {best_nll:.3f}")
    return best_result

def plot_fitted_causal_inference_joint(data, fitted_params, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
                                  standardVar, sensoryVar, conflictVar, intensityVariable, title="Causal Inference Model Fit"):
    """
    Plot fitted causal inference psychometric functions across all conditions.
    
    Parameters:
    -----------
    data : DataFrame
        Experimental data
    fitted_params : array
        Fitted causal inference parameters
    nLambda : int
        Number of lambda parameters
    nSigma : int
        Number of sigma parameters
    uniqueSensory : array
        Unique values of the sensory variable (e.g., auditory noise levels)
    uniqueStandard : array
        Unique values of the standard duration
    uniqueConflict : array
        Unique values of the conflict duration
    standardVar : str
        The name of the standard duration variable in the data
    sensoryVar : str
        The name of the sensory variable in the data
    conflictVar : str
        The name of the conflict variable in the data
    intensityVariable : str
        The name of the intensity variable (e.g., "delta_dur_percents")
    title : str
        Plot title
    """
    grouped_data = groupByChooseTest(data)
    
    # Get unique conditions
    unique_noise = sorted(data[sensoryVar].unique())
    unique_conflict = sorted(data[conflictVar].unique())
    
    n_noise = len(unique_noise)
    n_conflict = len(unique_conflict)
    
    fig, axes = plt.subplots(n_noise, n_conflict, figsize=(5*n_conflict, 4*n_noise))
    if n_noise == 1 and n_conflict == 1:
        axes = np.array([[axes]])
    elif n_noise == 1:
        axes = axes[np.newaxis, :]
    elif n_conflict == 1:
        axes = axes[:, np.newaxis]
    
    for i, noise in enumerate(unique_noise):
        for j, conflict in enumerate(unique_conflict):
            ax = axes[i, j]
            
            # Filter data for this condition
            condition_data = grouped_data[
                (grouped_data[sensoryVar] == noise) & 
                (grouped_data[conflictVar] == conflict)
            ]
            
            if len(condition_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get fitted parameters for this condition
            lambda_ci, sigma_av_a, sigma_av_v, p_common = getCausalInferenceParams(
                fitted_params, conflict, noise, nLambda, nSigma
            )
            
            # Plot data
            x_data = condition_data[intensityVariable]
            y_data = condition_data['p_choose_test']
            sizes = condition_data['total_responses'] * 5  # Scale point sizes by trials
            
            ax.scatter(x_data, y_data, s=sizes, alpha=0.6, color='blue', label='Data')
            
            # Plot fitted curve
            x_range = np.linspace(x_data.min() - 0.1, x_data.max() + 0.1, 100)
            y_pred = [causalInferencePsychometric(x, lambda_ci, sigma_av_a, sigma_av_v, p_common, conflict) 
                     for x in x_range]
            
            ax.plot(x_range, y_pred, 'r-', linewidth=2, label='Causal Inference Model')
            
            # Formatting
            ax.set_xlabel('Duration Difference (%)')
            ax.set_ylabel('P(Choose Test)')
            ax.set_title(f'Noise={noise:.2f}, Conflict={conflict:.3f}\n' +
                        f'λ={lambda_ci:.3f}, σₐ={sigma_av_a:.3f}, σᵥ={sigma_av_v:.3f}, p={p_common:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def example_usage():
    """
    Example usage of the fitting functions with simulated data.
    
    Returns:
    --------
    None
    """
    # Simulate some data
    np.random.seed(0)
    n_trials = 1000
    true_lambda = 0.1
    true_mu = 0.02
    true_sigma = 0.05
    
    # Simulate sensory variable (e.g., auditory noise levels)
    sensory_var = np.random.choice([0.01, 0.02, 0.03], size=n_trials)
    
    # Simulate conflict variable (e.g., visual conflict levels)
    conflict_var = np.random.choice([0.005, 0.01, 0.015], size=n_trials)
    
    # Simulate intensity variable (e.g., duration differences)
    delta_dur = sensory_var + conflict_var + np.random.normal(0, 0.01, size=n_trials)
    
    # Simulate responses (0 or 1) based on a psychometric function with added noise
    p_choose_test = psychometric_function(delta_dur, true_lambda, true_mu, true_sigma)
    responses = np.random.binomial(1, p_choose_test)
    
    # Create a DataFrame from the simulated data
    data = pd.DataFrame({
        'delta_dur_percents': delta_dur,
        'audNoise': sensory_var,
        'conflictDur': conflict_var,
        'responses': responses
    })
    
    # Fit the model to the simulated data
    grouped_data = groupByChooseTest(data)
    levels = grouped_data['delta_dur_percents'].values
    chooseTest = grouped_data['num_of_chose_test'].values
    totalResp = grouped_data['total_responses'].values
    conflictLevels = grouped_data['conflictDur'].values
    audioNoiseLevels = grouped_data['audNoise'].values
    
    # Standard fitting
    print("Fitting standard psychometric function...")
    standard_init_guesses = [0.05, 0.01, 0.02]  # Initial guesses for lambda, mu, sigma
    standard_fit = fit_psychometric_function(levels, chooseTest, totalResp, standard_init_guesses)
    print("Standard psychometric function fitted.")
    
    # Joint causal inference fitting
    print("Fitting joint causal inference model...")
    causal_init_guesses = [0.05, 0.2, 0.3, 0.7]  # Initial guesses for lambda_, sigma_av_a, sigma_av_v, p_common
    causal_fit = fitCausalInferenceMultipleStartingPoints(data, nStart=1)
    print("Joint causal inference model fitted.")
    
    # Compare models
    print("Comparing models...")
    grouped_data = groupByChooseTest(data)
    levels = grouped_data['delta_dur_percents'].values
    responses = grouped_data['num_of_chose_test'].values
    totalResp = grouped_data['total_responses'].values
    conflictLevels = grouped_data['conflictDur'].values
    noiseLevels = grouped_data['audNoise'].values
    
    # Standard model NLL
    standard_nll = nLLJoint(standard_fit.x, levels, responses, totalResp, conflictLevels, noiseLevels)
    
    # Causal inference model NLL
    causal_nll = causalInferenceNLLJoint(causal_fit.x, levels, responses, totalResp, conflictLevels, noiseLevels)
    
    print(f"Standard psychometric model NLL: {standard_nll:.2f}")
    print(f"Causal inference model NLL: {causal_nll:.2f}")
    print(f"Improvement: {standard_nll - causal_nll:.2f} (positive = causal model is better)")
    
    # Compute AIC
    n_params_standard = len(standard_fit.x)
    n_params_causal = len(causal_fit.x)
    
    aic_standard = 2 * n_params_standard + 2 * standard_nll
    aic_causal = 2 * n_params_causal + 2 * causal_nll
    
    print(f"\nAIC Comparison:")
    print(f"Standard model: {aic_standard:.2f} (params: {n_params_standard})")
    print(f"Causal inference model: {aic_causal:.2f} (params: {n_params_causal})")
    print(f"ΔAIC: {aic_causal - aic_standard:.2f} (negative favors causal model)")
    
    # Plot both models for comparison
    print(f"\n{'-'*40}")
    print("PLOTTING RESULTS")
    print(f"{'-'*40}")
    
    # Plot standard model
    plt.figure(figsize=(20, 12))
    plt.suptitle("Model Comparison: Standard vs Causal Inference", fontsize=16)
    
    # Standard model subplot
    plt.subplot(2, 1, 1)
    plot_fitted_psychometric(
        data, standard_fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
        standardVar, sensoryVar, conflictVar, intensityVariable
    )
    plt.title("Standard Psychometric Model", fontsize=14)
    
    # Causal inference model subplot  
    plt.subplot(2, 1, 2)
    plot_fitted_causal_inference_joint(
        data, causal_fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
        standardVar, sensoryVar, conflictVar, intensityVariable
    )
    plt.title("Joint Causal Inference Model", fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Show parameter details for a few conditions
    print(f"\n{'-'*40}")
    print("PARAMETER DETAILS (Sample Conditions)")
    print(f"{'-'*40}")
    
    for i, noise in enumerate(uniqueSensory[:2]):  # Show first 2 noise levels
        for j, conflict in enumerate(uniqueConflict[:3]):  # Show first 3 conflict levels
            # Standard model parameters
            lambda_std, mu_std, sigma_std = getParams(standard_fit.x, conflict, noise, nLambda, nSigma)
            
            # Causal inference parameters  
            lambda_ci, sigma_av_a, sigma_av_v, p_common = getCausalInferenceParams(
                causal_fit.x, conflict, noise, nLambda, nSigma
            )
            
            print(f"\nNoise: {noise}, Conflict: {conflict}")
            print(f"  Standard: λ={lambda_std:.3f}, μ={mu_std:.3f}, σ={sigma_std:.3f}")
            print(f"  Causal:   λ={lambda_ci:.3f}, σ_a={sigma_av_a:.3f}, σ_v={sigma_av_v:.3f}, P(c)={p_common:.3f}")
    
    return {
        'standard_fit': standard_fit,
        'causal_fit': causal_fit,
        'standard_nll': standard_nll,
        'causal_nll': causal_nll,
        'data': data
    }

# Add this to the main execution section
if __name__ == "__main__":
    # ...existing code...
    
    # Add example for joint causal inference fitting
    print("\n" + "="*80)
    print("RUNNING JOINT CAUSAL INFERENCE EXAMPLE")
    print("="*80)
    
    # Uncomment the line below to run the joint causal inference example
    #results = example_joint_causal_inference_fitting(dataName)
    
    print("To run joint causal inference fitting, uncomment the line above and modify as needed.")
