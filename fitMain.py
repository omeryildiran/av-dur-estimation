# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import norm
from scipy.optimize import minimize

# function for loading data
# function for loading data
def loadData(dataName, isShared, isAllIndependent):
	global data, sharedSigma, intensityVariable, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu, allIndependent
	sharedSigma = isShared  # Set to True if you want to use shared sigma across noise levels
	allIndependent = isAllIndependent  # Set to 1 if you want to use independent parameters for each condition
	intensityVariable="testDurS"  # Use raw test duration for log-normal model

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
	data['visualPSEBias'] = data['recordedDurVisualStandard'] -data["standardDur"]-data['conflictDur']
	data['visualPSEBiasTest'] = data['recordedDurVisualTest'] -data["testDurS"]


	data["unbiasedVisualStandardDur"]= data["recordedDurVisualStandard"] - data["visualPSEBias"]
	data["unbiasedVisualTestDur"]= data["recordedDurVisualTest"] - data["visualPSEBiasTest"]

	data["unbiasedVisualStandardDurMs"]= data["unbiasedVisualStandardDur"]*1000
	data["unbiasedVisualTestDurMs"]= data["unbiasedVisualTestDur"]*1000


	print(f"\n Total trials before cleaning\n: {len(data)}")
	data= data[data['audNoise'] != 0]
	data=data[data['standardDur'] != 0]
	data[standardVar] = data[standardVar].round(2)
	data[conflictVar] = data[conflictVar].round(3)
	uniqueSensory = data[sensoryVar].unique()
	uniqueStandard = data[standardVar].unique()
	uniqueConflict = sorted(data[conflictVar].unique())
	print(f"uniqueSensory: {uniqueSensory} \n uniqueStandard: {uniqueStandard} \n uniqueConflict: {uniqueConflict}")

	#data['avgAVDeltaS'] = (data['delta_dur_percents'] + (data['recordedDurVisualTest'] - data['recordedDurVisualStandard'])) / 2
	#data['deltaDurPercentVisual'] = ((data['recordedDurVisualTest'] - data['recordedDurVisualStandard']) / data['recordedDurVisualStandard'])
	#data['avgAVDeltaPercent'] = data[['delta_dur_percents', 'deltaDurPercentVisual']].mean(axis=1)

	# Define columns for chosing test or standard
	data['chose_test'] = (data['responses'] == data['order']).astype(int)
	data['chose_standard'] = (data['responses'] != data['order']).astype(int)

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

intensityVariable="testDurS"  # Use raw test duration for log-normal model

sensoryVar="audNoise"
standardVar="standardDur"
conflictVar="conflictDur"

def groupByChooseTest(x, groupArgs):
    #print(f"Grouping by {intensityVariable}, {sensoryVar}, {standardVar}, {conflictVar}")
    grouped = x.groupby(groupArgs).agg(
        num_of_chose_test=('chose_test', 'sum'),
        total_responses=('responses', 'count'),
        num_of_chose_standard=('chose_standard', 'sum'),
    ).reset_index()
    grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']
    
    # Add standard_dur column for compatibility
    if standardVar in grouped.columns:
        grouped['standard_dur'] = grouped[standardVar]

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
        # Convert seconds to milliseconds for plotting
        plt.scatter(grouped['x_mean']*1000, grouped['y_mean'], s=grouped['total_resp']/data['total_responses'].sum()*900, color=color)

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
        # each noise level has its own lambda and sigma, but mu varies with conflict
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

def psychometric_function(test_dur, standard_dur, lambda_, mu, sigma):
    """
    Log-normal observer model for duration estimation.
    
    This model is more appropriate for duration data than cumulative normal because:
    1. It gives zero probability for negative durations
    2. It naturally incorporates Weber's law-like behavior
    3. It's consistent with scalar timing theory
    
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

# Negative log-likelihood
def negative_log_likelihood(params, test_dur, standard_dur, chose_test, total_responses):
    lambda_, mu, sigma = params # Unpack parameters
    
    p = psychometric_function(test_dur, standard_dur, lambda_, mu, sigma) # Compute probability of choosing test
    epsilon = 1e-9 # Add a small number to avoid log(0) when calculating the log-likelihood
    p = np.clip(p, epsilon, 1 - epsilon) # Clip p to avoid log(0) and log(1)
    # Compute the negative log-likelihood
    log_likelihood = np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
    return -log_likelihood


# Fit psychometric function
def fit_psychometric_function(test_durations, standard_durations, nResp, totalResp, init_guesses=[0,0,0]):
    # then fits the psychometric function
    # order is lambda mu sigma
    #initial_guess = [0, -0.2, 0.05]  # Initial guess for [lambda, mu, sigma]
    bounds = [(0, 0.25), (-1.0, +1.0), (0.01, 2.0)]  # Bounds for log-normal model
    # fitting is done here
    result = minimize(
        negative_log_likelihood, x0=init_guesses, 
        args=(test_durations, standard_durations, nResp, totalResp),  # Pass the data and fixed parameters
        bounds=bounds,
        method='L-BFGS-B' 
    )
    # returns the fitted parameters lambda, mu, sigma
    return result.x


# Update nLLJoint to use getParams
def nLLJoint(params, test_durations, standard_durations, responses, total_responses, conflicts, noise_levels):
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
        lambda_, mu, sigma = getParams(params, conflict, audio_noise, nLambda, nSigma)
        
        # Calculate probability of choosing test using log-space model
        p = psychometric_function(test_dur, standard_dur, lambda_, mu, sigma)
        
        # Avoid numerical issues
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        
        # Add to negative log-likelihood
        nll += -1 * (chose_test * np.log(p) + (total_response - chose_test) * np.log(1 - p))
    
    return nll

# fitting function for joint model
def fitJoint(grouped_data,  initGuesses):
    if allIndependent:
        nCond = nSensoryVar * nConflictVar
        # Repeat initial guesses for each condition
        initGuesses = (initGuesses * nCond)[:nCond*3]
        test_durations = grouped_data[intensityVariable].values  # Raw test durations
        standard_durations = grouped_data[standardVar].values    # Standard durations
        chose_tests = grouped_data['num_of_chose_test'].values
        total_responses = grouped_data['total_responses'].values
        conflicts = grouped_data[conflictVar].values
        noise_levels = grouped_data[sensoryVar].values
        # Bounds for each parameter: lambda, mu, sigma per condition
        bounds = []
        for _ in range(nCond):
            bounds.extend([(0, 0.25), (-1.0, 1.0), (0.01, 2)])
    elif sharedSigma==True:
        # Initialize guesses for parameters 
        # lambda, mu,sigma
        initGuesses= [initGuesses[0]]*nLambda +  [initGuesses[1]]*nSensoryVar+[initGuesses[2]]*nSensoryVar*nConflictVar
        
        test_durations = grouped_data[intensityVariable].values  # Raw test durations
        standard_durations = grouped_data[standardVar].values    # Standard durations
        chose_tests = grouped_data['num_of_chose_test'].values
        total_responses = grouped_data['total_responses'].values
        conflicts = grouped_data[conflictVar].values
        noise_levels = grouped_data[sensoryVar].values
        
        
        # Set bounds for parameters (log-normal model bounds)
        # array of parameters in order of lambda, mu, sigma
        bounds = [(0,0.25)]*nLambda +  [(0.01, +2.0)]*nSensoryVar+[(-1.0, +1.0)]*nSensoryVar*nConflictVar 
    else:
        # Initialize guesses for parameters 
        # lambda, mu, sigma
        initGuesses= [initGuesses[0]]*nLambda + [initGuesses[1]]*nSensoryVar*nConflictVar+ [initGuesses[2]]*nSensoryVar*nConflictVar
        
        test_durations = grouped_data[intensityVariable].values  # Raw test durations
        standard_durations = grouped_data[standardVar].values    # Standard durations
        chose_tests = grouped_data['num_of_chose_test'].values
        total_responses = grouped_data['total_responses'].values
        conflicts = grouped_data[conflictVar].values
        noise_levels = grouped_data[sensoryVar].values
        
        
        # Set bounds for parameters (log-normal model bounds)
        # array of parameters in order of lambda, mu, sigma
        bounds = [(0, 0.25)]*nLambda + [(-1.0, +1.0)]*nSensoryVar*nConflictVar + [(0.01, +2.0)]*nSensoryVar*nConflictVar 

    #print(f" len initGuesses: {len(initGuesses)}")

    # Minimize negative log-likelihood
    result = minimize(
        nLLJoint,
        x0=initGuesses,
        args=(test_durations, standard_durations, chose_tests, total_responses, conflicts, noise_levels),
        bounds=bounds,
        method='L-BFGS-B'  # Use L-BFGS-B for bounded optimization
    )
    
    return result

def nLLJoint(params, test_durations, standard_durations, responses, total_responses, conflicts, noise_levels):
    """
    Compute negative log likelihood for all conditions using log-space psychometric function.
    (Duplicate function removed - using the one defined earlier)
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
        lambda_, mu, sigma = getParams(params, conflict, audio_noise, nLambda, nSigma)
        
        # Calculate probability of choosing test using log-space model
        p = psychometric_function(test_dur, standard_dur, lambda_, mu, sigma)
        
        # Avoid numerical issues
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        
        # Add to negative log-likelihood
        nll += -1 * (chose_test * np.log(p) + (total_response - chose_test) * np.log(1 - p))
    
    return nll

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
    groupedData = groupByChooseTest(data, 
									groupArgs=[intensityVariable, sensoryVar, standardVar, conflictVar])
    global nLambda, nSensoryVar, nConflictVar, uniqueSensory, uniqueConflict
    nSensoryVar = len(uniqueSensory)  # Number of sensory variables
    nConflictVar = len(uniqueConflict)  # Number of conflict variables
    uniqueSensory = data['audNoise'].unique()
    uniqueConflict = sorted(data[conflictVar].unique())
    
    test_durations = groupedData[intensityVariable].values  # Raw test durations
    standard_durations = groupedData[standardVar].values   # Standard durations
    responses = groupedData['num_of_chose_test'].values
    totalResp = groupedData['total_responses'].values
    conflictLevels = groupedData[conflictVar].values
    noiseLevels = groupedData[sensoryVar].values

    # For initial guess estimation, convert to percentage differences for compatibility
    percentage_diffs = (test_durations - standard_durations) / standard_durations
    singleInitGuesses = estimate_initial_guesses(percentage_diffs, responses, totalResp)

    multipleInitGuesses = multipleInitGuessesWEstimate(singleInitGuesses, nStart)

    # Fit the model with multiple starting points
    
    best_fit = None
    best_nll = float('inf')  # Initialize with infinity
    disable=False
    if len(multipleInitGuesses)==1:
        disable=True
    
    for i in tqdm(range(len(multipleInitGuesses)), desc="Fitting multiple starting points",disable=disable):
        
        fit = fitJoint(groupedData, initGuesses=multipleInitGuesses[i])
        nll = nLLJoint(fit.x, test_durations, standard_durations, responses, totalResp, conflictLevels, noiseLevels)

        if nll < best_nll:
            best_nll = nll
            best_fit = fit

    return best_fit

def plot_fitted_psychometric(data, best_fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict, standardVar, sensoryVar, conflictVar, intensityVariable):
    colors = sns.color_palette("viridis", n_colors=len(uniqueSensory))  # Use Set2 palette for different noise levels
    plt.figure(figsize=(12*2.5, 6*2))

    for i, standardLevel in enumerate(uniqueStandard):
        for j, audioNoiseLevel in enumerate(sorted(uniqueSensory)):
            print(f"standardLevel: {standardLevel}, audioNoiseLevel: {audioNoiseLevel}, uniqueConflict: {uniqueConflict}")
            for k, conflictLevel in enumerate(uniqueConflict):
                lambda_, mu, sigma = getParams(best_fit.x, conflictLevel, audioNoiseLevel, nLambda, nSigma)
                
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
                df = data[round(data[standardVar], 2) == round(standardLevel,2)]
                df = df[df[sensoryVar] == audioNoiseLevel]
                df = df[df[conflictVar] == conflictLevel]
                dfFiltered = groupByChooseTest(df,groupArgs=[intensityVariable, sensoryVar, standardVar, conflictVar])
                test_durations = dfFiltered[intensityVariable].values  # Raw test durations
                if len(test_durations) == 0:
                    continue
                responses = dfFiltered['num_of_chose_test'].values
                totalResponses = dfFiltered['total_responses'].values
                
                # Fit the psychometric function
                plt.subplot(1, 2, j+1)
                
                # Create range of test durations for smooth curve
                minX = min(test_durations) * 0.8  # Use raw duration range
                maxX = max(test_durations) * 1.2
                x_smooth = np.linspace(minX, maxX, 1000)
                
                # For plotting, we need to provide both test and standard durations
                standard_dur_array = np.full_like(x_smooth, standardLevel)  # Constant standard duration
                y = psychometric_function(x_smooth, standard_dur_array, lambda_, mu, sigma)
                
                color=sns.color_palette("viridis", as_cmap=True)(k / len(uniqueConflict))  # Use a colormap for different conflict levels
                plt.plot(x_smooth*1000, y, color=color, label=f"C: {int(conflictLevel*1000)}, $\lambda$: {lambda_:.2f} $\mu$: {mu:.2f}, $\sigma$: {sigma:.2f}", linewidth=4,)
                plt.axvline(x=standardLevel*1000, color='gray', linestyle='--')  # Show standard duration line
                plt.axhline(y=0.5, color='gray', linestyle='--')
                plt.xlabel(f"Test duration (ms)")
                plt.ylabel("P(chose test)")
                #plt.title(f" {pltTitle} ", fontsize=16)

                plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
                plt.legend(fontsize=14, title_fontsize=14)
                plt.grid()
                
                # Plot binned data points (convert to ms for display)
                dfFiltered['testDurMs'] = dfFiltered[intensityVariable] * 1000
                bin_and_plot(dfFiltered, bin_method='cut', bins=10, plot=True, color=color)

                plt.text(0.05, 0.8, f"Shared $\lambda$: {lambda_:.2f}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
                plt.tight_layout()
                plt.grid(True)

                # print the fitted parameters
                print(f"Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, Mu: {mu:.3f}, Sigma: {sigma:.3f}")
    #plt.show()


def simulate_dataset(params, gdf):
    """
    Produce one synthetic data set that has the SAME predictors
    (test_dur, standard_dur, conflict, noise, total_responses) as 'gdf' but with
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

    # Use log-space psychometric function
    test_durations = sim_gdf[intensityVariable].values
    standard_durations = sim_gdf[standardVar].values
    p_choose_test = psychometric_function(test_durations, standard_durations, lam, mu, sigma)

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
    In log-space model, PSE = standard * exp(mu), so PSE shift = standard * (exp(mu) - 1)
    """
    plt.figure(figsize=(12, 6))
    m = 0
    for i, standardLevel in enumerate(uniqueStandard):
        for j, audioNoiseLevel in enumerate(uniqueSensory):
            for k, conflictLevel in enumerate(uniqueConflict):
                lambda_, mu, sigma = getParams(best_fit.x, conflictLevel, audioNoiseLevel, nLambda, nSigma)
                
                # Calculate PSE shift in linear space (ms)
                pse_shift_ms = standardLevel * (np.exp(mu) - 1) * 1000
                
                m += 1        
                plt.subplot(1, 2, j + 1)
                color = sns.color_palette("viridis", as_cmap=True)(k / len(uniqueConflict))
                plt.scatter(conflictLevel * 1000, pse_shift_ms, color=color, s=100)
                plt.xlabel("Visual Conflict(ms)")
                plt.ylabel("PSE shift (ms)")
                plt.title(f"Standard: {standardLevel}, Noise: {audioNoiseLevel}")
                plt.grid()
                plt.axhline(y=0, color='gray', linestyle='--')
                plt.axvline(x=0, color='gray', linestyle='--')

                mu_all = []
                for fit in allBootedFits:
                    lambda_, muBooted, sigma = getParams(fit, conflictLevel, audioNoiseLevel, nLambda, nSigma)
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
    parser = argparse.ArgumentParser(description="Fit joint psychometric model.")
    parser.add_argument("--dataName", type=str, help="CSV file name in data/ directory")
    parser.add_argument("--sharedSigma", action="store_true", help="Use shared sigma across noise levels")
    args = parser.parse_args()

    dataName = args.dataName
    global sharedSigma
    global allIndependent
    allIndependent= 1
    sharedSigma = args.sharedSigma

    if not dataName:
        dataName = "all_main.csv"
    global pltTitle
    pltTitle=dataName.split("_")[1]
    pltTitle=dataName.split("_")[0]+str(" ")+pltTitle
        
    sharedSigma = False  # Set to True if you want to use shared sigma across noise levels
    # Example usage
    data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(dataName, sharedSigma,isAllIndependent=allIndependent)
    fit = fitMultipleStartingPoints(data, nStart=1)
    # print the fitted parameters
    print(f"Fitted parameters: {fit.x}")

    # Plot the fitted psychometric functions
    plot_fitted_psychometric(
        data, fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
        standardVar, sensoryVar, conflictVar, intensityVariable
    )
    plotStairCases(data)


    # Plot the relation between conflict and PSE (mu) with confidence intervals
    #allBootedFits = paramBootstrap(fit.x, nBoots=100)
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