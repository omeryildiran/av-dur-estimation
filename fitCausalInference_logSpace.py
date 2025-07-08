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
	
	data["logStandardDur"] = np.log(1000*data[standardVar])
	data["logConflictDur"] = np.log(1000*data[conflictVar])
	data["logTestDur"] = np.log(1000*data["testDurS"])
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
from tqdm import tqdm

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

# fitting function for joint model
def fitJoint(grouped_data,  initGuesses):
	if allIndependent:
		nCond = nSensoryVar * nConflictVar
		# Repeat initial guesses for each condition
		initGuesses = (initGuesses * nCond)[:nCond*3]
		intensities = grouped_data[intensityVariable]
		chose_tests = grouped_data['num_of_chose_test']
		total_responses = grouped_data['total_responses']
		conflicts = grouped_data[conflictVar]
		noise_levels = grouped_data[sensoryVar]
		# Bounds for each parameter: lambda, mu, sigma per condition
		bounds = []
		for _ in range(nCond):
			bounds.extend([(0, 0.25), (-2, 2), (0.01, 2)])
	elif sharedSigma==True:
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

def nLLJoint(params, delta_dur, responses, total_responses, conflicts, noise_levels):
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

# Fit the psychometric function to the grouped data
def multipleInitGuesses(singleInitGuesses, nStart):
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

# Regular psychometric function fitting with multiple starting points
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
		for j, audioNoiseLevel in enumerate(sorted(uniqueSensory)):
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
				plt.plot(x, y, color=color, label=f"C: {int(conflictLevel*1000)}, $\lambda$: {lambda_:.2f} $\mu$: {mu:.2f}, $\sigma$: {sigma:.2f}", linewidth=4,)
				plt.axvline(x=0, color='gray', linestyle='--')
				plt.axhline(y=0.5, color='gray', linestyle='--')
				plt.xlabel(f"({intensityVariable}) Test(stair-a)-Standard(a) Duration Difference Ratio(%)")
				plt.ylabel("P(chose test)")
				#plt.title(f" {pltTitle} ", fontsize=16)

				plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
				plt.legend(fontsize=14, title_fontsize=14)
				plt.grid()
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

				mu_all = []
				for fit in allBootedFits:
					lambda_, muBooted, sigma = getParams(fit, conflictLevel, audioNoiseLevel, nLambda, nSigma)
					mu_all.append(muBooted)
				mu_ci = np.percentile(mu_all, [2.5, 97.5])
				lower_err = np.maximum(mu*1000/2-mu_ci[0]*1000/2, 0)
				upper_err = np.maximum(mu_ci[1]*1000/2-mu*1000/2, 0)
				plt.errorbar(conflictLevel*1000, mu*1000/2, yerr=[[lower_err], [upper_err]], fmt='o', color=color, capsize=10
							, label=f"95% CI for mu: {mu_ci[0]*1000/2:.2f} - {mu_ci[1]*1000/2:.2f}", linewidth=2)
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



# ===============================
# JOINT CAUSAL INFERENCE FITTING
# ===============================

# get parameters func acording to the SNR(1+3x2=7 parameters)
def getParamsCausal(params, conflict, SNR):
	"""Extract causal inference parameters for a specific condition (conflict, noise)."""

	lambda_=params[0]
	if np.isclose(SNR, 0.1):  # High SNR (SNR=0.1 means 1/0.1=10, clean, more reliable)
		sigma_av_a=params[1]
		sigma_av_v=params[2]
		p_c=params[3]  # P(C=1) for high SNR
	elif np.isclose(SNR,1.2):  # Low SNR (SNR=1.2 means low reliability, noisy)
		sigma_av_a=params[4]
		sigma_av_v=params[5]
		p_c=params[6]  # P(C=1) for low SNR

	return lambda_,sigma_av_a,sigma_av_v,p_c

# likelihood function
def unimodalLikelihood( S, sigma):
	#S=np.log(S)  # convert S to log scale
	# P(m|s) # likelihood of measurements given the true duration
	m=np.linspace(0, S + 10*sigma, 500)
	p_m=norm.pdf(m, loc=S, scale=sigma) # uncomment
	return m, p_m


# probability density function of a Gaussian distribution
def gaussianPDF(x,S, sigma):
	return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-((x-S)**2)/(2*(sigma**2)))
# Fusion function
def fusionAV(sigma_av_a, sigma_av_v, S_a, conflict):
	S_v = S_a + conflict
	J_AV_A = 1 / sigma_av_a**2
	J_AV_V = 1 / sigma_av_v**2
	w_a = J_AV_A / (J_AV_A + J_AV_V)
	w_v = 1 - w_a
	fused_S_av = w_a * S_a + w_v * S_v
	sigma_S_AV_hat = np.sqrt(1 / (J_AV_A + J_AV_V))
	return fused_S_av, sigma_S_AV_hat

# Likelihood under common cause
def likelihood_C1(ma, mv, sigma_av_a, sigma_av_v):
	var_sum = sigma_av_a**2 + sigma_av_v**2
	return (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(ma - mv)**2 / (2 * var_sum))

# Likelihood under independent causes
def likelihood_C2(m_a,m_v,S_a, S_v, sigma_av_a, sigma_av_v):
	return norm.pdf(m_a, loc=S_a, scale=sigma_av_a) * norm.pdf(m_v, loc=S_v, scale=sigma_av_v)

# Posterior of common cause
def posterior_C1(likelihood_c1, likelihood_c2, p_c):
	return (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))

# Decision noise
def genDecisionNoiseCausal(sigma_av_a, sigma_av_v, p_common):
	var_fusion = 1 / (1/sigma_av_a**2 + 1/sigma_av_v**2)
	var_segregated = sigma_av_a**2
	var_estimate = p_common * var_fusion + (1 - p_common) * var_segregated
	return np.sqrt(2 * var_estimate)


def causalInference(S_a,conflict,sigma_av_a, sigma_av_v,  p_c):
	S_v= S_a+conflict
	
	#
	m_a = S_a# np.random.normal(S_a, sigma_av_a)
	m_v = S_v#np.random.normal(S_v, sigma_av_v)
	
	# LIKELIHOODS fo CC
	likelihood_c1 = likelihood_C1(m_a, m_v, sigma_av_a, sigma_av_v)
	likelihood_c2 = likelihood_C2(m_a, m_v,S_a,S_v, sigma_av_a, sigma_av_v)
	# POSTERIOR P Common
	posterior_c1 = posterior_C1(likelihood_c1, likelihood_c2, p_c)
	posterior_c2 = 1 - posterior_c1
	# Estimates
	# \hat S_{av,c=1} Fusion
	fused_S_av, sigma_S_AV_hat = fusionAV(sigma_av_a, sigma_av_v, S_a, conflict)
	# C=2 no CC
	hat_S_AV_A_No_CC=S_a
	# Model avaraging
	hat_S_AV_A_final = posterior_c1 * fused_S_av + posterior_c2 * hat_S_AV_A_No_CC

	return hat_S_AV_A_final

def probTestLonger(deltaDur, conflict, lambda_,sigma_av_a, sigma_av_v,p_c):
	
	S_a_s=0.5
	S_a_t=S_a_s+deltaDur

	"""Compute psychometric using causal inference model"""
	# take estimates
	est_standard= causalInference(S_a_s,conflict,sigma_av_a, sigma_av_v,  p_c) 
	est_test= fusionAV(sigma_av_a, sigma_av_v, S_a_t, conflict)[0]  # We compute the fused estimate for test duration because test duration do
	# not have a conflict, so we use the fusion function directly.
	
	# Decison decison
	deltaEstimates=est_test-est_standard
	sigma_decision=genDecisionNoiseCausal(sigma_av_a,sigma_av_v,p_c)
	p_choose_test = lambda_/2 + (1 - lambda_) * norm.cdf(deltaEstimates, loc=0, scale=sigma_decision)	
	return p_choose_test


def nLL_causal_inference(params, rawData):
	ll = 0.0
	lenData = len(rawData)
	y = np.empty(lenData)
	for i in range(lenData):
		currSNR = rawData["audNoise"][i]
		currConflict = rawData["conflictDur"][i]  # Using the same name as in the rest of the code
		currResp = rawData["chose_test"][i]
		currDeltaDur = rawData["deltaDurS"][i]  # Using the consistent naming
		y[i] = currResp
		lambda_, sigma_av_a, sigma_av_v, p_c = getParamsCausal(params, currConflict, currSNR)
		P = probTestLonger(currDeltaDur, currConflict, lambda_, sigma_av_a, sigma_av_v, p_c)
		
		# Add small epsilon to avoid log(0)
		epsilon = 1e-9
		P = np.clip(P, epsilon, 1 - epsilon)
		
		ll += y[i] * np.log(P) + (1 - y[i]) * np.log(1 - P)
	
	return -ll
def fitCausalInferenceModel(data, initGuesses, use_vectorized=True, show_progress=0):
	"""Fit the causal inference model to the data."""
	# parameters: lambda, sigma_a_high, sigma_v_high, p_c_high, sigma_a_low, sigma_v_low, p_c_low
	bounds = [
		(0, 0.25),      # lambda_
		(0.01, 1.0),    # sigma_a (high SNR = 0.1)
		(0.01, 1.0),    # sigma_v (high SNR = 0.1)
		(0.01, 0.95),   # p_c (high SNR = 0.1) - independent parameter
		(0.01, 1.0),    # sigma_a (low SNR = 1.2)
		(0.01, 1.0),    # sigma_v (low SNR = 1.2)
		(0.01, 0.95)    # p_c (low SNR = 1.2) - independent parameter
	]
	
	# Add bounds checking for initial guesses
	initGuesses = np.array(initGuesses)
	for i, (guess, (lower, upper)) in enumerate(zip(initGuesses, bounds)):
		if guess < lower or guess > upper:
			print(f"Warning: Initial guess {i} ({guess}) outside bounds [{lower}, {upper}]. Clipping.")
			initGuesses[i] = np.clip(guess, lower + 1e-6, upper - 1e-6)
	
	class TqdmMinimizeCallback:
		def __init__(self, total=100, show_progress=show_progress):
			self.show_progress = show_progress
			if self.show_progress:
				self.pbar = tqdm(total=total, desc="Fitting Causal Inference", leave=True)
			self.last_nfev = 0

		def __call__(self, xk):
			# This callback is called after each iteration
			if self.show_progress:
				self.pbar.update(1)

		def close(self):
			if self.show_progress:
				self.pbar.close()

	callback = TqdmMinimizeCallback(total=100, show_progress=0)
	
	# Choose which negative log-likelihood function to use
	if use_vectorized:
		nll_function = nLL_causal_inference_fully_vectorized
		#print("\nUsing fully vectorized causal inference fitting...\n")
	else:
		nll_function = nLL_causal_inference
		#print("\nUsing original (slower) causal inference fitting...\n")

	try:
		# Test the function once before optimization
		test_nll = nll_function(initGuesses, data)
		if not np.isfinite(test_nll):
			print(f"Warning: Initial NLL is not finite: {test_nll}")
			return None
		
		result = minimize(
			nll_function,
			x0=initGuesses,
			args=(data,),
			bounds=bounds,
			method='L-BFGS-B',
			callback=callback
		)
		callback.close()
		
		if not result.success:
			print(f"Optimization failed: {result.message}")
		
		return result.x
		
	except Exception as e:
		print(f"Error during fitting: {e}")
		callback.close()
		return None

# ===============================
# VECTORIZED CAUSAL INFERENCE FUNCTIONS
# ===============================


def causalInference_vectorized(S_a, conflict, sigma_av_a, sigma_av_v, p_c):
	"""Vectorized version of causal inference computation."""
	S_v = S_a + conflict
	
	# Measurements (in this case, just the true values)
	m_a = S_a
	m_v = S_v
	
	# LIKELIHOODS
	# Common cause likelihood
	var_sum = sigma_av_a**2 + sigma_av_v**2
	likelihood_c1 = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))
	
	# Independent causes likelihood
	likelihood_c2 = norm.pdf(m_a, loc=S_a, scale=sigma_av_a) * norm.pdf(m_v, loc=S_v, scale=sigma_av_v)
	
	# POSTERIOR probability of common cause
	posterior_c1 = (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))
	
	# ESTIMATES
	# Fusion estimate (common cause)
	J_AV_A = 1 / sigma_av_a**2
	J_AV_V = 1 / sigma_av_v**2
	w_a = J_AV_A / (J_AV_A + J_AV_V)
	w_v = 1 - w_a
	fused_S_av = w_a * S_a + w_v * S_v
	
	# Segregation estimate (independent causes)
	segregated_S_av = S_a
	
	# Model averaging
	hat_S_AV_A_final = posterior_c1 * fused_S_av + (1 - posterior_c1) * segregated_S_av
	
	return hat_S_AV_A_final

def probTestLonger_vectorized(deltaDur, conflict, lambda_, sigma_av_a, sigma_av_v, p_c):
	"""Vectorized version of probability computation."""
	S_a_s = 0.5
	S_a_t = S_a_s + deltaDur
	
	# Compute estimates
	est_standard = causalInference_vectorized(S_a_s, conflict, sigma_av_a, sigma_av_v, p_c)
	
	# For test duration, use fusion directly (no conflict)
	J_AV_A = 1 / sigma_av_a**2
	J_AV_V = 1 / sigma_av_v**2
	w_a = J_AV_A / (J_AV_A + J_AV_V)
	w_v = 1 - w_a
	est_test = w_a * S_a_t + w_v * S_a_t  # Since there's no conflict, S_v = S_a
	
	# Decision noise
	var_fusion = 1 / (1/sigma_av_a**2 + 1/sigma_av_v**2)
	var_segregated = sigma_av_a**2
	var_estimate = p_c * var_fusion + (1 - p_c) * var_segregated
	sigma_decision = np.sqrt(2 * var_estimate)
	
	# Decision
	deltaEstimates = est_test - est_standard
	p_choose_test = lambda_/2 + (1 - lambda_) * norm.cdf(deltaEstimates, loc=0, scale=sigma_decision)
	
	return p_choose_test

def nLL_causal_inference_fully_vectorized(params, rawData):
	"""Fully vectorized version - even faster by avoiding parameter loops."""
	# Extract data arrays - using correct columns
	snr_values = rawData["audNoise"].values
	conflict_values = rawData["conflictDur"].values  # Use conflictDur, not conflictDurMs
	responses = rawData["chose_test"].values
	delta_dur_values = rawData["deltaDurS"].values
	
	# Create boolean masks for different SNR conditions
	snr_01_mask = np.isclose(snr_values, 0.1)
	snr_12_mask = np.isclose(snr_values, 1.2)
	
	# Pre-allocate parameter arrays
	n_trials = len(rawData)
	lambda_arr = np.full(n_trials, params[0])  # lambda is shared
	sigma_av_a_arr = np.empty(n_trials)
	sigma_av_v_arr = np.empty(n_trials)
	p_c_arr = np.empty(n_trials)
	
	# Vectorized parameter assignment
	sigma_av_a_arr[snr_01_mask] = params[1]  # sigma_a for high SNR (0.1)
	sigma_av_v_arr[snr_01_mask] = params[2]  # sigma_v for high SNR (0.1)
	p_c_arr[snr_01_mask] = params[3]         # p_c for high SNR (0.1)
	
	sigma_av_a_arr[snr_12_mask] = params[4]  # sigma_a for low SNR (1.2)
	sigma_av_v_arr[snr_12_mask] = params[5]  # sigma_v for low SNR (1.2)
	p_c_arr[snr_12_mask] = params[6]         # p_c for low SNR (1.2)
	
	# Work in original scale, not log scale for causal inference
	S_a_s = 0.5  # Standard duration in seconds
	S_a_t = S_a_s + delta_dur_values  # Test durations
	
	# Standard estimates (vectorized)
	S_v_s = S_a_s + conflict_values  # Visual standard duration
	m_a_s = S_a_s
	m_v_s = S_v_s
	
	# Common cause likelihood for standard
	var_sum_s = sigma_av_a_arr**2 + sigma_av_v_arr**2
	likelihood_c1_s = (1 / np.sqrt(2 * np.pi * var_sum_s)) * np.exp(-(m_a_s - m_v_s)**2 / (2 * var_sum_s))
	
	# Independent causes likelihood for standard
	likelihood_c2_s = norm.pdf(m_a_s, loc=S_a_s, scale=sigma_av_a_arr) * norm.pdf(m_v_s, loc=S_v_s, scale=sigma_av_v_arr)
	
	# Posterior probability of common cause for standard
	posterior_c1_s = (likelihood_c1_s * p_c_arr) / (likelihood_c1_s * p_c_arr + likelihood_c2_s * (1 - p_c_arr))
	
	# Fusion estimate for standard
	J_AV_A = 1 / sigma_av_a_arr**2
	J_AV_V = 1 / sigma_av_v_arr**2
	w_a = J_AV_A / (J_AV_A + J_AV_V)
	w_v = 1 - w_a
	fused_S_av_s = w_a * S_a_s + w_v * S_v_s
	
	# Final estimate for standard (model averaging)
	est_standard = posterior_c1_s * fused_S_av_s + (1 - posterior_c1_s) * S_a_s
	
	# Test estimates (fusion only, no conflict for test)
	est_test = S_a_t  # No conflict for test duration
	
	# Decision noise
	var_fusion = 1 / (1/sigma_av_a_arr**2 + 1/sigma_av_v_arr**2)
	var_segregated = sigma_av_a_arr**2
	var_estimate = p_c_arr * var_fusion + (1 - p_c_arr) * var_segregated
	sigma_decision = np.sqrt(2 * var_estimate)
	
	# Decision
	deltaEstimates = est_test - est_standard
	P = lambda_arr/2 + (1 - lambda_arr) * norm.cdf(deltaEstimates, loc=0, scale=sigma_decision)
	
	# Clip probabilities to avoid log(0)
	epsilon = 1e-9
	P = np.clip(P, epsilon, 1 - epsilon)
	# Vectorized log-likelihood computation
	ll = np.sum(responses * np.log(P) + (1 - responses) * np.log(1 - P))
	return -ll

def multipleInitGuessesCausal(singleInitGuesses, nStart):
	if nStart == 1:
		return [singleInitGuesses]
	print(f"Generating {nStart} initial guesses for causal inference model...")
	guesses = []
	for _ in range(nStart):
		lambda_ = np.random.uniform(0.001, 0.2)
		sigma_a_high = np.random.uniform(0.01, 0.5)  # High SNR (0.1)
		sigma_v_high = np.random.uniform(0.01, 0.5)  # High SNR (0.1)
		p_c_high = np.random.uniform(0.05, 0.8)     # P(C=1) for high SNR (independent)
		sigma_a_low = np.random.uniform(0.01, 0.5)   # Low SNR (1.2)
		sigma_v_low = np.random.uniform(0.01, 0.5)   # Low SNR (1.2)
		p_c_low = np.random.uniform(0.05, 0.9)       # P(C=1) for low SNR (independent)
		
		guesses.append([
			lambda_, sigma_a_high, sigma_v_high, p_c_high,
			sigma_a_low, sigma_v_low, p_c_low
		])
	print(f"Generated {len(guesses)} initial guesses for causal inference model.")
	return guesses



def fitCausalInferenceMultipleStarts(data, singleInitGuesses, nStart=5, use_vectorized=True):
	"""
	Fit the causal inference model with multiple starting points.
	"""
	# Generate multiple initial guesses
	multipleInitGuesses = multipleInitGuessesCausal(singleInitGuesses, nStart)
	
	best_fit = None
	best_nll = float('inf')
	best_params = None
	successful_fits = 0
	
	disable_progress = (len(multipleInitGuesses) == 1)
	
	print(f"Fitting causal inference model with {len(multipleInitGuesses)} different starting points...")
	
	for i in tqdm(range(len(multipleInitGuesses)), desc="Fitting multiple starting points", disable=disable_progress):
		try:
			# Fit with current initial guess - disable individual progress bars
			fitted_params = fitCausalInferenceModel(data, multipleInitGuesses[i], use_vectorized=use_vectorized, show_progress=False)
			
			if fitted_params is None:
				print(f"Start {i+1}: Failed during optimization")
				continue
				
			# Calculate negative log-likelihood for this fit
			if use_vectorized:
				nll = nLL_causal_inference_fully_vectorized(fitted_params, data)
			else:
				nll = nLL_causal_inference(fitted_params, data)
			
			if not np.isfinite(nll):
				print(f"Start {i+1}: Non-finite NLL = {nll}")
				continue
				
			successful_fits += 1
			
			# Check if this is the best fit so far
			if nll < best_nll:
				best_nll = nll
				best_params = fitted_params
				best_fit = i
			
			print(f"Start {i+1}: NLL = {nll:.4f}")
			
		except Exception as e:
			print(f"Start {i+1}: Failed with error: {e}")
			continue
	
	if best_params is None:
		raise ValueError(f"All fitting attempts failed! Successful fits: {successful_fits}/{len(multipleInitGuesses)}")
	
	print(f"\nBest fit found at starting point {best_fit+1} with NLL = {best_nll:.4f}")
	print(f"Successful fits: {successful_fits}/{len(multipleInitGuesses)}")
	print(f"Best parameters: {best_params}")
	
	return best_params, best_nll

def fitCausalInferenceWrapper(data, initGuesses=None, nStart=1, use_vectorized=True, verbose=True):
	# Default initial guesses if none provided
	# [lambda, sigma_a_high, sigma_v_high, p_c_high, sigma_a_low, sigma_v_low, p_c_low]
	if initGuesses is None:
		initGuesses = [0.03, 0.1, 0.1, 0.3, 0.1, 0.1, 0.6]  # Let data determine the relationship
	
	if nStart == 1:
		# Single starting point
		if verbose:
			print("=== Fitting with single starting point ===")
		fitted_params = fitCausalInferenceModel(data, initGuesses, use_vectorized=use_vectorized, show_progress=verbose)
		
		# Calculate NLL
		if use_vectorized:
			nll = nLL_causal_inference_fully_vectorized(fitted_params, data)
		else:
			nll = nLL_causal_inference(fitted_params, data)
		
		if verbose:
			print(f"Fitted parameters: {fitted_params}")
			print(f"NLL: {nll:.4f}")
		
		return fitted_params, nll
	
	else:
		# Multiple starting points
		if verbose:
			print(f"=== Fitting with {nStart} starting points ===")
		
		fitted_params, nll = fitCausalInferenceMultipleStarts(
			data, initGuesses, nStart=nStart, use_vectorized=use_vectorized
		)
		
		return fitted_params, nll

def validate_causal_inference_params(fitted_params, verbose=True):
	"""
	Validate that the fitted causal inference parameters satisfy theoretical constraints.
	Specifically, P(C=1) should decrease as SNR increases (sensory reliability improves).
	In your experiment: SNR=0.1 means HIGH SNR (reliable), SNR=1.2 means LOW SNR (noisy).
	"""
	# Extract P(C=1) for both SNR conditions
	_, _, _, p_c_high = getParamsCausal(fitted_params, 0, 0.1)   # High SNR (reliable)
	_, _, _, p_c_low = getParamsCausal(fitted_params, 0, 1.2)    # Low SNR (noisy)
	
	if verbose:
		print(f"\nValidating causal inference parameters:")
		print(f"P(C=1) for high SNR (0.1): {p_c_high:.3f}")
		print(f"P(C=1) for low SNR (1.2): {p_c_low:.3f}")
		print(f"Difference (low - high): {p_c_low - p_c_high:.3f}")
	
	# Check if P(C=1) decreases with higher SNR (i.e., P(C=1) should be lower for SNR=0.1 than for SNR=1.2)
	if p_c_low > p_c_high:
		if verbose:
			print("✓ VALID: P(C=1) decreases with increasing SNR (as expected)")
		return True
	else:
		if verbose:
			print("✗ INVALID: P(C=1) does NOT decrease with increasing SNR")
			print("  This violates causal inference theory!")
		return False

def debug_data_issues(data):
	"""Debug function to check for common data issues."""
	print("=== Data Debugging ===")
	print(f"Data shape: {data.shape}")
	print(f"Columns: {list(data.columns)}")
	
	# Check for missing values
	missing_cols = data.isnull().sum()
	if missing_cols.sum() > 0:
		print(f"Missing values:\n{missing_cols[missing_cols > 0]}")
	
	# Check key columns
	key_cols = ['audNoise', 'conflictDur', 'deltaDurS', 'chose_test']
	for col in key_cols:
		if col in data.columns:
			print(f"{col}: min={data[col].min():.4f}, max={data[col].max():.4f}, "
			      f"mean={data[col].mean():.4f}, unique_count={data[col].nunique()}")
		else:
			print(f"WARNING: {col} not found in data!")
	
	# Check for infinite values
	for col in key_cols:
		if col in data.columns:
			inf_count = np.isinf(data[col]).sum()
			if inf_count > 0:
				print(f"WARNING: {col} has {inf_count} infinite values!")
	
	# Check conflict values specifically
	if 'conflictDur' in data.columns:
		conflict_vals = data['conflictDur'].unique()
		print(f"Unique conflict values: {sorted(conflict_vals)}")
		if any(val <= 0 for val in conflict_vals):
			print("WARNING: Conflict values include zero or negative values!")
			print("This will cause issues with log() operations!")
	
	print("=== End Debug ===\n")

# Example usage:
if __name__ == "__main__":
	# Load data
	loadDataVars = loadData("all_all.csv", 1, 1)
	data = loadDataVars[0]
	
	# Debug data issues
	debug_data_issues(data)
	
	# Initial guesses for [lambda, sigma_av_a_high, sigma_av_v_high, p_c_high, sigma_av_a_low, sigma_av_v_low, p_c_low]
	initGuesses = [0.03, 0.1, 0.1, 0.3, 0.1, 0.1, 0.6]  # Let data determine the relationship
	
	# Test with a single starting point first to debug
	print("=== Testing Single Starting Point ===")
	try:
		fitted_params_single, nll_single = fitCausalInferenceWrapper(data, initGuesses, nStart=1, verbose=True)
		print(f"Single fit successful: NLL = {nll_single:.4f}")
		
		# Validate the single fit
		validate_causal_inference_params(fitted_params_single, verbose=True)
		
	except Exception as e:
		print(f"Single fit failed: {e}")
		import traceback
		traceback.print_exc()
	
	# Option 2: Multiple starting points (more robust) - only if single works
	print("\n=== Multiple Starting Points ===")
	fitted_params_multi, nll_multi = fitCausalInferenceWrapper(data, initGuesses, nStart=10, use_vectorized=True)
	
	# Validate the multiple fits
	validate_causal_inference_params(fitted_params_multi, verbose=True)
	
	# Option 1: Single starting point (faster, but less robust)
	print("\n=== Single Starting Point ===")
	fitted_params_single, nll_single = fitCausalInferenceWrapper(data, initGuesses)
	
	# Use the best fit for plotting
	fitted_params = fitted_params_multi if nll_multi < nll_single else fitted_params_single
	
	# Final validation
	print("\n=== Final Model Validation ===")
	is_valid = validate_causal_inference_params(fitted_params, verbose=True)
	
	# Plot results (existing plotting code continues from here...)
	plt.figure(figsize=(16, 6))
	for i, standardLevel in enumerate(uniqueStandard):
		for j, audioNoiseLevel in enumerate(sorted(uniqueSensory)):

			for k, conflictLevel in enumerate(uniqueConflict):
				plt.subplot(1, 2, j+1)
				lambda_, sigma_av_a, sigma_av_v, p_c = getParamsCausal(fitted_params, conflictLevel, audioNoiseLevel)
				x = np.linspace(-0.9, 0.9, 500)
				y = probTestLonger_vectorized(x, conflictLevel, lambda_, sigma_av_a, sigma_av_v, p_c)
				color = sns.color_palette("viridis", as_cmap=True)(k / len(uniqueConflict))
				plt.plot(x, y, color=color, label=f"c: {int(conflictLevel*1000)}, $\sigma_a$: {sigma_av_a:.2f}, $\sigma_v$: {sigma_av_v:.2f}", linewidth=4)
				plt.axvline(x=0, color='gray', linestyle='--')
				plt.axhline(y=0.5, color='gray', linestyle='--')
				plt.xlabel(f"({intensityVariable}) Test(stair-a)-Standard(a) Duration Difference Ratio(%)")
				plt.ylabel("P(chose test)")
				plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
				plt.legend(fontsize=14, title_fontsize=14)
				plt.grid()
				groupedData = groupByChooseTest(data[(data[standardVar] == standardLevel) & (data[sensoryVar] == audioNoiseLevel) & (data[conflictVar] == conflictLevel)])
				bin_and_plot(groupedData, bin_method='cut', bins=10, plot=True, color=color)
				plt.text(0.05, 0.8, f"$\sigma_a$: {sigma_av_a:.2f}, $\sigma_v$: {sigma_av_v:.2f},", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
				plt.tight_layout()
				plt.grid(True)
				print(f"Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, Lambda: {lambda_:.3f}, Sigma_a: {sigma_av_a:.3f}, Sigma_v: {sigma_av_v:.3f}, p_c: {p_c:.3f}")
			plt.text(0.15, 0.9, f"P(C=1): {p_c:.2f}", fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
	plt.show()


delta_dur_values = data["deltaDurS"].values
conflict_values = data["conflictDur"].values
snr_values = data["audNoise"].values
best_params = fitted_params  # Use the best fitted parameters from the previous fitting

posterior_values = []
for delta, conflict, snr in zip(delta_dur_values, conflict_values, snr_values):
	λ, σa, σv, pc = getParamsCausal(best_params, conflict, snr)
	S_std = 0.5
	S_test = S_std + delta
	S_v = S_std + conflict

	m_a = S_std
	m_v = S_v

	L1 = likelihood_C1(m_a, m_v, σa, σv)
	L2 = likelihood_C2(m_a, m_v, S_std, S_v, σa, σv)
	posterior = posterior_C1(L1, L2, pc)
	posterior_values.append(posterior)




# Plot posterior probability vs conflict for noisy SNR (e.g., SNR == 0.1)
noisy_snr_value = 1.2  # Adjust if your noisy SNR value is different
mask_noisy = np.isclose(snr_values, noisy_snr_value)
conflicts_noisy = conflict_values[mask_noisy]
posteriors_noisy = np.array(posterior_values)[mask_noisy]

plt.figure(figsize=(8, 5))
plt.scatter(conflicts_noisy * 1000, posteriors_noisy, alpha=0.6, label='Posterior P(C=1) (Noisy SNR)')
plt.xlabel('Conflict (ms)')
plt.ylabel('Posterior Probability of Common Cause')
plt.title('Posterior P(C=1) vs Conflict (Noisy SNR)')
plt.axhline(y=0.5, color='gray', linestyle='--', label='P(C=1) = 0.5')
plt.legend()
plt.grid()
plt.show()