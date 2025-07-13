from fitMain import *
import loadData
# Steps of code
""" 1- Load data
2- Define initial Guesses () or do not define it just use nStart and give single init guesses within that
3-  fit
4- PLot the fitted psychometric
"""

# Steps for the monte carlo model
# get parameters func acording to the SNR(1+3x2=7 parameters)
def getParamsCausal(params, conflict, SNR):
	"""Extract causal inference parameters for a specific condition (conflict, noise)."""

	lambda_=params[0]
	if np.isclose(SNR, 0.1):
		sigma_av_a=params[1]
		sigma_av_v=params[2]
		p_c=params[3]
	elif np.isclose(SNR,1.2):
		sigma_av_a=params[4]
		sigma_av_v=params[5]
		p_c=params[6]

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
def fusionAV(m_a,m_v,sigma_av_a, sigma_av_v):
	J_AV_A = 1 / sigma_av_a**2
	J_AV_V = 1 / sigma_av_v**2
	w_a = J_AV_A / (J_AV_A + J_AV_V)
	w_v = 1 - w_a
	fused_S_av = w_a * m_a + w_v * m_v
	sigma_S_AV_hat = np.sqrt(1 / (J_AV_A + J_AV_V))
	return fused_S_av, sigma_S_AV_hat

# Likelihood under common cause
def likelihood_C1(m_a, m_v, sigma_av_a, sigma_av_v):
	var_sum = sigma_av_a**2 + sigma_av_v**2
	return (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))

# Likelihood under independent causes
def likelihood_C2(m_a,m_v,S_a, S_v, sigma_av_a, sigma_av_v):
	return norm.pdf(m_a, loc=S_a, scale=sigma_av_a) * norm.pdf(m_v, loc=S_v, scale=sigma_av_v)

# Posterior of common cause
def posterior_C1(likelihood_c1, likelihood_c2, p_c):
	return (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))


def causalInference(S_a,S_v,m_a,m_v,sigma_av_a, sigma_av_v,  p_c):

	# LIKELIHOODS fo CC
	likelihood_c1 = likelihood_C1(m_a, m_v, sigma_av_a, sigma_av_v)
	likelihood_c2 = likelihood_C2(m_a, m_v,S_a,S_v, sigma_av_a, sigma_av_v)
	# POSTERIOR P Common
	posterior_c1 = posterior_C1(likelihood_c1, likelihood_c2, p_c)
	posterior_c2 = 1 - posterior_c1
	# Estimates
	# \hat S_{av,c=1} Fusion
	fused_S_av, sigma_S_AV_hat = fusionAV(m_a,m_v,sigma_av_a, sigma_av_v)
	# C=2 no CC
	hat_S_AV_A_No_CC=S_a
	# Model avaraging
	hat_S_AV_A_final = posterior_c1 * fused_S_av + posterior_c2 * hat_S_AV_A_No_CC

	return hat_S_AV_A_final

def causalInfDecision(trueStims,measurements,sigma_av_a, sigma_av_v,p_c):
	S_a_s,S_a_t,S_v_s,S_v_t, = trueStims
	if measurements[0] is None:
		m_a_s,m_a_t,m_v_s,m_v_t = S_a_s, S_a_t, S_v_s, S_v_t
	else:
		m_a_s,m_a_t,m_v_s,m_v_t = measurements

	"""Compute psychometric using causal inference model"""
	# take estimates
	standardShat= causalInference(S_a_s,S_v_s,m_a_s,m_v_s,sigma_av_a, sigma_av_v,  p_c) 
	testShat= fusionAV(m_a_t,m_v_t, sigma_av_a, sigma_av_v)[0]  # We compute the fused estimate for test duration because test duration do

	# decision is a binary choice a boolean just based on the difference

	decision= testShat-standardShat>0
	
	return decision

def probTestLonger(trueStims,sigma_av_a, sigma_av_v,p_c):
	# its going to be fully vectorized monte carlo approximation
	nSimul=100
	S_a_s,S_a_t,S_v_s,S_v_t, = trueStims
	#maybe not necessary but deltaDur is the difference between the two durations
	conflict= S_v_s - S_a_s
	deltaDur = S_a_t - S_a_s

	p_c_arr=np.full(nSimul, p_c)

	m_a_s_arr=np.random.normal(S_a_s, sigma_av_a, nSimul)
	m_v_s_arr=np.random.normal(S_v_s, sigma_av_v, nSimul)
	m_a_t_arr=np.random.normal(S_a_t, sigma_av_a, nSimul)
	m_v_t_arr=np.random.normal(S_v_t, sigma_av_v, nSimul)
	measurementsArr = np.array([m_a_s_arr, m_a_t_arr, m_v_s_arr, m_v_t_arr])
	
	# S_v_s_arr = np.full(nSimul, S_v_s)
	# S_a_s_arr = np.full(nSimul, S_a_s)
	# S_v_t_arr = np.full(nSimul, S_v_t)
	# S_a_t_arr = np.full(nSimul, S_a_t)
	#stimArr = np.array([S_a_s_arr, S_a_t_arr, S_v_s_arr, S_v_t_arr])
	stimArr= np.array([S_a_s, S_a_t, S_v_s, S_v_t])

	sigma_av_v_arr = np.full(nSimul, sigma_av_v)
	sigma_av_a_arr = np.full(nSimul, sigma_av_a)

	decisionArr = causalInfDecision(stimArr, measurementsArr, sigma_av_a, sigma_av_v, p_c)
	# dimensions of decisionArr is (nSimul,)
	#print(f"Decision array shape: {decisionArr.shape}")

	# Calculate the proportion of decisions where the test duration is longer than the standard duration
	#decisionArr = decisionArr.astype(int)  # Convert boolean to int (True -> 1, False -> 0)
	# decision is avarage of monte carlo simulations
	return np.mean(decisionArr)

def nLLMonteCarloCausal(params,groupedData):
	ll=0
	lenData=len(groupedData)
	y=np.empty(lenData)
	choseTests=groupedData['num_of_chose_test'].values
	totalResp = groupedData['total_responses'].values

	for i in range(lenData):
		currSNR = groupedData["audNoise"][i]
		currConflict = groupedData["conflictDur"][i]
		currResp= groupedData['num_of_chose_test'][i]
		totalResponses = groupedData['total_responses'][i]
		currDeltaDur = groupedData["deltaDurS"][i]
		y[i]=currResp
		# Get the parameters for the current condition
		lambda_, sigma_av_a, sigma_av_v, p_c = getParamsCausal(params, currConflict, currSNR)

		# Get the true standard and test durations
		S_a_s = groupedData["standardDur"][i]
		S_v_s = groupedData["unbiasedVisualStandardDur"][i]
		S_a_t = groupedData["testDurS"][i]
		S_v_t = groupedData["unbiasedVisualTestDur"][i]
		trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
		# Calculate the probability of choosing the test duration being longer than the standard duration
		p_test_longer = probTestLonger(trueStims, sigma_av_a, sigma_av_v, p_c)
		# Calculate the likelihood for the current condition

		# add epsilon to avoid log(0)
		epsilon=1e-9
		P= np.clip(p_test_longer, epsilon, 1 - epsilon)
		ll += np.log(P) * currResp + np.log(1 - P) * (totalResponses - currResp)
	# Normalize the log-likelihood by the number of trials
	ll /= lenData
	# Return the negative log-likelihood
	return -ll

class TqdmMinimizeCallback:
	def __init__(self, total=100, show_progress=1):
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
def fitCausalInferenceMonteCarlo(data, nStart=10):

	bounds=[(0,0.25),#lambda_
		 (0.01,1.5), #sigma_av_a_1
		 (0.01,1.5), #sigma_av_v_1
		 (0,1), #p_c_1
		 (0.01,1.5), #sigma_av_a_2
		 (0.01,1.5), #sigma_av_v_2
		 (0,1)] #p_c_2
	
	callback = TqdmMinimizeCallback(total=100, show_progress=1)
	
	result=minimize(nLLMonteCarloCausal,
					x0=np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.3]),
					args=(data,),
					method='L-BFGS-B',
					bounds=bounds,
					callback=callback
	)
	callback.close()
	return result.x



def groupByChooseTest(x,groupArgs):
	#print(f"Grouping by {intensityVariable}, {sensoryVar}, {standardVar}, {conflictVar}")
	grouped = x.groupby(groupArgs).agg(
		num_of_chose_test=('chose_test', 'sum'),
		total_responses=('responses', 'count'),
		
		#num_of_chose_standard=('chose_standard', 'sum'),
	).reset_index()
	grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']

	return grouped

import time
if __name__ == "__main__":
	# Example usage
	data, dataName = loadData.loadData("dt_all.csv")


	intensityVariable="deltaDurS"

	sensoryVar="audNoise"
	standardVar="standardDur"
	conflictVar="conflictDur"

	# true stim vars to group by
	visualStandardVar="unbiasedVisualStandardDur"
	visualTestVar="unbiasedVisualTestDur"
	audioStandardVar="standardDur"
	audioTestVar="testDurS"

	groupedData = groupByChooseTest(data, [intensityVariable, sensoryVar, standardVar, conflictVar,visualStandardVar, visualTestVar, audioTestVar])

	timeStart = time.time()
	print(f"\nFitting Causal Inference Model for {dataName} with {len(groupedData)} unique conditions")
	fitted_params = fitCausalInferenceMonteCarlo(groupedData)
	print(f"\nFitted parameters for {dataName}: {fitted_params}")
	print(f"Time taken to fit: {time.time() - timeStart:.2f} seconds")