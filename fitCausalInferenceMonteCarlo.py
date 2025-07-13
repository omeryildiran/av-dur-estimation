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
	return fused_S_av

def fusionAV_vectorized(m_a, m_v, sigma_av_a, sigma_av_v):
	J_AV_A = 1 / sigma_av_a**2
	J_AV_V = 1 / sigma_av_v**2
	w_a = J_AV_A / (J_AV_A + J_AV_V)
	w_v = 1 - w_a
	fused_S_av = w_a * m_a + w_v * m_v
	return fused_S_av  # omit returning sigma unless needed


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


def causalInference(S_a, S_v, m_a, m_v, sigma_av_a, sigma_av_v, p_c):
	"""Vectorized causal inference computation"""
	
	# LIKELIHOODS for Common Cause
	likelihood_c1 = likelihood_C1(m_a, m_v, sigma_av_a, sigma_av_v)
	likelihood_c2 = likelihood_C2(m_a, m_v, S_a, S_v, sigma_av_a, sigma_av_v)
	
	# POSTERIOR P(Common Cause)
	posterior_c1 = posterior_C1(likelihood_c1, likelihood_c2, p_c)
	posterior_c2 = 1 - posterior_c1
	
	# Estimates
	# Fusion estimate (common cause)
	fused_S_av = fusionAV(m_a, m_v, sigma_av_a, sigma_av_v)
	
	# Independent causes estimate (use auditory measurement for auditory estimate)
	hat_S_AV_A_No_CC = m_a  # Use measurement, not true stimulus
	
	# Model averaging
	hat_S_AV_A_final = posterior_c1 * fused_S_av + posterior_c2 * hat_S_AV_A_No_CC

	return hat_S_AV_A_final

def causalInference_vectorized(S_a, S_v, m_a, m_v, sigma_av_a, sigma_av_v, p_c):
	var_sum = sigma_av_a**2 + sigma_av_v**2
	likelihood_c1 = (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-(m_a - m_v)**2 / (2 * var_sum))
	
	likelihood_c2 = norm.pdf(m_a, loc=S_a, scale=sigma_av_a) * norm.pdf(m_v, loc=S_v, scale=sigma_av_v)
	
	posterior_c1 = (likelihood_c1 * p_c) / (likelihood_c1 * p_c + likelihood_c2 * (1 - p_c))
	
	fused_S_av = fusionAV_vectorized(m_a, m_v, sigma_av_a, sigma_av_v)
	final_estimate = posterior_c1 * fused_S_av + (1 - posterior_c1) * S_a
	
	return final_estimate


def causalInfDecision(trueStims, measurements, sigma_av_a, sigma_av_v, p_c):
	S_a_s, S_a_t, S_v_s, S_v_t = trueStims
	
	if measurements[0] is None:
		m_a_s, m_a_t, m_v_s, m_v_t = S_a_s, S_a_t, S_v_s, S_v_t
	else:
		m_a_s, m_a_t, m_v_s, m_v_t = measurements

	"""Compute psychometric using causal inference model - fully vectorized"""
	# Apply causal inference to both standard and test intervals consistently
	standardShat = causalInference_vectorized(S_a_s, S_v_s, m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c) 
	testShat= fusionAV_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v)  # Use fusion for test interval

	# Decision is based on the difference between test and standard estimates
	decision = (testShat - standardShat) > 0
	
	return decision

def probTestLonger(trueStims, sigma_av_a, sigma_av_v, p_c, lambda_=0):
	"""Monte Carlo approximation of probability test duration judged longer than standard"""
	nSimul = 1000  # Increase for better approximation
	#print(f"\nRunning Monte Carlo simulation with {nSimul} samples...\n")
	S_a_s, S_a_t, S_v_s, S_v_t = trueStims
	
	# Generate measurements for all Monte Carlo samples
	m_a_s_arr = np.random.normal(S_a_s, sigma_av_a, nSimul)
	m_v_s_arr = np.random.normal(S_v_s, sigma_av_v, nSimul)
	m_a_t_arr = np.random.normal(S_a_t, sigma_av_a, nSimul)
	m_v_t_arr = np.random.normal(S_v_t, sigma_av_v, nSimul)
	
	measurementsArr = np.array([m_a_s_arr, m_a_t_arr, m_v_s_arr, m_v_t_arr])
	stimArr = np.array([S_a_s, S_a_t, S_v_s, S_v_t])

	# Get vectorized decisions
	decisionArr = causalInfDecision(stimArr, measurementsArr, sigma_av_a, sigma_av_v, p_c)
	
	# Calculate base probability (proportion choosing test longer)
	p_base = np.mean(decisionArr)
	
	# Apply lapse rate: p_final = (1-lambda) * p_base + lambda/2
	p_final = (1 - lambda_) * p_base + lambda_ / 2
	
	return p_final

from scipy.ndimage import gaussian_filter1d

def smooth_psychometric_curve(x_vals, conflict, sigma_av_a, sigma_av_v, p_c, lambda_, n_samples=2000, sigma_smooth=2):
	"""Returns a smoothed psychometric curve using Monte Carlo and Gaussian filtering"""
	S_a_s = 0.5
	S_v_s = S_a_s + conflict
	
	y_vals = np.zeros_like(x_vals)
	
	for i, delta in enumerate(x_vals):
		S_a_t = S_a_s + delta
		S_v_t = S_a_t  # no conflict
		true_stims = (S_a_s, S_a_t, S_v_s, S_v_t)
		y_vals[i] = probTestLonger(true_stims, sigma_av_a, sigma_av_v, p_c, lambda_)
	
	y_smoothed = gaussian_filter1d(y_vals, sigma=0.5)
	return y_smoothed




def probTestLonger_vectorized_mc(trueStims, sigma_av_a, sigma_av_v, p_c,lambda_ ):
	nSimul=1000
	S_a_s, S_a_t, S_v_s, S_v_t = trueStims

	# Sample Monte Carlo measurements
	m_a_s = np.random.normal(S_a_s, sigma_av_a, nSimul)
	m_v_s = np.random.normal(S_v_s, sigma_av_v, nSimul)
	m_a_t = np.random.normal(S_a_t, sigma_av_a, nSimul)
	m_v_t = np.random.normal(S_v_t, sigma_av_v, nSimul)

	# Estimate using causal inference
	est_standard = causalInference_vectorized(S_a_s, S_v_s, m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c)
	est_test = fusionAV_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v)  # C=1 assumed for test

	# Return probability test > standard
	p_base= np.mean(est_test > est_standard)
	# Apply lapse rate
	p_final = (1 - lambda_) * p_base + lambda_ / 2
	return p_final



def nLLMonteCarloCausal(params, groupedData):
	"""Negative log-likelihood for causal inference model"""
	ll = 0
	lenData = len(groupedData)
	
	# Check for invalid parameters
	if np.any(np.isnan(params)) or np.any(np.isinf(params)):
		return 1e10
	
	for i in range(lenData):
		currSNR = groupedData["audNoise"].iloc[i]
		currConflict = groupedData["conflictDur"].iloc[i]
		currResp = groupedData['num_of_chose_test'].iloc[i]
		totalResponses = groupedData['total_responses'].iloc[i]
		
		
		# Get the parameters for the current condition
		lambda_, sigma_av_a, sigma_av_v, p_c = getParamsCausal(params, currConflict, currSNR)

		# Get the true standard and test durations
		S_a_s = groupedData["standardDur"].iloc[i]
		S_v_s = groupedData["unbiasedVisualStandardDur"].iloc[i]
		S_a_t = groupedData["testDurS"].iloc[i]
		S_v_t = groupedData["unbiasedVisualTestDur"].iloc[i]
		trueStims = (S_a_s, S_a_t, S_v_s, S_v_t)
		
		# Calculate the probability of choosing the test duration being longer than the standard duration
		try:
			p_test_longer = probTestLonger_vectorized_mc(trueStims, sigma_av_a, sigma_av_v, p_c, lambda_)
		except:
			return 1e10
			
		# Calculate the likelihood for the current condition
		# add epsilon to avoid log(0)
		epsilon = 1e-10
		P = np.clip(p_test_longer, epsilon, 1 - epsilon)
		ll += np.log(P) * currResp + np.log(1 - P) * (totalResponses - currResp)
		
	# Check for invalid likelihood
	if np.isnan(ll) or np.isinf(ll):
		return 1e10
		
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
	"""Fit causal inference model with multiple random starts"""
	
	bounds = [(0, 0.25),      # lambda_ (lapse rate)
			  (0.1, 1.5),   # sigma_av_a_1 (SNR 0.1)
			  (0.1, 1.5),   # sigma_av_v_1 (SNR 0.1)
			  (0.05, 0.99),        # p_c_1 (SNR 0.1)
			  (0.1, 1.5),   # sigma_av_a_2 (SNR 1.2)  
			  (0.1, 1.5),   # sigma_av_v_2 (SNR 1.2)
			  (0.05, 0.99)]        # p_c_2 (SNR 1.2)
	
	best_result = None
	best_ll = np.inf
	
	print(f"Starting {nStart} optimization attempts...")
	
	# Loop for multiple random starts
	# Each attempt will start with a random initialization within the bounds
	#attempts_iter = tqdm(range(nStart), desc="Random Starts", leave=True)
	for attempt in tqdm(range(nStart), desc="Optimization Attempts"):
		#print(f"\nAttempt {attempt + 1}/{nStart}")
		
		# Random initialization within bounds
		x0 = []
		for i, (lower, upper) in enumerate(bounds):
			if i in [3, 6]:  # p_c_1 and p_c_2
				x0.append(np.random.uniform(0.1, 0.98))
			elif i in [1, 2, 4, 5]:  # sigma parameters
				x0.append(np.random.uniform(0.1, 1.4))
			else:  # lambda
				x0.append(np.random.uniform(0.01, 0.24))

		x0 = np.array(x0)
		if attempt == 0:
			print(f"Initial guess: {x0}")
		
		callback = TqdmMinimizeCallback(total=100, show_progress=0)
		
		try:
			result = minimize(nLLMonteCarloCausal,
							x0=x0,
							args=(data,),
							method='Powell',
							bounds=bounds,
							callback=callback,
							)
			
			callback.close()
			if attempt == 0:
				print(f"\nresult fitted params is{result.x}\n")
			
			#print(f"Attempt {attempt + 1} - Final LL: {result.fun:.6f}")
			#print(f"Success: {result.success}, Message: {result.message}")
			
			if result.fun < best_ll and result.success:
				best_ll = result.fun
				best_result = result
				#print(f"New best result found!")
				
		except Exception as e:
			callback.close()
			print(f"Attempt {attempt + 1} failed: {str(e)}")
			continue



	if best_result is None:
		raise RuntimeError("All optimization attempts failed!")

	print(f"\nBest result from {nStart} attempts:")
	print(f"Final parameters: {best_result.x}")
	print(f"Final log-likelihood: {best_result.fun:.6f}")
	
	return best_result.x



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
from tqdm import tqdm
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
	fittedParams = fitCausalInferenceMonteCarlo(groupedData)
	print(f"\nFitted parameters for {dataName}: {fittedParams}")
	print(f"Time taken to fit: {time.time() - timeStart:.2f} seconds")

	uniqueStandard= groupedData[standardVar].unique()
	uniqueSensory = groupedData[sensoryVar].unique()
	uniqueConflict = sorted(groupedData[conflictVar].unique())
	
		
	# Plot results (existing plotting code continues from here...)
	pltTitle = dataName + " Causal Inference Model Fit"
	plt.figure(figsize=(16, 6))
	for i, standardLevel in enumerate(uniqueStandard):
		for j, audioNoiseLevel in enumerate(sorted(uniqueSensory)):

			for k, conflictLevel in enumerate(uniqueConflict):
				plt.subplot(1, 2, j+1)
				lambda_, sigma_av_a, sigma_av_v, p_c = getParamsCausal(fittedParams, conflictLevel, audioNoiseLevel)
				x = np.linspace(-0.5, 0.5, 100)
				S_a_s=0.5
				#c_arr=np.full_like(x, conflictLevel)
				S_v_s= S_a_s+conflictLevel
				y= np.zeros_like(x)
				for i in range(len(x)):
					y[i] = probTestLonger([S_a_s,S_a_s+x[i],S_v_s,S_a_s+x[i]], sigma_av_a, sigma_av_v, p_c, lambda_)

				color = sns.color_palette("viridis", as_cmap=True)(k / len(uniqueConflict))
				plt.plot(x, y, color=color, label=f"c: {int(conflictLevel*1000)}, $\sigma_a$: {sigma_av_a:.2f}, $\sigma_v$: {sigma_av_v:.2f}", linewidth=4)
				
				plt.axvline(x=0, color='gray', linestyle='--')
				plt.axhline(y=0.5, color='gray', linestyle='--')
				plt.xlabel(f"({intensityVariable}) Test(stair-a)-Standard(a) Duration Difference Ratio(%)")
				plt.ylabel("P(chose test)")
				plt.title(f"{pltTitle} AV,A Duration Comp. Noise: {audioNoiseLevel}", fontsize=16)
				plt.legend(fontsize=14, title_fontsize=14)
				plt.grid()

				groupedData = groupByChooseTest(data[(data[standardVar] == standardLevel) & (data[sensoryVar] == audioNoiseLevel) & (data[conflictVar] == conflictLevel)], [intensityVariable, sensoryVar, standardVar, conflictVar,visualStandardVar, visualTestVar, audioTestVar])
				bin_and_plot(groupedData, bin_method='cut', bins=10, plot=True, color=color)
				plt.text(0.05, 0.8, f"$\sigma_a$: {sigma_av_a:.2f}, $\sigma_v$: {sigma_av_v:.2f},", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
				plt.tight_layout()
				plt.grid(True)
				print(f"Noise: {audioNoiseLevel}, Conflict: {conflictLevel}, Lambda: {lambda_:.3f}, Sigma_a: {sigma_av_a:.3f}, Sigma_v: {sigma_av_v:.3f}, p_c: {p_c:.3f}")
			plt.text(0.15, 0.9, f"P(C=1): {p_c:.2f}", fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)
	plt.show()



def plot_posterior_vs_conflict(data,fittedParams,snr_list=[1.2, 0.1]):

	delta_dur_values = data["deltaDurS"].values
	conflict_values = data["conflictDur"].values
	snr_values = data["audNoise"].values
	best_params = fittedParams  # Use the best fitted parameters from the previous fitting

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


	"""
	Plot posterior probability vs conflict for given SNR values.
	snr_list: list of SNR values to plot (default: [1.2, 0.1])
	"""
	plt.figure(figsize=(8, 5))
	for idx, noisy_snr_value in enumerate(snr_list):
		mask_noisy = np.isclose(snr_values, noisy_snr_value)
		conflicts_noisy = conflict_values[mask_noisy]
		posteriors_noisy = np.array(posterior_values)[mask_noisy]
		plt.subplot(1, 2, idx + 1)
		plt.scatter(conflicts_noisy * 1000, posteriors_noisy, alpha=0.6, label=f'Posterior P(C=1) (SNR={noisy_snr_value})')
		plt.xlabel('Conflict (ms)')
		plt.ylabel('Posterior Probability of Common Cause')
		plt.title(f'Posterior P(C=1) vs Conflict (SNR={noisy_snr_value})')
		plt.axhline(y=getParamsCausal(fittedParams, conflicts_noisy, noisy_snr_value)[3], color='gray', linestyle='--', label=f'P(C=1)={getParamsCausal(fittedParams, conflicts_noisy, noisy_snr_value)[3]:.2f}')
		plt.legend()
		plt.ylim(0, 1)
		plt.grid()
	plt.tight_layout()
	plt.show()
