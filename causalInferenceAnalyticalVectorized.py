def flexSigma(sigma,dur):
	return sigma*np.log((1 + dur))
def nLL_causal_inference_fully_vectorized_flexSigma(params, rawData, flexSigma=True):
	"""Fully vectorized version - even faster by avoiding parameter loops."""
	# Extract data arrays
	snr_values = rawData["audNoise"].values
	conflict_values = rawData["conflictDur"].values
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
	
    # flexible sigma
    

	# Vectorized parameter assignment
	sigma_av_a_arr[snr_01_mask] = params[1]
	sigma_av_v_arr[snr_01_mask] = params[2]
	p_c_arr[snr_01_mask] = params[3]
	
	sigma_av_a_arr[snr_12_mask] = params[4]
	sigma_av_v_arr[snr_12_mask] = params[5]
	p_c_arr[snr_12_mask] = params[6]
	
	# Vectorized causal inference computation
	S_a_s = 0.5
	S_a_t = S_a_s + delta_dur_values
	
	# Standard estimates (vectorized)
	S_v_s = rawData["unbiasedVisualStandardDur"].values  # Assuming this is the unbiased visual standard

	
	m_a_s = S_a_s # Measurements for standard
	m_v_s = S_v_s # Measurements for standard

	# Common cause likelihood for standard
	var_sum_s = sigma_av_a_arr**2 + sigma_av_v_arr**2
	likelihood_c1_s = (1 / np.sqrt(2 * np.pi * var_sum_s)) * np.exp(-(m_a_s - m_v_s)**2 / (2 * var_sum_s))
	
	# Independent causes likelihood for standard
	likelihood_c2_s = norm.pdf(m_a_s, loc=S_a_s, scale=sigma_av_a_arr) * norm.pdf(m_v_s, loc=S_v_s, scale=flexSigma(sigma_av_v_arr, conflict_values))
	
	# Posterior probability of common cause for standard
	posterior_c1_s = (likelihood_c1_s * p_c_arr) / (likelihood_c1_s * p_c_arr + likelihood_c2_s * (1 - p_c_arr))
	
	# Fusion estimate for standard
	J_AV_A = 1 / flexSigma(sigma_av_a_arr,S_a_s)**2
	J_AV_V = 1 / flexSigma(sigma_av_a_arr,S_a_s)**2
	w_a = J_AV_A / (J_AV_A + J_AV_V)
	w_v = 1 - w_a
	fused_S_av_s = w_a * S_a_s + w_v * S_v_s
	
	# Final estimate for standard (model averaging)
	est_standard = posterior_c1_s * fused_S_av_s + (1 - posterior_c1_s) * S_a_s
	
	# Test estimates (fusion only, no conflict)
	est_test = w_a * S_a_t + (1 - w_a) * S_a_t  # Simplifies to S_a_t

	# Decision noise
	var_fusion = 1 / (1/sigma_av_a_arr**2 + 1/sigma_av_v_arr**2)
	var_segregated = sigma_av_a_arr**2
	var_estimate = p_c_arr * var_fusion + (1 - p_c_arr) * var_segregated
	sigma_decision = np.sqrt(2 * var_estimate)
	
	# Decision
	deltaEstimates = est_test - est_standard
	P = lambda_arr/2 + (1 - lambda_arr) * norm.cdf(deltaEstimates, loc=0, scale=flexSigma(sigma_decision, conflict_values))
	
	# Clip probabilities to avoid log(0)
	epsilon = 1e-9
	P = np.clip(P, epsilon, 1 - epsilon)
	# Vectorized log-likelihood computation
	ll = np.sum(responses * np.log(P) + (1 - responses) * np.log(1 - P))
	return -ll

