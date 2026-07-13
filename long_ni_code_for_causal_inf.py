# For Long Ni
# === Lognormal (Log-Space Gaussian Measurement) Model === #

def posterior_C1(self,m_a,m_v,sigma_a,
                                sigma_v, p_c,
                                t_min,t_max):

    #likelihoods
    L1 = self.L_C1(m_a, m_v, sigma_a, sigma_v, t_min, t_max)  
    L2 = self.L_C2(m_a, m_v, sigma_a, sigma_v, t_min, t_max)  # Fixed: consistent parameter order
    
    # posterior with numerical stability
    denominator = L1*p_c + L2*(1-p_c)
    
    lratio = L1 / (L2+1e-20)
    postC1=lratio * p_c / (lratio * p_c + (1 - p_c))
    return postC1


def truncated_normal_mean(self, mu, sigma, t_min, t_max):
    """Posterior mean of the hidden duration under a Gaussian likelihood
    centered at `mu` (scale `sigma`) and a boxcar prior on [t_min, t_max].

    `mu` may be an array (Monte Carlo measurements); `sigma`, `t_min`,
    `t_max` are scalars in the same space as `mu` (log space for the
    log-normal / switching models, which pass log-transformed bounds)."""
    mu = np.asarray(mu, dtype=float)
    alpha = (t_min - mu) / sigma 
    beta = (t_max - mu) / sigma
    Z = norm.cdf(beta) - norm.cdf(alpha)
    num = norm.pdf(alpha) - norm.pdf(beta)
    # When the measurement is many sigma outside the box, the prior mass Z
    # underflows to exactly 0 (0/0); the posterior then collapses onto the
    # nearest boundary, which np.clip(mu, ...) recovers. The exact formula is
    # used everywhere Z is representable (i.e. every draw that occurs with
    # non-negligible probability).
    safe = Z > 1e-300
    est = mu + sigma * num / np.where(safe, Z, 1.0)
    est = np.where(safe & np.isfinite(est), est, np.clip(mu, t_min, t_max))
    # The true truncated mean always lies within the box.
    return np.clip(est, t_min, t_max)

def fusionAV_boxcar(self, m_a, m_v, sigma_a, sigma_v, t_min, t_max):
    """Common-cause (C=1) estimate under the boxcar prior: the precision-
    weighted fusion posterior N(mu_c, sigma_c^2) passed through the
    truncated-normal correction on [t_min, t_max]."""
    J_a = 1.0 / sigma_a**2
    J_v = 1.0 / sigma_v**2
    mu_c = (J_a * m_a + J_v * m_v) / (J_a + J_v)
    sigma_c = np.sqrt(1.0 / (J_a + J_v))
    return self.truncated_normal_mean(mu_c, sigma_c, t_min, t_max)


# Vectorized causal inference functions
def p_single(self,m,sigma,t_min,t_max):
    """p(m | C=2)     and Gaussian measurement noise N(m; y, sigma^2). 
    and Gaussian measurement noise N(m; y, sigma^2)."""
    hi_cdf= norm.cdf((t_max - m) /sigma)
    lo_cdf=norm.cdf((t_min-m)/sigma)
    return (hi_cdf-lo_cdf)/(t_max-t_min)
    
def L_C2(self, m_a,m_v,sigma_a,sigma_v,t_min,t_max):
    """ Likelihood of separate sources: product of two marginal likelihoods 
    two integral over two hidden duration y_a y_v"""

    return self.p_single(m_a,sigma_a,t_min,t_max) * self.p_single(m_v,sigma_v,t_min,t_max)


def L_C1(self,m_a,m_v,sigma_a,sigma_v,t_min,t_max):

    sigma_c_sq = (sigma_a**2 * sigma_v**2) / (sigma_a**2 + sigma_v**2)
    sigma_c = np.sqrt(sigma_c_sq)
    mu_c = (m_a / sigma_a**2 + m_v / sigma_v**2) / (1 / sigma_a**2 + 1 / sigma_v**2)

    hi_cdf = norm.cdf((t_max-mu_c)/sigma_c)
    lo_cdf = norm.cdf((t_min-mu_c)/sigma_c)
    
    expo = np.exp(-(m_a-m_v)**2/(2*(sigma_a**2+sigma_v**2)))
    
    prior = 1/(t_max-t_min)
    
    return prior * sigma_c/np.sqrt(sigma_a**2 * sigma_v**2) * (hi_cdf-lo_cdf) * expo

#print("Using lognormal distribution for measurements") Use Causal inference lognormal
if self.modelName == "lognorm":
    nSimul = self.nSimul
    S_a_s, S_a_t, S_v_s, S_v_t = trueStims
    m_a_s = np.random.normal(loc=np.log(S_a_s), scale=sigma_av_a, size=nSimul)
    m_v_s = np.random.normal(loc=np.log(S_v_s), scale=sigma_av_v, size=nSimul)
    m_a_t = np.random.normal(loc=np.log(S_a_t), scale=sigma_av_a, size=nSimul)
    m_v_t = np.random.normal(loc=np.log(S_v_t), scale=sigma_av_v, size=nSimul)
    est_standard = self.causalInference_vectorized(m_a_s, m_v_s, sigma_av_a, sigma_av_v, p_c, np.log(t_min), np.log(t_max))
    est_test = self.causalInference_vectorized(m_a_t, m_v_t, sigma_av_a, sigma_av_v, p_c, np.log(t_min), np.log(t_max))

elif self.modelName ==  "fusionOnlyLogNorm":
    nSimul = self.nSimul
    S_a_s, S_a_t, S_v_s, S_v_t = trueStims
    # Generate measurements in LOG space
    m_a_s = np.random.normal(loc=np.log(S_a_s), scale=sigma_av_a, size=nSimul)
    m_v_s = np.random.normal(loc=np.log(S_v_s), scale=sigma_av_v, size=nSimul)
    m_a_t = np.random.normal(loc=np.log(S_a_t), scale=sigma_av_a, size=nSimul)
    m_v_t = np.random.normal(loc=np.log(S_v_t), scale=sigma_av_v, size=nSimul)
    # Boxcar-prior fusion estimate in LOG space (bounds are log-transformed), then exp back
    est_standard = self.fusionAV_boxcar(m_a_s, m_v_s, sigma_av_a, sigma_av_v, np.log(t_min), np.log(t_max))
    est_standard=np.exp(est_standard)  # Convert back to linear space
    est_test = self.fusionAV_boxcar(m_a_t, m_v_t, sigma_av_a, sigma_av_v, np.log(t_min), np.log(t_max))
    est_test=np.exp(est_test)  # Convert back to linear space
