import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
# Psychometric function
def psychometric_function(intensities, lapse_rate, mu, sigma):
    cdf = norm.cdf(intensities, loc=mu, scale=sigma)
    return lapse_rate * 0.5 + (1 - lapse_rate) * cdf

# Negative log-likelihood
def negative_log_likelihood(params, delta_dur, chose_test, total_responses):
    lambda_, mu, sigma = params
    p = psychometric_function(delta_dur, lambda_, mu, sigma)
    epsilon = 1e-9
    p = np.clip(p, epsilon, 1 - epsilon)
    log_likelihood = np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
    return -log_likelihood

# Fit psychometric function
def fit_psychometric_function(data_grouped):
    initial_guess = [0.001, 0, 0.1]  # Initial guess for [lambda, mu, sigma]
    bounds = [(0.01, 0.2), (-0.5, 0.5), (0.01, 1)]  # Reasonable bounds
    result = minimize(
        negative_log_likelihood, initial_guess, 
        args=(data_grouped['delta_dur_adjusted'], data_grouped['num_of_chose_test'], data_aggregated_data['total_responses']),
        bounds=bounds
    )
    return result.x

# Load data
file_path = 'data/'+'_auditory_dur_estimate_2024-12-19_01h06.07.475.csv'
data = pd.read_csv(file_path)
data[:2]

# Step 1: Prepare the data
data['chose_test'] = (data['response'] == data['test_order']).astype(int)

# Step 2: Group by delta_dur_adjusted and rise_dur
grouped = data.groupby(['delta_dur_adjusted', 'rise_dur']).agg(
    total_responses=('response', 'count'),
    chose_test=('chose_test', 'sum')
).reset_index()

# Step 3: Calculate the proportion of "choose test" responses
grouped['p_choose_test'] = grouped['chose_test'] / grouped['total_responses']
grouped[:2]



