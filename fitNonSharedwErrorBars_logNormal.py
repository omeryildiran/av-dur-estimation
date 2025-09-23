# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse

# function for loading data
def loadData(dataName):
    intensityVariable="testDurS"  # Use raw test duration for log-normal model

    sensoryVar="audNoise"
    standardVar="standardDur"
    conflictVar="conflictDur"
    testDurVar="testDurS"



    data = pd.read_csv("data/"+dataName)
    # ignore firts 3 rows
    data= data[data['audNoise'] != 0]
    data=data[data['standardDur'] != 0]
    data["testDurMs"]= data["testDurS"]*1000
    data["standardDurMs"]= data["standardDur"]*1000
    
    #round standardDur to 2 decimal places
    data = data.round({'standardDur': 2, 'audNoise': 2, 'conflictDur': 2, 'delta_dur_percents': 2})
    
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

intensityVariable="testDurS"  # Use raw test duration for log-normal model
fixedMu = False  # Set to True to fix mu to 0 (no bias)



def groupByChooseTest(x):
    grouped = x.groupby([intensityVariable, sensoryVar, standardVar,conflictVar,"testDurMs"]).agg(
        num_of_chose_test=('chose_test', 'sum'),
        total_responses=('responses', 'count'),
        num_of_chose_standard=('chose_standard', 'sum'),
    ).reset_index()
    grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']

    return grouped


def groupByChooseTestWithParticipants(data):
    """
    Group data by intensity, sensory, standard, conflict AND participant to get individual participant responses
    """
    # First group by participant and conditions to get individual participant psychometric data
    participant_grouped = data.groupby([intensityVariable, sensoryVar, standardVar, conflictVar,'testDurMs', 'participantID']).agg(
        num_of_chose_test=('chose_test', 'sum'),
        total_responses=('responses', 'count'),
        num_of_chose_standard=('chose_standard', 'sum'),
    ).reset_index()
    participant_grouped['p_choose_test'] = participant_grouped['num_of_chose_test'] / participant_grouped['total_responses']
    
    return participant_grouped

# Compute sigma from slope
def compute_sigma_from_slope(slope, lapse_rate=0.02):
    sigma = (1 - lapse_rate) / (np.sqrt(2 * np.pi) * slope)*np.exp(-0.5)
    return sigma

def bin_and_plot_with_error_bars(data, bin_method='cut', bins=10, bin_range=None, plot=True, color="blue",binVar="delta_dur_percents"):
    
    """
    Bin data and plot with error bars calculated across participants
    """
    # First get participant-level data
    participant_data = groupByChooseTestWithParticipants(data)
    
    if bin_method == 'cut':
        participant_data['bin'] = pd.cut(participant_data[binVar], bins=bins, labels=False, include_lowest=True, retbins=False)
    elif bin_method == 'manual':
        participant_data['bin'] = np.digitize(participant_data[binVar], bins=bin_range) - 1
    
    # Group by bin and calculate statistics across participants
    bin_summary = participant_data.groupby('bin').agg(
        x_mean=(binVar, 'mean'),
        y_mean=('p_choose_test', 'mean'),
        y_sem=('p_choose_test', lambda x: np.std(x) / np.sqrt(len(x)) if len(x) > 1 else 0),  # Standard error
        y_std=('p_choose_test', 'std'),
        n_participants=('participantID', 'nunique'),
        total_resp=('total_responses', 'sum')
    ).reset_index()
    
    if plot and len(bin_summary) > 0:
        # Plot with error bars
        plt.errorbar(bin_summary['x_mean'], bin_summary['y_mean'], 
                   yerr=bin_summary['y_sem'], 
                   fmt='o', color=color, capsize=5, capthick=2,
                   markersize=8, alpha=0.8, elinewidth=2)
        
        #Add text showing number of participants
        if not bin_summary['n_participants'].empty:
            n_participants = bin_summary['n_participants'].iloc[0]
            plt.text(0.02, 0.95, f'N = {n_participants}', 
                   transform=plt.gca().transAxes, fontsize=16,)
                   #bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    return bin_summary


def bin_and_plot(data, bin_method='cut', bins=10, bin_range=None, plot=True, color="blue", add_error_bars=True,binVar="delta_dur_percents"):
    if add_error_bars and 'participantID' in data.columns:
        return bin_and_plot_with_error_bars(data, bin_method, bins, bin_range, plot, color,binVar)
    else:
        # Original behavior for backwards compatibility
        # If we have raw data (with participantID), we need to group it first
        if 'participantID' in data.columns and 'p_choose_test' not in data.columns:
            data = groupByChooseTest(data)
        
        if bin_method == 'cut':
            data['bin'] = pd.cut(data[binVar], bins=bins, labels=False, include_lowest=True, retbins=False)
        elif bin_method == 'manual':
            data['bin'] = np.digitize(data[binVar], bins=bin_range) - 1
        
        grouped = data.groupby('bin').agg(
            x_mean=(binVar, 'mean'),
            y_mean=('p_choose_test', 'mean'),
            total_resp=('total_responses', 'sum')
        )

        if plot:
            plt.scatter(grouped['x_mean'], grouped['y_mean'], 
                      s=grouped['total_resp']/data['total_responses'].sum()*900, 
                      color=color, alpha=0.8)
        
        return grouped

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

from scipy.stats import norm, lognorm
from scipy.optimize import minimize

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
    if fixedMu:
        mu = 0
    
    # Ensure positive durations
    test_dur = np.maximum(test_dur, 1e-10)
    standard_dur = np.maximum(standard_dur, 1e-10)
    
    # Calculate the log ratio of durations
    log_ratio = np.log(test_dur / standard_dur)
    
    # Apply bias and normalize by discrimination parameter
    z_score = (log_ratio - mu) / sigma
    
    # Use standard normal CDF (this ensures monotonicity)
    from scipy.stats import norm
    p_longer = norm.cdf(z_score)
    
    # Apply lapse rate
    p = lambda_/2 + (1 - lambda_) * p_longer
    
    return p

def mu_to_pse_shift(mu, standard_dur):
    """
    Convert log-space bias (mu) to linear PSE shift in same units as standard_dur.
    
    The PSE is the test duration where P(chose test) = 0.5 (ignoring lapse rate).
    In log-normal model: PSE = standard_dur * exp(mu)
    PSE shift = PSE - standard_dur = standard_dur * (exp(mu) - 1)
    
    Parameters:
    -----------
    mu : float
        Bias parameter in log space
    standard_dur : float
        Standard duration in seconds
        
    Returns:
    --------
    float
        PSE shift in same units as standard_dur (positive = overestimation)
    """
    pse = standard_dur * np.exp(mu)
    pse_shift = pse - standard_dur
    return pse_shift

def pse_shift_to_mu(pse_shift, standard_dur):
    """
    Convert linear PSE shift to log-space bias (mu).
    
    Inverse of mu_to_pse_shift function.
    
    Parameters:
    -----------
    pse_shift : float
        PSE shift in same units as standard_dur
    standard_dur : float
        Standard duration in seconds
        
    Returns:
    --------
    float
        Bias parameter in log space
    """
    pse = standard_dur + pse_shift
    mu = np.log(pse / standard_dur)
    return mu

def calculate_pse_stats(mu, sigma, lambda_, standard_dur):
    """
    Calculate comprehensive PSE statistics.
    
    Parameters:
    -----------
    mu, sigma, lambda_ : float
        Model parameters
    standard_dur : float
        Standard duration in seconds
        
    Returns:
    --------
    dict
        Dictionary with PSE statistics
    """
    # Pure PSE (ignoring lapse rate)
    pse_pure = standard_dur * np.exp(mu)
    pse_shift_pure = pse_pure - standard_dur
    
    # PSE accounting for lapse rate (solve for p = 0.5)
    # 0.5 = lambda_/2 + (1-lambda_) * Phi((log(PSE/std) - mu)/sigma)
    # Phi((log(PSE/std) - mu)/sigma) = (0.5 - lambda_/2) / (1-lambda_)
    if lambda_ < 1.0:  # Avoid division by zero
        target_p = (0.5 - lambda_/2) / (1 - lambda_)
        target_p = np.clip(target_p, 1e-10, 1-1e-10)  # Avoid edge cases
        z_target = norm.ppf(target_p)
        log_pse_ratio = mu + sigma * z_target
        pse_lapse_corrected = standard_dur * np.exp(log_pse_ratio)
        pse_shift_lapse_corrected = pse_lapse_corrected - standard_dur
    else:
        pse_lapse_corrected = standard_dur
        pse_shift_lapse_corrected = 0
    
    return {
        'pse_pure': pse_pure,
        'pse_shift_pure': pse_shift_pure,
        'pse_lapse_corrected': pse_lapse_corrected,
        'pse_shift_lapse_corrected': pse_shift_lapse_corrected,
        'pse_shift_percent': (pse_shift_pure / standard_dur) * 100
    }

# Negative log-likelihood
def negative_log_likelihood(params, test_dur, standard_dur, chose_test, total_responses):
    lambda_, mu, sigma = params # Unpack parameters
    if fixedMu:
        mu = 0
    
    p = psychometric_function(test_dur, standard_dur, lambda_, mu, sigma) # Compute probability of choosing test
    epsilon = 1e-9 # Add a small number to avoid log(0) when calculating thxe log-likelihood
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
    if fixedMu:
        bounds = [(0, 0.25), (0.01, 1.5)]
        init_guesses = [init_guesses[0],  init_guesses[2]]  # Set mu to 0 if fixed
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
    Compute negative log likelihood for all conditions.
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
        
        # Calculate probability of choosing test
        p = psychometric_function(test_dur, standard_dur, lambda_, mu, sigma)
        
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
    
    test_durations = grouped_data[intensityVariable]  # Raw test durations
    standard_durations = grouped_data[standardVar]    # Standard durations
    chose_tests = grouped_data['num_of_chose_test']
    total_responses = grouped_data['total_responses']
    conflicts = grouped_data[conflictVar]
    noise_levels = grouped_data[sensoryVar]
    
    
    # Set bounds for parameters
    bounds = [(0, 0.25)]*nLambda + [(0.01, +2.0)]*nSensoryVar*nConflictVar + [(-1, +1)]*nSensoryVar*nConflictVar


    # Minimize negative log-likelihood
    result = minimize(
        nLLJoint,
        x0=initGuesses,
        args=(test_durations, standard_durations, chose_tests, total_responses, conflicts, noise_levels),
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

def plot_fitted_psychometric(data, best_fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict, standardVar, sensoryVar, conflictVar, intensityVariable, show_error_bars=True):
    plottingCrossModal=1
    if plottingCrossModal:
        standardLabel="Visual:"
        testLabel="Auditory:"
        choseLabel="chose auditory"
        colors=[ "navy","maroon" ]

    else:
        standardLabel="Standard:"
        testLabel="Test:"
        choseLabel="chose test"
        colors=[ "black","navy","maroon" ]

    print(f"Fitted parameters: {best_fit.x}")
    #colors = sns.color_palette("viridis", n_colors=len(uniqueSensory))  # Use Set2 palette for different noise levels
    
    plt.figure(figsize=(10, 10))
    labeledStandard=0
    for i, standardLevel in enumerate(uniqueStandard):
        for j, audioNoiseLevel in enumerate(uniqueSensory):
            for k, conflictLevel in enumerate(uniqueConflict):
                lambda_, mu, sigma = getParams(best_fit.x, conflictLevel, audioNoiseLevel, nLambda, nSigma)
                weberFraction = sigma/np.sqrt(2)/standardLevel
                sigmaSensory = sigma/np.sqrt(2)
                sigmaSensoryLinear= standardLevel * (np.exp(sigmaSensory)-1)
                weberFractionLinear = sigmaSensoryLinear / standardLevel
                
                
                # Calculate PSE statistics
                pse_stats = calculate_pse_stats(mu, sigma, lambda_, standardLevel)
                
                print(f"\n=== Audio Noise: {audioNoiseLevel:.2f}, Conflict: {conflictLevel:.2f} ===")
                print(f"Sigma Sensory:{sigmaSensory:.3f}, Sigma sensory (linear): {sigmaSensoryLinear:.3f} s")
                print(f"Weber fraction (linear): {weberFractionLinear:.3f}")

                print(f"Raw Parameters - Lambda: {lambda_:.3f}, Mu: {mu:.3f}, Sigma: {sigma:.3f}")
                print(f"Sensory noise sigma (σ/√2): {sigma/np.sqrt(2):.3f}")
                print(f"Weber fraction (σ/√2 / standard): {weberFraction:.3f}")
                
                print(f"PSE Analysis:")
                print(f"  PSE (pure): {pse_stats['pse_pure']*1000:.1f} ms")
                print(f"  PSE shift (pure): {pse_stats['pse_shift_pure']*1000:+.1f} ms ({pse_stats['pse_shift_percent']:+.1f}%)")
                print(f"  PSE (lapse-corrected): {pse_stats['pse_lapse_corrected']*1000:.1f} ms")
                print(f"  PSE shift (lapse-corrected): {pse_stats['pse_shift_lapse_corrected']*1000:+.1f} ms")
                if pse_stats['pse_shift_pure'] > 0:
                    print(f"  → Overestimation: Test needs to be {abs(pse_stats['pse_shift_pure']*1000):.1f} ms longer than standard to appear equal")
                elif pse_stats['pse_shift_pure'] < 0:
                    print(f"  → Underestimation: Test needs to be {abs(pse_stats['pse_shift_pure']*1000):.1f} ms shorter than standard to appear equal")
                else:
                    print(f"  → No bias: Test and standard perceived equally when physically equal")
                # Filter the data for the current standard and audio noise levels
                df = data[round(data[standardVar], 2) == round(standardLevel,2)]
                df = df[df[sensoryVar] == audioNoiseLevel]
                df = df[df[conflictVar] == conflictLevel]
                dfFiltered = groupByChooseTest(df)
                test_durations = dfFiltered[intensityVariable].values  # Raw test durations
                if len(test_durations) == 0:
                    continue
                responses = dfFiltered['num_of_chose_test'].values
                totalResponses = dfFiltered['total_responses'].values
                
                # Create range of test durations for smooth curve
                minX = min(test_durations) * 0.8  # Use raw duration range
                maxX = max(test_durations) *1.2

                x_smooth = np.linspace(minX, maxX, 1000)
                # For plotting, we need to provide both test and standard durations
                standard_dur_array = np.full_like(x_smooth, standardLevel)  # Constant standard duration
                y = psychometric_function(x_smooth, standard_dur_array, lambda_, mu, sigma)

                color = colors[j]
                #plt.plot(x, y, color=color, label=f"Noise: {audioNoiseLevel}\n $\\mu$: {mu:.2f}, $\\sigma$: {sigma:.2f}", linewidth=4)
                labelsDict={0.1:"Auditory low noise",1.2:"Auditory high noise",99:"Visual",0.03:"High noise"}
                plt.plot(x_smooth*1000, y, color=color, linewidth=4, label=f"{labelsDict.get(audioNoiseLevel,audioNoiseLevel)}" )
                #plt.axvline(x=0, color='gray', linestyle='--')
                plt.axhline(y=0.5, color='gray', linestyle='--')
                binVar='testDurMs'
                fontSize=16
                #plt.xlabel(f"({intensityVariable}) Test(stair)-Standard(0.5s) Duration Difference Ratio")
                plt.xlabel(f"{testLabel} Duration (ms)",fontsize=fontSize)
                plt.ylabel(f"P({choseLabel} )",fontsize=fontSize)
                

                if not labeledStandard:
                    plt.axvline(standardLevel*1000, linestyle='--')  # Show standard duration line
                labeledStandard=1
                #plt.title(f" {pltTitle} ", fontsize=20)
                #plt.title("Unimodal-visual psychometric function",fontsize=fontSize)
                plt.xticks(fontsize=fontSize-2)
                plt.yticks(fontsize=fontSize-2)
                #plt.xticks(500*np.arange(0, 3, 0.5))
                
                plt.legend(fontsize=14, title_fontsize=fontSize)
                #plt.grid()
                # Use the raw data (df) instead of grouped data (dfFiltered) to preserve participantID for error bars
                bin_and_plot(df, bin_method='cut', bins=10, plot=True, color=color, add_error_bars=show_error_bars,binVar="testDurMs")
                plt.text(0.05, 0.9, f"{str(standardLabel)} {standardLevel*1000:.0f}ms ", fontsize=14, ha='left', va='top', transform=plt.gca().transAxes)
                #plt.text(0.05, 0.8, f"Shared $\\lambda$: {lambda_:.2f}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)

                # Set x-axis ticks based on actual duration range
                #plt.xticks(np.arange(0, 1001, step=250))
                plt.xlim(0, 1000)
                plt.tight_layout()
                
                #plt.grid(True)
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fit psychometric functions with optional error bars')
    parser.add_argument('--no-error-bars', action='store_true', 
                       help='Plot without error bars across participants')
    parser.add_argument('--data', default='all_crossModal.csv',
                       help='Data file to use (default: all_auditory.csv)')
    args = parser.parse_args()
    
    fixedMu = 0 # Set to True to ignore the bias in the model (overrides global setting)
    dataName = args.data
    show_error_bars  =  not args.no_error_bars  # Invert the flag
    
    # Load and prepare data
    print(f"Loading data from {dataName}...")
    data, sensoryVar, standardVar, conflictVar, uniqueSensory, uniqueStandard, uniqueConflict, nLambda, nSigma, nMu = loadData(dataName)
    
    # Create plot title
    pltTitle = dataName.split("_")[1]
    pltTitle = dataName.split("_")[0] + str(" ") + pltTitle
    
    # Print data summary
    print(f"Data loaded: {len(data)} trials from {data['participantID'].nunique()} participants")
    print(f"Participants: {sorted(data['participantID'].unique())}")
    
    # Group data for fitting (using traditional grouping for model fitting)
    grouped_data = groupByChooseTest(data)
    
    # Fit the psychometric model
    print("Fitting psychometric model...")
    fit = fitMultipleStartingPoints(data, nStart=1)
    print(f"Fitted parameters: {fit.x}")

    # Plot the fitted psychometric functions 
    if show_error_bars:
        print("Plotting psychometric functions with error bars across participants...")
    else:
        print("Plotting psychometric functions without error bars...")
        
    plot_fitted_psychometric(
        data, fit, nLambda, nSigma, uniqueSensory, uniqueStandard, uniqueConflict,
        standardVar, sensoryVar, conflictVar, intensityVariable, show_error_bars=show_error_bars)
    
    # Optionally plot staircase data
    # plotStairCases(data)




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

