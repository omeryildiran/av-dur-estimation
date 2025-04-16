# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import linregress
from scipy.stats import norm
from scipy.optimize import minimize


class analyzeData():
	def __init__(self,
			  dataName="_mainExpAvDurEstimate_2025-03-27_15h13.32.171.csv",
			  folderDir="mainExpAvDurEstimate/dataAvMain/",
			  intensityVar="avgAVDeltaPercent",
			  sensoryNoiseVar="avNoise",
			  conflictVar="conflictDur",
			  standardDurVar="standardDur",
	
			  data=None):
		self.dataName = dataName
		self.folderDir = folderDir
		self.intensityVar = intensityVar
		self.sensoryNoiseVar = sensoryNoiseVar
		self.conflictVar = conflictVar
		#print("Current directory:\n s", os.getcwd())
		currentDir = os.getcwd()
		# Load data from CSV file
		self.data = pd.read_csv(folderDir+ dataName)
		# Check if the data is loaded correctly
		if self.data.empty:
			print(f"\nData file {dataName} is empty or does not exist.\n")
			return
		
		self.data['avgAVDeltaS'] = (self.data['deltaDurS'] + (self.data['recordedDurVisualTest'] - self.data['recordedDurVisualStandard'])) / 2
		# Calculate deltaDurPercentVisual just as the difference between the test and standard visual durations over the standard visual duration
		self.data['deltaDurPercentVisual'] = ((self.data['recordedDurVisualTest'] - self.data['recordedDurVisualStandard']) / self.data['recordedDurVisualStandard'])
		self.data['avgAVDeltaPercent'] = self.data[['delta_dur_percents', 'deltaDurPercentVisual']].mean(axis=1)

		# Define columns for chosing test or standard
		self.data['chose_test'] = (self.data['responses']== self.data['order']).astype(int)
		self.data['chose_standard'] = (self.data['responses']!= self.data['order']).astype(int)
		self.data['standard_dur']=self.data['standardDur']

		
	def plotRawData(self):
		# Create a new figure
		plt.figure(figsize=(10, 6))
		# Plot the data scatter
		sns.scatterplot(data=self.data, x=self.intensityVar, y='chose_test',  alpha=0.5)
		plt.title('Raw Data')
		# Show the plot
		plt.show()
	
	# Bin and plot data
	def bin_and_plot(self,data, bin_method='cut', bins=10, bin_range=None, plot=True,color="blue"):
		if bin_method == 'cut':
			data['bin'] = pd.cut(data[self.intensityVar], bins=bins, labels=False, include_lowest=True, retbins=False)
		elif bin_method == 'manual':
			data['bin'] = np.digitize(data[self.intensityVar], bins=bin_range) - 1
		
		grouped = data.groupby('bin').agg(
			x_mean=(self.intensityVar, 'mean'),
			y_mean=('p_choose_test', 'mean'),
			total_resp=('total_responses', 'sum')
		)

		if plot:
			plt.scatter(grouped['x_mean'], grouped['y_mean'], s=grouped['total_resp']/data['total_responses'].sum()*900, color=color)

		
	
	def groupByChooseTest(self,x):
		grouped = x.groupby([self.intensityVar, 'riseDur', 'standardDur','conflictDur']).agg(
			num_of_chose_test=('chose_test', 'sum'),
			total_responses=('responses', 'count'),
			num_of_chose_standard=('chose_standard', 'sum'),
		).reset_index()
		grouped['p_choose_test'] = grouped['num_of_chose_test'] / grouped['total_responses']

		return grouped
	# Compute sigma from slope
	def compute_sigma_from_slope(self, slope, lapse_rate=0.02):
		sigma = (1 - lapse_rate) / (np.sqrt(2 * np.pi) * slope)*np.exp(-0.5)
		return sigma
	
	def psychometric_function(self,x, lambda_, mu, sigma):
		# Cumulative distribution function with mean mu and standard deviation sigma
		cdf = norm.cdf(x, loc=mu, scale=sigma) 
		# take into account of lapse rate and return the probability of choosing test
		p = lambda_/2 + (1-lambda_) * norm.cdf((x - mu) / sigma)
		#return lapse_rate * 0.5 + (1 - lapse_rate) * cdf 
		return p

		#p = lambda_/2 + (1-lambda_) * norm.cdf((x - mu) / sigma)

	# Negative log-likelihood
	def negative_log_likelihood(self,params, delta_dur, chose_test, total_responses):
		lambda_, mu, sigma = params # Unpack parameters
		
		p = self.psychometric_function(delta_dur, lambda_, mu, sigma) # Compute probability of choosing test
		epsilon = 1e-9 # Add a small number to avoid log(0) when calculating the log-likelihood
		p = np.clip(p, epsilon, 1 - epsilon) # Clip p to avoid log(0) and log(1)
		# Compute the negative log-likelihood
		log_likelihood = np.sum(chose_test * np.log(p) + (total_responses - chose_test) * np.log(1 - p))
		return -log_likelihood


	# Fit psychometric function
	def fit_psychometric_function(self, levels,nResp, totalResp,init_guesses=[0,0,0]):
		# then fits the psychometric function
		# order is lambda mu sigma
		#initial_guess = [0, -0.2, 0.05]  # Initial guess for [lambda, mu, sigma]
		bounds = [(0, 0.2), (-0.4, +0.4), (0.01, 1)]  # Reasonable bounds
		# fitting is done here
		result = minimize(
			self.negative_log_likelihood, x0=init_guesses, 
			args=(levels, nResp, totalResp),  # Pass the data and fixed parameters
			bounds=bounds,
			method='Nelder-Mead'
		)
		# returns the fitted parameters lambda, mu, sigma
		return result.x
	
	from scipy.stats import linregress
	def estimate_initial_guesses(self, levels,chooseTest,totalResp, max_sigma_ratio=0.2):
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
		sigma_guess= self.compute_sigma_from_slope(slope,lapse_rate_guess)-0.1

		# Regularize sigma to avoid overestimation
		intensity_range = np.abs(max(intensities)) - np.abs(min(intensities))
		

		#
		groupedData = self.groupByChooseTest(self.data)
		single_init_guesses = self.estimate_initial_guesses(
		self.groupedData[self.intensityVar],
		self.groupedData['num_of_chose_test'],
		self.groupedData['total_responses']
		)
		print("Initial guesses:", single_init_guesses)
	
		
		return [lapse_rate_guess, mu_guess, sigma_guess]


	def negative_log_likelihood_unified(self,params, delta_dur, chose_tests, total_responses, conflicts, noise_levels):
		lambda_ = params[0]  # Lapse rate
		sigmaA = params[1]   # Standard deviation for audioNoise 0.1
		sigmaB = params[2]   # Standard deviation for audioNoise 1.2
		
		# Mean parameters for different conditions
		muAminus50 = params[3]  # Mean for audioNoise 0.1 and conflict -0.05
		muA0 = params[4]        # Mean for audioNoise 0.1 and conflict 0
		muAplus50 = params[5]   # Mean for audioNoise 0.1 and conflict 0.05
		muBminus50 = params[6]  # Mean for audioNoise 1.2 and conflict -0.05
		muB0 = params[7]        # Mean for audioNoise 1.2 and conflict 0
		muBplus50 = params[8]   # Mean for audioNoise 1.2 and conflict 0.05
		
		# Initialize negative log-likelihood
		nll = 0
		
		# Loop through each data point
		for i in range(len(delta_dur)):
			x = delta_dur[i]
			conflict = round(conflicts[i], 2)
			audio_noise = round(noise_levels[i], 2)
			total_response = total_responses[i]
			chose_test = chose_tests[i]
			
			# Select appropriate parameters based on condition
			if audio_noise == 0.1:
				sigma = sigmaA
				if conflict == -0.05:
					mu = muAminus50
				elif conflict == 0:
					mu = muA0
				elif conflict == 0.05:
					mu = muAplus50
				else:
					raise ValueError(f"Unknown conflict level: {conflict}")
			elif audio_noise == 1.2:
				sigma = sigmaB
				if conflict == -0.05:
					mu = muBminus50
				elif conflict == 0:
					mu = muB0
				elif conflict == 0.05:
					mu = muBplus50
				else:
					raise ValueError(f"Unknown conflict level: {conflict}")
			else:
				raise ValueError(f"Unknown audioNoise level: {audio_noise}")
			
			# Calculate the probability of choosing the test stimulus at level x
			p = self.psychometric_function(x, lambda_, mu, sigma)
			
			# Avoid numerical issues with log(0) or log(1)
			epsilon = 1e-9
			p = np.clip(p, epsilon, 1 - epsilon)
			
			# Add to the negative log-likelihood
			nll += -1 * (chose_test * np.log(p) + (total_response - chose_test) * np.log(1 - p))
		
		return nll

	def fitPsychUnified(self, grouped_data, intensity_variable='deltaDur', initGuesses=[0.05, 0.1, 0.1, -0.2, 0, 0.2, -0.2, 0, 0.2]):
		"""
		Fit psychometric function to all conditions
		
		Parameters:
		-----------
		grouped_data : pandas DataFrame
			Grouped data with proportion of 'chose test' responses
		intensity_variable : str
			Column name for the intensity variable
			
		Returns:
		--------
		result : scipy.optimize.OptimizeResult
			Optimization result
		"""
		intensities = grouped_data[intensity_variable]
		chose_tests = grouped_data['num_of_chose_test']
		total_responses = grouped_data['total_responses']
		conflicts = grouped_data['conflictDur']
		noise_levels = grouped_data['riseDur']
		
		
		# Set bounds for parameters
		bounds = [(0, 0.2),      # lambda
				(0.02, 1),     # sigmaA
				(0.02, 1)]     # sigmaB
		bounds += [(-0.4, 0.4)] * 6  # All mu bounds

		# Minimize negative log-likelihood
		result = minimize(
			self.negative_log_likelihood_unified,  # Add 'self.' here
			x0=initGuesses,
			args=(intensities, chose_tests, total_responses, conflicts, noise_levels),
			bounds=bounds,
			method='L-BFGS-B'  # Use L-BFGS-B for bounded optimization
		)
		
		return result

	def fitMultipleStartingPoints(self, grouped_data, intensity_variable='deltaDur', multipleInitGuesses=None):
		# initial guesses
		nStarts = 2
		if multipleInitGuesses == None:
			multipleInitGuesses = []
			# 100 random multiple starts
			initLambdas = np.linspace(0, 0.2, nStarts)  # Different initial guesses for lambda
			initSigmaA = np.linspace(0.05, 1, nStarts)  # Different initial guesses for sigma
			initSigmaB = np.linspace(0.05, 1, nStarts)  # Different initial guesses for sigma
			initMuA1 = np.linspace(-0.4, 0.4, nStarts)  # Different initial guesses for mu
			initMuA2 = np.linspace(-0.4, 0.4, nStarts)  # Different initial guesses for mu
			initMuA3 = np.linspace(-0.4, 0.4, nStarts)  # Different initial guesses for mu
			initMuB1 = np.linspace(-0.4, 0.4, nStarts)  # Different initial guesses for mu
			initMuB2 = np.linspace(-0.4, 0.4, nStarts)  # Different initial guesses for mu
			initMuB3 = np.linspace(-0.4, 0.4, nStarts)  # Different initial guesses for mu
			
			#create combinations of the initial guesses combined in different ways
			initLambdas = np.linspace(0, 0.2, nStarts)  # Different initial guesses for lambda
			initMus = np.linspace(-0.15, 0.15, nStarts)  # Different initial guesses for mu
			initSigmas = np.linspace(0.05, 1, nStarts)  # Different initial guesses for sigma

			for initLambda in initLambdas:
				for initSigma in initSigmas:
					for initMu in initMus:
						multipleInitGuesses.append([initLambda, initSigma, initSigma, initMu, initMu, initMu,initMu, initMu, initMu])


		levels = grouped_data[self.intensityVar]
		nResp = grouped_data['num_of_chose_test']
		totalResp = grouped_data['total_responses']
		conflictLevels = grouped_data['conflictDur']
		noiseLevels = grouped_data['riseDur']
		
		groupedData=self.groupByChooseTest(self.data)
		best_fit = None
		best_nll = float('inf')  # Initialize with infinity

		from tqdm import tqdm
		# Replace the manual progress tracking with tqdm
		for i, init_guess in enumerate(tqdm(multipleInitGuesses, desc="Fitting models", unit="fit")):
			fit = self.fitPsychUnified(groupedData, self.intensityVar, initGuesses=init_guess)
			nll = self.negative_log_likelihood_unified(fit.x, levels, nResp, totalResp, conflictLevels, noiseLevels)

			if nll < best_nll:
				best_nll = nll
				best_fit = fit

		# Extract fitted parameters
		lambda_, sigmaA_fit, sigmaB_fit, muAminus50_fit, muA0_fit, muAplus50_fit, muBminus50_fit, muB0_fit, muBplus50_fit = best_fit.x
		print(f"Fitted parameters:\n lambda: {lambda_}\n sigmaA: {sigmaA_fit}\n sigmaB: {sigmaB_fit}\n muAminus50: {muAminus50_fit}\n muA0: {muA0_fit}\n muAplus50: {muAplus50_fit}\n muBminus50: {muBminus50_fit}\n muB0: {muB0_fit}\n muBplus50: {muBplus50_fit}")
		
		self.best_fit = best_fit
		
		return best_fit
	


	def plotPsychometricFunctionUnified(self, grouped_data, fit_params):
			# extract fitted parameters
			standardDurLevels = np.unique(grouped_data['standardDur'])
			noiseLevels = np.unique(grouped_data['riseDur'])
			conflictLevels = np.unique(grouped_data['conflictDur'])
			plt.figure(figsize=(24, 6))
			m = 0
			# Loop through each condition
			for i, standardLevel in enumerate(standardDurLevels):
				lambda_ = fit_params[0]
				print(f"At standard level {standardLevel} fitted lambda {lambda_:.4f}")
				for j, noiseLevel in enumerate(noiseLevels):
					sigma = fit_params[j+1]
					print(f"At noise level {noiseLevel} fitted sigma {sigma:.4f}")
					for k, conflictLevel in enumerate(conflictLevels):
						m += 1        
						mu = fit_params[m+2]
						# Filter data by condition
						df = self.data[self.data['conflictDur'] == conflictLevel]
						df = df[df["standardDur"] == standardLevel]
						df = df[df["riseDur"] == noiseLevel]
						dfFiltered = self.groupByChooseTest(df)
						# Select levels for plotting
						levels = dfFiltered[self.intensityVar].values
						responses = dfFiltered['num_of_chose_test'].values
						totalResponses = dfFiltered['total_responses'].values

						print(f"Noise: {noiseLevel}, Conflict: {conflictLevel}, fit sigma: {sigma:.4f}, fit mu: {mu:.4f}")
						plt.subplot(1, 2, j+1)
						maxX = max(levels) if len(levels) > 0 else 1.3  # Handle case with no levels
						if maxX < 1.3:
							maxX = 1.3
						plt.xlim(-1, maxX)
						plt.ylim(0, 1)
						x = np.linspace(-1, maxX, 100)

						y = self.psychometric_function(x, lambda_, mu, sigma)
						color = sns.color_palette("viridis", as_cmap=True)(k / len(conflictLevels))
						plt.plot(x, y, label=round(conflictLevel, 2), color=color, linewidth=3)
						plt.axvline(x=0, color='gray', linestyle='--')
						plt.axhline(y=0.5, color='gray', linestyle='--')
						plt.xlabel('Physical duration difference (Comp-Standard averaged sec)')
						plt.ylabel('Probability of Choosing Test')
						plt.title(f'Standard Auditory Duration: {round(standardLevel,2)}, Noise Level: {round(noiseLevel,2)}')
						plt.legend(title="Conflict Level", fontsize=14, title_fontsize=14)
						self.bin_and_plot(dfFiltered, bin_method='cut', bins=8, plot=True, color=color)
						
if __name__ == "__main__":
    # Create an instance of analyzeData
    analyzer = analyzeData(
        intensityVar="delta_dur_percents", 
        folderDir="mainExpAvDurEstimate/dataAvMain/",
        dataName="_mainExpAvDurEstimate_2025-03-27_15h13.32.171.csv"
    )
    
    # Basic data visualization
    print("Plotting raw data...")
    #analyzer.plotRawData()
    
    # Group the data
    grouped_data = analyzer.groupByChooseTest(analyzer.data)
    
    # Define initial guesses for the model parameters
    init_guesses = [0.05,   # lambda (lapse rate)
                    0.1,    # sigmaA (st. dev. for low noise)
                    0.2,    # sigmaB (st. dev. for high noise)
                    -0.1,   # muAminus50 (mean for low noise, negative conflict)
                    0.0,    # muA0 (mean for low noise, zero conflict)
                    0.1,    # muAplus50 (mean for low noise, positive conflict)
                    -0.1,   # muBminus50 (mean for high noise, negative conflict)
                    0.0,    # muB0 (mean for high noise, zero conflict)s
                    0.1]    # muBplus50 (mean for high noise, positive conflict)
    
    # Fit the unified psychometric function model
    print("Fitting unified psychometric function model...")
    fit_result = analyzer.fitMultipleStartingPoints(grouped_data, analyzer.intensityVar)
    
    # Extract fitted parameters and print them (4-decimal precision)
    fit_params = fit_result.x
    lambda_, sigmaA, sigmaB, muAminus50, muA0, muAplus50, muBminus50, muB0, muBplus50 = fit_params
    
    print("\nFitted Parameters:")
    print(f"  Lambda (lapse rate): {lambda_:.4f}")
    print(f"  SigmaA (low noise):   {sigmaA:.4f}")
    print(f"  SigmaB (high noise):  {sigmaB:.4f}")
    print("\nMeans (PSE shifts):")
    print(f"  Low noise, -5% conflict: {muAminus50:.4f}")
    print(f"  Low noise, 0% conflict:  {muA0:.4f}")
    print(f"  Low noise, +5% conflict: {muAplus50:.4f}")
    print(f"  High noise, -5% conflict: {muBminus50:.4f}")
    print(f"  High noise, 0% conflict:  {muB0:.4f}")
    print(f"  High noise, +5% conflict: {muBplus50:.4f}")
    
    # Plot the psychometric functions for all conditions
    print("\nPlotting psychometric functions...")
    analyzer.plotPsychometricFunctionUnified(grouped_data, fit_params)
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis complete!")