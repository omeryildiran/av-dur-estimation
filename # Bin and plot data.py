# Bin and plot data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def bin_and_plot(data, bin_method='cut', bins=10, bin_range=None, plot=True,color="blue",intensityVar,probVar,nResp):
	if bin_method == 'cut':
		data['bin'] = pd.cut(data[intensityVar], bins=bins, labels=False, include_lowest=True, retbins=False)
	elif bin_method == 'manual':
		data['bin'] = np.digitize(data[intensityVar], bins=bin_range) - 1
	
	grouped = data.groupby('bin').agg(
		x_mean=(self.intensityVar, 'mean'),
		y_mean=(probVar, 'mean'),
		total_resp=(nResp, 'sum')
	)

	if plot:
		plt.scatter(grouped['x_mean'], grouped['y_mean'], s=grouped['total_resp']/data[nResp].sum()*900, color=color)

	