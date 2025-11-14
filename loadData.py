# function for loading data
import pandas as pd
import numpy as np

def loadData(dataName, verbose=True):	
	sensoryVar="audNoise"
	standardVar="standardDur"
	conflictVar="conflictDur"
	pltTitle=dataName.split("_")[1]
	pltTitle=dataName.split("_")[0]+str(" ")+pltTitle

	def _print(*args, **kwargs):
		if verbose:
			print(*args, **kwargs)

	data = pd.read_csv("data/"+dataName)
	data["testDurMs"]= data["testDurS"]*1000
	data["standardDurMs"]= data["standardDur"]*1000
	data["conflictDurMs"]= data["conflictDur"]*1000
	data["conflictPreMod"]=data["conflictDur"]
	data["DeltaDurMs"]= data["testDurMs"] - data["standardDurMs"]

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

	data = data.round({'standardDur': 2, 'conflictDur': 2})

	# SV=SA+c+PSE
	data["realConflictDur"]=data['recordedDurVisualStandard'] -data["standardDur"]-data["VisualPSE"]
	data["realConflictDurMs"]=data["realConflictDur"]*1000

	_print(f"\n Total trials before cleaning\n: {len(data)}")
	data= data[data['audNoise'] != 0]
	data=data[data['standardDur'] != 0]
	data[standardVar] = data[standardVar].round(2)
	data[conflictVar] = data[conflictVar].round(3)
	uniqueSensory = data[sensoryVar].unique()
	uniqueStandard = data[standardVar].unique()
	uniqueConflict = sorted(data[conflictVar].unique())
	_print(f"uniqueSensory: {uniqueSensory} \n uniqueStandard: {uniqueStandard} \n uniqueConflict: {uniqueConflict}")

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
		data["standardDurCompare"] = abs(data['recordedDurVisualStandard'] - data['standardDur']-data["VisualPSE"]-data["conflictPreMod"])

	except:
		_print("Bias check failed!!!! No bias check columns found. Skipping bias check.")
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
		data=data[data['recordedDurVisualStandard'] <=998]
		data=data[data['recordedDurVisualStandard'] >=0]
		data=data[data['recordedDurVisualTest'] <=998]
		data=data[data['recordedDurVisualTest'] >=0]
		#clean trials where standardDurCheck and testDurSCheck are false
		data=data[data['standardDurCompare'] < 0.017]
		data=data[data['testDurSCompare']< 0.017]
	except:
		pass

	_print(f"total trials after cleaning: {len(data)}")
	nLambda=len(uniqueStandard)
	nSigma=len(uniqueSensory)
	nMu=len(uniqueConflict)*nSigma

	data["logStandardDur"] = np.log(data[standardVar])
	data["logConflictDur"] = np.log(data[conflictVar])
	data["logTestDur"] = np.log(data["testDurS"])
	data["logDeltaDur"] = data["logTestDur"] - data["logStandardDur"]
	data["logDeltaDurMs"] = np.log(data["testDurMs"]) - np.log(data["standardDurMs"])

	dataName = dataName.split(".")[0]
	return data, dataName
