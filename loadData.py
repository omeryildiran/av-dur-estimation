# function for loading data
import pandas as pd
import numpy as np

def loadData(dataName):	
	sensoryVar="audNoise"
	standardVar="standardDur"
	conflictVar="conflictDur"
	pltTitle=dataName.split("_")[1]
	pltTitle=dataName.split("_")[0]+str(" ")+pltTitle    



	data = pd.read_csv("data/"+dataName)
	data["testDurMs"]= data["testDurS"]*1000
	data["standardDurMs"]= data["standardDur"]*1000
	data["conflictDurMs"]= data["conflictDur"]*1000
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

	# SV=SA+c+PSE
	data["realConflictDur"]=data['recordedDurVisualStandard'] -data["standardDur"]-data["VisualPSE"]
	data["realConflictDurMs"]=data["realConflictDur"]*1000

	data = data.round({'standardDur': 3, 'conflictDur': 3})

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

	try: 
		data['biasCheckTest'] = np.isclose(data['visualPSEBiasTest'], data['VisualPSE'], atol=0.012)
		data['biasCheckStandard'] = np.isclose(data['visualPSEBias'], data['VisualPSE'], atol=0.012)
		data["testDurSCheck"] = (abs(data['recordedDurVisualTest'] - data['testDurS']-data["VisualPSE"]) < 0.03)
		data["testDurSCheckBias"] = (abs(data['recordedDurVisualTest'] - data['testDurS']-data["VisualPSE"]) < 0.03)

		data["standardDurCheck"] = (abs(data['recordedDurVisualStandard'] - data['standardDur']-data["VisualPSE"]-data['conflictDur']) < 0.03)
		data["testDurSCompare"] = abs(data['recordedDurVisualTest'] - data['testDurS']-data["VisualPSE"])
		data["standardDurCompare"] = abs(data['recordedDurVisualStandard'] - data['standardDur']-data["VisualPSE"]-data['conflictDur'])

		# #print len of testDurSCheck and standardDurCheck false
		# print("")
		# print(len(data[data['testDurSCheck'] == False]), " trials with testDurSCheck False")
		# print(len(data[data['standardDurCheck'] == False]), " trials with standardDurCheck False\n")
		# # print number of abs(testDurSCompare
		# print(len(data[abs(data['testDurSCompare']) > 0.03]), " trials with abs(testDurSCompare) > 0.05")
		# print(len(data[abs(data['standardDurCompare']) > 0.03]), " trials with abs(standardDurCompare) > 0.05")
		# print("")
		# print(len(data[data['testDurSCheckBias'] == False]), " trials with testDurSCheckBias False")

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

	data['standard_dur']=round(data['standardDur'],3)
	data["delta_dur_percents"]=round(data["delta_dur_percents"],3)
	# try:
	# 	print(len(data[data['recordedDurVisualTest']<0]), " trials with negative visual test duration")
	# 	print(len(data[data['recordedDurVisualStandard']<0]), " trials with negative visual standard duration")
	# except:
	# 	print("No negative visual test or standard duration found.")




	try:
		#print(f'testdurCompare > 0.05: {len(data[data["testDurSCompare"] > 0.05])} trials')

		#print(len(data[data['recordedDurVisualStandard']<0]), " trials with negative visual standard duration")
		#print(len(data[data['recordedDurVisualTest']<0]), " trials with negative visual test duration")


		data=data[data['recordedDurVisualStandard'] <=998]
		data=data[data['recordedDurVisualStandard'] >=0]
		data=data[data['recordedDurVisualTest'] <=998]
		data=data[data['recordedDurVisualTest'] >=0]
		#clean trials where standardDurCheck and testDurSCheck are false
		data=data[data['standardDurCompare'] < 0.003]
		data=data[data['testDurSCompare']< 0.003]
	except:
		pass

	print(f"total trials after cleaning: {len(data)}")
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
