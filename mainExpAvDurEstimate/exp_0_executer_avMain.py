isRecall=0
nTrialPerStairPerCondition=70


ExpTesting = 0
ExpTraining= 0
fullScreen=1
expName = 'mainExpAvDurEstimate'
modalityCue='A'

#avPSEseconds=-0.025 # audiovisual PSE in seconds (negative means visual perceived shorter
standardDur=0.5 # standard duration in seconds
avPSEsecondsLow= 0.20 *standardDur # Neg Visual Bias for high noise  audiovisual PSE in seconds (negative means visual perceived shorter
avPSEsecondsHigh= 0.21 *standardDur # Neg Visual Bias for high noise audiovisual PSE in seconds (negative means visual perceived shorter

# (when audio is shorter than visual, people perceive audio and visual in equal length) so to handle this, we need to add this value to the visual duration

conflicts=[ -0.25, -0.167, 0.083, 0, -0.083, 0.167, 0.25,]

conflictsBatch1=[ 0, -0.167,  0.25]
conflictsBatch2=[ -0.083, 0.167]

conflictsBatch3=[ -0.25,  0.083]

conflicts=conflictsBatch2


nLAPSE = 2 if isRecall else 7
print('n LAPSE:', nLAPSE)

import os
import sys  # to get file system encoding

# add child directory to path
try:
    os.chdir(os.path.dirname(__file__))
except:
    print("Current directory is already the directory of the script.")
# Change directory to the current directory
#os.chdir(os.path.dirname(__file__))

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# list of all the modules in the current directory
#print(os.listdir(os.path.dirname(os.path.abspath(__file__))))
# audio prefs
from psychopy import prefs
from psychopy.sound import backend_ptb as ptb
#prefs.general['audioLib'] = ['sounddevice', 'pyo', 'pygame']
prefs.hardware['audioLib'] = ['PTB']
prefs.general['audioLatencyMode'] = 4


# Path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\nCurrent directory\n: {current_dir}")
#parent directory
parent_dir = os.path.dirname(current_dir)
print(f"\nParent directory\n: {parent_dir}")
exec(open(parent_dir+"/intervalDurs.py").read())

volume=volume

# 1 - Inititate and welcome participantss
exec(open("exp_1_openScreen_avMain.py").read())

# 2 - Import components
exec(open("exp_2_importComponents_avMain.py").read()) 
 
# 3 - Run trial and response Loop
exec(open("exp_3_trialRoutine_avMain.py").read())  
  