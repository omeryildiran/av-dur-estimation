ExpTesting = False
ExpTraining= False
fullScreen=False  
expName = 'bimodal_audioVisual_durEst'
import os
# audio prefs
from psychopy import prefs
from psychopy.sound import backend_ptb as ptb
print(ptb.getDevices(kind='output'))
#prefs.general['audioLib'] = ['sounddevice', 'pyo', 'pygame']
prefs.hardware['audioLib'] = ['PTB']
#prefs.hardware['audioDevice'] = 2
prefs.general['audioLatencyMode'] = 4
volume=0.2


# Path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# 1 - Inititate and welcome participants
exec(open("exp_1_openScreen_bimodal.py").read())

# 2 - Import components
exec(open("exp_2_importComponents_av_bimodal.py").read()) 
 
# 3 - Run trial and response Loop
exec(open("exp_3_trialRoutine_av_bimodal.py").read(),)  
  