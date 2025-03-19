ExpTesting = False
ExpTraining= False
fullScreen=True 
expName = 'bimodal_audioVisual_durEst'
import os

# Path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# 1 - Inititate and welcome participants
exec(open("bimodal_audioVisual_durEst/exp_1_openScreen_bimodal.py").read())

# 2 - Import components
exec(open("exp_2_importComponents_av_bimodal.py").read()) 
 
# 3 - Run trial and response Loop
exec(open("exp_3_trialRoutine_av_bimodal.py").read(),)  
 