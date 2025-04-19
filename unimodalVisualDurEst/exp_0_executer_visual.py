ExpTesting = True
ExpTraining= False
fullScreen=False
expName = 'visualDurEst'
expCondition='visual'
conflicts=[0]
#riseDurs=[1]
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
# print(f"Current directory: {current_dir}")
parent_dir = os.path.dirname(current_dir)
print(f"\nParent directory\n: {parent_dir}")
exec(open(parent_dir+"/intervalDurs.py").read())
volume=volume
#prefs.hardware['audioDevice'] = [deviceIndex,1,2,3,4,5]
# 1 - Inititate and welcome participants
exec(open("exp_1_openScreen_v.py").read())

# 2 - Import components
exec(open("exp_2_importComponents_v.py").read()) 


# 3 - Run trial and response Loop
exec(open("exp_3_trialRoutine_v.py").read(),)  
