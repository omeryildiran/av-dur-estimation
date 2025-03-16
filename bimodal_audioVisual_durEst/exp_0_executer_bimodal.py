ExpTesting = False
ExpTraining= False
fullScreen=False
expName = 'bimodal_audioVisual_durEst'

# 1 - Inititate and welcome participants
exec(open("exp_1_openScreen_bimodal.py").read())

# 2 - Import components
exec(open("exp_2_importComponents_av_bimodal.py").read()) 

# 3 - Run trial and response Loop
exec(open("exp_3_trialRoutine_av_bimodal.py").read(),)  
