# Import components
# import the staircase
from my_staircase import stairCase
from conditions_av import create_conditions_matrix

#Handy timers
globalClock = core.Clock()
trialClock = core.Clock()

# Retrieve the conditions

rise_conds= [0.1] if ExpTraining==False else [0.1]
maxIntensityBurst=5
n_trial_per_condition=50 


#print('given trials number',len(conds_obj.intens))
#total_trials=(conds_obj.trial_per_condition)*2*4
"""
matrix columns:
0: Standard durations
1: Rise conditions
2: Conflict A-V
"""
conditions_matrix =create_conditions_matrix(totalTrialN=360,standards=[0.5], 
                                             background_levels=[0.1,0.85], conflicts=[0.05, 0, -0.05])
numberOfTrials=len(conditions_matrix)
print(f"Number of trials: {numberOfTrials}")
standardDurs = conditions_matrix[:, 0] # standard durations
riseDurs = conditions_matrix[:, 1] # Background noise level conditions should change riseDur to backgroundNoise everywhere
conflictDurs = conditions_matrix[:, 2] # conflict durations

uniqueRiseDurs=np.unique(riseDurs)
uniqueStandardDurs=np.unique(standardDurs)
uniqueConflictDurs=np.unique(conflictDurs)
totalUniqueConds=uniqueStandardDurs.shape[0]*uniqueRiseDurs.shape[0]*uniqueConflictDurs.shape[0]

# total stim durations
total_stim_durs=[]
# add columns for total audio duration, response, is_correct, RT



# Create general variables before the trial loop
endExpNow = False  # flag for 'escape' or other condition => quit the exp
mouse = event.Mouse(win=win,visible=False)
frameTolerance = 0.001  # how close to onset before 'same' frame
trialN=-1

# Responses
response = None
tolerance_trials=100
responses=np.zeros(numberOfTrials+tolerance_trials)
is_corrects=np.zeros(numberOfTrials+tolerance_trials)
response_rts=np.zeros(numberOfTrials+tolerance_trials)
response_keys=np.zeros(numberOfTrials+tolerance_trials)

responseKeys = keyboard.Keyboard(backend='PTB')

current_stairs=np.zeros(numberOfTrials+tolerance_trials)
deltaDurs=np.zeros(numberOfTrials+tolerance_trials)
testDurs=np.zeros(numberOfTrials+tolerance_trials)
realDeltaDurs=np.zeros(numberOfTrials+tolerance_trials)
intensities=np.zeros(numberOfTrials+tolerance_trials)

# create an array with names of the columns
column_names=columns=[
                'standardDur', 'riseDur','order', 'preDur', 'postDur', 'isiDur', 'trial_num',
                'total_audio_dur', 'delta_dur_percents', 'deltaDurS', 'testDurS', 'intensities',
                'current_stair', 'responses', 'is_correct', 'response_rts' , 'stair_num_reversal', 'stair_is_reversal', 'response_keys'
            'conflictDur', 
            'recordedOnsetVisualTest', 'recordedOffsetVisualTest', 'recordedDurVisualTest',
            'recordedOnsetVisualStandard', 'recordedOffsetVisualStandard', 'recordedDurVisualStandard','modalityPostCue']
print(f"column names: {len(column_names)}")
# create empty data matrix to save the data
exp_data=np.zeros((numberOfTrials+tolerance_trials, len(column_names)+1),dtype=object)

# region [rgba(2, 40, 30, 0.30)]
# Start the trial - response loop (there weill be)
""" Staircase Setup"""
stepFactor=0.6
initStep=0.2
maxReversals=100
max_level=0.95
initLevel=0.8

# Create the staircases
max_trial_per_stair=n_trial_per_condition#total_trials//5

print(f'rise unique: {np.unique(riseDurs)}')
stairCaseLonger = stairCase(init_level=initLevel, 
                            init_step=initStep, 
                            method="1D1U",  
                            step_factor=stepFactor, 
                            max_level=max_level+1, 
                            max_reversals=maxReversals, 
                            max_trials=max_trial_per_stair,)
                            
stairCaseLonger2D1U = stairCase(init_level=initLevel, init_step=initStep, method="2D1U",  step_factor=stepFactor, max_level=max_level+1, max_reversals=maxReversals, 
                                max_trials=max_trial_per_stair,)

stairCaseShorter = stairCase(init_level=-initLevel, init_step=initStep, method="1U1D",step_factor=stepFactor,
                              max_level=max_level, max_reversals=maxReversals, max_trials=max_trial_per_stair,)
stairCaseShorter2U1D = stairCase(init_level=-initLevel, init_step=initStep, method="2U1D",step_factor=stepFactor, 
                                 max_level=max_level, max_reversals=maxReversals, max_trials=max_trial_per_stair,)


stairCaseLapse = stairCase(init_level=0.6, init_step=initStep, method="lapse_rate", step_factor=stepFactor, max_level=max_level, max_reversals=maxReversals) # no need for it just decide on deltas

if ExpTraining:
    all_staircases=[stairCaseLapse]
else:
    all_staircases=[stairCaseLapse, stairCaseLonger,stairCaseShorter,stairCaseLonger2D1U,stairCaseShorter2U1D]#, stairCaseLonger_b,stairCaseShorter_b,] 
#all_staircases=[stairCaseShorter,stairCaseLonger,stairCaseLapse, ]#stairCaseLonger_b,stairCaseShorter_b,]
np.random.shuffle(all_staircases)
stopped_stair_count=0

def lapse_rate_cond_generate():
    lapse_deltas=[-0.9,0.9]
    all_conds=[]
    for i in uniqueStandardDurs: # standard durations 1.3, 1.6, 1.9
        for j in uniqueRiseDurs: # rise durations 0.05, 0.25
            for k in uniqueConflictDurs:
                for l in lapse_deltas: # lapse deltas -0.55, 0.55
                    all_conds.append([i,j,k,l])
    
    # in total 12 conditions
    
    # tile the lapse conditions
    all_conds=np.tile(all_conds,(7,1))
    np.random.shuffle(all_conds) 
    return all_conds
lapse_rate_conds=lapse_rate_cond_generate()
#print(f"lapse rate conditions: {lapse_rate_conds}")
total_trial_num=max_trial_per_stair*(len(all_staircases)-1)+len(lapse_rate_conds)

print(f'There are in total  {lapse_rate_conds.shape[0]} lapse rate conditions')
print(f"there are maximum {len(standardDurs)} normal trials in total of  {max_trial_per_stair*4} staircase trials" )

# Convert to list the standard durs and rise durs
standardDurs = standardDurs.tolist()
riseDurs = riseDurs.tolist()
conflictDurs = conflictDurs.tolist()
lapse_ended=False


# Set up fixation cross
fixation = visual.TextStim(win, text='+', color='white', height=deg2pix(1,monitor=win.monitor), pos=(0, 0))
# fixation.draw()
# win.flip()

#components for visual stimuli
visualStimSize=dva_to_px(size_in_deg=1.5,h=screen_height,d=screen_distance,r=sizeIs)

# Create Objects for the visual stimuli
visualStim=visual.Circle(win, radius=visualStimSize, fillColor=True, lineColor='black', colorSpace='rgb', units='pix',
                    pos=(0, 0), color='black')
visualStim.lineWidht=5


#Postcues
audioIcon =visual.ImageStim(win, image="audio_icon.png", pos=(0, 100), size=(50, 50))
visualIcon = visual.ImageStim(win, image="visual_icon.png", pos=(0, 100), size=(50, 50))
