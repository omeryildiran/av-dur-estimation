# Import components
# import the staircase

from betterStaircase import stairCase
from conditions_av import create_conditions_matrix

#Handy timers
globalClock = core.Clock()
trialClock = core.Clock()

# Retrieve the conditions

audNoiseConds= [0.1,1.2] 
maxIntensityBurst=5
nTrialPerStairPerCondition=45
conflicts=[0]
nTrialPerStair=len(audNoiseConds)*len(conflicts)*nTrialPerStairPerCondition
totalTrialN=len(audNoiseConds)*nTrialPerStair*(2+1)*len(conflicts)



#print('given trials number',len(conds_obj.intens))

"""
matrix columns:
0: Standard durations


1: Rise conditions
2: Conflict A-V
"""
conditions_matrix =create_conditions_matrix(totalTrialN=totalTrialN,standards=[0.5], 
                                             background_levels=audNoiseConds, conflicts=conflicts)#
numberOfTrials=len(conditions_matrix)
print(f"Number of trials: {numberOfTrials}")
standardDurs = conditions_matrix[:, 0] # standard durations
audNoises = conditions_matrix[:, 1] # Background noise level conditions should change riseDur to backgroundNoise everywhere
conflictDurs = conditions_matrix[:, 2] # conflict durations


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
                'standardDur', 'audNoise','order', 'preDur', 'postDur', 'isiDur', 'trial_num',
                'total_audio_dur', 'delta_dur_percents', 'deltaDurS', 'testDurS', 'intensities',
                'current_stair', 'responses', 'is_correct', 'response_rts' , 'stair_num_reversal', 'stair_is_reversal', 'response_keys'
            'conflictDur', 
            'recordedOnsetVisualTest', 'recordedOffsetVisualTest', 'recordedDurVisualTest',
            'recordedOnsetVisualStandard', 'recordedOffsetVisualStandard', 'recordedDurVisualStandard','modalityPostCue']
print(f"column names: {len(column_names)}")
# create empty data matrix to save the data
exp_data=np.zeros((numberOfTrials+tolerance_trials, len(column_names)+1),dtype=object)

""" Staircase Setup"""
"----------------------------------------------------------------------------------------------------------------------------------------------------"
# Start the trial - response loop (there weill be)
""" Staircase Setup"""


# Create the staircases
max_trial_per_stair=nTrialPerStair

print(f'rise unique: {np.unique(audNoises)}')



## 2 up or 2 down staircases
stairCaseLonger2D1U=stairCase( max_level=max_level, max_trials=max_trial_per_stair, step_factor=stepFactor,  init_step=initStep,    method="2D1U" ) # '1D2U', '2D1U'                    )

stairCaseShorter2U1D=stairCase( max_level=max_level, max_trials=max_trial_per_stair, step_factor=stepFactor,  init_step=initStep,    method="1D2U" ) # '1D2U', '2D1U'                    )

## 3 up or 3 down 
stairCaseLonger=stairCase( max_level=max_level, max_trials=max_trial_per_stair, step_factor=stepFactor,  init_step=initStep,    method="1D1U" ) # '1D2U', '2D1U'                    )

stairCaseShorter=stairCase( max_level=max_level, max_trials=max_trial_per_stair, step_factor=stepFactor,  init_step=initStep,    method="1U1D" ) # '1D2U', '2D1U'                    )



stairCaseLapse = stairCase( max_level=max_level, max_trials=max_trial_per_stair, step_factor=stepFactor,  init_step=initStep,    method="lapse_rate" ) # '1D2U', '2D1U'                    )


if ExpTraining:
    all_staircases=[stairCaseLapse]
else:
    all_staircases=[stairCaseLapse, stairCaseLonger,stairCaseShorter,stairCaseLonger2D1U,stairCaseShorter2U1D]#, stairCaseLonger_b,stairCaseShorter_b,] 

#all_staircases=[stairCaseLapse ]#stairCaseLonger_b,stairCaseShorter_b,]
#all_staircases=[stairCaseShorter,stairCaseLonger,stairCaseLapse, ]#stairCaseLonger_b,stairCaseShorter_b,]
np.random.shuffle(all_staircases)
stopped_stair_count=0

def lapse_rate_cond_generate():
    lapse_deltas=[-0.9,0.9]# or the method of constant stim deltas= [0.05,0.20,0.10,0.40,-0.05,-0.20,-0.10,-0.40,]
    all_conds=[]
    for i in np.unique(standardDurs): # standard durations 1.3, 1.6, 1.9
        for j in np.unique(audNoises): # rise durations 0.05, 0.25
            for k in lapse_deltas: # lapse deltas -0.55, 0.55
                all_conds.append([i,j,k])

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



# region [rgba(2, 40, 30, 0.30)]
# Convert to list the standard durs and rise durs
standardDurs = standardDurs.tolist()
audNoises = audNoises.tolist()
conflictDurs = conflictDurs.tolist()
lapse_ended=False

# audio samplerate
sampleRate = 48000

# Set up fixation cross
fixation = visual.TextStim(win, text='+', color='white', height=deg2pix(1,monitor=win.monitor), pos=(0, 0))
#fixation.draw()
#win.flip()

#components for visual stimuli
visualStimSize=dva_to_px(size_in_deg=1.5,h=screen_height,d=screen_distance,r=sizeIs)


# Create Objects for the visual stimuli
visualStim=visual.Circle(win, radius=visualStimSize, fillColor=True, lineColor='black', colorSpace='rgb', units='pix',
                    pos=(0, 0), color='black')
visualStim.lineWidht=5
visualStim.color='black'
# startEndAudioCue=sound.Sound(value='A', sampleRate=sampleRate, stereo=True, volume=volume, name='startEndAudioCue')
# startEndAudioCue.setVolume(volume)
# startEndAudioCue.setSound('A', secs=0.033)


