# Import components
# import condition generator
from create_conds_staircase import expConds
# import the staircase
from my_staircase import stairCase
from conditions_av import create_conditions_matrix


#Handy timers
globalClock = core.Clock()
trialClock = core.Clock()

# Retrieve the conditions

audNoiseConds= [0.1] 
maxIntensityBurst=5
nTrialPerStairPerCondition=70
nTrialPerStair=len(audNoiseConds)*len(conflicts)*nTrialPerStairPerCondition
totalTrialN=len(audNoiseConds)*nTrialPerStair*(2+1)*len(conflicts)

#totalTrialN=len(audNoiseConds)*nTrialPerStair*(2+1)*len(conflicts)
bin_dur=0.1

conds_obj = expConds(trial_per_condition=nTrialPerStairPerCondition*8,
                             rise_conds=rise_conds,
                             standard_durations=[0.5],
                             intens=maxIntensityBurst) 
#print('given trials number',len(conds_obj.intens))
#total_trials=(conds_obj.trial_per_condition)*2*4
"""
matrix columns:
0: Standard durations
1: Rise conditions
2: Order of test
3: Pre duration
4: Post duratione
5: ISI duration
6: Trial number
7: Total audio duration
8: Response
9: Is correct
10: Response RT
"""
conditions_matrix = conds_obj.gen_duration_matrix()

standardDurs = conditions_matrix[:, 0] # standard durations
riseDurs = conditions_matrix[:, 1] # rise conditions
orders = conditions_matrix[:, 2] # order of the test
preDurs = conditions_matrix[:, 3] # pre duration
postDurs = conditions_matrix[:, 4] # post duration
isiDurs = conditions_matrix[:, 5] # ISI duration
trial_num = conditions_matrix[:, 6] # trial number

# total stim durations
total_stim_durs=[]
# add columns for total audio duration, response, is_correct, RT
conditions_matrix = np.column_stack((conditions_matrix, np.nan * np.zeros((len(conditions_matrix), 4)))) # total_audio_dur, response, is_correct, RT
"""
7: 
7: Total audio duration
8: Response
9: Is correct
10: Response RT
"""


# Create general variables before the trial loop
endExpNow = False  # flag for 'escape' or other condition => quit the exp
mouse = event.Mouse(win=win,visible=False)
frameTolerance = 0.001  # how close to onset before 'same' frame
trialN=-1

# Responses
response = None
tolerance_trials=100
responses=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
is_corrects=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
response_rts=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
response_keys=np.zeros(conditions_matrix.shape[0]+tolerance_trials)

responseKeys = keyboard.Keyboard(backend='PTB')

current_stairs=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
deltaDurs=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
testDurs=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
realDeltaDurs=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
intensities=np.zeros(conditions_matrix.shape[0]+tolerance_trials)

# create empty data matrix to save the data
exp_data=np.zeros((conditions_matrix.shape[0]+tolerance_trials, 19),dtype=object)

# region [rgba(2, 40, 30, 0.30)]
# Start the trial - response loop (there weill be)
""" Staircase Setup"""
stepFactor=0.67
initStep=0.2
maxReversals=100
max_level=0.9
initLevel=0.65

# Create the staircases
max_trial_per_stair=nTrialPerStair#total_trials//5

print(f'rise unique: {np.unique(riseDurs)}')
stairCaseLonger = stairCase(init_level=initLevel, init_step=initStep, method="3D1U",  step_factor=stepFactor, max_level=max_level+1, max_reversals=maxReversals, max_trials=max_trial_per_stair, 
                            sigma_level=None,)
stairCaseLonger2D1U = stairCase(init_level=initLevel, init_step=initStep, method="2D1U",  step_factor=stepFactor, max_level=max_level+1, max_reversals=maxReversals, 
                                max_trials=max_trial_per_stair, sigma_level=None,)
stairCaseShorter = stairCase(init_level=initLevel, init_step=initStep, method="3U1D",step_factor=stepFactor,
                              max_level=max_level, max_reversals=maxReversals, max_trials=max_trial_per_stair, sigma_level=None,)
stairCaseShorter2U1D = stairCase(init_level=initLevel, init_step=initStep, method="2U1D",step_factor=stepFactor, 
                                 max_level=max_level, max_reversals=maxReversals, max_trials=max_trial_per_stair, sigma_level=None,)

stairCaseLapse = stairCase(init_level=0.6, init_step=initStep, method="lapse_rate", step_factor=stepFactor, max_level=max_level, max_reversals=maxReversals) # no need for it just decide on deltas

if ExpTraining:
    all_staircases=[stairCaseLapse]
else:
    all_staircases=[stairCaseLapse, stairCaseLonger,stairCaseShorter,stairCaseLonger2D1U,stairCaseShorter2U1D]#, stairCaseLonger_b,stairCaseShorter_b,] 
#all_staircases=[stairCaseShorter,stairCaseLonger,stairCaseLapse, ]#stairCaseLonger_b,stairCaseShorter_b,]
np.random.shuffle(all_staircases)
stopped_stair_count=0

def lapse_rate_cond_generate():
    lapse_deltas=[-0.8,0.8]
    all_conds=[]
    for i in np.unique(standardDurs): # standard durations 1.3, 1.6, 1.9
        for j in np.unique(riseDurs): # rise durations 0.05, 0.25
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

# Convert to list the standard durs and rise durs
standardDurs = standardDurs.tolist()
riseDurs = riseDurs.tolist()
lapse_ended=False


# Set up fixation cross
fixation = visual.TextStim(win, text='+', color='white', height=deg2pix(1,monitor=win.monitor), pos=(0, 0))
fixation.draw()
win.flip()

#components for visual stimuli
visualStimSize=dva_to_px(size_in_deg=1.5,h=screen_height,d=screen_distance,r=sizeIs)
