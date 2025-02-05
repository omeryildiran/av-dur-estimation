"""
Experiment for the PhD Project 1.2: Audiovisual integration for duration estimation

Authors: Omer Yildiran and Michael Landy, New York University(NYU)
Supervisor: Michael Landy, New York University(NYU)
Experiment Coded by Omer Yildiran, PhD candidate at NYU

Start Date: 11/2024
Last Update: 11/2024

"""
from psychopy.sound import backend_ptb as ptb
import matplotlib.pyplot as plt
# Importing Libraries
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, iohub, hardware
import os
import numpy as np
from numpy.random import choice as randchoice
from numpy.random import random, randint, normal, shuffle, choice as randchoice
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import sys  # to get file system encoding
#from psychopy import prefs/Users/omer/Library/CloudStorage/GoogleDrive-omerulfaruk97@gmail.com/My Drive/MyReposDrive/obsidian_Notes/Landy Omer Re 1/av-dur-estimation/exp_auditory_dur_estimate_binnedAudio.py
from psychopy import prefs
import psychopy.iohub as io
from psychopy.iohub.util import hideWindow, showWindow
from psychopy.tools.monitorunittools import deg2pix, pix2deg
from psychopy import monitors
from psychopy.hardware import keyboard
import random
import scipy.io as sio
import pandas as pd
from my_staircase import stairCase

# audio prefs
#prefs.general['audioLib'] = ['sounddevice', 'pyo', 'pygame']
print(ptb.getDevices(kind='output'))
prefs.hardware['audioLib'] = ['PTB']
prefs.hardware['audioDevice'] = 1

#prefs.general['audioLatencyMode'] = 4

# import condition generator
from create_conds_staircase import audioDurationGen
# import audio generator
from audi_cue_gen_bin_filtered_v3_LRA_fixed import AudioCueGenerator
# import the staircase
from my_staircase import stairCase

# Set the experiment directory
exp_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(exp_dir)
"""          Experiment INFO Setup"""
# Experiment Information
# ask for participant and session number
expName = 'auditory_dur_estimate_bin_stair'
expInfo = {'participant': '', 'session number': '001'}
# present a dialogue to change params
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
core.quit() if dlg.OK == False else None
# save expInfo to a file
expInfo['date'] = data.getDateStr()  # add a simple timestamp
filename = exp_dir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])


#setup screen properties
monitor_options = {
    "asusZenbook14": { "sizeIs": 1024, "screen_width": 30.5, "screen_height": 18, "screen_distance": 40 },
    "labMon": { "sizeIs": 1024, "screen_width": 28, "screen_height": 28, "screen_distance": 60 },
    "macAir": { "sizeIs": 800, "screen_width": 25, "screen_height": 20, "screen_distance": 40 }
}
def adj_delta(weber_fraction,delta,standard):
    # example usage
    # adj_delta(0.15,delta_dur_percent,standard_dur)
    return standard*delta/(weber_fraction+1)

monitorSpecs = monitor_options["macAir"]
sizeIs = monitorSpecs["sizeIs"]
screen_width = monitorSpecs["screen_width"]
screen_height = monitorSpecs["screen_height"]
screen_distance = monitorSpecs["screen_distance"]

# Define monitor
myMon = monitors.Monitor('macAir', width=screen_width, distance=screen_distance)
myMon.setSizePix((sizeIs, sizeIs))
# Create window
win = visual.Window(size=(sizeIs, sizeIs),
                    fullscr=True, monitor=myMon, units='pix', color="black", useFBO=True, screen=0, colorSpace='rgb')

# Set window properties
win.monitor.setWidth(screen_width)
win.monitor.setDistance(screen_distance)
win.monitor.setSizePix((sizeIs, sizeIs))

refreshRate=win.getActualFrameRate()


# frame rate
refreshRate=win.getActualFrameRate()
print('Frame Rate: ', refreshRate)
frame_dur = 1.0/round(refreshRate, 2) if refreshRate else 1.0/60.0

# Handy timers
globalClock = core.Clock()
trialClock = core.Clock()

# Set up welcome screen
welcome_text="""Welcome to the experiment!
Press any key to start the experiment."""
welcome_text_comp = visual.TextStim(win, text=welcome_text, color='white', height=30)
welcome_text_comp.draw()
win.flip()
event.waitKeys() # wait for a key press
#core.quit() if event.getKeys(keyList=['escape']) else None
    

# Set up fixation cross
fixation = visual.TextStim(win, text='+', color='white', height=deg2pix(1,monitor=win.monitor), pos=(0, 0))
fixation.draw()
win.flip()
#event.waitKeys() # wait for a key press


# Retrieve the conditions
# create the conditions matri x
rise_conds=[3.5,0.2]
intens=9
n_trial_per_condition=50
conds_obj = audioDurationGen(trial_per_condition=n_trial_per_condition*2,
                             rise_conds=rise_conds,
                             standard_durations=[0.8],
                             intens=intens)
bin_dur=0.1
#print('given trials number',len(conds_obj.intens))
#total_trials=(conds_obj.trial_per_condition)*2*4
"""
matrix columns:
0: Standard durations
1: Rise conditions
2: Order of test
3: Pre duration
4: Post duration
5: ISI duration
6: Trial number
7: Total audio duration
8: Response
9: Is correct
10: Response RT
"""
conditions_matrix = conds_obj.gen_duration_matrix()

standard_durs = conditions_matrix[:, 0] # standard durations
rise_durs = conditions_matrix[:, 1] # rise conditions
orders = conditions_matrix[:, 2] # order of the test
pre_durs = conditions_matrix[:, 3] # pre duration
post_durs = conditions_matrix[:, 4] # post duration
isi_durs = conditions_matrix[:, 5] # ISI duration
trial_num = conditions_matrix[:, 6] # trial number

# total stim durations
total_audio_durs=[]

# add columns for total audio duration, response, is_correct, RT
conditions_matrix = np.column_stack((conditions_matrix, np.nan * np.zeros((len(conditions_matrix), 4)))) # total_audio_dur, response, is_correct, RT
"""
7: 
7: Total audio duration
8: Response
9: Is correct
10: Response RT

"""

# Initialize the stimulus component
sampleRate = 44100
audio_cue_gen = AudioCueGenerator(sampleRate=sampleRate)


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
delta_durs=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
test_durs=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
real_delta_durs=np.zeros(conditions_matrix.shape[0]+tolerance_trials)
intensities=np.zeros(conditions_matrix.shape[0]+tolerance_trials)

# create empty data matrix to save the data
exp_data=np.zeros((conditions_matrix.shape[0]+tolerance_trials, 19),dtype=object)

# region [rgba(2, 40, 30, 0.30)]
# Start the trial - response loop (there weill be)
""" Staircase Setup"""
stepFactor=0.67
initStep=0.2
maxReversals=100
max_level=0.8

# Create the staircases
max_trial_per_stair=n_trial_per_condition#total_trials//5

print(f'rise unique: {np.unique(rise_durs)}')


stair_training = stairCase(init_level=0.6, init_step=initStep, method="training", step_factor=stepFactor, max_level=max_level, max_reversals=maxReversals) # no need for it just decide on deltas

all_staircases=[stair_training]
np.random.shuffle(all_staircases)
stopped_stair_count=0

def lapse_rate_cond_generate():
    lapse_deltas=[-0.7,0.7]
    all_conds=[]
    for i in np.unique(standard_durs): # standard durations 1.3, 1.6, 1.9
        for j in np.unique(rise_durs): # rise durations 0.05, 0.25
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
print(f"there are maximum {len(standard_durs)} normal trials in total of  {max_trial_per_stair*4} staircase trials" )

# Convert to list the standard durs and rise durs
standard_durs = standard_durs.tolist()
rise_durs = rise_durs.tolist()
lapse_ended=False

while not endExpNow and stopped_stair_count!=(len(all_staircases)):
    print(f"Trial {trialN} out of {total_trial_num}")
    
    
    stair=stair_training 
    if trialN%len(all_staircases)==0:
        shuffle(all_staircases)
    current_stair=stair.method

    # Get the current trial
    if current_stair=="lapse_rate" or current_stair=="training":
        if not lapse_ended:
            if lapse_rate_conds.shape[0]>0:
                standard_dur=lapse_rate_conds[0][0]
                rise_dur=lapse_rate_conds[0][1]
                delta_dur_percent=lapse_rate_conds[0][2]
                lapse_rate_conds=lapse_rate_conds[1:]
            else:
                stopped_stair_count+=1
                lapse_ended=True
                stair.stair_stopped=True
                print('lapse rate ended')
                continue
        else:
            continue

    else:
        standard_dur = standard_durs.pop()
        rise_dur = rise_durs.pop()

        delta_dur_percent = round(stair.next_trial(),5) # delta dur in terms of percentage of the standard duration (0.1, 0.2, 0.3, 0.4, 0.5)

    delta_dur_s= round(standard_dur*delta_dur_percent,5)  # delta dur in terms of seconds
    test_dur_s = round(standard_dur + delta_dur_s,5)

    
    delta_durs[trialN] = delta_dur_percent
    test_durs[trialN] = test_dur_s
    real_delta_durs[trialN] = delta_dur_s
    
    # Assign values calculate directly now
    order = int(np.random.choice([1,2])) # or orders[trialN]
    intensity = conds_obj.intens
    pre_dur = np.random.uniform(0.25, 0.55)
    post_dur = np.random.uniform(0.25, 0.55)
    isi_dur = np.random.uniform(0.3, 0.75)

    print(f'Current Stair: {current_stair}, Standard Dur: {standard_dur}, Test Dur: {test_dur_s}, Rise Dur: {rise_dur},Test in: {order} place,  Delta Dur: {delta_dur_percent},  delta_dur_s: {delta_dur_s}')

    #audio_stim=audio_cue_gen.whole_stimulus(test_dur_s, standard_dur, "white", intensity, rise_dur, order,pre_dur,post_dur,isi_dur) # create the audio stimulus
    
    audio_stim = audio_cue_gen.whole_stimulus_with_binning(
    test_dur=test_dur_s, standard_dur=standard_dur, noise_type='white', intensity=intens, 
    order=order, 
    pre_dur=pre_dur, post_dur=post_dur, isi_dur=isi_dur, 
    bin_dur=bin_dur, amp_mean=0, amp_var=rise_dur)    
    
    total_dur_of_audio = len(audio_stim) / sampleRate # calculate the total duration of the audio stimulus
    total_audio_durs.append(total_dur_of_audio) # save the total duration of the audio stimulus
    
    audio_stim_sound=sound.Sound(value=audio_stim, sampleRate=sampleRate, stereo=True)


    # For testing purposes uncomment the following line
    #audio_stim_sound=sound.Sound('A', sampleRate=sampleRate, stereo=False,secs=0.0001) 

    trialN += 1
    # have a rest screen
    if trialN % 30 == 0 and trialN > 0:
        block_num = trialN // 30
        total_blocks = total_trial_num // 30
        rest_text = f"""You can have a break now.
        You are in block {block_num} out of {total_blocks}.
        Press any key to continue."""
        rest_text_comp = visual.TextStim(win, text=rest_text, color='white', height=30)
        rest_text_comp.draw()
        win.flip()
        # comment for testing
        event.waitKeys()
    
    if current_stair=="training" and trialN>0:
        # draw correct or incorrect text
        if is_correct:
            feedback_text = "Correct!"
            color='green'
        else:
            feedback_text = "Incorrect!"
            color='red'
        feedback_text_comp = visual.TextStim(win, text=feedback_text, color=color, height=50)
        feedback_text_comp.draw()
        win.flip()
        # comment for testing
        #event.waitKeys()
        core.wait(0.5)
    # Check if the experiment is over
    if endExpNow or event.getKeys(keyList=['escape']):
        core.quit()


    # clear the screen and wait for 100 ms
    win.flip()
    core.wait(0.1)
    
    # timers
    trialClock.reset()
    globalClock.reset()
    t_start=globalClock.getTime()
    
    win.flip(clearBuffer=True)

    continueRoutine = True
    """ Run the trial routine""" 
    while continueRoutine:
        t = globalClock.getTime()
        # draw the fixation cross
        fixation.setAutoDraw(True)

        # audio stimulus
        if audio_stim_sound.status == NOT_STARTED and t >= 0:
            audio_stim_sound.play()

            audio_stim_sound.status = STARTED
            t_start = globalClock.getTime()
        elif audio_stim_sound.status == STARTED:
            if audio_stim_sound.isPlaying == False:
                t_dur=t-t_start
                audio_stim_sound.status = FINISHED
                continueRoutine = False
                #audio_stim_sound.stop()
                #break
        
        # check for quit (typically the Esc key)
        if event.getKeys(keyList=["escape"]):
            endExpNow = True
            audio_stim_sound.stop()
            break
        win.flip()

    stair_num_reversal=stair.reversals
    stair_is_reversal=stair.is_reversal
    # endregion

    # region [rgba(40, 10, 3, 0.30)]

    """ SAVE TRIAL DATA BEFORE RESPONSE"""
    exp_data[trialN, 0] = standard_dur
    exp_data[trialN, 1] = rise_dur
    exp_data[trialN, 2] = order
    exp_data[trialN, 3] = pre_dur
    exp_data[trialN, 4] = post_dur
    exp_data[trialN, 5] = isi_dur
    exp_data[trialN, 6] = trialN
    exp_data[trialN, 7] = round(total_dur_of_audio,6)
    exp_data[trialN, 8] = delta_dur_percent
    exp_data[trialN, 9] = delta_dur_s
    exp_data[trialN, 10] = test_dur_s
    exp_data[trialN, 11] = intensity
    
    # stair data
    exp_data[trialN, 12] = current_stair
    exp_data[trialN,16]= stair_num_reversal # num of reversals
    exp_data[trialN,17]= stair_is_reversal # is this trial a reversal (1: reversal 0: not reversal)


    # endregion

    # clear the screen
    fixation.setAutoDraw(False)
    waitingResponse = True
    # clear event buffer
    event.clearEvents(eventType='keyboard')
    responseKeys.clearEvents()
    # response timer
    t_start = globalClock.getTime()
    # Two interval forced choice response
    response_text = "First longer (<-) vs Second longer (->)"
    response_text_comp = visual.TextStim(win, text=response_text, color='white', height=30)
    
    # region [rgba(40, 10, 30, 0.30)]
    #plt.plot(audio_stim)
    #plt.show()
    
    while waitingResponse and not endExpNow:
        response_text_comp.draw()
        win.flip()

        response = responseKeys.getKeys(keyList=['left', 'right'], waitRelease=False)
        class simKeys:
            def __init__(self, keyList, rtRange):
                self.name=np.random.choice(keyList)
                self.rt = np.random.choice(np.linspace(rtRange[0], rtRange[1])/1000)

        # #  for testing purposesSimulate a key press
        # if not response:
        #     #fake a response responseKeys 
        #     response=[simKeys(keyList=['left', 'right'], rtRange=[200,1000])]

            
        # record the response
        if response:
            responses[trialN] = 1 if response[0].name=='left' else 2  # 1 for first longer, 2 for second longer
            is_correct=(test_dur_s>standard_dur and responses[trialN]==order) or (test_dur_s<standard_dur and responses[trialN]!=order) # 1 for correct, 0 for incorrect
            print(f"Response: {response[0].name} - Order: {order} - Is Correct: {is_correct}")
            is_corrects[trialN] = is_correct
            response_rts[trialN] = globalClock.getTime() - t_start

            exp_data[trialN, 13] = responses[trialN] # response as num
            exp_data[trialN, 14] = is_correct 
            exp_data[trialN, 15] = round(response_rts[trialN],6)
            exp_data[trialN, 18] = response[0].name # response key as name

            # constrain the conditions matrix to the max current trial
            exp_data_saved = exp_data[:trialN+1, :]
         
            # save exp_data as DataFrame
            data_saved = pd.DataFrame(exp_data_saved, columns=[
                'standard_dur', 'rise_dur', 'order', 'pre_dur', 'post_dur', 'isi_dur', 'trial_num',
                'total_audio_dur', 'delta_dur_percents', 'delta_dur_s', 'test_dur_s', 'intensities',
                'current_stair', 'responses', 'is_correct', 'response_rts' , 'stair_num_reversal', 'stair_is_reversal', 'response_keys'
            ])            
            data_saved.to_csv(filename + '.csv')

            # save as mat file with the same variable names
        # save as mat file with the same variable names
            sio.savemat(
                filename + '.mat', 
                {
                    'standard_dur': exp_data_saved[:, 0],
                    'rise_dur': exp_data_saved[:, 1],
                    'order': exp_data_saved[:, 2],
                    'pre_dur': exp_data_saved[:, 3],
                    'post_dur': exp_data_saved[:, 4],
                    'isi_dur': exp_data_saved[:, 5],
                    'trial_num': exp_data_saved[:, 6],
                    'total_audio_dur': exp_data_saved[:, 7],
                    'delta_dur_percents': exp_data_saved[:, 8],
                    'delta_dur_s': exp_data_saved[:, 9],
                    'test_dur_s': exp_data_saved[:, 10],
                    'intensities': exp_data_saved[:, 11],
                    'current_stair': exp_data_saved[:, 12],
                    'responses': exp_data_saved[:, 13],
                    'is_correct': exp_data_saved[:, 14],
                    'response_rts': exp_data_saved[:, 15],
                    'stair_num_reversal': exp_data_saved[:, 16],
                    'stair_is_reversal': exp_data_saved[:, 17],
                    'response_keys': exp_data_saved[:, 18]
                    
                }
            )

            waitingResponse = False
        
            # update staircase
            stair.update_staircase(is_correct)

            if stair.stair_stopped:
                stopped_stair_count+=1

        if endExpNow or event.getKeys(keyList=["escape"]):
            endExpNow = True
            break
    

# endregion
# clear the screen
win.flip()

waitEndExp = True
# End of Experiment Routine
while waitEndExp:
    #save data
    # data.to_csv(filename + '.csv')
    # sio.savemat(filename + '.mat', {'data': conditions_matrix})
    # end of experiment text
    print("end of exp")
    end_text = """End of the experiment.
    Thank you for your participation. Press any key to exit."""
    end_text_comp = visual.TextStim(win, text=end_text, color='white', height=30)
    end_text_comp.draw()
    win.flip()
    event.waitKeys()
    waitEndExp = False
core.quit()
    
        
            





                


        # if sound has stopped, break the loop
    


    # """Run Response Routine"""
    # # Clear the event buffer before entering the waitResponse phase
    # event.clearEvents(eventType='keyboard')
    # responseKeys.keys = []
    # responseKeys.rt = []

        