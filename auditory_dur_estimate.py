"""
Experiment for the PhD Project 1.2: Audiovisual integration for duration estimation

Authors: Omer Yildiran and Michael Landy, New York University(NYU)
Supervisor: Michael Landy, New York University(NYU)
Experiment Coded by Omer Yildiran, PhD candidate at NYU

Start Date: 11/2024
Last Update: 11/2024

"""

# Importing Libraries
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, iohub, hardware
import os
import numpy as np
from numpy.random import choice as randchoice
from numpy.random import random, randint, normal, shuffle, choice as randchoice
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import sys  # to get file system encoding
from psychopy import prefs
import psychopy.iohub as io
from psychopy.iohub.util import hideWindow, showWindow
from psychopy.tools.monitorunittools import deg2pix, pix2deg
from psychopy import monitors
from psychopy.hardware import keyboard
import random
import scipy.io as sio
import pandas as pd

# audio prefs
prefs.general['audioLib'] = ['sounddevice', 'pyo', 'pygame']

# import condition generator
from create_conds_sound_dur import audioDurationGen
# import audio generator
from audio_cue_generator import AudioCueGenerator

# Set the experiment directory
exp_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(exp_dir)
"""          Experiment INFO Setup"""
# Experiment Information
expName = 'auditory_dur_estimate'
expInfo = {'participant': '', 'session number': '001'}
expInfo['date'] = data.getDateStr()  # add a simple timestamp
#dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
#core.quit() if dlg.OK == False else None
filename = exp_dir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])


#setup screen properties
monitor_options = {
    "asusZenbook14": { "sizeIs": 1024, "screen_width": 30.5, "screen_height": 18, "screen_distance": 40 },
    "labMon": { "sizeIs": 1024, "screen_width": 28, "screen_height": 28, "screen_distance": 60 },
    "macAir": { "sizeIs": 800, "screen_width": 25, "screen_height": 20, "screen_distance": 40 }
}

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
                    fullscr=True, monitor=myMon, units='pix', color="black", useFBO=True, screen=1, colorSpace='rgb')

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
gen = audioDurationGen(trial_per_condition=8,rise_conds=[0.20],intens=2.5)
conditions_matrix = gen.gen_duration_matrix()
std_durs = conditions_matrix[:, 0] # standard durations
delta_durs = conditions_matrix[:, 1] # relative durations
real_delta_durs = conditions_matrix[:, 2] # real relative durations
test_durs = conditions_matrix[:, 3] # test durations
rise_durs = conditions_matrix[:, 4] # rise conditions
orders = conditions_matrix[:, 5] # order of the test
intensities = conditions_matrix[:, 6] # intensity of the test
pre_durs = conditions_matrix[:, 7] # pre duration
post_durs = conditions_matrix[:, 8] # post duration
isi_durs = conditions_matrix[:, 9] # ISI duration


trial_num = conditions_matrix[:, -1] # trial number

# total stim durations
total_audio_durs=np.zeros(conditions_matrix.shape[0])

# Create a dataframe to store the results
# data = pd.DataFrame(columns=['standard_dur', 'delta_dur', 'delta_dur_adjusted', 'test_dur', 'rise_dur', 
#                              'test_order', 'intensity','pre_dur','post_dur','isi_dur','trial_num', 'total_dur','response', 'RT'])
# or add response and RT to the conditions matrix full of NaNs
conditions_matrix = np.column_stack((conditions_matrix, np.nan * np.zeros((len(conditions_matrix), 3)))) # total_audio_dur, response, RT

# # update the data frame with the conditions matrix
# data = pd.DataFrame(conditions_matrix, columns=['standard_dur', 'delta_dur', 'delta_dur_adjusted', 'test_dur', 'rise_dur',
#                             'test_order', 'intensity','pre_dur','post_dur','isi_dur','trial_num', 'total_dur','response', 'response_rt'])


# Initialize the stimulus component
sampleRate = 44100
audio_cue_gen = AudioCueGenerator(sampleRate=sampleRate)
# audio_stim=audio_cue_gen.whole_stimulus(test_durs[0], std_durs[0], "white", 3, rise_durs[0], order=orders[0])
# audio_stim_sound=sound.Sound(audio_stim, sampleRate=sampleRate, stereo=False)
#audio_stim_sound.play()
#print(intensities[0])
#audio_cue_gen.play_sound(audio_stim)

# Create general variables before the trial loop
endExpNow = False  # flag for 'escape' or other condition => quit the exp
mouse = event.Mouse(win=win,visible=False)
frameTolerance = 0.001  # how close to onset before 'same' frame
trialN=-1

# Responses
response = None
responses=np.zeros(conditions_matrix.shape[0])
response_rts=np.zeros(conditions_matrix.shape[0])
responseKeys = keyboard.Keyboard(backend='iohub')


# Start the trial - response loop (there weill be)
while trialN <= conditions_matrix.shape[0] and not endExpNow:
    trialN += 1
    # have a rest screen
    if trialN % 30 == 0 and trialN > 0:
        block_num = trialN // 30
        total_blocks = conditions_matrix.shape[0] // 30
        rest_text = f"""You can have a break now.
        You are in block {block_num} out of {total_blocks}.
        Press any key to continue."""
        rest_text_comp = visual.TextStim(win, text=rest_text, color='white', height=30)
        rest_text_comp.draw()
        win.flip()
        event.waitKeys()

    # Check if the experiment is over
    if endExpNow or event.getKeys(keyList=['escape']):
        core.quit()

    # Get the current trial
    std_dur = std_durs[trialN]
    delta_dur = delta_durs[trialN]
    real_delta_dur = real_delta_durs[trialN]
    test_dur = test_durs[trialN]
    rise_dur = rise_durs[trialN]
    order = orders[trialN]
    intensity = intensities[trialN]
    pre_dur = pre_durs[trialN]
    post_dur = post_durs[trialN]
    isi_dur = isi_durs[trialN]


    audio_stim=audio_cue_gen.whole_stimulus(test_dur, std_dur, "white", intensity, rise_dur, order,pre_dur,post_dur,isi_dur)
    total_dur_of_audio = len(audio_stim) / sampleRate
    total_audio_durs[trialN]=total_dur_of_audio
    audio_stim_sound=sound.Sound(audio_stim, sampleRate=sampleRate, stereo=False)

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
    print(f"Trial {trialN} out of {conditions_matrix.shape[0]}")
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
                #print(f"Measured audio duration: {t_dur}")
                #print(f"Aimed audio duration: {total_dur_of_audio}")

                audio_stim_sound.status = FINISHED
                continueRoutine = False
                audio_stim_sound.stop()
                break
        
        # check for quit (typically the Esc key)
        if event.getKeys(keyList=["escape"]):
            endExpNow = True
            audio_stim_sound.stop()
            break
        win.flip()

    # clear the screen
    fixation.setAutoDraw(False)

    waitingResponse = True

    # clear event buffer
    event.clearEvents(eventType='keyboard')
    responseKeys.clearEvents()

    # response timer
    t_start = globalClock.getTime()
    #response_text = f'Is sound {int(order)} longer than the {int(order+1)} sound?\n Press left for no and right for yes.'
    # Two interval forced choice response
    response_text = "First longer (<-) vs Second longer (->)"
    response_text_comp = visual.TextStim(win, text=response_text, color='white', height=30)
    
    while waitingResponse and not endExpNow:
        response_text_comp.draw()
        win.flip()

        response = responseKeys.getKeys(keyList=['left', 'right'], waitRelease=False)
        if response:
            responses[trialN] = 1 if response[0].name=='left' else 2  # 0 for no, 1 for yes
            response_rts[trialN] = globalClock.getTime() - t_start

            # add the response to the data
            conditions_matrix[trialN, -3] = round(total_dur_of_audio,6)
            conditions_matrix[trialN, -2] = responses[trialN]
            conditions_matrix[trialN, -1] = round(response_rts[trialN],6)

            # save conditions matrix as data
            data = pd.DataFrame(conditions_matrix, columns=['standard_dur', 'delta_dur', 'delta_dur_adjusted', 'test_dur', 'rise_dur',
                                        'test_order', 'intensity','pre_dur','post_dur','isi_dur','trial_num', 'total_dur','response', 'response_rt'])
            data.to_csv(filename + '.csv')

            # save as mat file with the same variable names
            sio.savemat(filename + '.mat', {'standard_dur': std_durs, 'delta_dur': delta_durs, 'delta_dur_adjusted': real_delta_durs,
                                            'test_dur': test_durs, 'rise_dur': rise_durs, 'test_order': orders, 'intensity': intensities,
                                            'pre_dur': pre_durs, 'post_dur': post_durs, 'isi_dur': isi_durs, 'trial_num': trial_num,
                                            'total_dur': conditions_matrix[:, -1], 'response': conditions_matrix[:, -2], 'response_rt': conditions_matrix[:, -3]})
            
            

            waitingResponse = False

        if endExpNow or event.getKeys(keyList=["escape"]):
            endExpNow = True
            break
    
# clear the screen
win.flip()

waitEndExp = True
# End of Experiment Routine
while waitEndExp:
    #save data
    # data.to_csv(filename + '.csv')
    # sio.savemat(filename + '.mat', {'data': conditions_matrix})
    # end of experiment text
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

        






