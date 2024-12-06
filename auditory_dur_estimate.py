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
#dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
#core.quit() if dlg.OK == False else None
filename = exp_dir + os.sep + u'data\%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'],)


#setup screen properties
monitor_options = {
    "asusZenbook14": { "sizeIs": 1024,    "screen_width": 30.5,   "screen_height": 18,    "screen_distance": 40 },
    "labMon": {  "sizeIs": 1024,        "screen_width": 28,        "screen_height": 28,        "screen_distance": 60 },
    "macAir": {  "sizeIs": 1024,        "screen_width": 25,        "screen_height": 20,        "screen_distance": 40  }}
monitorSpecs=monitor_options["macAir"]
sizeIs=monitorSpecs["sizeIs"] # 1024
screen_width=monitorSpecs["screen_width"] #31 asuSs 14 # actual size of my screen in cm is 28x17
screen_height=monitorSpecs["screen_height"] #28 # 16.5 asus
screen_distance=monitorSpecs["screen_distance"] #60 # 57 asus
# define monitor
myMon=monitors.Monitor('asusMon', width=screen_width, distance=57)
#myMon.setSizePix((sizeIs, sizeIs))
selectedMon=myMon
win = visual.Window(size=(sizeIs,sizeIs),
                    fullscr=False,  monitor=myMon, units='pix',  color="black", useFBO=True, screen=1, colorSpace='rgb')

# Set window properties
win.monitor.setWidth(screen_width)
win.monitor.setDistance(screen_distance)

refreshRate=win.getActualFrameRate()


# create the conditions matrix
gen = audioDurationGen(trial_per_condition=1,rise_conds=[0.1,2.5])
conditions_matrix = gen.gen_duration_matrix()
std_durs = conditions_matrix[:, 0]
delta_durs = conditions_matrix[:, 1]
real_delta_durs = conditions_matrix[:, 2]
test_durs = conditions_matrix[:, 3]
rise_conds = conditions_matrix[:, 4]
orders = conditions_matrix[:, 5]


