

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
from dva_to_pix import arcmin_to_px, dva_to_px



# Set the experiment directory
exp_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(exp_dir)
"""          Experiment INFO Setup"""
# Experiment Information
# ask for participant and session number
expInfo = {'participant': '', 'session number': '001'}
if ExpTesting:
    expInfo['participant'] = 'test'
# present a dialogue to change params
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
core.quit() if dlg.OK == False else None
# save expInfo to a file
expInfo['date'] = data.getDateStr()  # add a simple timestamp
filename = exp_dir + os.sep + u'dataAuditory/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])


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
                fullscr=fullScreen, monitor=myMon, units='pix', color="gray", useFBO=False, screen=0, colorSpace='rgb')



# Set window properties
win.monitor.setWidth(screen_width)
win.monitor.setDistance(screen_distance)
win.monitor.setSizePix((sizeIs, sizeIs))

#refreshRate=win.getActualFrameRate()
#print('Frame Rate: ', refreshRate)
#frame_dur = 1.0/round(refreshRate, 2) if refreshRate else 1.0/60.0


# frame rate
refreshRate=win.getActualFrameRate()
refreshRate=120 if refreshRate is None else refreshRate
print('Refresh Rate: ', refreshRate)
frame_dur = 1.0/round(refreshRate, 2) 
frameRate=refreshRate





welcome_text="""Welcome to the experiment!
Press any key to start the experiment."""
welcome_text_comp = visual.TextStim(win, text=welcome_text, color='white', height=30)
welcome_text_comp.draw()
win.flip()
event.waitKeys() # wait for a key press
#core.quit() if event.getKeys(keyList=['escape']) else None

