"""
Coded by Omer Yildiran subject to attribution- noncommercial 4.0 International (CC BY-NC 4.0) license
Start Date: 12/2023

"""

import numpy as np

def generate_beep_sound(duration, sample_rate, beep_frequency):
    t = np.arange(0, duration, 1.0 / sample_rate)
    beep_signal = np.sin(2.0 * np.pi * beep_frequency * t)
    return beep_signal

def create_panning_beep_array(duration, sample_rate, beep_frequency):
    t = np.arange(0, duration, 1.0 / sample_rate)
    pan_factor = np.linspace(-1, 1, len(t))  # Linear panning from left to right

    left_channel = (1 - pan_factor) * generate_beep_sound(duration, sample_rate, beep_frequency)
    right_channel = pan_factor * generate_beep_sound(duration, sample_rate, beep_frequency)

    stereo_array = np.column_stack((left_channel, right_channel))
    return stereo_array


def create_panning_beep_array(duration, sample_rate, beep_frequency, pan_exponent=2):
    t = np.arange(0, duration, 1.0 / sample_rate)
    fade_dur_ind = len(t) // 3  # Index for half of the sound duration

    # Exponential panning from left to right for the first half of the sound
    pan_factor = np.linspace(-1, 1, fade_dur_ind) # ** pan_exponent
    # Keep playing on the panned side for the second half of the sound
    pan_factor = np.concatenate((pan_factor, np.full(len(t) - fade_dur_ind, pan_factor[-1])))

    left_channel = (1 - pan_factor) * generate_beep_sound(duration, sample_rate, beep_frequency)
    right_channel = pan_factor * generate_beep_sound(duration, sample_rate, beep_frequency)

    stereo_array = np.column_stack((left_channel, right_channel))
    return stereo_array

def create_stereo_sound(duration, sample_rate, beep_frequency,channel='left'):
    t = np.arange(0, duration, 1.0 / sample_rate)
    if channel=='left':
        left_channel = generate_beep_sound(duration, sample_rate, beep_frequency)  # Full sound on the left channel
        right_channel = np.zeros(len(t))
    elif channel=='right':
        left_channel = np.zeros(len(t))
        right_channel = generate_beep_sound(duration, sample_rate, beep_frequency)

    stereo_array = np.column_stack((left_channel, right_channel))
    return stereo_array


def generate_white_noise(duration, sample_rate):
    return np.random.normal(0, 1, int(duration * sample_rate))

def generate_a_note(duration, sample_rate):
    frequency = 440  # Frequency of A4
    t = np.arange(0, duration, 1.0 / sample_rate)
    a_note = np.sin(2.0 * np.pi * frequency * t)
    return a_note


def positional_audio(duration, sample_rate, relPosX,relPosY=0.5):
    #sample_rate=sample_rate*((relPosY+0.5)*2)
    t=np.arange(0, duration, 1.0 / sample_rate)
    #sound=generate_white_noise(duration, sample_rate)
    # Generate a note with frequency based on relPosY
    frequency = 440 * (2 ** relPosY)  # Frequency of A4 multiplied by an octave shift
    sound = np.sin(2.0 * np.pi * frequency * t)
    
    if relPosX<0:
        left_channel = (abs(relPosX)+0.5) * sound
        right_channel = (0.5-abs(relPosX)) * sound
    elif relPosX>=0:
        left_channel = (0.5-abs(relPosX)) * sound
        right_channel = (abs(relPosX)+0.5) * sound
    stereo_array = np.column_stack((left_channel, right_channel))
    return stereo_array


# def play_audio(array, sample_rate):
#     sd.play(array, samplerate=sample_rate)
#     sd.wait()


# import sounddevice as sd

# if __name__ == "__main__":
#     duration = 2 # seconds
#     sample_rate = 44100  # Hz
#     beep_frequency = 440  # Hz

#     stereo_array = positional_audio(duration, sample_rate,-0.5)

#     play_audio(stereo_array, sample_rate)

# # save audio
# import soundfile as sf
# # left diyor
# sf.write('left.wav', stereo_array, sample_rate)
