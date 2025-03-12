# generate noise with desired intensity
import numpy as np

def auditory_noise(intensity, dur, sample_rate):
    clear_high_noise = np.random.normal(0, 1, int(dur * sample_rate))
    clear_high_noise = clear_high_noise / np.max(np.abs(clear_high_noise))# 
    noise = clear_high_noise * intensity
    # set the boundary minimum amplitude to be +-2
    mask = np.abs(noise) < 2
    noise[mask] = 2 * np.sign(noise[mask])


