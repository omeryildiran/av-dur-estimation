from scipy.signal import butter, filtfilt
from scipy.signal import butter, sosfilt

def broadband_filter( signal, low_cut, high_cut, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoffs = [low_cut / nyquist, high_cut / nyquist]
    sos = butter(order, normal_cutoffs, btype="band", analog=False, output="sos")
    return sosfilt(sos, signal)