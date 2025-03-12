import numpy as np
from scipy.signal import hilbert

def generate_S2_stimulus(total_dur=20,          # total duration in seconds (for the full signal)
                         stim_dur=4,            # duration (in seconds) of the extracted stimulus
                         sample_rate=44100,     # sampling rate in Hz
                         peak_am=4.0,           # desired peak AM frequency (mode) in Hz 
                         sigma=0.5,             # standard deviation of log(f) (smaller -> more concentrated power)
                         target_rms=0.1,        # desired RMS of the final stimulus (arbitrary units)
                         seed=None):
    """
    Generate an S2 auditory stimulus with a designated peak AM frequency and temporal regularity.
    
    The processing pipeline is as follows:
      1. Define a lognormal amplitude spectrum:
            A(f) = exp( -((ln(f - b) - mu)^2/(2*sigma^2)) )
         with b = -1.2813 * peak_am and mu = ln(peak_am) + sigma^2.
         (A(0) is set to zero.)
         
      2. Create a complex spectrum with the above amplitude and random phases.
      
      3. Apply an inverse FFT to obtain a 20-s time series.
      
      4. Use the Hilbert transform to extract an amplitude envelope.
      
      5. Modulate a white-noise (LNN) carrier with the envelope.
      
      6. Extract the middle stim_dur seconds from the modulated signal.
      
      7. Normalize the RMS of the extracted stimulus.
      
    Parameters:
      total_dur   : float
                    Total duration of the generated signal (in seconds).
      stim_dur    : float
                    Duration (in seconds) of the extracted stimulus segment.
      sample_rate : int
                    Sampling rate in Hz.
      peak_am     : float
                    The desired peak (mode) of the AM spectrum in Hz.
      sigma       : float
                    The standard deviation of the log of the spectrum (smaller values yield a spectrum 
                    with power narrowly concentrated around peak_am, i.e. higher temporal regularity).
      target_rms  : float
                    The desired root-mean-square value of the output stimulus.
      seed        : int or None
                    Seed for reproducibility.
                    
    Returns:
      stimulus    : 1D numpy array containing the final amplitude-modulated noise stimulus.
    """
    if seed is not None:
        np.random.seed(seed)
        
    total_samples = int(total_dur * sample_rate)
    
    # === Step 1: Build the target AM spectrum using a lognormal function ===
    # Frequency vector for the positive frequencies (rFFT frequencies)
    freqs = np.fft.rfftfreq(total_samples, d=1/sample_rate)
    
    # Compute the lognormal parameters:
    mu = np.log(peak_am) + sigma**2      # ensures mode = exp(mu - sigma^2) equals peak_am
    b = -1.2813 * peak_am                # x-shift parameter
    
    # Compute the amplitude spectrum
    amp_spectrum = np.zeros_like(freqs)
    # Avoid f - b <= 0 (set amplitude to zero for those frequencies)
    valid = (freqs - b) > 0
    amp_spectrum[valid] = np.exp( - ((np.log(freqs[valid] - b) - mu)**2) / (2 * sigma**2) )
    # (Optionally, you might apply additional scaling/multiplication here if needed.)
    
    # === Step 2: Impose random phases ===
    # For rfft, the first element (DC) is real, and if total_samples is even, the Nyquist frequency is real.
    random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, size=len(freqs)))
    random_phases[0] = 1.0  # DC component: no phase
    if total_samples % 2 == 0:
        random_phases[-1] = 1.0  # ensure Nyquist is real if present
    
    # Create the complex spectrum
    complex_spectrum = amp_spectrum * random_phases

    # === Step 3: Inverse FFT to obtain a time-domain AM signal ===
    # Use the inverse real FFT to get a real-valued time series of length total_samples.
    am_signal = np.fft.irfft(complex_spectrum, n=total_samples)
    
    # === Step 4: Extract the amplitude envelope ===
    # Although the IFFT yields a signal that fluctuates around zero, we want a strictly nonnegative envelope.
    # The Hilbert transform provides an analytic signal from which we take the magnitude.
    analytic_signal = hilbert(am_signal)
    envelope = np.abs(analytic_signal)
    
    # === Step 5: Generate the low-noise noise (LNN) carrier and modulate it ===
    # Here we generate a white noise carrier. (It is “low-noise” in the sense that its envelope is flat.)
    carrier = np.random.randn(total_samples)
    modulated_signal = envelope * carrier
    
    # === Step 6: Extract the middle stim_dur segment ===
    # For example, for a 20-s signal extracting the middle 4 s means taking the samples
    # starting at (total_dur - stim_dur)/2 seconds.
    start_idx = int(((total_dur - stim_dur) / 2) * sample_rate)
    end_idx = start_idx + int(stim_dur * sample_rate)
    stimulus = modulated_signal[start_idx:end_idx]
    
    # === Step 7: Normalize the stimulus RMS ===
    current_rms = np.sqrt(np.mean(stimulus**2))
    if current_rms > 0:
        stimulus = stimulus * (target_rms / current_rms)
    
    return stimulus


import random
from psychopy import sound, core

# =============================================================================
# Example usage:
if __name__ == '__main__':
    # Generate one S2 stimulus with a peak AM frequency of 4 Hz and sigma of 0.5.
    s2 = generate_S2_stimulus(total_dur=1, stim_dur=1, sample_rate=44100,
                                peak_am=4.0, sigma=0.5, target_rms=0.1 )

    s1=generate_S2_stimulus(total_dur=1, stim_dur=1, sample_rate=44100,
                                peak_am=4.0, sigma=2.5, target_rms=0.1)
    
    pure_noise = np.random.randn(round(44100*0.5))
    pure_noise= pure_noise/np.max(np.abs(pure_noise))
    pure_noise = pure_noise*0.2 
    whole_stim = np.concatenate((pure_noise, s2, pure_noise,s1,pure_noise))
    #pure_noise= sound.Sound(value=pure_noise, sampleRate=44100,stereo=True)
    #pure_noise.play()
    # (For instance, you could write the stimulus to a WAV file using soundfile or similar.)
    # import soundfile as sf
    # sf.write('S2_stimulus.wav', stim, 44100)
        # play the stimulus
    a=sound.Sound(value=whole_stim, sampleRate=44100,stereo=True)
    a.play()
    core.wait(1)
    # For demonstration, print some basic info:
    print("S2 stimulus generated:")
    print(" - Duration (s):", len(whole_stim)/44100)
    print(" - RMS:", np.sqrt(np.mean(whole_stim**2)))
    # plot the stimulus
    import matplotlib.pyplot as plt
    t= np.arange(len(whole_stim)) / 44100
    plt.plot(t,whole_stim)
    plt.title('S2 Stimulus Waveform')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.show()




# =============================================================================