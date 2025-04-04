import numpy as np
from psychopy import sound, core
from scipy.signal import butter, filtfilt
from scipy.signal import butter, sosfilt

class AudioCueGenerator:
    def __init__(self, sampleRate=44100):
        
        self.sample_rate = sampleRate  # in Hz

    def generate_noise(self,dur=2,  noise_type="white"):
        """
        Generate various types of noise (white, pink, brownian, blue, violet).
        
        :param dur: Duration of the noise (in seconds).
        :param sample_rate: Sampling rate (Hz).
        :param noise_type: Type of noise ('white', 'pink', 'brownian', 'blue', 'violet').
        :return: The generated noise signal.
        """
        # Generate raw white noise
        noise_signal = np.random.normal(0, 1, int(dur * self.sample_rate))
        
        if noise_type == "white":
            # White noise: no filtering needed
            return noise_signal / np.max(np.abs(noise_signal))
        elif noise_type == "pink":
            # Pink noise: 1 / sqrt(f) filter
            num_samples = len(noise_signal)
            freqs = np.fft.rfftfreq(num_samples, d=1/self.sample_rate)
            fft_values = np.fft.rfft(noise_signal)
            pink_filter = np.where(freqs > 0, 1 / np.sqrt(freqs), 0)  # Avoid division by zero
            filtered_fft_values = fft_values * pink_filter
            noise_signal = np.fft.irfft(filtered_fft_values, n=num_samples)

        elif noise_type == "tone":
            # Frequency of A note in Hz
            frequency = 440.0
            # Time array
            t = np.linspace(0, dur, int(dur * self.sample_rate), endpoint=False)
            # Generate sine wave
            sine_wave = np.sin(2 * np.pi * frequency * t)
            # Normalize the sine wave to range [-1, 1]
            sine_wave /= np.max(np.abs(sine_wave))
            return sine_wave
        else:
            raise ValueError("Unsupported noise type. Use 'white', 'pink', 'brownian', 'blue', or 'violet'.")
        
        # Normalize final signal to range [-1, 1]
        noise_signal = noise_signal / np.max(np.abs(noise_signal))
        return noise_signal


    def play_sound(self, sound_array):
        duration=len(sound_array)/self.sample_rate
        beep = sound.Sound(value=sound_array, sampleRate=self.sample_rate, stereo=True)
        beep.play()
        core.wait(duration)

    def gaussian_pdf(self, x, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)



    def generate_standard_sound(self, total_dur=2.5, noise_type="white", intensity=3.0):
        # Generate base noise
        noise_signal = self.generate_noise(dur=total_dur, noise_type=noise_type)
        noise = noise_signal * intensity #envelope
        # mask = np.abs(noise) < 2
        # noise[mask] = 2 * np.sign(noise[mask])
        return noise

    # Apply lowpass filtering (similar to MATLAB `lowpass(samples, 700, sampRate)`)
    def lowpass_filter(self,signal, cutoff, sample_rate, order=4):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
        return filtfilt(b, a, signal)

    def lowpass_filter(self, signal, cutoff, sample_rate, window_size=None):
        """
        Apply a moving average lowpass filter.
        
        :param signal: Input signal
        :param cutoff: Cutoff frequency (Hz)
        :param sample_rate: Sampling rate (Hz)
        :param window_size: Size of the moving average window (in samples)
        :return: Filtered signal
        """
        if window_size is None:
            # Set window size based on cutoff frequency
            window_size = int(sample_rate / cutoff)
        
        # Apply moving average
        window = np.ones(window_size) / window_size
        return np.convolve(signal, window, mode='same')

    def lowpass_filter(self, signal, cutoff, sample_rate, order=4):
        """
        Apply a lowpass filter using second-order sections (SOS).
        
        :param signal: Input signal
        :param cutoff: Cutoff frequency (Hz)
        :param sample_rate: Sampling rate (Hz)
        :param order: Filter order
        :return: Filtered signal
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        return sosfilt(sos, signal)

    # # Apply lowpass filtering (similar to MATLAB `lowpass(samples, 700, sampRate)`)
    # def lowpass_filter(self,signal, cutoff, sample_rate, order=4):
    #     nyquist = 0.5 * sample_rate
    #     normal_cutoff = cutoff / nyquist
    #     b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    #     return filtfilt(b, a, signal)

    def generate_binned_noise(self, total_dur=2.5, noise_type="white", 
                                                    intensity=3.0, 
                                                    bin_dur=0.1, amp_mean=1, amp_var=0.5):
        
        # Generate base noise
        total_samples = int(total_dur * self.sample_rate)
        bin_samples = int(bin_dur * self.sample_rate)
        n_bins = total_samples // bin_samples
        noise_signal = np.zeros(total_samples)
        # Handle remaining samples if total_samples is not divisible by bin_samples
        remaining_samples = total_samples - n_bins * bin_samples

        for i in range(n_bins):
            start_idx = i * bin_samples
            end_idx = start_idx + bin_samples
            # Generate raw white noise for this bin
            bin_noise = np.random.normal(0, 1, bin_samples)
            # Sample amplitude from a uniform or normal distribution
            # bin_amplitude = np.random.uniform(intensity-amp_var+.5, intensity + amp_var-.5)
            if i == 0 :
                bin_amplitude = max(np.random.normal(intensity, amp_var),intensity)
            elif i==n_bins-1 and remaining_samples == 0:
                bin_amplitude = max(np.random.normal(intensity, amp_var),intensity)
            else:
                # or with normal
                bin_amplitude = max(np.random.normal(intensity, amp_var),intensity+amp_var)
                #bin_amplitude = np.random.uniform(intensity-amp_var+.5, intensity + amp_var-.5)
            # # Scale the noise in this bin
            noise_signal[start_idx:end_idx] = bin_amplitude * bin_noise / np.max(np.abs(bin_noise))


        if remaining_samples > 0:
            last_bin_noise = np.random.normal(0, 1, remaining_samples)
            bin_amplitude = max(np.random.normal(intensity, amp_var),intensity)
            noise_signal[-remaining_samples:] = bin_amplitude * last_bin_noise / np.max(np.abs(last_bin_noise))
        noise_signal = noise_signal / np.max(np.abs(noise_signal))

        # Apply the envelope to the noise
        noise_signal = noise_signal * (intensity) #*  envelope

        # reshape to get a 1D signal sample
        noise_signal = noise_signal.reshape(-1)
        # low pass filter

        #noise_signal=self.lowpass_filter(noise_signal,700,self.sample_rate,4)
        # low pass filter
        #noise_signal = np.convolve(noise_signal, np.ones(100)/100, mode='same')


        return noise_signal
    


    def whole_stimulus_with_binning(self, test_dur, standard_dur, noise_type, intensity,
                                order, pre_dur=0.1, post_dur=0.1, isi_dur=0.3,
                                bin_dur=0.1, amp_mean=1, amp_var=0.5):
        """
        Generate the full stimulus including pre-cue, test sound, ISI, standard sound, and post-cue,
        with binned amplitude modulation for the test sound.
        """
        # 1. Generate pre-cue sound noise
        pre_cue_sound = self.generate_noise(dur=pre_dur, noise_type=noise_type)

        # pre_cue_sound = self.generate_binned_noise(total_dur=pre_dur,
        #                                             noise_type=noise_type,
        #                                             intensity=1,
        #                                             bin_dur=bin_dur,
        #                                             min_amp=1,
        #                                             amp_var=0.1)
  
        # 4. Generate standard sound noise
        standard_sound = self.generate_standard_sound(total_dur=standard_dur, 
                                                    noise_type=noise_type, 
                                                    intensity=intensity)
        
        # standard_sound= self.generate_binned_noise(total_dur=standard_dur, 
        #                                             noise_type=noise_type, 
        #                                             intensity=intensity, 
        #                                             bin_dur=bin_dur, 
        #                                             min_amp=3, 
        #                                             amp_var=0.1)
        # self.standard_bounds = [min(standard_sound), max(standard_sound)]

        # 2. Generate test sound with binning and raised cosine envelope
        test_sound = self.generate_binned_noise(
            total_dur=test_dur, noise_type="white", 
            intensity=intensity, bin_dur=bin_dur, amp_mean=amp_mean, amp_var=amp_var)
        # test_sound = self.generate_binned_noise(
        #     total_dur=test_dur, noise_type=noise_type, 
        #     intensity=intensity, bin_dur=bin_dur, min_amp=3, amp_var=amp_var)

        # 3. Generate interstimulus interval (ISI) noise
        isi_sound = self.generate_noise(dur=isi_dur, noise_type=noise_type)
        # isi_sound= self.generate_binned_noise(total_dur=pre_dur,
        #                                             noise_type=noise_type,
        #                                             intensity=1,
        #                                             bin_dur=bin_dur,
        #                                             min_amp=1,
        #                                             amp_var=0.1)

        # 5. Generate post-cue sound noise
        post_cue_sound = self.generate_noise(dur=post_dur, noise_type=noise_type)
        # post_cue_sound = self.generate_binned_noise(total_dur=post_dur,
        #                                             noise_type=noise_type,
        #                                             intensity=1,
        #                                             bin_dur=bin_dur,
        #                                             min_amp=1,
        #                                             amp_var=0.1)
        jitter_sound = np.zeros(int(0.01 * self.sample_rate))
        # Concatenate all segments depending on the order
 
        if order == 1:
            stim_sound = np.concatenate([jitter_sound,pre_cue_sound, test_sound, isi_sound, standard_sound, post_cue_sound,jitter_sound])
        elif order == 2:
            stim_sound = np.concatenate([jitter_sound,pre_cue_sound, standard_sound, isi_sound, test_sound, post_cue_sound,jitter_sound])
        else:
            raise ValueError("Invalid order value. Use 1 or 2.")
        # Normalize the signal
        # lowpass filter stim
        stim_sound=self.lowpass_filter(stim_sound,700,self.sample_rate,4)
        #stim_sound=np.concatenate([stim_sound])
        #print(max(stim_sound))
        stim_sound = stim_sound / np.max(np.abs(stim_sound))
        return stim_sound



import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

audio_cue = AudioCueGenerator(sampleRate=48000)
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
from psychopy.sound import backend_ptb as ptb
#print(ptb.getDevices(kind='output'))
#prefs.hardware['audioDevice'] = 3

# Generate whole stimulus with binning
test_dur = 1
standard_dur = 1
noise_type = "white"
intensity = 4
order = 1
bin_dur = 0.1# 100 ms bins
amp_mean = 0
amp_var = 5.5# to increase the perceptual noise in the test stimuli just modify the amplitude variance value. 
# the higher the value the more perceptual noise will be added to the test stimuli.
# Bin duration should stay the same as the standard stimuli.

pre_post_dur=0.4
stim_sound = audio_cue.whole_stimulus_with_binning(
    test_dur, standard_dur, noise_type, intensity, order, 
    pre_dur=pre_post_dur, post_dur=pre_post_dur, isi_dur=pre_post_dur, 
    bin_dur=bin_dur, amp_mean=amp_mean, amp_var=amp_var
)

#Play and plot the stimulus
audio_cue.play_sound(stim_sound)

# Plot the stimulus
t = np.linspace(0, len(stim_sound) / audio_cue.sample_rate, len(stim_sound))
plt.figure(figsize=(10, 4))
plt.plot(t, stim_sound)
plt.title("Stimulus (Low-rel test+Standard) Sound with Binning")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.axvspan(pre_post_dur, pre_post_dur+test_dur, color="blue", alpha=0.5, label="Test Sound")
plt.axvspan(pre_post_dur+test_dur+pre_post_dur, pre_post_dur+test_dur+pre_post_dur+standard_dur, color="purple", alpha=0.5, label="Standard Sound")

plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
plt.show()

