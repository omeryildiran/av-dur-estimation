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

    def generate_binned_noise_v2(self, total_dur=2.5, noise_type="white", 
                                bin_dur=0.1, min_amp=1, amp_var=0.5, sound_type='noise'):        # Calculate the number of bins and samples per bin
        bin_samples = int(bin_dur * self.sample_rate)
        n_bins = int(total_dur // bin_dur)  # Using floor division
        real_duration = n_bins * bin_dur

        # Generate Gaussian noise for each bin
        bin_v1 = np.random.randn(n_bins, bin_samples)

        # Generate amplitude noise from a uniform distribution
        amp_noise = np.random.uniform(min_amp, min_amp + amp_var, n_bins)

        # Apply amplitude to each bin
        bin_v2 = np.empty_like(bin_v1)
        for i in range(n_bins):
            bin_v2[i, :] = bin_v1[i, :] * amp_noise[i]

        # Flatten to create a single 1D array
        samples = bin_v2.flatten()

        # Handling remaining samples if total_dur is not a perfect multiple of bin_dur
        remaining_samples = int((total_dur - real_duration) * self.sample_rate)
        if remaining_samples > 0:
            # Generate noise for the remaining samples
            last_bin_noise = np.random.randn(remaining_samples)
            # Sample amplitude for the last bin
            last_bin_amp = np.random.uniform(min_amp, min_amp + amp_var)
            last_bin_noise *= last_bin_amp
            # Append to the main signal
            samples = np.concatenate([samples, last_bin_noise])

        return samples


    def whole_stimulus_with_binning(self, test_dur, standard_dur, noise_type, intensity,
                                order, pre_dur=0.1, post_dur=0.1, isi_dur=0.3,
                                bin_dur=0.1,  amp_var=0.5):
        """
        Generate the full stimulus including pre-cue, test sound, ISI, standard sound, and post-cue,
        with binned amplitude modulation for the test sound.
        """
        # 1. Generate pre-cue sound noise

        pre_cue_sound = self.generate_binned_noise_v2(total_dur=pre_dur,
                                                    noise_type=noise_type,
                                                    bin_dur=bin_dur,
                                                    min_amp=2,
                                                    amp_var=0.1)
          # 3. Generate interstimulus interval (ISI) noise
        isi_sound= self.generate_binned_noise_v2(total_dur=pre_dur,
                                                    noise_type=noise_type,
                                                    bin_dur=bin_dur,
                                                    min_amp=2,
                                                    amp_var=0.1)

        # 5. Generate post-cue sound noise
        post_cue_sound = self.generate_binned_noise_v2(total_dur=post_dur,
                                                    noise_type=noise_type,
                                                    bin_dur=bin_dur,
                                                    min_amp=2,
                                                    amp_var=0.1)
        jitter_sound = np.zeros(int(0.015 * self.sample_rate))

        noiseScaler=0.9
        pre_cue_sound=pre_cue_sound*noiseScaler
        isi_sound=isi_sound*noiseScaler
        post_cue_sound=post_cue_sound*noiseScaler

        
        # 2. Generate test sound with binning and raised cosine envelope
        min_amp_unreliable = 4
        min_amp_reliable = 4

        standard_sound= self.generate_binned_noise_v2(
                                                    total_dur=standard_dur, 
                                                    noise_type=noise_type, 
                                                    bin_dur=bin_dur, 
                                                    min_amp=min_amp_reliable, 
                                                    amp_var=0.1,
                                                    sound_type='reliable_signal')
        
        
        # 4. Generate standard sound noise
        test_sound= self.generate_binned_noise_v2(total_dur=test_dur, 
                                                    noise_type=noise_type, 
                                                    bin_dur=bin_dur, 
                                                    min_amp=min_amp_unreliable, 
                                                    amp_var=amp_var,
                                                    sound_type='unreliable_signal')
        

        test_sound = test_sound*(np.mean(abs(standard_sound))/np.mean(abs(test_sound)))
        
        print('\n')
        print(f"Mean of test sound: {np.mean(abs(test_sound)):.3f}, Min: {min(np.abs(test_sound)):.3f}, Max: {max(np.abs(standard_sound)):.3f}, Std: {np.std(test_sound):.3f}, Var: {np.var(test_sound):.3f}, Sum: {np.sum(test_sound):.3f}")
        print(f"Mean of standard sound: {np.mean(abs(standard_sound)):.3f}, Min: {min(np.abs(standard_sound)):.3f}, Max: {max(np.abs(standard_sound)):.3f}, Std: {np.std(standard_sound):.3f}, Var: {np.var(standard_sound):.3f}, Sum: {np.sum(standard_sound):.3f}")

        # Concatenate all segments depending on the order
        if order == 1:
            stim_sound = np.concatenate([jitter_sound,pre_cue_sound, test_sound, isi_sound, standard_sound, post_cue_sound,jitter_sound])
        elif order == 2:
            stim_sound = np.concatenate([jitter_sound,pre_cue_sound, standard_sound, isi_sound, test_sound, post_cue_sound,jitter_sound])
        else:
            raise ValueError("Invalid order value. Use 1 or 2.")

        stim_sound=stim_sound*0.15
        return stim_sound



import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

audio_cue = AudioCueGenerator(sampleRate=48000)
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
from psychopy.sound import backend_ptb as ptb
print(ptb.getDevices(kind='output'))
prefs.hardware['audioDevice'] = 1

# Generate whole stimulus with binning
test_dur = 1
standard_dur = 0.5
noise_type = "white"
intensity = 9
order = 1
bin_dur = 0.1# 100 ms bins
amp_var = 0.2# to increase the perceptual noise in the test stimuli just modify the amplitude variance value. 
# the higher the value the more perceptual noise will be added to the test stimuli.
# Bin duration should stay the same as the standard stimuli.
pre_post_dur=0.4123


## PLot different sounds with different amplitude variance
def plot_sounds():
    plt.figure(figsize=(16, 10))
    for idx, amp_var in enumerate([0.5, 5]):
        stim_sound = audio_cue.whole_stimulus_with_binning(
        test_dur=test_dur , standard_dur=standard_dur, noise_type='white', intensity=9, order=order,
        pre_dur=pre_post_dur, post_dur=pre_post_dur, isi_dur=pre_post_dur,
        bin_dur=bin_dur, amp_var=amp_var
        )
        t=np.linspace(0, len(stim_sound) / audio_cue.sample_rate, len(stim_sound))
        # if idx in [0,1]:
        #     audio_cue.play_sound(stim_sound)
        
        plt.subplot(3, 1, idx + 1)
        plt.plot(t, stim_sound)
        plt.title(f"Amplitude Variance: {amp_var}")
        plt.ylim(-5,5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.axvspan(pre_post_dur, pre_post_dur+test_dur, color="#5a0000", alpha=0.5, label="Unreliable signal")
        plt.axvspan(pre_post_dur+test_dur+pre_post_dur, pre_post_dur+test_dur+pre_post_dur+standard_dur, color="#1c3403", alpha=0.2, label="Reliable signal")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.show()
    
plot_sounds()