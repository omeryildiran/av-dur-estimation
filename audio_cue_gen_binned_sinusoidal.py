import numpy as np
from psychopy import sound, core

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

        return noise

    # Sinusoidal envelope for gradual transitions
    def sinusoidal_envelope(self, rise_dur, sample_rate=44100):
        """
        Generate a sinusoidal envelope in the time domain.
        :param rise_dur: Duration of the rising and falling phase (seconds).
        :param t_total: Total duration of the total envelope (seconds).
        :param sample_rate: Sampling rate (Hz).
        :return: Sinusoidal envelope as a numpy array.
        """
        def time_to_samples(t, sample_rate=44100):
            return int(t * sample_rate)

        # Convert durations and times to samples
        total_samples = time_to_samples(rise_dur*2, sample_rate)
        rise_samples = time_to_samples(rise_dur, sample_rate)
        fall_samples = rise_samples
        event_samples = 2 * rise_samples  # Total event duration: rise + fall

        # Ensure the event duration does not exceed the total duration
        if event_samples > total_samples:
            print("Event Duration: ", event_samples / sample_rate)
            print("Total Duration: ", total_samples / sample_rate)
            print("Rise Duration: ", rise_samples / sample_rate)
            raise ValueError("The event duration exceeds the total duration of the envelope.")

        # Create time axis for the envelope
        envelope = np.zeros(total_samples)

        # Rising phase
        t_rise = np.arange(rise_samples)
        envelope[0:rise_samples] = np.sin((np.pi / 2) * t_rise / rise_samples)

        # Falling phase
        t_fall_start = 0 + rise_samples
        t_fall = np.arange(fall_samples)
        envelope[t_fall_start:t_fall_start + fall_samples] = np.sin((np.pi / 2) * (1 - t_fall / fall_samples))

        return envelope



    def generate_binned_noise(self, total_dur=2.5, noise_type="white",                                                  
                                                bin_dur=0.1,  amp_var=0.5):
        
        # Generate base noise
        total_samples = int(total_dur * self.sample_rate)
        bin_samples = int(bin_dur * self.sample_rate)
        n_bins = total_samples // bin_samples
        noise_signal = np.zeros(total_samples)
        
        for i in range(n_bins):
            start_idx = i * bin_samples
            end_idx = start_idx + bin_samples
            # Generate raw white noise for this bin
            bin_noise = np.random.normal(0, 1, bin_samples)
            # Sample amplitude from a uniform or normal distribution
            # bin_amplitude = np.random.uniform(intensity-amp_var+.5, intensity + amp_var-.5)
            envelope = self.sinusoidal_envelope(bin_dur/2, self.sample_rate)+amp_var
            #envelope = envelope*(intensity)

            noise_signal[start_idx:end_idx] =envelope * bin_noise

        # Handle remaining samples if total_samples is not divisible by bin_samples
        remaining_samples = total_samples - n_bins * bin_samples
        if remaining_samples > 0:
            last_bin_noise = np.random.normal(0, 1, remaining_samples)
            bin_amplitude = intensity#np.random.uniform(intensity-amp_var+.5, intensity + amp_var-.5)
            noise_signal[-remaining_samples:] = bin_amplitude * last_bin_noise / np.max(np.abs(last_bin_noise))
        noise_signal = noise_signal / np.max(np.abs(noise_signal))


        return noise_signal
    
    def whole_stimulus_with_binning(self, test_dur, standard_dur, noise_type,
                                order, pre_dur=0.1, post_dur=0.1, isi_dur=0.3,
                                bin_dur=0.1, amp_mean=1, amp_var=0.5):
        """
        Generate the full stimulus including pre-cue, test sound, ISI, standard sound, and post-cue,
        with binned amplitude modulation for the test sound.
        """
        # 1. Generate pre-cue sound noise
        pre_cue_sound = self.generate_noise(dur=pre_dur, noise_type=noise_type)
        # 3. Generate interstimulus interval (ISI) noise
        isi_sound = self.generate_noise(dur=isi_dur, noise_type=noise_type)
        # 5. Generate post-cue sound noise
        post_cue_sound = self.generate_noise(dur=post_dur, noise_type=noise_type)

        # 2. Generate test sound with binning and raised cosine envelope
        test_sound = self.generate_binned_noise(
            total_dur=test_dur, noise_type=noise_type, 
            bin_dur=bin_dur,  amp_var=amp_var)


        # 4. Generate standard sound noise
        standard_sound = self.generate_binned_noise(
            total_dur=standard_dur, noise_type=noise_type, 
            bin_dur=bin_dur,  amp_var=0.5)

        

        # test_sound = test_sound * 2
        # standard_sound = standard_sound*(np.mean(abs(test_sound))/np.mean(abs(standard_sound)))
        standard_sound = standard_sound * 3
        test_sound = test_sound*(np.mean(abs(standard_sound))/np.mean(abs(test_sound)))

        print('\n')
        print(f"Mean of test sound: {np.mean(abs(test_sound)):.3f}, Min: {min(np.abs(test_sound)):.3f}, Max: {max(np.abs(standard_sound)):.3f}, Std: {np.std(test_sound):.3f}, Var: {np.var(test_sound):.3f}, Sum: {np.sum(test_sound):.3f}")
        print(f"Mean of standard sound: {np.mean(abs(standard_sound)):.3f}, Min: {min(np.abs(standard_sound)):.3f}, Max: {max(np.abs(standard_sound)):.3f}, Std: {np.std(standard_sound):.3f}, Var: {np.var(standard_sound):.3f}, Sum: {np.sum(standard_sound):.3f}")


        # Concatenate all segments depending on the order
        if order == 1:
            stim_sound = np.concatenate([pre_cue_sound, test_sound, isi_sound, standard_sound, post_cue_sound])
        elif order == 2:
            stim_sound = np.concatenate([pre_cue_sound, standard_sound, isi_sound, test_sound, post_cue_sound])
        else:
            raise ValueError("Invalid order value. Use 1 or 2.")
        
        # Normalize the signal
        #stim_sound = stim_sound / np.max(np.abs(stim_sound))
        stim_sound = stim_sound *0.3
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
amp_var = 0.5# to increase the perceptual noise in the test stimuli just modify the amplitude variance value. 
# the higher the value the more perceptual noise will be added to the test stimuli.
# Bin duration should stay the same as the standard stimuli.
pre_post_dur=0.4123


## PLot different sounds with different amplitude variance
def plot_sounds():
    plt.figure(figsize=(16, 10))
    for idx, amp_var in enumerate([0.5, 5]):
        stim_sound = audio_cue.whole_stimulus_with_binning(
        test_dur=test_dur , standard_dur=standard_dur, noise_type='white',order=order,
        pre_dur=pre_post_dur, post_dur=pre_post_dur, isi_dur=pre_post_dur,
        bin_dur=bin_dur, amp_var=amp_var
        )
        t=np.linspace(0, len(stim_sound) / audio_cue.sample_rate, len(stim_sound))
        # if idx in [0,1]:
        #     audio_cue.play_sound(stim_sound)
        
        plt.subplot(2, 1, idx + 1)
        plt.plot(t, stim_sound)
        plt.title(f"Amplitude Variance: {amp_var}")
        #plt.ylim(-5,5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.axvspan(pre_post_dur, pre_post_dur+test_dur, color="#5a0000", alpha=0.5, label="Unreliable signal")
        plt.axvspan(pre_post_dur+test_dur+pre_post_dur, pre_post_dur+test_dur+pre_post_dur+standard_dur, color="#1c3403", alpha=0.2, label="Reliable signal")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.show()
    
plot_sounds()