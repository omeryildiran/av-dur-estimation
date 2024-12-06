"""
Coded by Omer Yildiran subject to attribution-noncommercial 4.0 International (CC BY-NC 4.0) license
Start Date: 12/2023
Last Update: 11/2024

"""

import numpy as np
from psychopy import sound, core

class AudioCueGenerator:
    def __init__(self, sampleRate=44100):
        
        self.sample_rate = sampleRate  # in Hz

    def generate_beep_sound(self, dur=2,  beep_frequency=440):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        beep_signal = np.sin(2.0 * np.pi * beep_frequency * t)
        return beep_signal

    def create_panning_beep_array(self, dur=2,  beep_frequency=440, pan_exponent=2):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        fade_dur_ind = len(t) // 3

        pan_factor = np.linspace(-1, 1, fade_dur_ind)
        pan_factor = np.concatenate((pan_factor, np.full(len(t) - fade_dur_ind, pan_factor[-1])))

        left_channel = (1 - pan_factor) * self.generate_beep_sound(dur, self.sample_rate, beep_frequency)
        right_channel = pan_factor * self.generate_beep_sound(dur, self.sample_rate, beep_frequency)

        stereo_array = np.column_stack((left_channel, right_channel))
        return stereo_array

    def create_stereo_sound(self, dur=2,  beep_frequency=440, channel='left'):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        if channel == 'left':
            left_channel = self.generate_beep_sound(dur, self.sample_rate, beep_frequency)
            right_channel = np.zeros(len(t))
        elif channel == 'right':
            left_channel = np.zeros(len(t))
            right_channel = self.generate_beep_sound(dur, self.sample_rate, beep_frequency)

        stereo_array = np.column_stack((left_channel, right_channel))
        return stereo_array

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

        elif noise_type == "brownian":
            # Brownian noise: cumulative sum of white noise
            noise_signal = np.cumsum(noise_signal)
            
        elif noise_type == "blue":
            # Blue noise: sqrt(f) filter
            num_samples = len(noise_signal)
            freqs = np.fft.rfftfreq(num_samples, d=1/self.sample_rate)
            fft_values = np.fft.rfft(noise_signal)
            blue_filter = np.sqrt(freqs)
            blue_filter[freqs == 0] = 0  # Avoid modifying DC component
            filtered_fft_values = fft_values * blue_filter
            noise_signal = np.fft.irfft(filtered_fft_values, n=num_samples)

        elif noise_type == "violet":
            # Violet noise: f filter
            num_samples = len(noise_signal)
            freqs = np.fft.rfftfreq(num_samples, d=1/self.sample_rate)
            fft_values = np.fft.rfft(noise_signal)
            violet_filter = freqs
            violet_filter[freqs == 0] = 0  # Avoid modifying DC component
            filtered_fft_values = fft_values * violet_filter
            noise_signal = np.fft.irfft(filtered_fft_values, n=num_samples)

        else:
            raise ValueError("Unsupported noise type. Use 'white', 'pink', 'brownian', 'blue', or 'violet'.")
        
        # Normalize final signal to range [-1, 1]
        noise_signal = noise_signal / np.max(np.abs(noise_signal))
        return noise_signal


    def generate_a_note(self, dur=2, frequency=440):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        a_note = np.sin(2.0 * np.pi * frequency * t)
        return a_note

    def positional_audio(self, dur=2, relPosX=0, relPosY=0.5):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        frequency = 440 * (2 ** relPosY)
        sound = np.sin(2.0 * np.pi * frequency * t)

        if relPosX < 0:
            left_channel = (abs(relPosX) + 0.5) * sound
            right_channel = (0.5 - abs(relPosX)) * sound
        elif relPosX >= 0:
            left_channel = (0.5 - abs(relPosX)) * sound
            right_channel = (abs(relPosX) + 0.5) * sound

        stereo_array = np.column_stack((left_channel, right_channel))
        return stereo_array

    def play_sound(self, sound_array):
        duration=len(sound_array)/self.sample_rate
        beep = sound.Sound(value=sound_array, sampleRate=self.sample_rate, stereo=True)
        beep.play()
        core.wait(duration)

    def gaussian_pdf(self, x, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def sum_of_gaussians(self,x, mu_list, sigma, intensity=1.0):
        gaussians = [self.gaussian_pdf(x, mu, sigma) for mu in mu_list]
        combined_gaussian = np.zeros_like(x)
        
        for i in range(len(mu_list) - 1):
            mask1 = (x >= mu_list[i]) & (x <= mu_list[i + 1])
            mask2 = x > mu_list[i + 1]
            gaussians[i][mask1] = np.max(gaussians[i]) #+ np.random.normal(0, 0.05, len(gaussians[i][mask1]))
            gaussians[i][mask2] = 0
            gaussians[i + 1][~mask2] = 0
            combined_gaussian += gaussians[i] + gaussians[i + 1]
            # normalize the combined gaussian
            combined_gaussian /= np.max(combined_gaussian)

        return combined_gaussian * intensity

    def generate_gaussian_envelope(self,total_dur, mu_list, sigma, intensity=1.0, peak_amplitude=1.0):
        envelope = np.zeros(int(total_dur * self.sample_rate))
        x = np.linspace(0, total_dur, len(envelope))
        envelope = self.sum_of_gaussians(x, mu_list, sigma, intensity) # Generate the Gaussian envelope
        envelope = np.convolve(envelope, np.ones(int(peak_amplitude)) / peak_amplitude, mode='same')
        envelope += 1
        return envelope

    def smooth_envelope(self, envelope, window_size=50):
        window = np.hanning(window_size)
        return np.convolve(envelope, window / window.sum(), mode="same")

    # Raised cosine envelope for precise transition and perceptual salience
    def raised_cosine_envelope(self,t_start, rise_dur, peak_dur, t_total, sample_rate=44100):
        """
        Generate a raised-cosine envelope in the time domain.
        :param t_start: Start time of the event (seconds).
        :param rise_dur: Duration of the rising phase (seconds).
        :param peak_dur: Duration of the event peak (seconds).
        :param t_total: Total duration of the total envelope (seconds).
        :param sample_rate: Sampling rate (Hz).
        :return: Raised-cosine envelope as a numpy array.
        """
        def time_to_samples(t, sample_rate=44100):
            return int(t * sample_rate)

        # Convert durations and times to samples
        total_samples = time_to_samples(t_total, sample_rate)
        start_sample = time_to_samples(t_start, sample_rate)
        rise_samples = time_to_samples(rise_dur, sample_rate)
        peak_samples = time_to_samples(peak_dur, sample_rate)
        fall_samples = rise_samples
        event_samples = 2 * rise_samples + peak_samples

        # Ensure the event duration does not exceed the total duration
        if start_sample + event_samples > total_samples:
            raise ValueError("The event duration exceeds the total duration of the envelope.")

        # Create time axis for the envelope
        envelope = np.zeros(total_samples)

        # Rising phase
        t_rise = np.arange(rise_samples)
        envelope[start_sample:start_sample + rise_samples] = 0.5 * (1 - np.cos(np.pi * t_rise / rise_samples))

        # Peak phase
        t_peak_start = start_sample + rise_samples
        envelope[t_peak_start:t_peak_start + peak_samples] = 1.0

        # Falling phase
        t_fall_start = t_peak_start + peak_samples
        t_fall = np.arange(fall_samples)
        envelope[t_fall_start:t_fall_start + fall_samples] = 0.5 * (1 + np.cos(np.pi * t_fall / fall_samples))

        return envelope

    def low_reliability_test_sound(self, total_dur=2.5, noise_type="pink", 
                                rise_dur=0.2, intensity=3.0):
        """
        Generate a low-reliability noise signal with a raised-cosine envelope.
        :param total_dur: Total duration of the noise (seconds).
        :param signal_start: Start time of the event (seconds).
        :param noise_type: Type of noise ('white', 'pink', etc.).
        :param rise_dur: Duration of the rising phase of the envelope (seconds).
        :param peak_dur: Duration of the peak phase of the envelope (seconds).
        :param intensity: Peak amplitude of the envelope.
        :return: Modulated noise signal as a numpy array.
        """
        peak_dur = total_dur - 2*rise_dur  # Ensure the peak phase does not exceed the total duration
        # Generate base noise
        noise_signal = self.generate_noise(dur=total_dur, noise_type=noise_type)

        # Generate the raised-cosine envelope
        envelope = self.raised_cosine_envelope(0, rise_dur, peak_dur, total_dur, self.sample_rate)
        
        # Scale the envelope
        envelope = envelope * (intensity - 1) + 1  # Ensure baseline is 1, and peak reaches the desired intensity

        # Apply the envelope to the noise
        modulated_noise = noise_signal * envelope
        return modulated_noise
        
    def whole_stimulus(self, test_dur, standard_dur, noise_type, intensity, rise_dur,order):

        # 1. generate pre-cue sound noise for 0.1 seconds
        pre_cue_sound = self.generate_noise(dur=np.random.uniform(0.05, 0.2), noise_type=noise_type)

        # 2. generate test sound noise for 2.5 seconds
        test_sound = self.low_reliability_test_sound(total_dur=test_dur, 
                                                    rise_dur=rise_dur, 
                                                    noise_type=noise_type, 
                                                    intensity=intensity)
        # 3. interstimulus interval noise 
        isi_sound = self.generate_noise(dur=np.random.uniform(0.2, 0.6), noise_type=noise_type)
        
        # 4. generate standard sound noise
        standard_sound = self.low_reliability_test_sound(total_dur=standard_dur, 
                                                    rise_dur=0, 
                                                    noise_type=noise_type, 
                                                    intensity=intensity)
        # 5. generate post-cue sound noise for 0.1 seconds
        post_cue_sound = self.generate_noise(dur=np.random.uniform(0.05, 0.2), noise_type=noise_type) 

        # concatenate all bins into one continuous signal depending on the order
        if order == 1:
            stim_sound = np.concatenate([pre_cue_sound, test_sound, isi_sound, standard_sound, post_cue_sound])
        elif order == 2:
            stim_sound = np.concatenate([pre_cue_sound, standard_sound, isi_sound, test_sound, post_cue_sound])
        else:
            raise ValueError("Invalid order value. Use 1 or 2.")
        
        # normalize the signal
        # stim_sound = stim_sound / np.max(np.abs(stim_sound))
        
        return stim_sound
    




# import for plotting
import matplotlib.pyplot as plt


# Example usage with raised-cosine envelope
audio_cue = AudioCueGenerator(sampleRate=96000)

# generate whole stim
test_dur = 1
standard_dur = 1
noise_type = "white"
intensity = 2.5
rise_dur = 0.2
order = 1

stim_sound = audio_cue.whole_stimulus(test_dur, standard_dur, noise_type, intensity, rise_dur, order)
audio_cue.play_sound(stim_sound)

t=np.linspace(0, len(stim_sound)/44100, len(stim_sound))

# Plot the sound
plt.figure(figsize=(10, 4))
plt.plot(t,stim_sound, label="Modulated Noise")
plt.show
"""
# start a timer
import time
start = time.time()

# Parameters
total_duration = 1.3  # Total duration of the sound
rise_duration = 0   # Duration of the rising phase
noise_type = "pink"  # Noise type
intensity = 3       # Peak amplitude of the envelope

# Generate low-reliability test sound
test_sound = audio_cue.low_reliability_test_sound(total_dur=total_duration, 
                                                  rise_dur=0.3, 
                                                  noise_type=noise_type, 
                                                  intensity=2.5)

standard_sound = audio_cue.low_reliability_test_sound(total_dur=1, 
                                                  rise_dur=0, 
                                                  noise_type=noise_type, 
                                                  intensity=4)
# generate pre-cue sound noise for 0.1 seconds
pre_cue_sound = audio_cue.generate_noise(dur=np.random.uniform(0.15, 0.25), noise_type=noise_type)
post_cue_sound = audio_cue.generate_noise(dur=np.random.uniform(0.15, 0.25), noise_type=noise_type)
isi_sound = audio_cue.generate_noise(dur=np.random.uniform(0.5, 1), noise_type=noise_type)

# concatenate all bins into one continuous signal
stim_sound = np.concatenate([pre_cue_sound,test_sound, isi_sound,standard_sound,post_cue_sound])
# normalize the signal
stim_sound = stim_sound / np.max(np.abs(stim_sound))
t_end = time.time() - start
print("Time to generate sound: ", t_end)
audio_cue.play_sound(stim_sound)

t=np.linspace(0, len(stim_sound)/44100, len(stim_sound))

# Plot the sound
plt.figure(figsize=(10, 4))
plt.plot(t,stim_sound, label="Modulated Noise")
# shade the event regions
plt.axvspan(0, len(pre_cue_sound) / 44100, color='gray', alpha=0.3, label='Pre Cue')
plt.axvspan(len(pre_cue_sound) / 44100, (len(pre_cue_sound) + len(test_sound)) / 44100, color='red', alpha=0.3, label='Test Audio')
plt.axvspan((len(pre_cue_sound) + len(test_sound)) / 44100, (len(pre_cue_sound) + len(test_sound) + len(isi_sound)) / 44100, color='blue', alpha=0.3, label='ISI')
plt.axvspan((len(pre_cue_sound) + len(test_sound) + len(isi_sound)) / 44100, (len(pre_cue_sound) + len(test_sound) + len(isi_sound) + len(standard_sound)) / 44100, color='green', alpha=0.3, label='Standard Sound')
plt.axvspan((len(pre_cue_sound) + len(test_sound) + len(isi_sound) + len(standard_sound)) / 44100, len(stim_sound) / 44100, color='gray', alpha=0.3, label='Post Cue')

plt.title("Low-Reliability Test Sound with Raised-Cosine Envelope")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.legend( bbox_to_anchor=(1.05, 1), loc='upper right')
plt.show()




"""



""" Sample code for generating different types of noise """
# # generate different high-reliability noise stimuli
# white_HRA= audio_cue.low_reliability_test_sound( dur=2.5, sample_rate=44100, increment_start=1, increment_duration=0.5, noise_type="white" )
# pink_HRA= audio_cue.low_reliability_test_sound( dur=2.5, sample_rate=44100, increment_start=1, increment_duration=0.5, noise_type="pink" )
# brownian_HRA= audio_cue.low_reliability_test_sound( dur=2.5, sample_rate=44100, increment_start=1, increment_duration=0.5, noise_type="brownian" )
# blue_HRA= audio_cue.low_reliability_test_sound( dur=2.5, sample_rate=44100, increment_start=1, increment_duration=0.5, noise_type="blue" )
# violet_HRA= audio_cue.low_reliability_test_sound( dur=2.5, sample_rate=44100, increment_start=1, increment_duration=0.5, noise_type="violet" )


# # Plot the noises
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 2, 1)
# plt.plot(white_HRA)
# plt.title("White Low Reliability Sound")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")

# plt.subplot(3, 2, 2)
# plt.plot(pink_HRA)
# plt.title("Pink Low Reliability Sound")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")

# plt.subplot(3, 2, 3)
# plt.plot(brownian_HRA)
# plt.title("Brownian Low Reliability Sound")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")

# plt.subplot(3, 2, 4)
# plt.plot(blue_HRA)
# plt.title("Blue Low Reliability Sound")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")

# plt.subplot(3, 2, 5)
# plt.plot(violet_HRA)
# plt.title("Violet Low Reliability Sound")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")

# plt.tight_layout()
# plt.show()


"TODO: Open comments for testing"

# test_noise_stimulus = audio_cue.generate_test_audio(dur=2,increment_start=0.5, increment_duration=1, noise_type="white")
# audio_cue.play_sound(test_noise_stimulus)

# import matplotlib.pyplot as plt
# # Plot test noise stimulus
# plt.figure()
# plt.plot(test_noise_stimulus)
# plt.title("Test Noise Stimulus with Intensity Increment")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")

# plt.show()

# stereo_array = audio_cue.positional_audio(0)
# # #audio_cue.play_sound(stereo_array)
# # white_noise = audio_cue.generate_white_noise()
# # pink_noise = audio_cue.generate_pink_noise()
# # audio_cue.play_sound(pink_noise)


# # Create high-reliability noise stimulus
# #high_reliability_noise = audio_cue.generate_high_reliability_noise(dur=2,fade_duration=0.01, noise_type="white")
# #audio_cue.play_sound(high_reliability_noise)

# # Create noise test stimulus with intensity increment
# test_noise_stimulus = audio_cue.generate_intensity_increment_noise(dur=2,increment_start=1, increment_duration=0.3, noise_type="white")
# audio_cue.play_sound(test_noise_stimulus)

# # Plot high-reliability noise
# plt.figure()
# #plt.plot(high_reliability_noise)
# plt.title("High-Reliability Noise")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")

# # Plot test noise stimulus
# plt.figure()
# plt.plot(test_noise_stimulus)
# plt.title("Test Noise Stimulus with Intensity Increment")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")

# plt.show()
