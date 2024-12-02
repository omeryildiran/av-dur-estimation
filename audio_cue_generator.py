"""
Coded by Omer Yildiran subject to attribution-noncommercial 4.0 International (CC BY-NC 4.0) license
Start Date: 12/2023
Last Update: 11/2024

"""

import numpy as np
from psychopy import sound, core

class AudioCueGenerator:
    def __init__(self):
        pass

    def generate_beep_sound(self, dur=2, sample_rate=44100, beep_frequency=440):
        t = np.arange(0, dur, 1.0 / sample_rate)
        beep_signal = np.sin(2.0 * np.pi * beep_frequency * t)
        return beep_signal

    def create_panning_beep_array(self, dur=2, sample_rate=44100, beep_frequency=440, pan_exponent=2):
        t = np.arange(0, dur, 1.0 / sample_rate)
        fade_dur_ind = len(t) // 3

        pan_factor = np.linspace(-1, 1, fade_dur_ind)
        pan_factor = np.concatenate((pan_factor, np.full(len(t) - fade_dur_ind, pan_factor[-1])))

        left_channel = (1 - pan_factor) * self.generate_beep_sound(dur, sample_rate, beep_frequency)
        right_channel = pan_factor * self.generate_beep_sound(dur, sample_rate, beep_frequency)

        stereo_array = np.column_stack((left_channel, right_channel))
        return stereo_array

    def create_stereo_sound(self, dur=2, sample_rate=44100, beep_frequency=440, channel='left'):
        t = np.arange(0, dur, 1.0 / sample_rate)
        if channel == 'left':
            left_channel = self.generate_beep_sound(dur, sample_rate, beep_frequency)
            right_channel = np.zeros(len(t))
        elif channel == 'right':
            left_channel = np.zeros(len(t))
            right_channel = self.generate_beep_sound(dur, sample_rate, beep_frequency)

        stereo_array = np.column_stack((left_channel, right_channel))
        return stereo_array

    def generate_noise(self,dur=2, sample_rate=44100, noise_type="white"):
        """
        Generate various types of noise (white, pink, brownian, blue, violet).
        
        :param dur: Duration of the noise (in seconds).
        :param sample_rate: Sampling rate (Hz).
        :param noise_type: Type of noise ('white', 'pink', 'brownian', 'blue', 'violet').
        :return: The generated noise signal.
        """
        # Generate raw white noise
        noise_signal = np.random.normal(0, 1, int(dur * sample_rate))
        
        if noise_type == "white":
            # White noise: no filtering needed
            return noise_signal / np.max(np.abs(noise_signal))

        elif noise_type == "pink":
            # Pink noise: 1 / sqrt(f) filter
            num_samples = len(noise_signal)
            freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
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
            freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
            fft_values = np.fft.rfft(noise_signal)
            blue_filter = np.sqrt(freqs)
            blue_filter[freqs == 0] = 0  # Avoid modifying DC component
            filtered_fft_values = fft_values * blue_filter
            noise_signal = np.fft.irfft(filtered_fft_values, n=num_samples)

        elif noise_type == "violet":
            # Violet noise: f filter
            num_samples = len(noise_signal)
            freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
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


    def generate_a_note(self, dur=2, sample_rate=44100, frequency=440):
        t = np.arange(0, dur, 1.0 / sample_rate)
        a_note = np.sin(2.0 * np.pi * frequency * t)
        return a_note

    def positional_audio(self, dur=2, sample_rate=44100, relPosX=0, relPosY=0.5):
        t = np.arange(0, dur, 1.0 / sample_rate)
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

    def play_sound(self, sound_array, sample_rate=44100, dur=2):
        beep = sound.Sound(value=sound_array, sampleRate=sample_rate, stereo=True)
        beep.play()
        core.wait(dur)

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

    def generate_gaussian_envelope(self,total_dur, sample_rate, mu_list, sigma, intensity=1.0, peak_amplitude=1.0):
        envelope = np.zeros(int(total_dur * sample_rate))
        x = np.linspace(0, total_dur, len(envelope))
        envelope = self.sum_of_gaussians(x, mu_list, sigma, intensity) # Generate the Gaussian envelope
        envelope = np.convolve(envelope, np.ones(int(peak_amplitude)) / peak_amplitude, mode='same')
        envelope += 1
        return envelope


    def low_reliability_test_sound(self, total_dur=2.5, sample_rate=44100, 
                                        signal_start=1,
                                        signal_duration=0.5, 
                                        noise_type="pink", 
                                        peak_amplitude=100, sigma=0.1):
        # Generate noise
        noise_signal = self.generate_noise(total_dur, sample_rate, noise_type)

        # Apply intensity increment envelope
        envelope = self.generate_gaussian_envelope(total_dur, sample_rate, 
                                                [signal_start, signal_start + signal_duration], sigma,peak_amplitude)
        noise_signal *= envelope


        return noise_signal

# # Example usage:
audio_cue = AudioCueGenerator()

test_sound = audio_cue.low_reliability_test_sound(
    total_dur=3, sample_rate=44100, signal_start=0.7, signal_duration=0.5, noise_type="white", sigma=0.1, peak_amplitude=3)
audio_cue.play_sound(test_sound)

import matplotlib.pyplot as plt
# Plot high-reliability audio stimulus
plt.figure()
plt.plot(test_sound)
plt.title("Low-Reliability Audio with Intensity Increment")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()



import matplotlib.pyplot as plt

# Parameters
duration = 2  # in seconds
sample_rate = 44100  # in Hz


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
