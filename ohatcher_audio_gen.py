"""
Coded by Omer Yildiran subject to attribution-noncommercial 4.0 International (CC BY-NC 4.0) license
Start Date: 12/2023
Last Update: 11/2024

"""

import numpy as np
from psychopy import sound, core
from scipy.signal import butter, filtfilt
from scipy.signal import butter, sosfilt


class AudioCueGenerator:
    def __init__(self, sampleRate=44100):
        
        self.sample_rate = sampleRate  # in Hz

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
    def broadband_filter(self, signal, low_cut, high_cut, sample_rate, order=4):
        nyquist = 0.5 * sample_rate
        normal_cutoffs = [low_cut / nyquist, high_cut / nyquist]
        sos = butter(order, normal_cutoffs, btype="band", analog=False, output="sos")
        return sosfilt(sos, signal)

    def generate_beep_sound(self, dur=2,  beep_frequency=440):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        beep_signal = np.sin(2.0 * np.pi * beep_frequency * t)
        return beep_signal


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

    def play_sound(self, sound_array):
        duration=len(sound_array)/self.sample_rate
        beep = sound.Sound(value=sound_array, sampleRate=self.sample_rate, stereo=True)
        beep.play()
        core.wait(duration)



    # Raised cosine envelope for precise transition and perceptual salience
    def raised_cosine_envelope(self,t_start, rise_dur, peak_dur, t_total, sample_rate=44100):

        def time_to_samples(t, sample_rate=44100):
            return int(t * sample_rate)

        # Convert durations and times to samples
        start_sample = time_to_samples(t_start, sample_rate)
        rise_samples = time_to_samples(rise_dur, sample_rate)
        peak_samples = time_to_samples(peak_dur, sample_rate)
        rise_fall_samples = time_to_samples(2*rise_dur, sample_rate)
        

        event_samples = round(rise_fall_samples + peak_samples)
        total_samples = event_samples 
        fall_samples = total_samples - peak_samples - rise_samples
        # Ensure the event duration does not exceed the total duration
        if start_sample + event_samples > total_samples:
            # print durations
            print("Event Duration: ", event_samples / sample_rate)
            print("Total Duration: ", total_samples / sample_rate)
            print("Rise Duration: ", rise_samples / sample_rate)
            
            raise ValueError("The event duration exceeds the total duration of the envelope.")

        # Create time axis for the envelope
        envelope = np.zeros(total_samples)

        # Rising phase
        t_rise = np.arange(rise_samples)
        envelope[start_sample:start_sample + rise_samples] = 0.5 * (1 - np.cos(np.pi * t_rise / rise_samples))

        # Peak phase
        t_peak_start = start_sample + rise_samples
        #print("Peak Start: ", t_peak_start) 
        envelope[t_peak_start:t_peak_start + peak_samples] = 1.0

        # Falling phase
        t_fall_start = t_peak_start + peak_samples
        #print("\nFall Start: ", t_fall_start)

        t_fall = np.arange(fall_samples)
        envelope[t_fall_start:] = 0.5 * (1 + np.cos(np.pi * t_fall / fall_samples))

        return envelope

    def low_reliability_test_sound(self, total_dur=2.5, noise_type="pink", 
                                rise_dur=0.2, intensity=3.0):
        peak_dur = total_dur - 2*rise_dur  # Ensure the peak phase does not exceed the total duration
        # Generate the raised-cosine envelope
        envelope = self.raised_cosine_envelope(0, rise_dur, peak_dur, total_dur, self.sample_rate)

        # Generate base noise
        noise_signal = np.random.normal(0, 1,len(envelope))#self.generate_noise(dur=total_dur, noise_type=noise_type)
        noise_signal = noise_signal / np.max(np.abs(noise_signal))
        # Scale the envelope
        envelope = envelope * (intensity - 1) + 1  # Ensure baseline is 1, and peak reaches the desired intensity
        # Apply the envelope to the noise
        modulated_noise = noise_signal * envelope
        return modulated_noise
        
    def whole_stimulus(self, test_dur, standard_dur, noise_type, intensity, rise_dur,order, pre_dur=0.1, post_dur=0.1,isi_dur=0.3,intensity_background=0.1):

        # 1. generate pre-cue sound noise for 0.1 seconds
        pre_cue_sound = self.generate_noise(dur=pre_dur, noise_type=noise_type)
      # 4. generate standard sound noise
        standard_sound = self.low_reliability_test_sound(total_dur=standard_dur, 
                                                    rise_dur=0.005, 
                                                    noise_type=noise_type, 
                                                    intensity=intensity)
        
        # 2. generate test sound noise for 2.5 seconds
        test_sound = self.low_reliability_test_sound(total_dur=test_dur, 
                                                    rise_dur=0.005, 
                                                    noise_type=noise_type, 
                                                    intensity=intensity)       

        # ensure that test sound and standard sound have same standard deviation
        test_sound = test_sound * (np.std(standard_sound) / np.std(test_sound))
        # 3. interstimulus interval noise 
        isi_sound = self.generate_noise(dur=isi_dur, noise_type=noise_type)
        # 5. generate post-cue sound noise for 0.1 seconds
        post_cue_sound = self.generate_noise(dur=post_dur, noise_type=noise_type) 
        jitter_sound = np.zeros(int(0.0001 * self.sample_rate))

        # concatenate all bins into one continuous signal depending on the order
        if order == 1:
            stim_sound = np.concatenate([jitter_sound,pre_cue_sound, test_sound, isi_sound, standard_sound, post_cue_sound,jitter_sound])
        elif order == 2:
            stim_sound = np.concatenate([jitter_sound,pre_cue_sound, standard_sound, isi_sound, test_sound, post_cue_sound,jitter_sound])
        else:
            raise ValueError("Invalid order value. Use 1 or 2.")
        
        # normalize the signal
        stim_sound = stim_sound / np.max(np.abs(stim_sound))

        # scale background noise to match the amplitude of the stimulus
        background_noise = np.random.normal(0, 1, len(stim_sound))
        background_noise = background_noise * (np.max(abs(stim_sound)) / np.max(abs(background_noise))) * intensity_background

        #print(max(background_noise))
        #print(max(stim_sound))

        # smooth the sound waveform
        background_noise = self.broadband_filter(background_noise,10, 610, self.sample_rate, order=4)
        stim_sound = self.broadband_filter(stim_sound, 150, 775, self.sample_rate, order=4)

        # plot the sound waveforms
        time= np.linspace(0, len(stim_sound) / self.sample_rate, len(stim_sound))

        plt.plot(time,stim_sound, label='Signal Sound', color='forestgreen', alpha=0.7)
        plt.plot(time,background_noise, label='Background Noise', color='black', alpha=0.7)
        plt.title("Stimulus Sound Waveform")
        plt.xlabel("Time (s)")
        plt.xlim(0, len(stim_sound) / self.sample_rate)
        plt.xticks(np.arange(0, len(stim_sound) / self.sample_rate, 0.1))
        plt.legend()
        plt.ylabel("Amplitude")
        plt.show()
        
        # #Background noise of same totaal duration

        # # mix the sounds
        stim_sound = stim_sound + background_noise
        stim_sound = stim_sound / np.max(np.abs(stim_sound))
        stim_sound=np.concatenate([jitter_sound,stim_sound,jitter_sound])

        return stim_sound
    





# # Example usage with raised-cosine envelope
audio_cue = AudioCueGenerator(sampleRate=44100)

#generate whole stim
test_dur = 0.4
standard_dur = 0.6
noise_type = "white"
intensity = 5
rise_dur = 1.2
order = 1
pre_cue_sound=0.25
pre_post_dur=pre_cue_sound
test_sound=test_dur
isi_sound=0.25
#audio_cue.play_sound(stim_sound)

#import for plotting
import matplotlib.pyplot as plt

## PLot different sounds with different amplitude variance
def plot_sounds():
    plt.figure(figsize=(12, 4))
    for idx, rise in enumerate([1.2]):
        stim_sound = audio_cue.whole_stimulus(test_dur, standard_dur, noise_type, intensity, rise, order, pre_dur=pre_cue_sound, post_dur=pre_cue_sound,isi_dur=pre_cue_sound,
                                              intensity_background=rise)
        #save sound
        from scipy.io import wavfile
        wavfile.write(f"stim_sound_rise_lowSNR.wav", audio_cue.sample_rate, (stim_sound * 32767).astype(np.int16))
        #stim_sound*=0.5

        t=np.linspace(0, len(stim_sound) / audio_cue.sample_rate, len(stim_sound))
        #if idx in [1]:
        #audio_cue.play_sound(stim_sound)
        
        plt.subplot(1, 2, idx + 1)
        plt.plot(t, stim_sound)
        plt.title(f"Rise duration: {rise}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.axvspan(pre_post_dur, pre_post_dur+test_dur, color="red", alpha=0.5, label="Reliable signal")
        plt.axvspan(pre_post_dur+test_dur+pre_post_dur, pre_post_dur+test_dur+pre_post_dur+standard_dur, color="forestgreen", alpha=0.2, label="unreliable signal")
    #plt.ylim(-2,3)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
    plt.show()
    
#plot_sounds()















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
