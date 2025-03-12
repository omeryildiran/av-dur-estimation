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


    def whole_stimulus_with_binning(self, test_dur, standard_dur, noise_type, intensity,
                                order, pre_dur=0.1, post_dur=0.1, isi_dur=0.3):
        """
        Generate the full stimulus including pre-cue, test sound, ISI, standard sound, and post-cue,
        with binned amplitude modulation for the test sound.
        """
        # 1. Generate pre-cue sound noise
        pre_cue_sound = self.generate_noise(dur=pre_dur, noise_type=noise_type)
  
        # 4. Generate standard sound noise
        standard_sound = self.generate_noise(dur=standard_dur, noise_type=noise_type)

        # 2. Generate test sound with binning and raised cosine envelope
        test_sound =self.generate_noise(dur=test_dur, noise_type=noise_type)

        # 3. Generate interstimulus interval (ISI) noise
        isi_sound = self.generate_noise(dur=isi_dur, noise_type=noise_type)

        # 5. Generate post-cue sound noise
        post_cue_sound = self.generate_noise(dur=post_dur, noise_type=noise_type)

        # intensity modulation
        standard_sound = standard_sound * 2
        test_sound = test_sound * intensity


        # Concatenate all segments depending on the order
        if order == 1:
            stim_sound = np.concatenate([pre_cue_sound, test_sound, isi_sound, standard_sound, post_cue_sound])
        elif order == 2:
            stim_sound = np.concatenate([pre_cue_sound, standard_sound, isi_sound, test_sound, post_cue_sound])
        else:
            raise ValueError("Invalid order value. Use 1 or 2.")
        
        # Normalize the signal
        stim_sound = stim_sound / np.max(np.abs(stim_sound))
        return stim_sound

import matplotlib.pyplot as plt

audio_cue = AudioCueGenerator(sampleRate=44100)

# Generate whole stimulus with binning
test_dur = 0.5
standard_dur = 0.5
noise_type = "white"
intensity = 1.5
order = 1


stim_sound = audio_cue.whole_stimulus_with_binning(
    test_dur, standard_dur, noise_type, intensity, order, 
    pre_dur=0.5, post_dur=0.5, isi_dur=0.5, 

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
plt.axvspan(0.5, 0.5+test_dur, color="blue", alpha=0.5, label="Test Sound")
plt.axvspan(0.5+test_dur+0.5, 0.5+test_dur+0.5+standard_dur, color="purple", alpha=0.5, label="Standard Sound")

plt.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
plt.show()
