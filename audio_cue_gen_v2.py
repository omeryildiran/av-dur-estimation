import numpy as np
from scipy.signal import butter, filtfilt
from psychopy import sound, core

class StimulusGenerator:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate  # Sampling rate in Hz

    def generate_pure_tone(self, freq=440, dur=1.0):
        """
        Generate a high-reliability pure tone with fixed amplitude.
        :param freq: Frequency of the tone (Hz).
        :param dur: Duration of the tone (seconds).
        :return: Numpy array of the tone signal.
        """
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        return tone

    def generate_low_reliability_noise(self, dur=2.0, bin_dur=0.1, min_amp=0.5, max_amp=1.5, sigma=0.1):
        """
        Generate low-reliability noise with bin-based amplitude modulation.
        :param dur: Total duration of the noise (seconds).
        :param bin_dur: Duration of each amplitude-modulated bin (seconds).
        :param min_amp: Minimum amplitude in the uniform distribution.
        :param max_amp: Maximum amplitude in the uniform distribution.
        :param sigma: Controls the variability of amplitudes.
        :return: Numpy array of the noise signal.
        """
        num_bins = int(dur / bin_dur)
        bin_samples = int(bin_dur * self.sample_rate)

        # Generate noise in bins with random amplitudes
        noise = []
        for _ in range(num_bins):
            bin_noise = np.random.normal(0, 1, bin_samples)
            bin_amp = np.random.uniform(min_amp, max_amp) + np.random.normal(0, sigma)
            noise.append(bin_amp * bin_noise)
        
        # Concatenate all bins into one continuous signal
        noise_signal = np.concatenate(noise)
        return noise_signal

    def apply_event_envelope(self, noise, dur=2.0, event_start=1.0, event_dur=0.5, increment_factor=2.0):
        """
        Apply a smooth amplitude increment-decrement envelope to a noise signal.
        :param noise: The noise signal (numpy array).
        :param dur: Total duration of the signal (seconds).
        :param event_start: Time when the intensity increment starts (seconds).
        :param event_dur: Duration of the intensity increment (seconds).
        :param increment_factor: The scaling factor for the increment.
        :return: Modulated noise signal with the envelope applied.
        """
        total_samples = len(noise)
        t = np.linspace(0, dur, total_samples, endpoint=False)

        # Create the envelope
        envelope = np.ones(total_samples)
        start_sample = int(event_start * self.sample_rate)
        event_samples = int(event_dur * self.sample_rate)
        
        # Smooth ramp for increment
        ramp = np.linspace(1, increment_factor, event_samples // 2)
        envelope[start_sample:start_sample + len(ramp)] = ramp
        envelope[start_sample + len(ramp):start_sample + event_samples] = ramp[::-1]

        # Apply the envelope
        modulated_noise = noise * envelope
        return modulated_noise

    def play_sound(self, signal, dur):
        """
        Play a sound using PsychoPy.
        :param signal: The audio signal to play (numpy array).
        :param dur: Duration of the sound in seconds.
        """
        sound_obj = sound.Sound(value=signal, sampleRate=self.sample_rate, stereo=False)
        sound_obj.play()
        core.wait(dur)

# Example Usage
generator = StimulusGenerator(sample_rate=44100)

# Generate a high-reliability pure tone (standard)
high_reliability_tone = generator.generate_pure_tone(freq=440, dur=2.0)

# Generate low-reliability noise (test stimulus)
low_reliability_noise = generator.generate_low_reliability_noise(dur=2.0, bin_dur=0.1, min_amp=0.5, max_amp=1.5, sigma=0.01)

# Apply an event envelope to the low-reliability noise
low_reliability_test = generator.apply_event_envelope(low_reliability_noise, dur=2.0, event_start=1, event_dur=1, increment_factor=3.0)

# Play the sounds (optional)
# generator.play_sound(high_reliability_tone, dur=2.0)
generator.play_sound(low_reliability_test, dur=2.0)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(high_reliability_tone)
plt.title("High-Reliability Pure Tone (Standard)")

plt.subplot(3, 1, 2)
plt.plot(low_reliability_noise)
plt.title("Low-Reliability Noise (Before Envelope)")

plt.subplot(3, 1, 3)
plt.plot(low_reliability_test)
plt.title("Low-Reliability Noise with Event Envelope (Test Stimulus)")

plt.tight_layout()
plt.show()
