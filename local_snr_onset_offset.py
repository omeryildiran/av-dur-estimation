import numpy as np
import matplotlib.pyplot as plt

# Parameters
duration = 5.0  # seconds
sample_rate = 44100  # Hz
total_samples = int(duration * sample_rate)
t = np.arange(total_samples) / sample_rate

# Original signal: a sine wave
frequency = 1000  # Hz
signal = np.sin(2 * np.pi * frequency * t)

# Add temporal jitter: randomly shift the onset of the signal by up to ±20 ms on each trial.
# For a single trial, you can shift by a random amount:
max_jitter = 0.02  # 20 ms maximum jitter
jitter_samples = int(np.random.uniform(-max_jitter, max_jitter) * sample_rate)

# Shift the signal in time: prepend or remove samples as needed.
if jitter_samples > 0:
    # Delay signal: prepend zeros
    signal = np.concatenate((np.zeros(jitter_samples), signal))[:total_samples]
elif jitter_samples < 0:
    # Advance signal: remove initial samples and pad at the end
    signal = np.concatenate((signal[abs(jitter_samples):], np.zeros(abs(jitter_samples))))
# (Now signal has been shifted randomly)

# Create an amplitude modulation (AM) envelope with random fluctuations
# This envelope will randomly modulate the signal amplitude (e.g., ±20% variation)
random_modulation = 1 + 0.2 * (np.random.rand(total_samples) - 0.5)
# Optionally smooth the modulation to avoid abrupt changes:
from scipy.ndimage import uniform_filter1d
modulation_envelope = uniform_filter1d(random_modulation, size=int(0.05 * sample_rate))
signal_modulated = signal * modulation_envelope

# Generate noise as before and boost noise at boundaries
noise = np.random.normal(0, 1, total_samples)
n_boundary = int(0.1 * sample_rate)  # 100 ms boundaries
ramp = np.linspace(1.5, 1.0, n_boundary)
envelope = np.ones(total_samples)
envelope[:n_boundary] = ramp
envelope[-n_boundary:] = ramp[::-1]
modified_noise = noise * envelope

# Increase the overall noise level as desired (for example, globally boost by 50%)
global_noise_boost = 1.5
modified_noise *= global_noise_boost

# Combine the modulated signal with the noise.
# Optionally, you can set an overall target SNR:
signal_power = np.mean(signal_modulated**2)
noise_power = np.mean(modified_noise**2)
desired_SNR_dB = 5  # Very low SNR (5 dB), making the sound less reliable
desired_SNR_linear = 10 ** (desired_SNR_dB / 10)
target_noise_power = signal_power / desired_SNR_linear
scaling_factor = np.sqrt(target_noise_power / noise_power)
scaled_noise = modified_noise * scaling_factor

# Final mixture
mixture = signal_modulated + scaled_noise

# Plot a short segment to inspect the effect (first 200 ms)
plt.figure(figsize=(10, 4))
plt.plot(t[:int(0.2 * sample_rate)], mixture[:int(0.2 * sample_rate)], label='Mixture')
plt.plot(t[:int(0.2 * sample_rate)], signal_modulated[:int(0.2 * sample_rate)], label='Signal (modulated)', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Final Mixture with Increased Unreliability')
plt.legend()
plt.show()
