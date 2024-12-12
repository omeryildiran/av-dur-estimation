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
        envelope[t_peak_start:t_peak_start + peak_samples] = 1.0

        # Falling phase
        t_fall_start = t_peak_start + peak_samples
        t_fall = np.arange(fall_samples)
        envelope[t_fall_start:t_fall_start + fall_samples] = 0.5 * (1 + np.cos(np.pi * t_fall / fall_samples))

        return envelope
    def generate_binned_noise(self, dur=2, noise_type="white", bin_dur=0.1, amp_mean=1, amp_var=0.5):
            """
            Generate noise with binned amplitude modulation.
            :param dur: Total duration of the noise (seconds).
            :param noise_type: Type of noise ('white' only for now).
            :param bin_dur: Duration of each bin (seconds).
            :param amp_mean: Mean amplitude of the bins.
            :param amp_var: Variability (standard deviation) of the amplitude for each bin.
            :return: The generated noise signal as a numpy array.
            """
            total_samples = int(dur * self.sample_rate)
            bin_samples = int(bin_dur * self.sample_rate)
            n_bins = total_samples // bin_samples

            # Generate noise for each bin
            noise_signal = np.zeros(total_samples)
            
            for i in range(n_bins):
                start_idx = i * bin_samples
                end_idx = start_idx + bin_samples
                
                # Generate raw white noise for this bin
                bin_noise = np.random.normal(0, 1, bin_samples)
                
                # Sample amplitude from a uniform or normal distribution
                bin_amplitude = np.random.uniform(amp_mean - amp_var, amp_mean + amp_var)
                
                # Scale the noise in this bin
                noise_signal[start_idx:end_idx] = bin_amplitude * bin_noise / np.max(np.abs(bin_noise))
            
            # If total_samples is not a perfect multiple of bin_samples, fill the rest
            remaining_samples = total_samples - n_bins * bin_samples
            if remaining_samples > 0:
                last_bin_noise = np.random.normal(0, 1, remaining_samples)
                bin_amplitude = np.random.uniform(amp_mean - amp_var, amp_mean + amp_var)
                noise_signal[-remaining_samples:] = bin_amplitude * last_bin_noise / np.max(np.abs(last_bin_noise))
            
            return noise_signal
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
        
    def whole_stimulus(self, test_dur, standard_dur, noise_type, intensity, rise_dur,order, pre_dur=0.1, post_dur=0.1,isi_dur=0.3):

        # 1. generate pre-cue sound noise for 0.1 seconds
        pre_cue_sound = self.generate_noise(dur=pre_dur, noise_type=noise_type)

        # 2. generate test sound noise for 2.5 seconds
        test_sound = self.low_reliability_test_sound(total_dur=test_dur, 
                                                    rise_dur=rise_dur, 
                                                    noise_type=noise_type, 
                                                    intensity=intensity)
        # 3. interstimulus interval noise 
        isi_sound = self.generate_noise(dur=isi_dur, noise_type=noise_type)
        
        # 4. generate standard sound noise
        standard_sound = self.low_reliability_test_sound(total_dur=standard_dur, 
                                                    rise_dur=0, 
                                                    noise_type=noise_type, 
                                                    intensity=intensity)
        # 5. generate post-cue sound noise for 0.1 seconds
        post_cue_sound = self.generate_noise(dur=post_dur, noise_type=noise_type) 

        # concatenate all bins into one continuous signal depending on the order
        if order == 1:
            stim_sound = np.concatenate([pre_cue_sound, test_sound, isi_sound, standard_sound, post_cue_sound])
        elif order == 2:
            stim_sound = np.concatenate([pre_cue_sound, standard_sound, isi_sound, test_sound, post_cue_sound])
        else:
            raise ValueError("Invalid order value. Use 1 or 2.")
        
        # normalize the signal
        stim_sound = stim_sound / np.max(np.abs(stim_sound))

        return stim_sound
    def low_reliability_test_sound_with_binning(self, total_dur=2.5, noise_type="white", 
                                                rise_dur=0.2, intensity=3.0, 
                                                bin_dur=0.1, amp_mean=1, amp_var=0.5):
        """
        Generate a low-reliability noise signal with binned amplitude modulation
        and a raised-cosine envelope for perceptual salience.
        :param total_dur: Total duration of the noise (seconds).
        :param noise_type: Type of noise ('white', 'pink', etc.).
        :param rise_dur: Duration of the rising phase of the envelope (seconds).
        :param intensity: Peak amplitude of the envelope.
        :param bin_dur: Duration of each bin (seconds).
        :param amp_mean: Mean amplitude of the bins.
        :param amp_var: Variability (standard deviation) of the amplitude for each bin.
        :return: Modulated noise signal as a numpy array.
        """
        peak_dur = total_dur - 2 * rise_dur  # Ensure the peak phase does not exceed the total duration
        
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
            bin_amplitude = np.random.uniform(amp_mean - amp_var, amp_mean + amp_var)
            
            # Scale the noise in this bin
            noise_signal[start_idx:end_idx] = bin_amplitude * bin_noise / np.max(np.abs(bin_noise))
        
        # Handle remaining samples if total_samples is not divisible by bin_samples
        remaining_samples = total_samples - n_bins * bin_samples
        if remaining_samples > 0:
            last_bin_noise = np.random.normal(0, 1, remaining_samples)
            bin_amplitude = np.random.uniform(amp_mean - amp_var, amp_mean + amp_var)
            noise_signal[-remaining_samples:] = bin_amplitude * last_bin_noise / np.max(np.abs(last_bin_noise))
        
        # Generate the raised-cosine envelope
        envelope = self.raised_cosine_envelope(0, rise_dur, peak_dur, total_dur, self.sample_rate)
        
        # Scale the envelope
        envelope = envelope * (intensity - 1) + 1  # Ensure baseline is 1, and peak reaches the desired intensity
        
        # Apply the envelope to the noise
        modulated_noise = noise_signal * envelope

        # clip to the intensity
        modulated_noise = np.clip(modulated_noise, -intensity, intensity)

        return modulated_noise
    
    def whole_stimulus_with_binning(self, test_dur, standard_dur, noise_type, intensity, rise_dur, 
                                order, pre_dur=0.1, post_dur=0.1, isi_dur=0.3,
                                bin_dur=0.1, amp_mean=1, amp_var=0.5):
        """
        Generate the full stimulus including pre-cue, test sound, ISI, standard sound, and post-cue,
        with binned amplitude modulation for the test sound.
        """
        # 1. Generate pre-cue sound noise
        pre_cue_sound = self.generate_noise(dur=pre_dur, noise_type=noise_type)

        # 2. Generate test sound with binning and raised cosine envelope
        test_sound = self.low_reliability_test_sound_with_binning(
            total_dur=test_dur, rise_dur=rise_dur, noise_type="pink", 
            intensity=intensity, bin_dur=bin_dur, amp_mean=amp_mean, amp_var=amp_var
        )

        # 3. Generate interstimulus interval (ISI) noise
        isi_sound = self.generate_noise(dur=isi_dur, noise_type=noise_type)

        # 4. Generate standard sound noise
        standard_sound = self.low_reliability_test_sound(total_dur=standard_dur, 
                                                    rise_dur=0, 
                                                    noise_type=noise_type, 
                                                    intensity=intensity)

        # 5. Generate post-cue sound noise
        post_cue_sound = self.generate_noise(dur=post_dur, noise_type=noise_type)

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
test_dur = 1.5
standard_dur = 1.5
noise_type = "white"
intensity = 2
rise_dur = 0.1
order = 1
bin_dur = 0.05 # 100 ms bins
amp_mean = 1
amp_var = 0.125

stim_sound = audio_cue.whole_stimulus_with_binning(
    test_dur, standard_dur, noise_type, intensity, rise_dur, order, 
    pre_dur=0.25, post_dur=0.25, isi_dur=0.3, bin_dur=bin_dur, amp_mean=amp_mean, amp_var=amp_var
)

# Play and plot the stimulus
audio_cue.play_sound(stim_sound)

# Plot the stimulus
t = np.linspace(0, len(stim_sound) / audio_cue.sample_rate, len(stim_sound))
plt.figure(figsize=(10, 4))
plt.plot(t, stim_sound, label="Modulated Noise")
plt.title("Stimulus (Low-rel test+Standard) Sound with Binning")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
