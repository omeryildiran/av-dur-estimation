import numpy as np
from psychopy import sound, core



class generateAudioClass:
    def __init__(self, sampleRate=44100):
        
        self.sample_rate = sampleRate  # in Hz
    def generateNote(self, dur=2,  beep_frequency=440):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        beep_signal = np.sin(2.0 * np.pi * beep_frequency * t)
        return beep_signal

    def createPanningBeep(self, dur=2,  beep_frequency=440, pan_exponent=2):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        fade_dur_ind = len(t) // 3

        pan_factor = np.linspace(-1, 1, fade_dur_ind)
        pan_factor = np.concatenate((pan_factor, np.full(len(t) - fade_dur_ind, pan_factor[-1])))

        left_channel = (1 - pan_factor) * self.generate_beep_sound(dur, self.sample_rate, beep_frequency)
        right_channel = pan_factor * self.generate_beep_sound(dur, self.sample_rate, beep_frequency)

        stereo_array = np.column_stack((left_channel, right_channel))
        return stereo_array

    def createStereoSound(self, dur=2,  beep_frequency=440, channel='left'):
        t = np.arange(0, dur, 1.0 / self.sample_rate)
        if channel == 'left':
            left_channel = self.generate_beep_sound(dur, self.sample_rate, beep_frequency)
            right_channel = np.zeros(len(t))
        elif channel == 'right':
            left_channel = np.zeros(len(t))
            right_channel = self.generate_beep_sound(dur, self.sample_rate, beep_frequency)

        stereo_array = np.column_stack((left_channel, right_channel))
        return stereo_array

    def generateNoise(self,dur=2,  noise_type="white"):
        """
        Generate various types of noise (white, pink, brownian, blue, violet).
        
        :param dur: Duration of the noise (in seconds).
        :param sample_rate: Sampling rate (Hz).
        :param noise_type: Type of noise ('white', 'pink', 'brownian', 'blue', 'violet').
        :return: The generated noise signal.
        """
        # Generate raw white noise
        noise_signal = np.random.normal(0, 3, int(dur * self.sample_rate))
        
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