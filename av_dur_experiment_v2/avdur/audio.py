"""Auditory stimulus, built like v1's ohatcher_audio_gen.whole_stimulus.

A trial is a noise stream (pre | interval 1 | ISI | interval 2 | post). An auditory
interval holds a burst: noise at BURST_INTENSITY times the stream level with 5 ms
raised-cosine ramps. A background noise scaled by the noise level is added; the stream is
band-passed 150-775 Hz and the background 10-610 Hz (4th-order Butterworth, causal sosfilt).

Level modes (AUDIO_LEVEL_MODE):
  "peak"   v1 exactly: each waveform divided by its own peak, so levels jitter by about
           +-1 dB from trial to trial and depend on how many bursts a trial has.
  "fixed"  v2: unit-SD noise floor, fixed background scale and a fixed output gain per noise
           level, so every burst of a noise level has the same level in every task and trial.
           The constants are calibrated (tests/calibrate_audio_levels.py) to v1's average burst
           level at each noise level (v1's peak normalisation made bursts quieter in high
           noise) and to v1's signal-to-noise ratio.
Both modes scale the test burst to the RMS of a standard-duration burst. v1 did that in
the auditory and main tasks but not the cross-modal task.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

import experiment_settings as S


class AudioGenerator:
    def __init__(self, sample_rate=S.SAMPLE_RATE, level_mode=S.AUDIO_LEVEL_MODE,
                 background_scale=S.FIXED_BACKGROUND_SCALE, output_gain=S.FIXED_OUTPUT_GAIN, clip_protect=True):
        if level_mode not in ("peak", "fixed"):
            raise ValueError(level_mode)
        self.sr, self.mode = sample_rate, level_mode
        self.background_scale, self.output_gain, self.clip_protect = background_scale, output_gain, clip_protect
        nyq = 0.5 * sample_rate
        self._sos_signal = butter(S.FILTER_ORDER, [f / nyq for f in S.SIGNAL_BAND_HZ], btype="band", output="sos")
        self._sos_background = butter(S.FILTER_ORDER, [f / nyq for f in S.BACKGROUND_BAND_HZ], btype="band", output="sos")
        self.pad = np.zeros(int(0.0001 * sample_rate))      # v1 "jitter" padding (0.1 ms of silence)
        self.last_gain_reduction_db = 0.0

    def _unit_noise(self, n, rng):
        x = rng.normal(0, 1, n)
        return x / np.max(np.abs(x)) if self.mode == "peak" else x

    def _noise(self, dur_s, rng):
        return self._unit_noise(int(dur_s * self.sr), rng)

    def _burst(self, dur_s, rng):
        """v1 raised-cosine envelope from 1 to BURST_INTENSITY applied to the stream noise."""
        rise = int(S.BURST_RISE_S * self.sr)
        peak = int((dur_s - 2 * S.BURST_RISE_S) * self.sr)
        n = round(int(2 * S.BURST_RISE_S * self.sr) + peak)
        fall = n - peak - rise
        env = np.zeros(n)
        env[:rise] = 0.5 * (1 - np.cos(np.pi * np.arange(rise) / rise))
        env[rise:rise + peak] = 1.0
        env[rise + peak:] = 0.5 * (1 + np.cos(np.pi * np.arange(fall) / fall))
        return self._unit_noise(n, rng) * (env * (S.BURST_INTENSITY - 1) + 1)

    def gain_for(self, noise_level):
        """Output gain: the calibrated value at a calibrated noise level; for a participant's own
        level, log-log interpolation along the calibrated points (and the same line beyond them)."""
        g = self.output_gain
        if not isinstance(g, dict):
            return g
        noise = float(noise_level)
        if noise in g:
            return g[noise]
        pts = sorted(g.items())
        if len(pts) == 1:
            return pts[0][1]
        (x0, y0), (x1, y1) = (pts[0], pts[1]) if noise < pts[1][0] else (pts[-2], pts[-1])
        slope = (np.log(y1) - np.log(y0)) / (np.log(x1) - np.log(x0))
        return float(np.exp(np.log(y0) + slope * (np.log(noise) - np.log(x0))))

    def trial_parts(self, schedule, noise_level, seed):
        """Waveform for one trial plus intermediate quantities (used by the calibration)."""
        rng = np.random.default_rng(seed)
        fr = schedule.frame_rate
        target_sd = np.std(self._burst(S.STANDARD_DUR_S, rng))       # level reference for every burst

        def interval(i, dur_frames):
            if schedule.audio_burst[i]:
                burst = self._burst(dur_frames / fr, rng)
                return burst * (target_sd / np.std(burst))
            return self._noise(dur_frames / fr, rng)

        pre = self._noise(schedule.pre / fr, rng)
        stim = np.concatenate([self.pad, pre, interval(0, schedule.dur1), self._noise(schedule.isi / fr, rng),
                               interval(1, schedule.dur2), self._noise(schedule.post / fr, rng), self.pad])
        background = rng.normal(0, 1, len(stim))
        reduction = 0.0
        if self.mode == "peak":                                         # v1
            scale = np.max(np.abs(stim))
            stim, pre = stim / scale, pre / scale
            background = background * (np.max(np.abs(stim)) / np.max(np.abs(background))) * noise_level
            mix = sosfilt(self._sos_signal, stim) + sosfilt(self._sos_background, background)
            mix = mix / np.max(np.abs(mix))
        else:                                                           # v2
            background = background * noise_level * self.background_scale
            gain = self.gain_for(noise_level)
            mix = (sosfilt(self._sos_signal, stim) + sosfilt(self._sos_background, background)) * gain
            peak = np.max(np.abs(mix))
            if self.clip_protect and peak > 0.99:                       # clip protection (rare; logged)
                mix *= 0.99 / peak
                reduction = 20 * np.log10(0.99 / peak)
        self.last_gain_reduction_db = reduction
        output = np.concatenate([self.pad, mix, self.pad]).astype(np.float32)
        return dict(output=output, floor_sd=float(np.std(pre)), background_sd=float(np.std(background)),
                    gain_reduction_db=reduction)

    def trial_sound(self, schedule, noise_level, seed):
        """Mono waveform; sample 0 is frame 0 of the schedule (plus v1's 0.2 ms of padding)."""
        return self.trial_parts(schedule, noise_level, seed)["output"]


def playback_volume():
    """PsychoPy volume for the active level mode (both give v1's average loudness)."""
    return S.VOLUME if S.AUDIO_LEVEL_MODE == "peak" else S.FIXED_VOLUME
