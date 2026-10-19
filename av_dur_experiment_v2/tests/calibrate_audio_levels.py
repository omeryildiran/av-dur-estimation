"""Calibrate the v2 fixed-level audio constants against v1's peak-normalised stimuli.

v1 divided every waveform by its own peak, so absolute levels depended on the random
peak (about +-1 dB from trial to trial), on how many bursts the trial had (cross-modal
tests were up to 1.3 dB louder than main-experiment tests), and on the noise level (the
louder background raised the peak, so bursts were quieter in high noise). v2 uses:

    FIXED_BACKGROUND_SCALE  background SD per unit noise level, relative to the unit-SD noise floor
    FIXED_OUTPUT_GAIN       gain after filtering, per noise level: reproduces v1's mean burst level
                            at each noise level, with the loudest calibration trial at 0.95 full scale
    FIXED_VOLUME            PsychoPy volume giving the same played level as v1 at VOLUME

    python tests/calibrate_audio_levels.py       # prints the values to paste into experiment_settings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur.audio import AudioGenerator  # noqa: E402
from avdur.schedule import make_schedule  # noqa: E402

FR, SR = S.EXPECTED_FRAME_RATE_HZ, S.SAMPLE_RATE


def burst_rms(x, sched, pad):
    on, off = sched.interval_frames(0)
    seg = x[pad + int(on / FR * SR) + int(0.02 * SR): pad + int(off / FR * SR) - int(0.02 * SR)]
    return np.sqrt(np.mean(seg ** 2))


def trials(rng, n, durations):
    for _ in range(n):
        yield make_schedule("main", 1 + rng.integers(2), rng.uniform(*durations), 0.0, 0.0, rng, FR), int(rng.integers(2**31))


def main(n=400, seed=0):
    rng = np.random.default_rng(seed)
    v1 = AudioGenerator(level_mode="peak")
    pad = 2 * len(v1.pad)

    ratio, v1_level = [], {}
    for noise in S.NOISE_LEVELS:
        rms = []
        for sched, s in trials(rng, n, (0.2, 0.8)):
            parts = v1.trial_parts(sched, noise, s)
            ratio.append(parts["background_sd"] / noise / parts["floor_sd"])
            rms.append(burst_rms(parts["output"], sched, pad))
        v1_level[noise] = 20 * np.log10(np.mean(rms))
    background_scale = float(np.mean(ratio))

    fixed = AudioGenerator(level_mode="fixed", background_scale=background_scale, output_gain=1.0, clip_protect=False)
    rel_gain, peak = {}, {}
    for noise in S.NOISE_LEVELS:
        rms, peaks = [], []
        for sched, s in trials(rng, n, (0.05, 0.95)):
            x = fixed.trial_sound(sched, noise, s)
            rms.append(burst_rms(x, sched, pad))
            peaks.append(np.max(np.abs(x)))
        rel_gain[noise] = 10 ** ((v1_level[noise] - 20 * np.log10(np.mean(rms))) / 20)   # gain that reproduces v1's level
        peak[noise] = max(peaks)
    headroom = 0.95 / max(rel_gain[k] * peak[k] for k in S.NOISE_LEVELS)                  # common scale, no clipping
    gains = {k: round(rel_gain[k] * headroom, 5) for k in S.NOISE_LEVELS}
    volume = S.VOLUME / headroom
    print(f"FIXED_BACKGROUND_SCALE = {background_scale:.4f}")
    print(f"FIXED_OUTPUT_GAIN = {gains}")
    print(f"FIXED_VOLUME = {volume:.4f}")
    for k in S.NOISE_LEVELS:
        print(f"  noise {k}: v1 mean burst {v1_level[k]:.2f} dBFS; v2 burst {v1_level[k] + 20 * np.log10(headroom):.2f} dBFS "
              f"(played level identical to v1)")


if __name__ == "__main__":
    main()
