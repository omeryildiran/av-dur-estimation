"""Simulated observer standing in for PsychoPy, to test the pipeline without a display.

Observer (log-duration space): auditory noise sigma_a[noise], visual noise sigma_v; vision
is perceived `true_bias[noise]` s shorter than its physical duration, so the ideal
correction equals true_bias; in the main task each interval is a reliability-weighted
average of vision and audition. interval_bias > 0 favours answering "first interval".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

import experiment_settings as S

DEFAULT_OBSERVER = dict(sigma_a={0.1: 0.12, 1.2: 0.30}, sigma_v=0.22, true_bias={0.1: 0.10, 1.2: -0.05},
                        lapse=0.03, interval_bias=0.0)


class SimulatedPresenter:
    def __init__(self, seed=0, observer=None, needs_audio=False):
        self.rng = np.random.default_rng(seed)
        self.obs = dict(DEFAULT_OBSERVER, **(observer or {}))
        self.needs_audio = needs_audio
        self.task = None

    def start(self, task, meta):
        self.task = task
        return meta["frame_rate_nominal_hz"]

    def instructions(self, task, practice):
        pass

    def rest(self, k, n):
        pass

    def order_cue(self, order, k, n):
        pass

    def feedback(self, correct):
        pass

    def finish(self, completed):
        pass

    def _interval_estimate(self, sched, i, noise):
        """Mean and SD of the log-duration percept of interval i."""
        o = self.obs
        fr = sched.frame_rate
        a_dur = (sched.dur1 if i == 0 else sched.dur2) / fr
        vis = sched.visual[i]
        v_dur = None if vis is None else (vis[1] - vis[0]) / fr
        if v_dur is not None:
            bias = o["true_bias"].get(noise, 0.0) if noise is not None else 0.0
            v_perc = max(v_dur - bias, 0.005)
        if self.task == "unimodal_auditory" or (self.task == "crossmodal" and sched.audio_burst[i]):
            return np.log(a_dur), o["sigma_a"][noise]
        if self.task in ("unimodal_visual", "crossmodal"):
            return np.log(v_perc if self.task == "crossmodal" else v_dur), o["sigma_v"]
        sa, sv = o["sigma_a"][noise], o["sigma_v"]          # main: forced fusion
        w = sa**2 / (sa**2 + sv**2)
        return w * np.log(v_perc) + (1 - w) * np.log(a_dur), np.sqrt(w**2 * sv**2 + (1 - w)**2 * sa**2)

    def run_trial(self, spec, sched, wave):
        m1, s1 = self._interval_estimate(sched, 0, spec.noise)
        m2, s2 = self._interval_estimate(sched, 1, spec.noise)
        d = (m1 - m2) / np.sqrt(s1**2 + s2**2) + self.obs["interval_bias"]
        p_first = self.obs["lapse"] / 2 + (1 - self.obs["lapse"]) * norm.cdf(d)
        response = 1 if self.rng.random() < p_first else 2
        flips = {}
        for i, v in enumerate(sched.visual):
            if v is not None:
                flips[(i, "on")] = v[0] / sched.frame_rate + self.rng.normal(0, 0.0003)
                flips[(i, "off")] = v[1] / sched.frame_rate + self.rng.normal(0, 0.0003)
        rt = float(self.rng.lognormal(np.log(0.55), 0.35))           # from the end of interval 2
        remaining = (sched.total - sched.second_interval_end()) / sched.frame_rate
        return dict(quit=False, response=response, key="left" if response == 1 else "right",
                    rt=rt, prompt_shown=rt > remaining, flips=flips,
                    audio_onset_minus_flip0_s=S.AUDIO_ONSET_OFFSET_S if spec.task != "unimodal_visual" else np.nan,
                    flip0_error_s=0.0, n_dropped_frames=0, max_frame_interval_ms=1000 / sched.frame_rate)
