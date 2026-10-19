"""PsychoPy presenter: window, stimuli, frame loop, sound and keyboard.

v2 timing: the sound is scheduled on the trial's first flip (frame 0), every frame is
flipped explicitly, the returned flip timestamps of each visual onset/offset are logged,
and frame intervals are checked for dropped frames. v1 started the sound from inside the
frame loop without scheduling and logged the loop time instead of the flip time.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import experiment_settings as S
from avdur.audio import playback_volume

ASSETS = Path(__file__).resolve().parents[1] / "assets"

INSTRUCTIONS = {
    "unimodal_auditory": "You will hear two sounds in noise.\nWhich sound was longer?",
    "unimodal_visual": "You will see two dark disks.\nWhich disk was shown longer?",
    "crossmodal": "Each trial has one sound and one disk.\nWhich one was longer?",
    "main": "You will hear two sounds; disks will flash with them.\nJudge only the SOUNDS: which sound was longer?",
}


def dva_to_px(size_deg, height_cm, distance_cm, window_px):
    """v1 conversion (dva_to_pix.py), kept so v1_profile reproduces the v1 disk size."""
    deg_per_px = math.degrees(math.atan2(0.5 * height_cm, distance_cm)) / (0.5 * window_px)
    return size_deg / deg_per_px


class PsychoPyPresenter:
    needs_audio = True

    def __init__(self, monitor_profile=S.DEFAULT_MONITOR):
        self.profile = S.MONITOR_PROFILES[monitor_profile]
        self.task = None

    # ── setup ─────────────────────────────────────────────────────────────────
    def start(self, task, meta):
        from psychopy import prefs
        prefs.hardware["audioLib"] = ["PTB"]
        prefs.hardware["audioLatencyMode"] = S.AUDIO_LATENCY_MODE
        from psychopy import core, monitors, sound, visual
        from psychopy.hardware import keyboard
        from psychopy.tools.monitorunittools import deg2pix

        self.core, self.sound, self.task = core, sound, task
        p = self.profile
        mon = monitors.Monitor("avdur_v2", width=p["width_cm"], distance=p["distance_cm"])
        mon.setSizePix((p["window_px"], p["window_px"]))
        self.win = visual.Window(size=(p["window_px"], p["window_px"]), fullscr=p["fullscreen"], screen=p["screen"],
                                 monitor=mon, units="pix", color=S.BACKGROUND_COLOR, colorSpace="rgb",
                                 waitBlanking=True, allowGUI=False)
        self.win.mouseVisible = False
        self.text = visual.TextStim(self.win, text="", color="white", height=S.TEXT_HEIGHT_PX,
                                    wrapWidth=0.9 * p["window_px"])
        self.kb = keyboard.Keyboard(backend="ptb")

        measured = self.win.getActualFrameRate(nIdentical=60, nMaxFrames=300, nWarmUpFrames=30, threshold=1)
        meta["sound_backend"] = getattr(sound, "audioLib", None)
        if measured is None or abs(measured - S.EXPECTED_FRAME_RATE_HZ) > S.FRAME_RATE_TOLERANCE_HZ:
            self._message(f"Measured refresh rate: {measured} Hz (expected {S.EXPECTED_FRAME_RATE_HZ} Hz).\n"
                          "Fix the display settings before running.\n\nPress any key to quit.")
            self.win.close()
            raise RuntimeError(f"Refresh rate {measured} Hz is not {S.EXPECTED_FRAME_RATE_HZ} Hz")

        radius = dva_to_px(S.DISK_RADIUS_DEG, p["height_cm"], p["distance_cm"], p["window_px"])
        self.disk = visual.Circle(self.win, radius=radius, fillColor=S.BACKGROUND_COLOR, lineColor="black",
                                  lineWidth=S.DISK_LINE_WIDTH_PX, units="pix", pos=(0, 0))
        self.fixation = visual.TextStim(self.win, text="+", color="white", height=deg2pix(1, mon))
        self.audio_icon = visual.ImageStim(self.win, image=str(ASSETS / "audio_icon.png"), pos=(0, 100), size=(50, 50))
        meta["disk_radius_px"] = radius
        return measured

    # ── screens ───────────────────────────────────────────────────────────────
    def _message(self, text, color="white", wait_keys=True, min_wait=0.0):
        self.text.text, self.text.color = text, color
        self.text.draw()
        self.win.flip()
        if min_wait:
            self.core.wait(min_wait)
        if wait_keys:
            self.kb.clearEvents()
            self.kb.waitKeys()

    def instructions(self, task, practice):
        header = "PRACTICE\n\n" if practice else ""
        self._message(f"{header}Welcome to the experiment!\n\n{INSTRUCTIONS[task]}\n\n"
                      "Left arrow = FIRST longer, right arrow = SECOND longer.\n\nPress any key to start.")

    def rest(self, k, n):   # v1 wording and colours
        self._message(f"You can have a break now.\nYou are in block {k} out of {n}.\nPlease wait a few seconds.",
                      color="maroon", wait_keys=False, min_wait=S.BREAK_MIN_WAIT_S)
        self._message("Now you can press any key to continue!", color="teal")

    def order_cue(self, order, k, n):   # v1 cross-modal mini-block cue
        av = "A -> V" if order == 1 else "V -> A"
        self._message(f"This is going to be block {k + 1} out of {n}.\n\nIn this block the order of modalities is\n\n"
                      f"{av}\n\nPress any key to continue.")

    def feedback(self, correct):
        self._message("Correct!" if correct else "Incorrect!", color="green" if correct else "red",
                      wait_keys=False, min_wait=S.FEEDBACK_DUR_S)

    def finish(self, completed):
        text = ("End of the experiment.\nThank you for your participation.\n\nPress any key to exit."
                if completed else "The session was stopped.\n\nPress any key to exit.")
        self._message(text)
        self.win.close()

    # ── trial ─────────────────────────────────────────────────────────────────
    def run_trial(self, spec, sched, wave):
        win, core = self.win, self.core
        win.flip()
        core.wait(S.PRE_TRIAL_WAIT_S)

        snd = None
        if wave is not None:
            snd = self.sound.Sound(value=wave, sampleRate=S.SAMPLE_RATE, stereo=True,
                                   hamming=S.HAMMING_WINDOW, volume=playback_volume())
        black = np.zeros(sched.total + 1, dtype=bool)
        for v in sched.visual:
            if v is not None:
                black[v[0]:v[1]] = True
        show_disk = self.task != "unimodal_auditory"

        flips = np.full(sched.total + 1, np.nan)
        predicted_flip0 = audio_when = flip_ptb = None
        win.frameIntervals = []
        win.recordFrameIntervals = True
        quit_ = False
        for n in range(sched.total + 1):
            if show_disk:
                self.disk.fillColor = "black" if black[n] else S.BACKGROUND_COLOR
                self.disk.draw()
            else:
                self.fixation.draw()
            if n == 0:
                predicted_flip0 = win.getFutureFlipTime()
                if snd is not None:
                    flip_ptb = win.getFutureFlipTime(clock="ptb")
                    audio_when = flip_ptb + S.AUDIO_ONSET_OFFSET_S
                    snd.play(when=audio_when)
            flips[n] = win.flip()
            if self.kb.getKeys([S.QUIT_KEY], waitRelease=False):
                quit_ = True
                break
        win.recordFrameIntervals = False
        intervals = np.asarray(win.frameIntervals[1:], dtype=float)
        if quit_:
            if snd is not None:
                snd.stop()
            return dict(quit=True)

        win.flip()
        core.wait(S.POST_STIMULUS_WAIT_S)
        if snd is not None:
            snd.stop()

        recorded = {}
        for i, v in enumerate(sched.visual):
            if v is not None:
                recorded[(i, "on")] = flips[v[0]] - flips[0]
                recorded[(i, "off")] = flips[v[1]] - flips[0]

        # response screen (v1 text; the audio icon reminds participants to judge sound in the main task)
        self.text.text, self.text.color = "1st Longer (<-) vs 2nd Longer (->)", "white"
        self.kb.clearEvents()
        win.callOnFlip(self.kb.clock.reset)
        keys = []
        while not keys:
            self.text.draw()
            if self.task == "main":
                self.audio_icon.draw()
            win.flip()
            keys = self.kb.getKeys(list(S.RESPONSE_KEYS) + [S.QUIT_KEY], waitRelease=False)
        key = keys[0]
        if key.name == S.QUIT_KEY:
            return dict(quit=True)

        frame = 1.0 / S.EXPECTED_FRAME_RATE_HZ
        return dict(
            quit=False, response=S.RESPONSE_KEYS[key.name], key=key.name, rt=key.rt, flips=recorded,
            audio_onset_minus_flip0_s=(audio_when - flip_ptb) if snd is not None else np.nan,
            flip0_error_s=flips[0] - predicted_flip0,
            n_dropped_frames=int(np.sum(intervals > 1.5 * frame)) if intervals.size else 0,
            max_frame_interval_ms=float(intervals.max() * 1000) if intervals.size else np.nan,
        )
