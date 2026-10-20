"""PsychoPy presenter: window, stimuli, frame loop, sound and keyboard.

v2 timing: the sound is scheduled on the trial's first flip (frame 0), every frame is
flipped explicitly, the returned flip timestamps of each visual onset/offset are logged,
and frame intervals are checked for dropped frames. v1 started the sound from inside the
frame loop without scheduling and logged the loop time instead of the flip time.

v2 pacing: no fixed waits; answers count from the end of the second interval (keys pressed
earlier are discarded), the remaining noise plays in full, and the prompt appears only if
no answer came before the sound ended.
"""

from __future__ import annotations

import ctypes
import math
import sys
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
PROMPT = "1st Longer (<-) vs 2nd Longer (->)"       # v1


def dva_to_px(size_deg, height_cm, distance_cm, window_px):
    """v1 conversion (dva_to_pix.py), kept so v1_profile reproduces the v1 disk size."""
    deg_per_px = math.degrees(math.atan2(0.5 * height_cm, distance_cm)) / (0.5 * window_px)
    return size_deg / deg_per_px


def load_psychtoolbox():
    """Import psychtoolbox, preloading the libraries its macOS pip build expects.

    psychtoolbox 3.0.19 (pip, macOS arm64) does not link libusb and PortAudio itself; it
    expects them to be loaded already. Without them both its keyboard and its audio fail:
    PsychoPy then falls back silently and crashes at exit with
    "'KeyboardDevice' object has no attribute '_buffers'".
    """
    loaded = []
    if sys.platform == "darwin":
        folders = [Path(sys.prefix) / "lib", Path("/opt/homebrew/lib"), Path("/usr/local/lib")]
        for names in (("libusb-1.0.0.dylib",), ("libportaudio.2.dylib", "libportaudio.dylib")):
            path = next((f / n for f in folders for n in names if (f / n).exists()), None)
            if path is not None:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(str(path))
    try:
        import psychtoolbox  # noqa: F401
        from psychtoolbox import audio  # noqa: F401
    except ImportError as err:
        raise RuntimeError(
            "PsychToolbox could not be loaded, so neither precise audio nor the PTB keyboard is available.\n"
            f"{err}\nInstall libusb and portaudio in this environment "
            "(conda install -c conda-forge libusb portaudio) and try again.") from err
    return loaded


class PsychoPyPresenter:
    needs_audio = True

    def __init__(self, monitor_profile=S.DEFAULT_MONITOR):
        self.profile_name = monitor_profile
        self.profile = S.MONITOR_PROFILES[monitor_profile]
        self.fr = self.profile["frame_rate_hz"]
        self.task = None

    # ── setup ─────────────────────────────────────────────────────────────────
    def start(self, task, meta):
        meta["ptb_libraries_preloaded"] = load_psychtoolbox()
        from psychopy import prefs
        prefs.hardware["audioLib"] = ["PTB"]          # needed for sound onset scheduled on a flip
        prefs.hardware["audioLatencyMode"] = S.AUDIO_LATENCY_MODE
        from psychopy import core, monitors, sound, visual
        from psychopy.hardware import keyboard
        from psychopy.tools.monitorunittools import deg2pix

        if str(getattr(sound, "audioLib", "")).lower() != "ptb":
            raise RuntimeError(f"PsychoPy chose the '{getattr(sound, 'audioLib', None)}' audio backend, not PTB.")
        meta["sound_backend"] = "ptb"
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
        measured = self._check_refresh(meta)

        radius = dva_to_px(S.DISK_RADIUS_DEG, p["height_cm"], p["distance_cm"], p["window_px"])
        self.disk = visual.Circle(self.win, radius=radius, fillColor=S.BACKGROUND_COLOR, lineColor="black",
                                  lineWidth=S.DISK_LINE_WIDTH_PX, units="pix", pos=(0, 0))
        self.fixation = visual.TextStim(self.win, text="+", color="white", height=deg2pix(1, mon))
        self.audio_icon = visual.ImageStim(self.win, image=str(ASSETS / "audio_icon.png"), pos=(0, 100), size=(50, 50))
        meta["disk_radius_px"] = radius
        return measured

    def _measure_refresh(self, n_frames=180, warmup=30):
        """Median refresh rate from our own flip timestamps, share of irregular intervals, SD (ms).

        Replaces win.getActualFrameRate(), which returns None (rate "cannot be measured")
        whenever it does not find enough identical intervals in a row.
        """
        self.text.text, self.text.color = "Checking the display...", "white"
        times = []
        for i in range(warmup + n_frames):
            self.text.draw()
            t = self.win.flip()
            if i >= warmup:
                times.append(t)
        intervals = np.diff(np.asarray(times, dtype=float))
        median = float(np.median(intervals))
        irregular = float(np.mean(np.abs(intervals - median) > 0.0015))
        return (1.0 / median if median > 0 else float("nan")), irregular, float(intervals.std() * 1000)

    def _check_refresh(self, meta):
        rate, irregular, jitter = self._measure_refresh()
        meta.update(frame_rate_measured_hz=rate, irregular_frame_share=irregular, frame_interval_sd_ms=jitter)
        rate_ok = abs(rate - self.fr) <= S.FRAME_RATE_TOLERANCE_HZ
        steady = irregular <= S.MAX_IRREGULAR_FRAMES
        if rate_ok and steady:
            return rate
        report = (f"Display check: {rate:.2f} Hz, {100 * irregular:.0f}% irregular frames (SD {jitter:.2f} ms).\n"
                  f"The '{self.profile_name}' monitor profile needs a steady {self.fr:g} Hz.\n\n")
        if not rate_ok:
            others = [n for n, p in S.MONITOR_PROFILES.items() if abs(p["frame_rate_hz"] - rate) <= S.FRAME_RATE_TOLERANCE_HZ]
            report += (f"This display runs at about {rate:.0f} Hz. "
                       + (f"Choose the '{others[0]}' profile, " if others else "Add a profile for it, ")
                       + f"or set the display to {self.fr:g} Hz.\n")
        else:
            report += ("Frames are irregular: run full screen, close other programs, and use a fixed refresh rate "
                       "(not ProMotion / variable refresh).\n")
        if rate_ok and self.profile.get("testing_only"):
            self._message(report + "\nTesting profile: press C to continue anyway, Escape to quit.", wait_keys=False)
            self.kb.clearEvents()
            if self.kb.waitKeys(keyList=["c", S.QUIT_KEY])[0].name == "c":
                meta["refresh_warning_overridden"] = True
                return rate
        else:
            self._message(report + "\nPress any key to quit.")
        self.win.close()
        raise RuntimeError(report.strip())

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
                      "Left arrow = FIRST longer, right arrow = SECOND longer.\n"
                      "You can answer as soon as the second one has ended.\n\nPress any key to start.")

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
    def _draw_frame(self, black):
        if self.task == "unimodal_auditory":
            self.fixation.draw()
        else:
            self.disk.fillColor = "black" if black else S.BACKGROUND_COLOR
            self.disk.draw()

    def run_trial(self, spec, sched, wave):
        win, kb = self.win, self.kb
        if S.PRE_TRIAL_WAIT_S:
            win.flip()
            self.core.wait(S.PRE_TRIAL_WAIT_S)

        snd = None
        if wave is not None:
            snd = self.sound.Sound(value=wave, sampleRate=S.SAMPLE_RATE, stereo=True,
                                   hamming=S.HAMMING_WINDOW, volume=playback_volume())
        black = np.zeros(sched.total + 1, dtype=bool)
        for v in sched.visual:
            if v is not None:
                black[v[0]:v[1]] = True
        end2 = sched.second_interval_end()          # answers count from this frame's flip
        response_keys = list(S.RESPONSE_KEYS) + [S.QUIT_KEY]

        flips = np.full(sched.total + 1, np.nan)
        predicted_flip0 = audio_when = flip_ptb = None
        key, stream_done_before_key = None, False
        kb.clearEvents()
        win.frameIntervals = []
        win.recordFrameIntervals = True
        for n in range(sched.total + 1):
            self._draw_frame(black[n])
            if n == 0:
                predicted_flip0 = win.getFutureFlipTime()
                if snd is not None:
                    flip_ptb = win.getFutureFlipTime(clock="ptb")
                    audio_when = flip_ptb + S.AUDIO_ONSET_OFFSET_S
                    snd.play(when=audio_when)
            if n == end2:                            # discard earlier keys, start the RT clock
                win.callOnFlip(kb.clearEvents)
                win.callOnFlip(kb.clock.reset)
            flips[n] = win.flip()
            if n < end2:
                if kb.getKeys([S.QUIT_KEY], waitRelease=False):
                    key = "quit"
                    break
            elif key is None:
                keys = kb.getKeys(response_keys, waitRelease=False)
                if keys:
                    key = keys[0]
                    if key.name == S.QUIT_KEY:
                        break
        win.recordFrameIntervals = False
        intervals = np.asarray(win.frameIntervals[1:], dtype=float)
        if key == "quit" or (key is not None and key.name == S.QUIT_KEY):
            if snd is not None:
                snd.stop()
            return dict(quit=True)

        if key is None:                              # sound over, no answer yet: show the prompt
            stream_done_before_key = True
            self.text.text, self.text.color = PROMPT, "white"
            while key is None:
                self.text.draw()
                if self.task == "main":
                    self.audio_icon.draw()
                win.flip()
                keys = kb.getKeys(response_keys, waitRelease=False)
                key = keys[0] if keys else None
            if key.name == S.QUIT_KEY:
                return dict(quit=True)
        else:
            self._draw_frame(False)                  # outline/fixation stays on into the next trial
            win.flip()
        if snd is not None:
            snd.stop()

        recorded = {}
        for i, v in enumerate(sched.visual):
            if v is not None:
                recorded[(i, "on")] = flips[v[0]] - flips[0]
                recorded[(i, "off")] = flips[v[1]] - flips[0]
        frame = 1.0 / self.fr
        return dict(
            quit=False, response=S.RESPONSE_KEYS[key.name], key=key.name, rt=key.rt, flips=recorded,
            prompt_shown=stream_done_before_key,
            audio_onset_minus_flip0_s=(audio_when - flip_ptb) if snd is not None else np.nan,
            flip0_error_s=flips[0] - predicted_flip0,
            n_dropped_frames=int(np.sum(intervals > 1.5 * frame)) if intervals.size else 0,
            max_frame_interval_ms=float(intervals.max() * 1000) if intervals.size else np.nan,
        )
