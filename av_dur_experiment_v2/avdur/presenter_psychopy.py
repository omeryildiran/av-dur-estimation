"""PsychoPy presenter: window, stimuli, frame loop, sound and keyboard.

v2 timing: every trial starts with a sync flip; the sound is then scheduled for the flip of
frame 0 (noise onset), far enough ahead for the sound card's output latency. Every frame is
flipped explicitly, the flip timestamps of each visual onset/offset are logged, frame
intervals are checked for dropped frames, and PTB's estimate of when the sound actually
started is logged against the frame-0 flip. v1 started the sound from inside the frame loop
without scheduling and logged the loop time instead of the flip time.

Pacing (as v1, shorter waits): the prompt appears on every trial, POST_STIM_WAIT_S after the
sound (and the last flash) has ended; keys pressed before it are discarded, and the response
time is measured from its first flip. Each trial starts PRE_TRIAL_WAIT_S after the answer.

Sound card: chosen explicitly (AUDIO_OUTPUT_DEVICE) and opened before the first trial, so a
sound problem stops the session before any file is written.

Autotest: answers come from the simulated observer at a random time after the second
interval, and message screens continue by themselves. Escape still quits during trials.
"""

from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import numpy as np

import experiment_settings as S
from avdur.audio import playback_volume

ASSETS = Path(__file__).resolve().parents[1] / "assets"

INSTRUCTIONS = {
    "noise_calibration": "You will hear two sounds in noise.\nWhich sound was longer?\n\n"
                         "The noise will get louder and softer during this part.",
    "unimodal_auditory": "You will hear two sounds in noise.\nWhich sound was longer?",
    "unimodal_visual": "You will see two dark disks.\nWhich disk was shown longer?",
    "crossmodal": "Each trial has one sound and one disk.\nWhich one was longer?",
    "main": "You will hear two sounds; disks will flash with them.\nJudge only the SOUNDS: which sound was longer?",
}
PROMPT = "1st Longer (<-) vs 2nd Longer (->)"       # v1


def stimulus_geometry(profile):
    """Pixel sizes of the stimuli and, if the picture size and distance are known, their true size."""
    out = dict(disk_radius_px=S.DISK_RADIUS_PX, fixation_height_px=S.FIXATION_HEIGHT_PX)
    height, distance = profile.get("picture_height_cm"), profile.get("viewing_distance_cm")
    if height and distance:
        cm_per_px = height / profile["window_px"]
        radius_cm = S.DISK_RADIUS_PX * cm_per_px
        out.update(disk_diameter_cm=2 * radius_cm, disk_diameter_deg=2 * math.degrees(math.atan2(radius_cm, distance)))
        if profile.get("picture_width_cm"):
            out["pixel_aspect_w_over_h"] = profile["picture_width_cm"] / height
    return out


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


def find_output_device(wanted, devices):
    """The output device named `wanted`, or the only one whose name contains it (any case)."""
    exact = [d for d in devices if d["DeviceName"].lower() == str(wanted).lower()]
    matches = exact or [d for d in devices if str(wanted).lower() in d["DeviceName"].lower()]
    if len(matches) != 1:
        what = "No output device matches" if not matches else "More than one output device matches"
        raise LookupError(f"{what} AUDIO_OUTPUT_DEVICE = {wanted!r}.")
    return matches[0]


def audio_problem_text(err, wanted, devices):
    names = ", ".join(repr(d["DeviceName"]) for d in devices) or "none"
    target = repr(wanted) if wanted is not None else "the Mac's current output (System Settings > Sound)"
    return (f"The sound card could not be started: {err}\n\n"
            f"Requested output: {target}.\nOutput devices found: {names}.\n\n"
            "Check that the speakers or audio interface are connected and not in use by another program "
            "(the line starting with PTB-ERROR above gives PortAudio's reason). Choose another output in "
            "System Settings > Sound, or set AUDIO_OUTPUT_DEVICE (or the profile's audio_device) in "
            "experiment_settings.py to part of a device name.")


class PsychoPyPresenter:
    needs_audio = True

    def __init__(self, monitor_profile=S.DEFAULT_MONITOR, autotest_seed=None, screen=None):
        self.profile_name = monitor_profile
        self.profile = S.MONITOR_PROFILES[monitor_profile]
        self.fr = self.profile["frame_rate_hz"]
        self.frame_period = 1.0 / self.fr           # replaced by the measured period at start
        self.task, self.speaker, self.lead_flips = None, None, 1
        self.win, self.setup_meta, self.measured = None, {}, None
        self.screen_override = screen
        self.auto = self.auto_rng = None
        if autotest_seed is not None:
            from avdur.presenter_sim import SimulatedPresenter
            self.auto = SimulatedPresenter(seed=autotest_seed)
            self.auto_rng = np.random.default_rng(autotest_seed + 1)

    # ── setup ─────────────────────────────────────────────────────────────────
    def start(self, task, meta):
        if self.win is not None:                     # a later block in the same window: reuse the setup
            return self._next_block(task, meta)
        before = set(meta)
        measured = self._start(task, meta)
        self.measured = measured
        self.setup_meta = {k: meta[k] for k in set(meta) - before}
        return measured

    def _next_block(self, task, meta):
        from avdur.session import NEEDS_AUDIO
        self.task = task
        if self.auto is not None:
            self.auto.task = task
        if task in NEEDS_AUDIO and self.speaker is None:
            before = set(meta)
            self._open_audio(meta)
            self.setup_meta.update({k: meta[k] for k in set(meta) - before})
        meta.update(self.setup_meta)
        meta["same_window_as_previous_block"] = True
        return self.measured

    def _start(self, task, meta):
        meta["ptb_libraries_preloaded"] = load_psychtoolbox()
        from psychopy import prefs
        prefs.hardware["audioLib"] = ["PTB"]          # needed for sound onset scheduled on a flip
        prefs.hardware["audioLatencyMode"] = S.AUDIO_LATENCY_MODE
        from psychopy import core, monitors, sound, visual
        from psychopy.hardware import keyboard
        from avdur.session import NEEDS_AUDIO

        if str(getattr(sound, "audioLib", "")).lower() != "ptb":
            raise RuntimeError(f"PsychoPy chose the '{getattr(sound, 'audioLib', None)}' audio backend, not PTB.")
        meta["sound_backend"] = "ptb"
        self.core, self.sound, self.task = core, sound, task
        if self.auto is not None:
            self.auto.task = task
        p = self.profile
        from avdur import screens
        found = screens.list_screens()
        screen, how = screens.choose(p, self.screen_override, found)
        meta.update(screen_index=screen, screen_choice=how, screens_found=[screens.describe(s) for s in found])
        print(f"Display: screen {screen} ({how}); screens found: "
              + ("; ".join(screens.describe(s) for s in found) or "unknown"), flush=True)
        mon = monitors.Monitor("avdur_v2", width=p.get("picture_width_cm") or p.get("picture_height_cm") or 30.0,
                               distance=p.get("viewing_distance_cm") or 57.0)
        mon.setSizePix((p["window_px"], p["window_px"]))
        self.win = visual.Window(size=(p["window_px"], p["window_px"]), fullscr=p["fullscreen"], screen=screen,
                                 monitor=mon, units="pix", color=S.BACKGROUND_COLOR, colorSpace="rgb",
                                 waitBlanking=True, allowGUI=False)
        self.win.mouseVisible = False
        self.text = visual.TextStim(self.win, text="", color="white", height=S.TEXT_HEIGHT_PX,
                                    wrapWidth=0.9 * p["window_px"])
        self.kb = keyboard.Keyboard(backend="ptb")
        measured = self._check_refresh(meta)
        # PsychoPy predicts flips as last flip + monitorFramePeriod, and assumes 60 Hz when its own
        # measurement fails ("cannot be measured"); use the period measured above instead.
        self.frame_period = 1.0 / measured
        self.win.monitorFramePeriod = self.frame_period
        self.win.refreshThreshold = 1.2 * self.frame_period
        if task in NEEDS_AUDIO:
            self._open_audio(meta)

        self.disk = visual.Circle(self.win, radius=S.DISK_RADIUS_PX, fillColor=S.BACKGROUND_COLOR, lineColor="black",
                                  lineWidth=S.DISK_LINE_WIDTH_PX, units="pix", pos=(0, 0))
        self.fixation = visual.TextStim(self.win, text="+", color="white", height=S.FIXATION_HEIGHT_PX)
        self.audio_icon = visual.ImageStim(self.win, image=str(ASSETS / "audio_icon.png"), pos=(0, 100), size=(50, 50))
        meta.update(stimulus_geometry(p))
        return measured

    def _open_audio(self, meta):
        """Open the chosen output now: a sound problem stops the session before its first trial."""
        from psychtoolbox import audio
        wanted = self.profile.get("audio_device", S.AUDIO_OUTPUT_DEVICE)
        devices = [d for d in audio.get_devices() if d["NrOutputChannels"] > 0]
        meta.update(audio_output_requested=wanted, audio_devices_found=[d["DeviceName"] for d in devices])
        try:
            # -1 = PortAudio's default output, i.e. the one chosen in System Settings > Sound
            self.speaker = -1 if wanted is None else int(find_output_device(wanted, devices)["DeviceIndex"])
            silent = self.sound.Sound(value=np.zeros((S.SAMPLE_RATE // 10, 2)), sampleRate=S.SAMPLE_RATE,
                                      stereo=True, hamming=False, volume=0.0, speaker=self.speaker)
            status = silent.stream.status
        except (OSError, LookupError) as err:
            self._fail(audio_problem_text(err, wanted, devices))
        index = int(status["OutDeviceIndex"])
        name = next((d["DeviceName"] for d in devices if int(d["DeviceIndex"]) == index), f"device {index}")
        rate, latency = float(status["SampleRate"]), float(status["PredictedLatency"])
        if abs(rate - S.SAMPLE_RATE) > 0.5:
            self._fail(f"The sound card '{name}' runs at {rate:.0f} Hz, not {S.SAMPLE_RATE} Hz, so every sound "
                       f"would play at the wrong speed. Set it to {S.SAMPLE_RATE} Hz in Audio MIDI Setup.")
        self.lead_flips = max(1, math.ceil((latency + S.AUDIO_SCHEDULE_MARGIN_S) / self.frame_period))
        meta.update(audio_device=name, audio_device_index=index, audio_sample_rate_hz=rate,
                    audio_output_latency_s=latency, audio_schedule_lead_frames=self.lead_flips)
        print(f"Sound output: {name} (PTB device {index}), {rate:.0f} Hz, output latency {1000 * latency:.1f} ms; "
              f"sound scheduled {self.lead_flips} frame(s) ahead.", flush=True)
        silent.play()                                 # wake the output path before the first trial
        self.core.wait(0.15)
        silent.stop()

    def _fail(self, report) -> NoReturn:
        """Show why the session cannot start, close the window and stop."""
        self._message(report + "\n\nPress any key to quit.", height=0.6 * S.TEXT_HEIGHT_PX)
        self.win.close()
        raise RuntimeError(report)

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
            if self._wait_keys(["c", S.QUIT_KEY]) == "c":
                meta["refresh_warning_overridden"] = True
                return rate
            self.win.close()
            raise RuntimeError(report.strip())
        self._fail(report.strip())

    # ── screens ───────────────────────────────────────────────────────────────
    def _wait_keys(self, keys=None):
        """Name of the key pressed; an autotest continues by itself with the first allowed key."""
        self.kb.clearEvents()
        if self.auto is None:
            return self.kb.waitKeys(keyList=keys)[0].name
        self.core.wait(S.AUTOTEST_SCREEN_S)
        return keys[0] if keys else "space"

    def _message(self, text, color="white", wait_keys=True, min_wait=0.0, height=S.TEXT_HEIGHT_PX):
        self.text.text, self.text.color, self.text.height = text, color, height
        self.text.draw()
        self.win.flip()
        self.text.height = S.TEXT_HEIGHT_PX
        if min_wait:
            self.core.wait(min_wait if self.auto is None else min(min_wait, S.AUTOTEST_SCREEN_S))
        if wait_keys:
            self._wait_keys()

    def instructions(self, task, practice):
        header = ("AUTOTEST: automatic answers\n\n" if self.auto is not None else "") + ("PRACTICE\n\n" if practice else "")
        self._message(f"{header}Welcome to the experiment!\n\n{INSTRUCTIONS[task]}\n\n"
                      "Left arrow = FIRST longer, right arrow = SECOND longer.\n"
                      "You can answer as soon as the second one has ended.\n\nPress any key to start.")

    def rest(self, k, n):   # v1 wording and colours
        if self.auto is not None:
            print(f"[autotest] {self.task}: break {k} of {n}", flush=True)
        self._message(f"You can have a break now.\nYou are in block {k} out of {n}.\nPlease wait a few seconds.",
                      color="maroon", wait_keys=False, min_wait=S.BREAK_MIN_WAIT_S)
        self._message("Now you can press any key to continue!", color="black")

    def order_cue(self, order, k, n):   # v1 cross-modal mini-block cue
        av = "A -> V" if order == 1 else "V -> A"
        self._message(f"This is going to be block {k + 1} out of {n}.\n\nIn this block the order of modalities is\n\n"
                      f"{av}\n\nPress any key to continue.")

    def feedback(self, correct):
        self._message("Correct!" if correct else "Incorrect!", color="green" if correct else "red",
                      wait_keys=False, min_wait=S.FEEDBACK_DUR_S)

    def wait_screen(self, text, ready, min_wait=0.0):
        """Show `text` until ready() is true and at least min_wait s have passed (the fit runs meanwhile)."""
        if self.auto is not None:
            min_wait = min(min_wait, S.AUTOTEST_SCREEN_S)
        self.text.text, self.text.color = text, "white"
        clock = self.core.Clock()
        while not (ready() and clock.getTime() >= min_wait):
            self.text.draw()
            self.win.flip()
            self.core.wait(0.1)

    def experimenter_check(self):
        """'Please wait for the experimenter'; C continues, Escape ends. Returns True to continue."""
        self._message("Please wait for the experimenter.\n\n(Experimenter: C = continue, Escape = end the session)",
                      wait_keys=False)
        return self._wait_keys(["c", S.QUIT_KEY]) == "c"

    def finish(self, completed):
        text = ("End of the experiment.\nThank you for your participation.\n\nPress any key to exit."
                if completed else "The session was stopped.\n\nPress any key to exit.")
        self._message(text)
        self.win.close()

    # ── trial ─────────────────────────────────────────────────────────────────
    def _draw_frame(self, black):
        if self.task in ("unimodal_auditory", "noise_calibration"):
            self.fixation.draw()
        else:
            self.disk.fillColor = "black" if black else S.BACKGROUND_COLOR
            self.disk.draw()

    def _auto_answer(self, spec, sched):
        """Autotest: (key, time after the prompt appeared) from the simulated observer, or None."""
        if self.auto is None:
            return None
        response = self.auto.run_trial(spec, sched, None)["response"]
        rt = float(self.auto_rng.uniform(*S.AUTOTEST_RT_RANGE_S))
        return ("left" if response == 1 else "right"), rt

    def _poll_answer(self, keys, auto):
        """A pressed answer key (or Escape), or the autotest answer once its time has come."""
        pressed = self.kb.getKeys(keys, waitRelease=False)
        if pressed:
            key, seen = pressed[0], self.kb.clock.getTime()          # reset on the prompt's first flip
            # On macOS, PsychoPy's PTB keyboard can time keys on another clock than the one it was
            # reset with (RTs around -1e6 s). Then use the time the key was first seen (1-frame
            # resolution). The answer itself is unaffected.
            rt = getattr(key, "rt", None)
            if rt is None or not (-0.005 <= rt <= seen + 0.05):
                return SimpleNamespace(name=key.name, rt=seen, rt_source="first seen (keyboard time invalid)")
            return key
        if auto is not None:
            t = self.kb.clock.getTime()                  # reset on the prompt's first flip
            if t >= auto[1]:
                return SimpleNamespace(name=auto[0], rt=t)
        return None

    def _frames(self, seconds):
        return int(round(seconds / self.frame_period))

    def run_trial(self, spec, sched, wave):
        win, kb = self.win, self.kb
        snd = None
        if wave is not None:
            snd = self.sound.Sound(value=wave, sampleRate=S.SAMPLE_RATE, stereo=True,
                                   hamming=S.HAMMING_WINDOW, volume=playback_volume(), speaker=self.speaker)
        black = np.zeros(sched.total + 1, dtype=bool)
        for v in sched.visual:
            if v is not None:
                black[v[0]:v[1]] = True
        response_keys = list(S.RESPONSE_KEYS) + [S.QUIT_KEY]
        auto = self._auto_answer(spec, sched)

        # Pre-trial wait and sync flips (outline/fixation on), then frame 0 (noise onset)
        # `lead_flips` refreshes later: the sound is scheduled for that flip, at least the output
        # latency ahead. Predicting from fresh flips keeps the frame-period error from adding up.
        for _ in range(self._frames(S.PRE_TRIAL_WAIT_S) + S.SYNC_FLIPS):
            self._draw_frame(False)
            win.flip()
        predicted_flip0 = win.getFutureFlipTime() + (self.lead_flips - 1) * self.frame_period
        flip0_ptb = win.getFutureFlipTime(clock="ptb") + (self.lead_flips - 1) * self.frame_period
        if snd is not None:
            snd.play(when=flip0_ptb + S.AUDIO_ONSET_OFFSET_S)
        for _ in range(self.lead_flips - 1):
            self._draw_frame(False)
            win.flip()

        # Stimulus: answer keys are ignored (only Escape is read) until the prompt appears.
        flips = np.full(sched.total + 1, np.nan)
        win.frameIntervals = []
        win.recordFrameIntervals = True
        quit_ = False
        for n in range(sched.total + 1):
            self._draw_frame(black[n])
            flips[n] = win.flip()
            if kb.getKeys([S.QUIT_KEY], waitRelease=False):
                quit_ = True
                break
        win.recordFrameIntervals = False
        intervals = np.asarray(win.frameIntervals[1:], dtype=float)
        if not quit_:
            for _ in range(self._frames(S.POST_STIM_WAIT_S)):   # blank pause after the sound has ended
                self._draw_frame(False)
                win.flip()

        # Prompt on every trial: keys pressed before it are discarded; RT counts from its first flip.
        key = SimpleNamespace(name=S.QUIT_KEY) if quit_ else None
        prompt_onset = np.nan
        if key is None:
            self.text.text, self.text.color = PROMPT, "white"
            win.callOnFlip(kb.clearEvents)
            win.callOnFlip(kb.clock.reset)
            first = True
            while key is None:
                self.text.draw()
                if self.task == "main":
                    self.audio_icon.draw()
                t = win.flip()
                if first:
                    prompt_onset, first = t - flips[0], False
                key = self._poll_answer(response_keys, auto)
        if key.name == S.QUIT_KEY:
            if snd is not None:
                snd.stop()
            return dict(quit=True)
        audio = self._audio_check(snd, flips[0] + (flip0_ptb - predicted_flip0))

        recorded = {}
        for i, v in enumerate(sched.visual):
            if v is not None:
                recorded[(i, "on")] = flips[v[0]] - flips[0]
                recorded[(i, "off")] = flips[v[1]] - flips[0]
        return dict(
            quit=False, response=S.RESPONSE_KEYS[key.name], key=key.name, rt=key.rt,
            rt_source=getattr(key, "rt_source", "automatic" if self.auto is not None else "keyboard"), flips=recorded,
            prompt_shown=True, prompt_onset_s=prompt_onset, **audio,
            flip0_error_s=flips[0] - predicted_flip0,
            n_dropped_frames=int(np.sum(intervals > 1.5 * self.frame_period)) if intervals.size else 0,
            max_frame_interval_ms=float(intervals.max() * 1000) if intervals.size else np.nan,
        )

    @staticmethod
    def _audio_check(snd, flip0_ptb):
        """PTB's estimate of when the sound reached the output, relative to the frame-0 flip, and
        the stream's buffer underruns so far; then stop the sound (it has played in full)."""
        if snd is None:
            return {}
        out = {}
        try:
            out["audio_start_minus_flip0_s"] = float(snd.statusDetailed["StartTime"]) - flip0_ptb
            out["audio_xruns"] = int(snd.stream.status["XRuns"])
        except (AttributeError, KeyError, TypeError):
            pass
        snd.stop()
        return out
