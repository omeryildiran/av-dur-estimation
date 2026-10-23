"""Response logic of the PsychoPy presenter, tested with fake window/keyboard/sound objects.

    python tests/test_presenter_logic.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur.design import TrialSpec  # noqa: E402
from avdur.presenter_psychopy import PsychoPyPresenter, find_output_device, stimulus_geometry  # noqa: E402
from avdur.schedule import make_schedule  # noqa: E402

FRAME = 1 / S.EXPECTED_FRAME_RATE_HZ


class Time:
    now = 0.0


class Clock:
    def __init__(self):
        self.last_reset = 0.0

    def reset(self):
        self.last_reset = Time.now

    def getTime(self):
        return Time.now - self.last_reset


class Win:
    def __init__(self, frame=FRAME, every_nth_doubled=0):
        self.frameIntervals, self.recordFrameIntervals, self._on_flip = [], False, []
        self.frame, self.every_nth_doubled, self.n = frame, every_nth_doubled, 0

    def getFutureFlipTime(self, clock=None):
        return Time.now + self.frame

    def close(self):
        pass

    def callOnFlip(self, fn):
        self._on_flip.append(fn)

    def flip(self):
        self.n += 1
        step = self.frame * (2 if self.every_nth_doubled and self.n % self.every_nth_doubled == 0 else 1)
        Time.now += step
        if self.recordFrameIntervals:
            self.frameIntervals.append(step)
        for fn in self._on_flip:
            fn()
        self._on_flip = []
        return Time.now


class Keyboard:
    """Scripted key presses: [(absolute time, key name)]."""

    def __init__(self, presses):
        self.clock, self.presses = Clock(), [dict(t=t, name=n, used=False) for t, n in presses]
        self.wait_answer = "space"

    def _pending(self, names):
        return [p for p in self.presses if not p["used"] and p["t"] <= Time.now and p["name"] in names]

    def getKeys(self, keyList, waitRelease=False):
        out = []
        for p in self._pending(keyList):
            p["used"] = True
            out.append(SimpleNamespace(name=p["name"], rt=p["t"] - self.clock.last_reset))
        return out

    def clearEvents(self):
        for p in self.presses:
            if p["t"] <= Time.now:
                p["used"] = True

    def waitKeys(self, keyList=None):
        if self.wait_answer is None:
            raise AssertionError("an autotest must not wait for a key")
        return [SimpleNamespace(name=self.wait_answer)]


class Sound:
    """Starts exactly when scheduled (the fake window's PTB clock equals its own)."""
    last = None

    def __init__(self, **kwargs):
        self.kwargs, self.when, self.stopped = kwargs, None, False
        self.stream = SimpleNamespace(status={"XRuns": 0})
        Sound.last = self

    def play(self, when=None):
        self.when = when

    @property
    def statusDetailed(self):
        return {"StartTime": self.when}

    def stop(self):
        self.stopped = True


def presenter(task, presses, profile="lab_crt", win=None, autotest_seed=None, lead=1):
    Time.now = 0.0
    p = PsychoPyPresenter(profile, autotest_seed=autotest_seed)
    p.task, p.win, p.kb, p.lead_flips = task, win or Win(), Keyboard(presses), lead
    if p.auto is not None:
        p.auto.task = task
        p.kb.wait_answer = None
    def wait(s):
        Time.now += s

    p.core = SimpleNamespace(wait=wait, Clock=Clock)
    p.sound = SimpleNamespace(Sound=Sound)
    drawable = SimpleNamespace(draw=lambda: None, fillColor=None, text="", color="")
    p.disk, p.fixation, p.audio_icon, p.text = drawable, drawable, drawable, drawable
    return p


def schedule(task, order=2):
    return make_schedule(task, order, 0.6, 0.083, 0.1, np.random.default_rng(1), S.EXPECTED_FRAME_RATE_HZ)


def times(task, lead=1):
    """Flip times of the end of interval 2 and of the last frame. Before frame 0 come the pre-trial
    wait and SYNC_FLIPS sync flips, then `lead` - 1 idle flips, so frame n flips at (n + k) frames."""
    sched = schedule(task)
    k = round(S.PRE_TRIAL_WAIT_S / FRAME) + S.SYNC_FLIPS + lead
    return (sched.second_interval_end() + k) * FRAME, (sched.total + k) * FRAME


def prompt_time(task, lead=1):
    """First flip of the prompt: POST_STIM_WAIT_S of blank frames after the last stimulus frame."""
    return times(task, lead)[1] + (round(S.POST_STIM_WAIT_S / FRAME) + 1) * FRAME


def spec(task):
    return TrialSpec(task, None if task == "unimodal_visual" else 0.1, 0.083 if task == "main" else 0.0,
                     "staircase", "2D1U", None, 2)


def run(task, presses, lead=1, p=None):
    p = p or presenter(task, presses, lead=lead)
    end2, stream_end = times(task, lead)
    wave = None if task == "unimodal_visual" else np.zeros(10, np.float32)
    return p.run_trial(spec(task), schedule(task), wave), end2, stream_end


def check_sound_and_flip_timing():
    for lead in (1, 2, 3):
        r, _, _ = run("main", [(10.0, "left")], lead=lead)
        assert abs(r["flip0_error_s"]) < 1e-9, (lead, r["flip0_error_s"])
        k = round(S.PRE_TRIAL_WAIT_S / FRAME) + S.SYNC_FLIPS + lead
        assert abs(Sound.last.when - k * FRAME) < 1e-9, "sound scheduled on the frame-0 flip"
        assert abs(r["audio_start_minus_flip0_s"]) < 1e-9 and r["audio_xruns"] == 0 and Sound.last.stopped
        assert Sound.last.kwargs["speaker"] is None or isinstance(Sound.last.kwargs["speaker"], int)


def check_autotest():
    for task in ("main", "crossmodal", "unimodal_auditory", "unimodal_visual"):
        end2, stream_end = times(task)
        for rt in (0.0, 0.12, 0.7):                          # automatic answers count from the prompt
            p = presenter(task, [], autotest_seed=3)
            p._auto_answer = lambda spec_, sched_, rt=rt: ("left", rt)
            r, _, _ = run(task, [], p=p)
            assert r["key"] == "left" and rt <= r["rt"] < rt + FRAME + 1e-9 and r["prompt_shown"], (task, rt, r)
        p = presenter(task, [], autotest_seed=3)           # real automatic answers from the observer
        keys = {run(task, [], p=p)[0]["key"] for _ in range(30)}
        assert keys == {"left", "right"}, keys
        p = presenter(task, [(stream_end * 0.5, "escape")], autotest_seed=3)
        assert run(task, [], p=p)[0]["quit"], "Escape still stops an autotest"
    p = presenter("main", [], autotest_seed=3)               # screens continue by themselves
    p.instructions("main", False)
    p.rest(1, 10)
    p.order_cue(1, 0, 4)
    p.feedback(True)
    p.finish(True)
    p = presenter("main", [], profile="office_test", win=Win(frame=1 / 75, every_nth_doubled=10), autotest_seed=3)
    meta = {}
    assert abs(p._check_refresh(meta) - 75) < 0.5 and meta["refresh_warning_overridden"]


def check_block_screens():
    """Between blocks: the rest screen lasts until the fit is ready; the experimenter key decides."""
    p = presenter("main", [])
    ready_at = 4.0
    p.wait_screen("rest", lambda: Time.now >= ready_at, min_wait=10.0)
    assert Time.now >= 10.0, "minimum rest"
    Time.now = 0.0
    p.wait_screen("rest", lambda: Time.now >= ready_at, min_wait=1.0)
    assert ready_at <= Time.now < ready_at + 0.2, "waits for the fit"
    p.kb.wait_answer = "c"
    assert p.experimenter_check()
    p.kb.wait_answer = "escape"
    assert not p.experimenter_check()
    p = presenter("main", [], autotest_seed=1)
    assert p.experimenter_check(), "an autotest continues by itself"
    p.win, p.measured, p.setup_meta, p.speaker = p.win, 120.0, {"audio_device": "X"}, -1
    meta = {}
    assert p.start("crossmodal", meta) == 120.0 and meta["audio_device"] == "X" and p.task == "crossmodal"


def check_screen_choice():
    """The participant's screen: the launcher's choice, else the one with the profile's resolution."""
    from avdur import screens
    lab = S.MONITOR_PROFILES["lab_crt"]
    two = [dict(index=0, width=800, height=800, x=0, y=0), dict(index=1, width=1920, height=1080, x=800, y=0)]
    assert screens.choose(lab, screens=two)[0] == 0, "the 800 x 800 CRT wins over the profile's screen number"
    assert screens.choose(lab, screens=list(reversed(two)))[0] == 0
    assert screens.choose(lab, override=1, screens=two)[0] == 1, "the launcher's choice wins"
    assert screens.choose(lab, screens=[])[0] == lab["screen"], "no screen list: the profile's number"
    assert screens.choose(S.MONITOR_PROFILES["office_test"], screens=two)[0] == 0
    for bad in ([dict(index=0, width=2560, height=1440, x=0, y=0), dict(index=1, width=1920, height=1080, x=2560, y=0)],):
        try:
            screens.choose(lab, screens=bad)
            raise AssertionError("no 800 x 800 screen must stop the session")
        except RuntimeError as err:
            assert "800 x 800" in str(err) and "screen 1: 1920 x 1080" in str(err)
    try:
        screens.choose(lab, override=5, screens=two)
        raise AssertionError("a screen that does not exist")
    except ValueError:
        pass


def check_device_choice_and_geometry():
    devices = [dict(DeviceName="DELL S2722DZ", DeviceIndex=0.0), dict(DeviceName="Mac mini Speakers", DeviceIndex=1.0),
               dict(DeviceName="External Headphones", DeviceIndex=3.0), dict(DeviceName="Speakers", DeviceIndex=4.0)]
    assert find_output_device("headphones", devices)["DeviceIndex"] == 3
    assert find_output_device("Speakers", devices)["DeviceIndex"] == 4          # exact name wins
    assert find_output_device("mac mini", devices)["DeviceIndex"] == 1
    for wanted in ("e", "Scarlett"):
        try:
            find_output_device(wanted, devices)
            raise AssertionError(wanted)
        except LookupError:
            pass
    lab = stimulus_geometry(S.MONITOR_PROFILES["lab_crt"])
    assert abs(lab["disk_radius_px"] - 42.75) < 0.01 and abs(lab["fixation_height_px"] - 22.34) < 0.01
    assert abs(lab["disk_diameter_deg"] - 2.96) < 0.01, lab
    assert "disk_diameter_deg" not in stimulus_geometry(S.MONITOR_PROFILES["office_test"])


def check_refresh():
    p = presenter("main", [])                                   # lab profile, 120 Hz display
    assert abs(p._check_refresh({}) - 120) < 0.5
    p = presenter("main", [], win=Win(frame=1 / 75))            # lab profile on the 75 Hz office display
    try:
        p._check_refresh({})
        raise AssertionError("a 75 Hz display must not pass the 120 Hz profile")
    except RuntimeError as err:
        assert "office_test" in str(err), err
    p = presenter("main", [], profile="office_test", win=Win(frame=1 / 75, every_nth_doubled=10))
    p.kb.wait_answer = "c"                                      # irregular but right rate: C continues (test only)
    meta = {}
    assert abs(p._check_refresh(meta) - 75) < 0.5 and meta["refresh_warning_overridden"]


def main():
    p = presenter("main", [(prompt_time("main") + 0.3, "left")])      # macOS PTB keyboard: RT on another clock
    real_get = p.kb.getKeys
    p.kb.getKeys = lambda keyList, waitRelease=False: [SimpleNamespace(name=k.name, rt=-1.26e6) for k in real_get(keyList)]
    r, _, _ = run("main", [], p=p)
    assert abs(r["rt"] - 0.3) < FRAME + 1e-9 and r["rt_source"].startswith("first seen"), r
    for task in ("main", "crossmodal", "unimodal_auditory", "unimodal_visual"):
        end2, stream_end = times(task)
        prompt = prompt_time(task)
        assert prompt - stream_end >= S.POST_STIM_WAIT_S, "blank pause between the stimulus and the prompt"

        # keys during the stimulus and the pause are discarded; the first key after the prompt counts
        r, _, _ = run(task, [(end2 - 0.05, "left"), (end2 + 0.10, "left"), (stream_end + 0.01, "left"),
                             (prompt + 0.25, "right")])
        assert not r["quit"] and r["key"] == "right" and r["prompt_shown"], (task, r)
        assert abs(r["rt"] - 0.25) < FRAME + 1e-9, (task, r["rt"])          # RT from the prompt's first flip
        k = round(S.PRE_TRIAL_WAIT_S / FRAME) + S.SYNC_FLIPS + 1
        assert abs(r["prompt_onset_s"] - (prompt - k * FRAME)) < 1e-9, (task, r["prompt_onset_s"])

        # escape quits, during the stimulus and during the prompt
        r, _, _ = run(task, [(end2 * 0.5, "escape")])
        assert r["quit"], task
        r, _, _ = run(task, [(prompt + 0.2, "escape")])
        assert r["quit"], task
    check_refresh()
    check_sound_and_flip_timing()
    check_autotest()
    check_device_choice_and_geometry()
    check_block_screens()
    check_screen_choice()
    print("Presenter response logic, sound scheduling, autotest, device choice and display check: all checks passed.")


if __name__ == "__main__":
    main()
