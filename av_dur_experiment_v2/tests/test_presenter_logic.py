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
from avdur.presenter_psychopy import PsychoPyPresenter  # noqa: E402
from avdur.schedule import make_schedule  # noqa: E402

FRAME = 1 / S.EXPECTED_FRAME_RATE_HZ


class Time:
    now = 0.0


class Clock:
    def __init__(self):
        self.last_reset = 0.0

    def reset(self):
        self.last_reset = Time.now


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
        return [SimpleNamespace(name=self.wait_answer)]


class Sound:
    def __init__(self, **kwargs):
        self.played = self.stopped = False

    def play(self, when=None):
        self.played = True

    def stop(self):
        self.stopped = True


def presenter(task, presses, profile="v1_profile", win=None):
    Time.now = 0.0
    p = PsychoPyPresenter(profile)
    p.task, p.win, p.kb = task, win or Win(), Keyboard(presses)
    p.core = SimpleNamespace(wait=lambda s: None)
    p.sound = SimpleNamespace(Sound=Sound)
    drawable = SimpleNamespace(draw=lambda: None, fillColor=None, text="", color="")
    p.disk, p.fixation, p.audio_icon, p.text = drawable, drawable, drawable, drawable
    return p


def schedule(task, order=2):
    return make_schedule(task, order, 0.6, 0.083, 0.1, np.random.default_rng(1), S.EXPECTED_FRAME_RATE_HZ)


def times(task):
    sched = schedule(task)
    end2_time = (sched.second_interval_end() + 1) * FRAME          # flip time of that frame (frame 0 flips at 1 frame)
    return end2_time, (sched.total + 1) * FRAME


def run(task, presses):
    p = presenter(task, presses)
    end2, stream_end = times(task)
    return p.run_trial(SimpleNamespace(task=task), schedule(task), np.zeros(10, np.float32)), end2, stream_end


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
    for task in ("main", "crossmodal", "unimodal_auditory", "unimodal_visual"):
        end2, stream_end = times(task)
        assert stream_end - end2 > 0.15, "post-stimulus noise must remain after interval 2"

        # key during interval 2 is discarded; the next key (during the post noise) counts
        r, _, _ = run(task, [(end2 - 0.05, "left"), (end2 + 0.10, "right")])
        assert not r["quit"] and r["key"] == "right" and not r["prompt_shown"], (task, r)
        assert abs(r["rt"] - 0.10) < FRAME + 1e-9, (task, r["rt"])

        # no key until after the sound: prompt shown, RT still from the end of interval 2
        late = stream_end + 0.30
        r, _, _ = run(task, [(late, "left")])
        assert r["prompt_shown"] and r["key"] == "left" and abs(r["rt"] - (late - end2)) < FRAME + 1e-9, (task, r)

        # escape quits, during the stimulus and during the prompt
        r, _, _ = run(task, [(end2 * 0.5, "escape")])
        assert r["quit"], task
        r, _, _ = run(task, [(stream_end + 0.2, "escape")])
        assert r["quit"], task
    check_refresh()
    print("Presenter response logic and display check: all checks passed.")


if __name__ == "__main__":
    main()
