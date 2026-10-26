"""Trial timing: the gaps (pre, ISI, post) never change what is shown or heard.

    python tests/test_schedule.py        # a few seconds

Brute force over every task, 120 and 60 Hz, test durations from the floor to the ceiling,
conflicts up to +/-450 ms and visual bias corrections up to +/-350 ms (BIAS max_abs_correction_s):
  - each sound lasts exactly its intended number of frames (test and standard);
  - each flash lasts its sound's duration plus its extension (main: bias, plus the conflict on
    the standard), is centred on its sound (within half a frame), and the flashes keep
    MIN_VISUAL_GAP_S apart and from the end of the trial;
  - no event comes before MIN_LEAD_S;
  - pre, ISI and post are never shorter than their random draw: make_schedule only lengthens
    a gap to make room for an extended flash;
  - the trial sound is as long as the schedule (audio and frames stay in step to the end).
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ["AVDUR_NO_NOTIFY"] = "1"
sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur.audio import AudioGenerator  # noqa: E402
from avdur.schedule import make_schedule, to_frames  # noqa: E402

CONFLICTS = (-0.45, -0.25, -0.167, -0.083, 0.0, 0.083, 0.167, 0.25, 0.45)
BIASES = (-0.35, -0.2, -0.1, -0.02, 0.0, 0.05, 0.12, 0.28, 0.35)


def drawn_gaps(seed, fr):
    """The pre / ISI / post make_schedule draws before any lengthening (same generator calls)."""
    r = np.random.default_rng(seed)
    return (to_frames(r.uniform(*S.PRE_RANGE_S), fr), to_frames(r.uniform(*S.ISI_RANGE_S), fr),
            to_frames(r.uniform(*S.POST_RANGE_S), fr))


def check_one(task, order, test_s, conflict, bias, fr, seed):
    s = make_schedule(task, order, test_s, conflict, bias, np.random.default_rng(seed), fr)
    test, std = to_frames(test_s, fr), to_frames(S.STANDARD_DUR_S, fr)
    # sounds (or the visual intervals of the unimodal visual task) last exactly as intended
    assert (s.dur1, s.dur2) == ((test, std) if order == 1 else (std, test)), (task, order, test_s)
    assert s.total == s.pre + s.dur1 + s.isi + s.dur2 + s.post
    pre0, isi0, post0 = drawn_gaps(seed, fr)
    assert s.pre >= pre0 and s.isi >= isi0 and s.post >= post0, "a gap was shortened"
    lead, gap = to_frames(S.MIN_LEAD_S, fr), to_frames(S.MIN_VISUAL_GAP_S, fr)
    first_events = []
    for i in (0, 1):
        on, off = s.interval_frames(i)
        if s.audio_burst[i]:
            first_events.append(on)
        v = s.visual[i]
        if v is None:
            continue
        is_test = (i == 0) == (order == 1)
        ext = 0
        if task == "main":
            ext = to_frames(bias, fr) + (0 if is_test else to_frames(conflict, fr))
        assert v[1] - v[0] == (off - on) + ext, ("flash duration", task, i, v, on, off, ext)
        assert abs((v[0] + v[1]) / 2 - (on + off) / 2) <= 0.5, "flash not centred on its sound"
        first_events.append(v[0])
    assert min(first_events) >= lead, "an event before MIN_LEAD_S"
    if s.visual[0] is not None and s.visual[1] is not None:
        assert s.visual[1][0] - s.visual[0][1] >= gap, "flashes too close"
    if s.visual[1] is not None:
        assert s.total - s.visual[1][1] >= gap, "flash 2 too close to the end"
    return s, (s.pre > pre0, s.isi > isi0, s.post > post0)


def run():
    n, lengthened = 0, np.zeros(3)
    for fr in (120.0, 60.0):
        for task in ("unimodal_auditory", "unimodal_visual", "crossmodal", "main"):
            cfg = S.TASKS[task]
            floor = cfg["floor"]
            levels = np.r_[np.linspace(floor, S.STAIRCASE["ceiling"], 12), 0.0]
            conflicts, biases = (CONFLICTS, BIASES) if task == "main" else ((0.0,), (0.0,))
            for d, c, b, order in itertools.product(levels, conflicts, biases, (1, 2)):
                test_s = S.STANDARD_DUR_S * (1 + d)
                flash_test = test_s + (b if task == "main" else 0)
                flash_std = S.STANDARD_DUR_S + (b + c if task == "main" else 0)
                if min(flash_test, flash_std) < S.MIN_VISUAL_DUR_S:      # never shown (session floor, design)
                    continue
                _, grew = check_one(task, order, test_s, c, b, fr, seed=n)
                lengthened += grew
                n += 1
    # the trial sound has the schedule's length (audio and frames stay in step)
    gen = AudioGenerator()
    for k, (c, b) in enumerate(((0.45, 0.35), (-0.45, 0.0), (0.0, -0.35), (0.25, 0.12))):
        s = make_schedule("main", 1 + k % 2, 0.3 + 0.2 * k, c, b, np.random.default_rng(k), 120.0)
        out = gen.trial_parts(s, 0.1, seed=k)["output"]
        want = s.total * S.SAMPLE_RATE / 120.0 + 4 * len(gen.pad)
        assert abs(len(out) - want) <= 6, ("sound length", len(out), want)
    return n, lengthened / n


if __name__ == "__main__":
    n, share = run()
    print(f"Schedules: {n} checked (all tasks, 120 and 60 Hz, conflicts to +/-450 ms, corrections to "
          f"+/-350 ms): durations exact, flashes centred and separated, no gap shortened.")
    print(f"Gaps lengthened to make room for flashes in this extreme grid: pre {share[0]:.0%}, "
          f"ISI {share[1]:.0%}, post {share[2]:.0%} of schedules.")
