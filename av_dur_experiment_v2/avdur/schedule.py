"""Frame-by-frame schedule of one trial.

Frame 0 is the first flip of the trial; the auditory stream (if any) is scheduled to start
on that flip, so frame n and audio sample n * sample_rate / frame_rate coincide.

The stream is: pre | interval 1 | ISI | interval 2 | post. In the main task each visual
flash is centred on its own auditory interval and extended symmetrically by
E = conflict + bias correction (standard) or E = bias correction (test). v1 anchored the
second flash to the first flash's offset instead, which displaced it from its sound.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import experiment_settings as S


def to_frames(seconds, frame_rate):
    return int(round(seconds * frame_rate))


def _split(extension):
    """Split a signed extension (frames) into onset and offset parts; |onset| <= |offset|."""
    onset = extension // 2 if extension >= 0 else -((-extension) // 2)
    return onset, extension - onset


@dataclass
class Schedule:
    frame_rate: float
    pre: int
    dur1: int
    isi: int
    dur2: int
    post: int
    audio_burst: tuple          # (interval 1 has a burst, interval 2 has a burst)
    visual: tuple               # ((onset, offset), (onset, offset)); None where no flash
    total: int

    def interval_frames(self, i):
        on = self.pre if i == 0 else self.pre + self.dur1 + self.isi
        return on, on + (self.dur1 if i == 0 else self.dur2)

    def seconds(self, frames):
        return frames / self.frame_rate

    def to_dict(self):
        return asdict(self)


def make_schedule(task, order, test_s, conflict_s, bias_s, rng, frame_rate):
    """Schedule for one trial. order 1 = test in interval 1. bias_s is the visual correction."""
    test = to_frames(test_s, frame_rate)
    standard = to_frames(S.STANDARD_DUR_S, frame_rate)
    dur1, dur2 = (test, standard) if order == 1 else (standard, test)

    pre = to_frames(rng.uniform(*S.PRE_RANGE_S), frame_rate)
    isi = to_frames(rng.uniform(*S.ISI_RANGE_S), frame_rate)
    post = to_frames(rng.uniform(*S.POST_RANGE_S), frame_rate)
    gap = to_frames(S.MIN_VISUAL_GAP_S, frame_rate)

    if task == "unimodal_auditory":
        bursts, ext = (True, True), (None, None)
    elif task == "unimodal_visual":
        bursts, ext = (False, False), (0, 0)
    elif task == "crossmodal":       # auditory test, visual standard (no correction, no conflict)
        bursts = (True, False) if order == 1 else (False, True)
        ext = (None, 0) if order == 1 else (0, None)
    elif task == "main":
        bias = to_frames(bias_s, frame_rate)
        conflict = to_frames(conflict_s, frame_rate)
        e_test, e_std = bias, bias + conflict
        bursts, ext = (True, True), ((e_test, e_std) if order == 1 else (e_std, e_test))
    else:
        raise ValueError(task)

    split = [(_split(e) if e is not None else (0, 0)) for e in ext]
    pre = max(pre, max(split[0][0], 0) + gap)
    isi = max(isi, max(split[0][1], 0) + max(split[1][0], 0) + gap)
    post = max(post, max(split[1][1], 0) + gap)

    visual = []
    for i, e in enumerate(ext):
        if e is None:
            visual.append(None)
            continue
        on = pre if i == 0 else pre + dur1 + isi
        off = on + (dur1 if i == 0 else dur2)
        visual.append((on - split[i][0], off + split[i][1]))
    total = pre + dur1 + isi + dur2 + post
    return Schedule(frame_rate, pre, dur1, isi, dur2, post, bursts, tuple(visual), total)
