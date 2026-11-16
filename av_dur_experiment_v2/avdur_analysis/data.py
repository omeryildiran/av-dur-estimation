"""Trials of the v2 experiment, cleaned the same way for every analysis.

Participants are those in data/participants.json (the v2 protocol: fixed high noise 1.0, no bias
correction); the P1xx pilots ran an older protocol and are left out. A participant is "complete" with
both unimodal blocks and every main part (S.TASKS["main"]["blocks"]), otherwise "partial".

Cleaning (clean()):
    - warm-up trials out (they start every block and part and are not part of the design)
    - only answered trials (responses 1 or 2)
    - trials with a dropped frame, or whose first frame flipped more than half a frame off its
      prediction (sound and flash misaligned), out
    - answers faster than MIN_RT_S after the prompt out. MIN_RT_S is 0 (all kept): answers count only
      after the prompt, and the fast ones are as accurate as the rest (16 Nov 2026, main experiment:
      BQ, 21% of answers under 100 ms, easy trials 94% correct vs 93%, staircase 69% vs 67%), so they
      are decided during the noise after the second interval, not guesses. analysis/main_bias_plots.py
      dropped them (100 ms); set MIN_RT_S = 0.1 to do the same.
Only completed sessions are used; a session stopped part-way (and restarted) is left out.

Columns added: x = log(test / standard), noise (NaN for the visual block), conflict (s, visual standard
minus auditory standard, exact frames), conflict_ms (rounded, for labels), condition (a readable key).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                   # av_dur_experiment_v2/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur import storage  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
MIN_RT_S = 0.0                      # see the module docstring
STANDARD_S = S.STANDARD_DUR_S
NOISE_LOW, NOISE_HIGH = S.NOISE_LOW, S.NOISE_HIGH
UNIMODAL_TASKS = ("unimodal_auditory", "unimodal_visual")
MAIN_PARTS = tuple(S.TASKS["main"]["blocks"])


def participants(root=DATA):
    """v2 participants in the order they were registered."""
    registry = json.loads((Path(root) / "participants.json").read_text())
    return sorted(registry, key=registry.get)


def status(pid, root=DATA):
    """Which blocks and parts this participant completed, and complete / partial."""
    done = {t: [m.get("block") for m in storage.sessions(pid, t, root)] for t in storage.TASK_ORDER}
    parts = tuple(p for p in MAIN_PARTS if p in done["main"])
    has = dict(auditory=bool(done["unimodal_auditory"]), visual=bool(done["unimodal_visual"]), main_parts=parts)
    has["complete"] = has["auditory"] and has["visual"] and parts == MAIN_PARTS
    return has


def status_table(pids=None, root=DATA):
    rows = []
    for pid in pids or participants(root):
        s = status(pid, root)
        rows.append(dict(participant=pid, auditory=s["auditory"], visual=s["visual"],
                         main_parts=",".join(s["main_parts"]) or "-",
                         status="complete" if s["complete"] else "partial"))
    return pd.DataFrame(rows)


def clean(d):
    """(cleaned trials, counts of what was removed). See the module docstring."""
    d = d[d["kind"] != "warmup"]
    n0 = len(d)
    d = d[d["responses"].isin([1, 2])]
    unanswered = n0 - len(d)
    half_frame = 0.5 / d["frame_rate_hz"].fillna(S.EXPECTED_FRAME_RATE_HZ)
    glitch = (d["n_dropped_frames"].fillna(0) > 0) | (d["flip0_error_s"].abs() > half_frame)
    fast = d["response_rts"] < MIN_RT_S
    out = d[~glitch & ~fast].copy()
    return out, dict(trials=n0, unanswered=unanswered, glitch=int(glitch.sum()), fast=int((fast & ~glitch).sum()),
                     kept=len(out))


def _annotate(d, task):
    d = d.copy()
    d["task"] = task
    d["x"] = np.log(d["testDurS"].astype(float) / d["standardDur"].astype(float))
    d["noise"] = np.nan if task == "unimodal_visual" else d["audNoise"].astype(float)
    d["conflict"] = d["conflictDur"].fillna(0.0).astype(float) if task == "main" else 0.0
    d["conflict_ms"] = np.round(1000 * d["conflict"]).astype(int)
    d["chose_test"] = d["chose_test"].astype(int)
    if task == "unimodal_visual":
        d["condition"] = "V"
    elif task == "unimodal_auditory":
        d["condition"] = np.where(np.isclose(d["noise"], NOISE_LOW), "A_low", "A_high")
    else:
        d["condition"] = [f"AV_{'low' if np.isclose(n, NOISE_LOW) else 'high'}_{c:+d}"
                          for n, c in zip(d["noise"], d["conflict_ms"])]
    return d


def load(pid, task, root=DATA, cleaned=True):
    """Trials of every completed session of `task` (all main parts pooled; column `block` names the part).
    Returns (trials, removal counts); empty trials if the participant has no such session."""
    d = storage.load_trials(task, pid, root)
    if d.empty:
        return d, dict(trials=0, unanswered=0, glitch=0, fast=0, kept=0)
    if cleaned:
        d, counts = clean(d)
    else:
        d, counts = d[d["kind"] != "warmup"].copy(), {}
    return _annotate(d, task), counts


def load_unimodal(pid, root=DATA, cleaned=True):
    """Auditory (both noise levels) and visual blocks together."""
    frames, counts = [], {}
    for task in UNIMODAL_TASKS:
        d, c = load(pid, task, root, cleaned)
        counts[task] = c
        if not d.empty:
            frames.append(d)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), counts


def cleaning_table(pids=None, root=DATA):
    """Trials removed per participant and task, for the notebook."""
    rows = []
    for pid in pids or participants(root):
        for task in storage.TASK_ORDER:
            _, c = load(pid, task, root)
            if c["trials"]:
                rows.append(dict(participant=pid, task=task, **c))
    return pd.DataFrame(rows)
