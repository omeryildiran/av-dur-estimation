"""One run of one task: design -> staircases -> schedule -> stimulus -> response -> saved row.

The presenter does the showing and the response collection (PsychoPy or a simulated
observer); everything else, including all logging, happens here so both behave the same.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

import experiment_settings as S
from avdur import bias as bias_mod
from avdur import design, storage
from avdur.audio import AudioGenerator
from avdur.schedule import make_schedule, to_frames
from avdur.staircase import Staircase

NEEDS_AUDIO = {"unimodal_auditory", "crossmodal", "main"}


class Session:
    def __init__(self, task, pid, presenter, block=None, practice=False, bias=None, bias_file=None,
                 seed=None, root=None, monitor=S.DEFAULT_MONITOR, simulate=False):
        if task == "main" and bias is None:
            raise ValueError("The main experiment needs a confirmed bias file (run the cross-modal task first).")
        if task == "main" and block not in S.TASKS["main"]["blocks"]:
            raise ValueError(f"Main-experiment block must be one of {list(S.TASKS['main']['blocks'])}, got {block!r}")
        self.task, self.pid, self.presenter = task, pid, presenter
        self.block, self.practice, self.bias, self.bias_file = block, practice, bias, bias_file
        self.seed = seed if seed is not None else int(dt.datetime.now().timestamp() * 1000) % 2**31
        self.root = root or (storage.SIM_ROOT if simulate else storage.DATA_ROOT)
        self.monitor, self.simulate = monitor, simulate
        self.cfg = S.TASKS[task]
        self.fr = S.EXPECTED_FRAME_RATE_HZ

    # ── helpers ───────────────────────────────────────────────────────────────
    def correction_s(self, noise):
        """Visual correction actually applied (whole frames), and the unrounded estimate."""
        if self.task != "main":
            return 0.0, float("nan")
        est = bias_mod.correction_for(self.bias, noise)
        return to_frames(est, self.fr) / self.fr, est

    def floor(self, noise):
        """Lowest staircase level: test at least MIN_VISUAL_DUR_S, and its visual flash too."""
        floor = self.cfg["floor"]
        if self.task == "main":
            applied, _ = self.correction_s(noise)
            floor = max(floor, (S.MIN_VISUAL_DUR_S - min(applied, 0.0)) / S.STANDARD_DUR_S - 1)
        return math.ceil(floor * 100) / 100

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self):
        rng = np.random.default_rng(self.seed)
        first = design.crossmodal_first_order(self.pid) if self.task == "crossmodal" else 1
        trials = design.build_trials(self.task, rng, self.block, self.practice, first)
        csv_path, json_path = storage.session_paths(self.pid, self.task, self.block, self.practice, self.root)
        meta = storage.base_metadata(self.pid, self.task, self.block, self.practice, self.seed,
                                     self.monitor, self.simulate)
        meta.update(n_trials_planned=len(trials), csv=csv_path.name, bias_file=self.bias_file,
                    corrections_applied_s={str(n): self.correction_s(n)[0] for n in self.cfg["noise_levels"]
                                           if n is not None},
                    staircase_floors={str(n): self.floor(n) for n in self.cfg["noise_levels"]})
        storage.write_json(json_path, meta)

        meta["frame_rate_measured_hz"] = self.presenter.start(self.task, meta)
        storage.write_json(json_path, meta)
        stairs = {(n, c, r): Staircase(r, self.floor(n), **S.STAIRCASE)
                  for n, c in design.conditions(self.task, self.block) for r in self.cfg["staircases"]}
        audio = AudioGenerator() if (self.task in NEEDS_AUDIO and self.presenter.needs_audio) else None
        writer = storage.TrialWriter(csv_path)
        n_done, quit_ = 0, False
        try:
            self.presenter.instructions(self.task, self.practice)
            prev_block = None
            for i, spec in enumerate(trials):
                if i and i % S.BREAK_EVERY_N_TRIALS == 0:
                    self.presenter.rest(i // S.BREAK_EVERY_N_TRIALS,
                                        math.ceil(len(trials) / S.BREAK_EVERY_N_TRIALS))
                if spec.order_block is not None and spec.order_block != prev_block:
                    n_blocks = max(t.order_block for t in trials) + 1
                    self.presenter.order_cue(spec.order, spec.order_block, n_blocks)
                    prev_block = spec.order_block
                row = self._trial(i, spec, stairs, rng, audio)
                if row is None:
                    quit_ = True
                    break
                writer.write(row)
                n_done += 1
                if self.practice:
                    self.presenter.feedback(bool(row["is_correct"]))
        finally:
            writer.close()
            meta.update(n_trials_done=n_done, completed=(n_done == len(trials) and not quit_),
                        quit_by_user=quit_, finished=dt.datetime.now().isoformat(timespec="seconds"))
            storage.write_json(json_path, meta)
            self.presenter.finish(meta["completed"])
        return meta["completed"]

    def _trial(self, i, spec, stairs, rng, audio):
        stair = stairs.get((spec.noise, spec.conflict, spec.staircase)) if spec.kind == "staircase" else None
        if stair is not None:
            level = stair.level
        else:
            level = self.floor(spec.noise) if spec.fixed_level == "floor" else float(spec.fixed_level)
        test_frames = max(to_frames(S.STANDARD_DUR_S * (1 + level), self.fr), 1)
        test_s = test_frames / self.fr
        applied, estimate = self.correction_s(spec.noise)
        sched = make_schedule(self.task, spec.order, test_s, spec.conflict, applied, rng, self.fr)
        seed = int(rng.integers(2**31))
        wave = audio.trial_sound(sched, spec.noise, seed) if audio is not None else None

        result = self.presenter.run_trial(spec, sched, wave)
        if result.get("quit"):
            return None
        chose_test = result["response"] == spec.order
        standard_s = to_frames(S.STANDARD_DUR_S, self.fr) / self.fr
        is_correct = (test_s > standard_s and chose_test) or (test_s < standard_s and not chose_test)
        if stair is not None:
            stair.update(chose_test)

        test_i = 0 if spec.order == 1 else 1
        vis = sched.visual
        flips = result.get("flips", {})

        def visual_cols(i_interval, label):
            if vis[i_interval] is None:
                return {f"intendedDurVisual{label}": np.nan, f"recordedOnsetVisual{label}": np.nan,
                        f"recordedOffsetVisual{label}": np.nan, f"recordedDurVisual{label}": np.nan}
            on, off = flips.get((i_interval, "on"), np.nan), flips.get((i_interval, "off"), np.nan)
            return {f"intendedDurVisual{label}": (vis[i_interval][1] - vis[i_interval][0]) / self.fr,
                    f"recordedOnsetVisual{label}": on, f"recordedOffsetVisual{label}": off,
                    f"recordedDurVisual{label}": off - on}

        a1, a2 = sched.interval_frames(0), sched.interval_frames(1)
        row = dict(
            participantID=self.pid, task=self.task, block=self.block or "", practice=self.practice,
            trial_num=i, kind=spec.kind, current_stair=spec.staircase or ("lapse_rate" if spec.kind == "lapse" else spec.kind),
            condition=f"{spec.noise}/{spec.conflict}",
            standardDur=standard_s, testDurS=test_s, deltaDurS=test_s - standard_s,
            delta_dur_percents=round(level, 4), audNoise=np.nan if spec.noise is None else spec.noise,
            conflict_nominal=spec.conflict,
            conflictDur=to_frames(spec.conflict, self.fr) / self.fr if self.task == "main" else 0.0,
            VisualPSE=applied if self.task == "main" else np.nan, VisualPSE_estimate=estimate,
            order=spec.order, order_block=spec.order_block if spec.order_block is not None else "",
            preDur=sched.pre / self.fr, isiDur=sched.isi / self.fr, postDur=sched.post / self.fr,
            totalDur=sched.total / self.fr,
            frames_pre=sched.pre, frames_dur1=sched.dur1, frames_isi=sched.isi, frames_dur2=sched.dur2,
            frames_post=sched.post,
            audio_interval1=f"{a1[0]}-{a1[1]}" if sched.audio_burst[0] else "",
            audio_interval2=f"{a2[0]}-{a2[1]}" if sched.audio_burst[1] else "",
            visual_interval1="" if vis[0] is None else f"{vis[0][0]}-{vis[0][1]}",
            visual_interval2="" if vis[1] is None else f"{vis[1][0]}-{vis[1][1]}",
            **visual_cols(test_i, "Test"), **visual_cols(1 - test_i, "Standard"),
            audio_seed=seed if wave is not None else "",
            audio_level_mode=audio.mode if audio is not None else "",
            audio_gain_reduction_db=audio.last_gain_reduction_db if audio is not None else np.nan,
            audio_onset_minus_flip0_s=result.get("audio_onset_minus_flip0_s", np.nan),
            flip0_error_s=result.get("flip0_error_s", np.nan),
            n_dropped_frames=result.get("n_dropped_frames", np.nan),
            max_frame_interval_ms=result.get("max_frame_interval_ms", np.nan),
            responses=result["response"], response_keys=result.get("key", ""),
            response_rts=round(float(result.get("rt", np.nan)), 4),
            chose_test=int(chose_test), is_correct=int(is_correct),
            stair_num_reversal=stair.reversals if stair else "", stair_is_reversal=stair.is_reversal if stair else "",
            wall_time=dt.datetime.now().isoformat(timespec="milliseconds"),
        )
        return row
