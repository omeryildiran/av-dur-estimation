"""One run of one task: design -> staircases -> schedule -> stimulus -> response -> saved row.

The presenter does the showing and the response collection (PsychoPy or a simulated
observer); everything else, including all logging, happens here so both behave the same.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_settings as S
from avdur import bias as bias_mod
from avdur import design, notify, storage
from avdur.audio import AudioGenerator
from avdur.schedule import make_schedule, to_frames
from avdur.quest import QuestStair
from avdur.staircase import Staircase

NEEDS_AUDIO = {"noise_calibration", "unimodal_auditory", "crossmodal", "main"}


class Session:
    def __init__(self, task, pid, presenter, block=None, practice=False, bias=None, bias_file=None,
                 seed=None, root=None, monitor=S.DEFAULT_MONITOR, simulate=False, autotest=False,
                 max_trials=None, solid_root=None, close_at_end=True, verbose=False, noise_high=None, noise_file=None,
                 standard_s=None, design=None, more_standards=None):
        if (standard_s is not None or more_standards) and task != "crossmodal":
            raise ValueError("Only the cross-modal task can use another standard duration.")
        if task == "main" and block not in S.TASKS["main"]["blocks"]:
            raise ValueError(f"Main-experiment block must be one of {list(S.TASKS['main']['blocks'])}, got {block!r}")
        if max_trials and not (simulate or autotest):
            raise ValueError("A trial limit is for simulated runs and autotests only.")
        self.task, self.pid, self.presenter = task, pid, presenter
        self.block, self.practice, self.bias, self.bias_file = block, practice, bias, bias_file
        self.seed = seed if seed is not None else int(dt.datetime.now().timestamp() * 1000) % 2**31
        self.root = root or storage.root_for(simulate, monitor, autotest)
        self.monitor, self.simulate, self.autotest = monitor, simulate, autotest
        self.max_trials, self.solid_root = max_trials, solid_root
        self.close_at_end, self.verbose = close_at_end, verbose     # False: more blocks follow in this window
        self.meta = self.csv_path = self.json_path = None
        self._last_notify = 0.0
        # v2: low noise fixed, high noise = the participant's calibrated level (NOISE_HIGH_DEFAULT without one)
        self.noise_high = round(float(noise_high if noise_high is not None else S.NOISE_HIGH_DEFAULT), 3)
        self.noise_file = noise_file
        self.cfg = self._task_cfg(design)                   # design: e.g. a shorter design for extra standards
        self.standard_s = float(standard_s if standard_s is not None else S.STANDARD_DUR_S)
        self.design_override = design
        # [(standard_s, cfg)]: more than one = the cross-modal task with its standards interleaved in one session
        self.more_standards = [(float(std), des) for std, des in (more_standards or ())]
        self.standards = [(self.standard_s, self.cfg)] + [(std, self._task_cfg(des)) for std, des in self.more_standards]
        self.schedule_task = task
        self.fr = S.MONITOR_PROFILES[monitor]["frame_rate_hz"]     # every duration is computed at this rate
        if task == "main" and not practice and not max_trials:
            self.earlier_parts()                            # fail before any window opens

    # ── helpers ───────────────────────────────────────────────────────────────
    def _task_cfg(self, design):
        cfg = dict(S.TASKS[self.task], **(design or {}))
        if any(n is not None for n in cfg["noise_levels"]):
            cfg["noise_levels"] = (S.NOISE_LOW, self.noise_high)
        return cfg

    def _std(self, spec):
        """The trial's standard duration (s)."""
        return self.standard_s if spec.standard is None else spec.standard

    def _key(self, spec, rule):
        return (spec.noise, spec.conflict, rule, self._std(spec))

    def correction_s(self, noise):
        """Visual correction actually applied (whole frames), and the unrounded estimate. From 28 Oct 2026 the
        main experiment runs without one (bias None: the bias is modelled); older bias files still work."""
        if self.task != "main" or self.bias is None:
            return 0.0, float("nan")
        est = bias_mod.correction_for(self.bias, noise)
        return to_frames(est, self.fr) / self.fr, est

    def floor(self, noise):
        """Lowest staircase level: test at least MIN_VISUAL_DUR_S, and its visual flash too."""
        floor = self.cfg["floor"]
        if self.task == "main":
            applied, _ = self.correction_s(noise)
            floor = max(floor, (S.MIN_VISUAL_DUR_S - min(applied, 0.0)) / self.standard_s - 1)
        return math.ceil(floor * 100) / 100

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self):
        rng = np.random.default_rng(self.seed)
        design.participant_number(self.pid, self.root, assign=True)    # initials: number fixed at the first session
        first = design.crossmodal_first_order(self.pid, self.root) if self.task == "crossmodal" else 1
        trials = self._build_trials(rng, first)
        meta = storage.base_metadata(self.pid, self.task, self.block, self.practice, self.seed,
                                     self.monitor, self.simulate, self.autotest)
        if self.max_trials and self.max_trials < len(trials):
            meta.update(trial_limit=self.max_trials, n_trials_full=len(trials))
            trials = trials[:self.max_trials]
        meta.update(noise_levels=[n for n in self.cfg["noise_levels"] if n is not None], noise_file=self.noise_file,
                    standard_s=self.standard_s, design_override=self.design_override,
                    standards_s=[std for std, _ in self.standards],
                    more_standards=[[std, des] for std, des in self.more_standards])
        meta.update(n_trials_planned=len(trials), bias_file=self.bias_file, frame_rate_nominal_hz=self.fr,
                    testing_only_profile=bool(S.MONITOR_PROFILES[self.monitor].get("testing_only")),
                    corrections_applied_s={str(n): self.correction_s(n)[0] for n in self.cfg["noise_levels"]
                                           if n is not None},
                    staircase_floors={str(n): self.floor(n) for n in self.cfg["noise_levels"]})
        try:                        # display/audio checks first: no session files for a failed start
            meta["frame_rate_measured_hz"] = self.presenter.start(self.task, meta)
        except Exception as err:
            storage.log_failed_start(self.root, meta, err)
            self._notify(f"⚠ {self._name()}: could not start ({str(err).splitlines()[0][:200]})", wait=True)
            raise

        csv_path, json_path = storage.session_paths(self.pid, self.task, self.block, self.practice, self.root)
        self.meta, self.csv_path, self.json_path = meta, csv_path, json_path
        meta["csv"] = csv_path.name
        storage.write_json(json_path, meta)
        stairs = self._make_stairs(rng)
        several = len(self.standards) > 1
        meta["staircase_starts"] = {f"{n}/{c}/{r}" + (f"/S{round(1000 * std)}" if several else ""): st.start
                                    for (n, c, r, std), st in stairs.items()}
        if getattr(self, "continued_from", None):
            meta["staircases_continue"] = self.continued_from
        storage.write_json(json_path, meta)
        audio = AudioGenerator() if (self.task in NEEDS_AUDIO and self.presenter.needs_audio) else None
        writer = storage.TrialWriter(csv_path)
        n_done, quit_, rows, t0 = 0, False, [], None
        try:
            self.presenter.instructions(self.task, self.practice)
            prev_block = None
            for i, spec in enumerate(trials):
                if i and i % S.BREAK_EVERY_N_TRIALS == 0:
                    if self.verbose:
                        print(self.progress(rows, len(trials), t0, stairs), flush=True)
                    if time.monotonic() - self._last_notify > 600:          # Slack: at most every 10 min
                        self._notify(self.progress(rows, len(trials), t0, stairs))
                    self.presenter.rest(i // S.BREAK_EVERY_N_TRIALS,
                                        math.ceil(len(trials) / S.BREAK_EVERY_N_TRIALS))
                if spec.order_block is not None and spec.order_block != prev_block:
                    n_blocks = max(t.order_block for t in trials) + 1
                    self.presenter.order_cue(spec.order, spec.order_block, n_blocks)
                    prev_block = spec.order_block
                t0 = t0 or time.monotonic()
                row = self._trial(i, spec, stairs, rng, audio)
                if row is None:
                    quit_ = True
                    break
                writer.write(row)
                rows.append(row)
                n_done += 1
                if self.practice:
                    self.presenter.feedback(bool(row["is_correct"]))
        finally:
            writer.close()
            meta.update(n_trials_done=n_done, completed=(n_done == len(trials) and not quit_),
                        quit_by_user=quit_, finished=dt.datetime.now().isoformat(timespec="seconds"))
            if meta["completed"]:
                self._after_run(meta, rows)
            storage.write_json(json_path, meta)
            self.save_meta()
            if self.verbose and rows:
                print(self.progress(rows, len(trials), t0, stairs), flush=True)
            if meta["completed"]:
                pass                        # the block's fit summary (sent by the runner) says it is done
            elif quit_:
                self._notify(f"⏹ {self._name()} stopped with Escape after {n_done}/{len(trials)} trials", wait=True)
            else:
                self._notify(f"❌ {self._name()} ended with an error after {n_done}/{len(trials)} trials", wait=True)
            if self.close_at_end or not meta["completed"]:
                self.presenter.finish(meta["completed"])
        return meta["completed"]

    # ── hooks (overridden by the noise calibration) ───────────────────────────
    def _build_trials(self, rng, first_order):
        return design.build_trials(self.task, rng, self.block, self.practice, first_order, self.cfg["noise_levels"],
                                   self.cfg, standards=self.standards if len(self.standards) > 1 else None)

    def _make_stairs(self, rng):
        """{(noise, conflict, rule, standard_s): staircase}."""
        several, stairs = len(self.standards) > 1, {}
        for std, cfg in self.standards:
            key_std = std if several else self.standard_s
            starts = design.staircase_starts(self.task, rng, self.block, cfg["noise_levels"], cfg)
            stairs.update({key + (key_std,): Staircase(key[2], self.floor(key[0]), start=start, **S.STAIRCASE)
                           for key, start in starts.items()})
            if "QUEST" in cfg["staircases"]:                   # QUEST+ chooses the test level (avdur.quest)
                for noise, conflict in design.conditions(self.task, self.block, cfg["noise_levels"], cfg):
                    stairs[(noise, conflict, "QUEST", key_std)] = QuestStair(self.floor(noise), S.STAIRCASE["ceiling"],
                                                                             std, self.fr)
        if self.task == "main" and not self.practice and not self.max_trials:
            self._continue_stairs(stairs)
        return stairs

    def earlier_parts(self):
        """Completed earlier parts of the main experiment (session metadata); an error if one is missing."""
        parts = list(S.TASKS["main"]["blocks"])
        earlier = parts[:parts.index(self.block)]
        metas = [m for m in storage.sessions(self.pid, "main", self.root)
                 if m.get("block") in earlier and not m.get("practice") and not m.get("trial_limit")]
        missing = [b for b in earlier if b not in {m["block"] for m in metas}]
        if missing:
            raise ValueError(f"{self.pid}: run main-experiment {', '.join(missing)} before {self.block} "
                             "(its staircases continue the earlier parts').")
        return metas

    def _continue_stairs(self, stairs):
        """Main experiment in parts: replay the answers of the participant's completed earlier parts (in the
        order they were given), so this part's staircases go on where they stopped."""
        metas = self.earlier_parts()
        keys = {(round(k[0], 4), round(k[1], 4), k[2]): k for k in stairs}
        replayed = []
        for meta in metas:
            if [n for n in meta.get("noise_levels", []) if n is not None] != [n for n in self.cfg["noise_levels"] if n is not None]:
                raise ValueError(f"{self.pid}: {meta['block']} ran at noise levels {meta.get('noise_levels')}, this part "
                                 f"would run at {list(self.cfg['noise_levels'])}; the staircases cannot continue.")
            d = pd.read_csv(meta["csv"])
            d = d[d["kind"] == "staircase"].sort_values("trial_num")
            for n, c, r, y in zip(d["audNoise"], d["conflict_nominal"], d["current_stair"], d["chose_test"]):
                stairs[keys[(round(float(n), 4), round(float(c), 4), r)]].update(bool(y))
            replayed.append(Path(meta["csv"]).name)
        self.continued_from = replayed

    def _trial_level(self, spec, stair):
        if stair is not None:
            return stair.level
        return self.floor(spec.noise) if spec.fixed_level == "floor" else float(spec.fixed_level)

    def _trial_noise(self, spec, stair):
        return spec.noise

    def _update_stair(self, stair, chose_test, is_correct):
        stair.update(chose_test)

    def _after_run(self, meta, rows):
        pass

    def _name(self):
        return (f"{self.pid} {self.task}" + (f" {self.block}" if self.block else "")
                + (" practice" if self.practice else "") + (" (autotest)" if self.autotest else ""))

    def _notify(self, text, wait=False):
        if not self.simulate and not self.practice:
            self._last_notify = time.monotonic()
            notify.send(text, wait=wait)

    def save_meta(self):
        """Write the session metadata (and refresh the data_solid/ copy of a complete real session)."""
        storage.write_json(self.json_path, self.meta)
        if self.solid_root is not False and storage.is_solid(self.meta, self.root):
            storage.copy_to_solid([self.csv_path, self.json_path], self.pid, self.task, self.solid_root)

    def progress(self, rows, n_total, t0, stairs):
        """Terminal lines: trials and time, easy-trial accuracy, RT, dropped frames, staircase levels."""
        n = len(rows)
        elapsed = time.monotonic() - t0 if t0 else 0.0
        left = elapsed / n * (n_total - n) if n else float("nan")
        name = self.task + (f" {self.block}" if self.block else "") + (" practice" if self.practice else "")
        lines = [f"[{self.pid} {name}] {n}/{n_total} trials, {elapsed / 60:.1f} min, about {left / 60:.0f} min left"]
        easy = {}
        for r in rows:
            if r["kind"] == "lapse":
                key = "visual" if self.task == "unimodal_visual" else f"noise {r['audNoise']:g}"
                easy.setdefault(key, []).append(r["is_correct"])
        rts = [r["response_rts"] for r in rows if r["response_rts"] == r["response_rts"]]
        dropped = sum(int(r["n_dropped_frames"]) for r in rows if r["n_dropped_frames"] == r["n_dropped_frames"])
        lines.append("    easy trials correct: " + (", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in easy.items()) or "none yet")
                     + (f"; median RT {statistics.median(rts):.2f} s" if rts else "") + f"; dropped frames {dropped}")
        if self.task == "main":
            levels = [st.level for st in stairs.values()]
            lines.append(f"    staircase levels: {min(levels):+.2f} to {max(levels):+.2f}")
        else:
            by_noise = {}
            for (noise, _, rule, std), st in stairs.items():
                if isinstance(st, QuestStair):
                    mu, sigma, _ = st.estimate()
                    text = f"QUEST PSE {1000 * std * math.exp(mu):.0f} ms, sigma {sigma:.2f}"
                else:
                    text = f"{rule} {st.level:+.2f}"
                name = "visual" if noise is None else f"noise {noise:g}"
                if len(self.standards) > 1:
                    name = f"S{round(1000 * std)} {name}"
                by_noise.setdefault(name, []).append(text)
            lines.append("    staircases: " + " | ".join(f"{k}: {' '.join(v)}" for k, v in by_noise.items()))
        return "\n".join(lines)

    def _trial(self, i, spec, stairs, rng, audio):
        stair = stairs.get(self._key(spec, spec.staircase)) if spec.kind == "staircase" else None
        quest_key = self._key(spec, "QUEST")
        level = self._trial_level(spec, stair)
        spec = dataclasses.replace(spec, noise=self._trial_noise(spec, stair))     # the noise actually played
        std = self._std(spec)
        test_frames = max(to_frames(std * (1 + level), self.fr), 1)
        test_s = test_frames / self.fr
        applied, estimate = self.correction_s(spec.noise)
        sched = make_schedule(self.schedule_task, spec.order, test_s, spec.conflict, applied, rng, self.fr,
                              standard_s=std)
        seed = int(rng.integers(2**31))
        wave = audio.trial_sound(sched, spec.noise, seed) if audio is not None else None

        result = self.presenter.run_trial(spec, sched, wave)
        if result.get("quit"):
            return None
        chose_test = result["response"] == spec.order
        standard_s = to_frames(std, self.fr) / self.fr
        is_correct = (test_s > standard_s and chose_test) or (test_s < standard_s and not chose_test)
        quest = stairs.get(quest_key)
        if stair is not None:
            self._update_stair(stair, chose_test, is_correct)
        elif spec.kind == "lapse" and quest is not None:                                  # easy trials inform
            quest.observe(test_s / standard_s - 1, chose_test)                            # its lapse

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
            frame_rate_hz=self.fr,
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
            audio_start_minus_flip0_s=result.get("audio_start_minus_flip0_s", np.nan),
            audio_xruns=result.get("audio_xruns", np.nan),
            flip0_error_s=result.get("flip0_error_s", np.nan),
            n_dropped_frames=result.get("n_dropped_frames", np.nan),
            max_frame_interval_ms=result.get("max_frame_interval_ms", np.nan),
            responses=result["response"], response_keys=result.get("key", ""),
            response_rts=round(float(result.get("rt", np.nan)), 4),     # from the prompt's first flip (as v1)
            rt_source=result.get("rt_source", "simulated"),
            prompt_shown=int(bool(result.get("prompt_shown", True))),
            answer_allowed_from_s=result.get("prompt_onset_s", np.nan),   # prompt onset, from frame 0
            chose_test=int(chose_test), is_correct=int(is_correct),
            stair_start=stair.start if stair else "",
            stair_num_reversal=stair.reversals if stair else "", stair_is_reversal=stair.is_reversal if stair else "",
            **quest_cols(quest),
            wall_time=dt.datetime.now().isoformat(timespec="milliseconds"),
        )
        return row


def quest_cols(quest):
    """QUEST+ posterior means after this trial (empty when the condition has no QUEST+)."""
    if quest is None:
        return dict(quest_mu="", quest_sigma="", quest_lapse="")
    mu, sigma, lapse = quest.estimate()
    return dict(quest_mu=round(mu, 4), quest_sigma=round(sigma, 4), quest_lapse=round(lapse, 4))
