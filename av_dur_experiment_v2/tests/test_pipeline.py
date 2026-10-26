"""End-to-end checks with simulated participants (no PsychoPy needed).

    python tests/test_pipeline.py              # writes to data_simulated/ and prints a summary
    python -m pytest tests/test_pipeline.py    # same checks under pytest
"""

from __future__ import annotations

import shutil
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ["AVDUR_NO_NOTIFY"] = "1"                # tests never post to Slack/ntfy
sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur import bias as bias_mod  # noqa: E402
from avdur import design, storage  # noqa: E402
from avdur.audio import AudioGenerator  # noqa: E402
from avdur.presenter_sim import DEFAULT_OBSERVER, SimulatedPresenter  # noqa: E402
from avdur.schedule import make_schedule  # noqa: E402
from avdur.session import Session  # noqa: E402
from avdur.staircase import Staircase  # noqa: E402

SIM_ROOT = ROOT / "data_simulated"
FR = S.EXPECTED_FRAME_RATE_HZ


def simulate_participant(pid, seed, observer=None, root=SIM_ROOT):
    obs = dict(DEFAULT_OBSERVER, **(observer or {}))
    for task in ("unimodal_auditory", "unimodal_visual", "crossmodal"):
        assert Session(task, pid, SimulatedPresenter(seed, obs), seed=seed, root=root, simulate=True).run()
    est = bias_mod.estimate_for_participant(pid, root, n_boot=60)
    bias_path = bias_mod.save(pid, est, confirmed_by="test", root=root)
    payload = storage.latest_bias(pid, root)[0]
    for k, b in enumerate(design.block_order(pid)):
        assert Session("main", pid, SimulatedPresenter(seed + 10 + k, obs), block=b, bias=payload,
                       bias_file=str(bias_path), seed=seed + 10 + k, root=root, simulate=True).run()
    return est, obs


def _intervals(col):
    return col.str.split("-", expand=True).astype(float)


# ── checks ────────────────────────────────────────────────────────────────────

def n_trials(task, n_conditions):
    cfg = S.TASKS[task]
    return n_conditions * (len(cfg["staircases"]) * cfg["trials_per_staircase"]
                           + len(cfg["lapse_levels"]) * cfg["lapse_trials_per_level"])


def check_counts(root, pid):
    expected = {"unimodal_auditory": n_trials("unimodal_auditory", 2), "unimodal_visual": n_trials("unimodal_visual", 1),
                "crossmodal": 388}
    for task, n in expected.items():
        d = storage.load_trials(task, pid, root)
        assert len(d) == n, (task, len(d), n)
    m = storage.load_trials("main", pid, root)
    per_block = m.groupby("block").size().to_dict()
    assert per_block == {"B1": 924, "B2": 616, "B3": 616}, per_block
    assert (m.groupby(["block", "condition"]).size() == 154).all()


def check_order_balance(root, pid):
    for task in ("unimodal_auditory", "unimodal_visual", "crossmodal", "main"):
        d = storage.load_trials(task, pid, root)
        stair = d.groupby(["session_file", "condition", "current_stair"]).order.apply(lambda s: abs((s == 1).sum() - (s == 2).sum()))
        assert stair.max() <= 1, (task, stair.max())
        cond = d.groupby(["session_file", "condition"]).order.apply(lambda s: abs((s == 1).sum() - (s == 2).sum()))
        assert cond.max() == 0, (task, cond[cond > 0])


def check_crossmodal_blocks(root, pid):
    d = storage.load_trials("crossmodal", pid, root).sort_values("trial_num")
    runs = (d.order != d.order.shift()).cumsum()
    sizes = d.groupby(runs).size()
    assert sizes.iloc[:-2].eq(S.TASKS["crossmodal"]["order_block_size"]).all()
    assert d.order.iloc[0] == design.crossmodal_first_order(pid)


def check_randomised(root, pid):
    m = storage.load_trials("main", pid, root)
    for _, s in m.groupby("session_file"):
        seq = s.sort_values("trial_num").condition.tolist()
        periodic = any(all(seq[i] == seq[i + p] for i in range(len(seq) - p)) for p in range(1, 13))
        assert not periodic


def check_main_alignment(root, pid):
    m = storage.load_trials("main", pid, root)
    worst = 0.0
    for k in (1, 2):
        a, v = _intervals(m[f"audio_interval{k}"]), _intervals(m[f"visual_interval{k}"])
        centre_shift = (v[0] + v[1]) / 2 - (a[0] + a[1]) / 2
        worst = max(worst, centre_shift.abs().max())
        is_std = (m.order == 2) if k == 1 else (m.order == 1)
        ext = (v[1] - v[0]) - (a[1] - a[0])
        want = np.round(m.VisualPSE * FR) + np.where(is_std, np.round(m.conflictDur * FR), 0)
        assert (ext == want).all(), "visual extension != bias (+ conflict for the standard)"
    assert worst <= 0.5, f"visual flash off-centre by {worst} frames"
    v1, v2 = _intervals(m.visual_interval1), _intervals(m.visual_interval2)
    gap = round(S.MIN_VISUAL_GAP_S * FR)
    assert (v1[0] >= gap).all() and ((v2[0] - v1[1]) >= gap).all()
    assert ((m.frames_pre + m.frames_dur1 + m.frames_isi + m.frames_dur2 + m.frames_post) - v2[1] >= gap).all()
    assert ((v1[1] - v1[0]) >= 1).all() and ((v2[1] - v2[0]) >= 1).all()
    return worst


def check_preparation_and_answer_window(root, pid):
    """No event in the first MIN_LEAD_S of a trial; answers only after interval 2 is fully over."""
    lead = round(S.MIN_LEAD_S * FR)
    shortest = np.inf
    for task in ("unimodal_auditory", "unimodal_visual", "crossmodal", "main"):
        d = storage.load_trials(task, pid, root)
        firsts = []
        for col in ("audio_interval1", "visual_interval1"):
            s = d[col].fillna("").astype(str)
            firsts.append(np.where(s != "", pd.to_numeric(s.str.split("-").str[0], errors="coerce"), np.inf))
        first_event = np.minimum(*firsts)
        assert (first_event >= lead).all(), (task, first_event.min())
        shortest = min(shortest, first_event.min() / FR)
        ends = []
        for col in ("audio_interval2", "visual_interval2"):
            s = d[col].fillna("").astype(str)
            ends.append(np.where(s != "", pd.to_numeric(s.str.split("-").str[1], errors="coerce"), -np.inf))
        prompt = (d.totalDur * FR + 1 + round(S.POST_STIM_WAIT_S * FR)) / FR      # after the whole stimulus
        assert (d.prompt_shown == 1).all() and np.allclose(d.answer_allowed_from_s, prompt), task
        assert (d.answer_allowed_from_s * FR >= np.maximum(*ends) + 1).all(), task
    return shortest


def check_bias(root, pid, est, obs):
    m = storage.load_trials("main", pid, root)
    assert m.groupby("audNoise").VisualPSE.nunique().eq(1).all(), "one correction per noise level"
    for noise, e in est["per_noise"].items():
        true = obs["true_bias"][float(noise)]
        assert abs(e["correction_s"] - true) < 0.04, (noise, e["correction_s"], true)
        applied = m.loc[m.audNoise == float(noise), "VisualPSE"].iloc[0]
        assert abs(applied - round(e["correction_s"] * FR) / FR) < 1e-9
    files = {meta["bias_file"] for meta in storage.sessions(pid, "main", root)}
    assert len(files) == 1, files


def check_audio():
    """Fixed-level mode: equal burst levels across tasks/trials; loudness and SNR as in v1."""
    sr = S.SAMPLE_RATE
    rng = np.random.default_rng(3)

    def measure(gen, task, noise, test, reps=100):
        pad = 2 * len(gen.pad)
        sched = make_schedule(task, 1, test, 0.0, 0.0, rng, FR)

        def seg(x, i):
            on, off = sched.interval_frames(i)
            return x[pad + int(on / FR * sr) + int(0.02 * sr): pad + int(off / FR * sr) - int(0.02 * sr)]

        vals = []
        for _ in range(reps):
            x = gen.trial_sound(sched, noise, int(rng.integers(2**31)))
            t, s = np.sqrt(np.mean(seg(x, 0) ** 2)), np.sqrt(np.mean(seg(x, 1) ** 2))
            floor = np.sqrt(np.mean(x[pad + int(0.02 * sr): pad + int(sched.pre / FR * sr) - int(0.02 * sr)] ** 2))
            vals.append((20 * np.log10(t), 20 * np.log10(t / s), 20 * np.log10(t / floor)))
        v = np.array(vals)
        return dict(level=v[:, 0].mean(), level_sd=v[:, 0].std(), test_vs_std=v[:, 1].mean(), snr=v[:, 2].mean())

    fixed, v1 = AudioGenerator(level_mode="fixed"), AudioGenerator(level_mode="peak")
    out = {}
    # windows of >= 0.26 s, so the RMS estimate itself is precise enough for 0.3 dB comparisons
    for noise in S.NOISE_LEVELS:
        for test in (0.3, 0.5, 0.9):
            m, c = measure(fixed, "main", noise, test), measure(fixed, "crossmodal", noise, test)
            p, pc = measure(v1, "main", noise, test), measure(v1, "crossmodal", noise, test)
            assert abs(m["test_vs_std"]) < 0.3, "test and standard bursts must have equal RMS"
            assert abs(m["level"] - c["level"]) < 0.3, ("cross-modal test level must match main", m["level"], c["level"])
            assert m["level_sd"] < p["level_sd"] - 0.1, ("per-trial level jitter must drop", m["level_sd"], p["level_sd"])
            played_v2 = m["level"] + 20 * np.log10(S.FIXED_VOLUME)
            played_v1 = p["level"] + 20 * np.log10(S.VOLUME)
            assert abs(played_v2 - played_v1) < 0.8, ("loudness must match v1", played_v2, played_v1)
            assert abs(m["snr"] - p["snr"]) < 1.0, ("SNR must match v1", m["snr"], p["snr"])
            out[(noise, test)] = (m, p, c, pc)
    tests = (0.3, 0.5, 0.9)
    snr = {n: (np.mean([out[(n, t)][0]["snr"] for t in tests]),
               np.mean([out[(n, t)][1]["snr"] for t in tests])) for n in S.NOISE_LEVELS}
    jitter = (np.mean([out[k][0]["level_sd"] for k in out]), np.mean([out[k][1]["level_sd"] for k in out]))
    xm_gap = (np.mean([out[k][2]["level"] - out[k][0]["level"] for k in out]),
              np.mean([out[k][3]["level"] - out[k][1]["level"] for k in out]))
    return snr, jitter, xm_gap


def check_staircase_rules():
    s = Staircase("2D1U", floor=-0.9)
    levels = []
    for chose in [False, False, True, True, False, True, True, True, True]:
        s.update(chose)
        levels.append(round(s.level, 4))
    # up 0.16, up 0.16, down (reversal -> step 0.08), up (reversal -> 0.04), down 0.04, down 0.04
    assert levels == [-0.74, -0.58, -0.58, -0.66, -0.62, -0.62, -0.66, -0.66, -0.7], levels
    top = Staircase("1D2U", floor=-0.9, start="ceiling")
    assert top.level == 0.9 and round(Staircase("1D1U_b", floor=-0.9).level, 2) == -0.9
    top.update(True)                                   # 1D: one "test longer" steps down from the ceiling
    assert round(top.level, 2) == 0.74
    rng = np.random.default_rng(0)
    for task in (t for t in S.TASKS if t != "noise_calibration"):      # its staircases move the noise
        for block in (S.TASKS["main"]["blocks"] if task == "main" else [None]):
            starts = design.staircase_starts(task, rng, block)
            rules = [r for r in S.TASKS[task]["staircases"] if r != "QUEST"]
            for noise, conflict in design.conditions(task, block):
                for k in range(0, len(rules), 2):
                    pair = {starts[(noise, conflict, rules[k])], starts[(noise, conflict, rules[k + 1])]}
                    assert pair == {"floor", "ceiling"}, (task, block, rules[k])
            n_cond = len(design.conditions(task, block))
            for r in rules:
                sides = {v for (_, _, rule), v in starts.items() if rule == r}
                side = design.staircase_side(r)
                if side:                                # starts opposite its convergence point
                    assert sides == {"floor" if side > 0 else "ceiling"}, (task, r, sides)
                else:                                   # 50% pair: balanced over conditions
                    n_top = sum(v == "ceiling" for (_, _, rule), v in starts.items() if rule == r)
                    assert abs(n_top - n_cond / 2) <= 0.5, (task, block, r, n_top, n_cond)
    try:
        Staircase("1U1D", floor=-0.9)
    except ValueError:
        return
    raise AssertionError("v1's diverging '1U1D' rule must be rejected")


def check_staircase_starts(root, pid):
    """Every staircase's first trial is at its logged starting end; pairs start at opposite ends."""
    for task in (t for t in storage.TASK_ORDER if t != "noise_calibration"):
        d = storage.load_trials(task, pid, root)
        d = d[(d.kind == "staircase") & (d.current_stair != "QUEST")].sort_values("trial_num")
        if d.empty:                                     # QUEST+ only: check_quest
            continue
        first = d.groupby(["session_file", "condition", "current_stair"]).first()
        assert (first.stair_start.isin(["floor", "ceiling"])).all()
        top = first[first.stair_start == "ceiling"]
        assert np.allclose(top.delta_dur_percents, S.STAIRCASE["ceiling"]), task
        bottom = first[first.stair_start == "floor"].delta_dur_percents
        assert (bottom < -0.5).all(), (task, bottom.max())
        per_cond = first.groupby(["session_file", "condition"]).stair_start.apply(lambda x: (x == "ceiling").sum())
        assert (per_cond == len([r for r in S.TASKS[task]["staircases"] if r != "QUEST"]) // 2).all(), task


def check_quest(root, pid):
    """QUEST+ conditions: every test a whole number of frames between the floor and the ceiling,
    the posterior logged on every trial, and its final sigma close to the screening fit's."""
    from avdur import screening
    for task in (t for t in S.TASKS if "QUEST" in S.TASKS[t]["staircases"]):
        for meta in storage.sessions(pid, task, root):
            d = pd.read_csv(meta["csv"])
            q = d[d.current_stair == "QUEST"]
            assert len(q) == len(d.condition.unique()) * S.TASKS[task]["trials_per_staircase"], task
            frames = q.testDurS * q.frame_rate_hz
            assert np.allclose(frames, frames.round()), "QUEST+ test not a whole number of frames"
            assert q.delta_dur_percents.between(S.TASKS[task]["floor"] - 1e-9, S.STAIRCASE["ceiling"] + 1e-9).all()
            assert q[["quest_mu", "quest_sigma", "quest_lapse"]].notna().all().all()
            fit = {e["noise"]: e["sigma"] for e in screening.block_fit(task, meta["csv"], None)["conditions"]}
            for cond, g in q.groupby("condition"):
                noise = None if cond.startswith("None") else float(cond.split("/")[0])
                ratio = g.quest_sigma.iloc[-1] / fit[noise]
                assert 0.5 < ratio < 2.0, (task, cond, ratio)


def check_display_profiles(root):
    assert storage.root_for(False, "lab_crt") == storage.DATA_ROOT
    assert storage.root_for(False, "office_test") == storage.TEST_ROOT
    assert storage.root_for(True, "office_test") == storage.SIM_ROOT
    assert storage.root_for(False, "lab_crt", autotest=True) == storage.TEST_ROOT
    assert Session("crossmodal", "SIM75", SimulatedPresenter(5), seed=5, root=root, monitor="office_test",
                   simulate=True).run()
    d = storage.load_trials("crossmodal", "SIM75", root)
    assert (d.frame_rate_hz == 75).all()
    frames = d.intendedDurVisualStandard * 75
    assert np.allclose(frames, np.round(frames)), "durations must be whole 75 Hz frames"


class FailingPresenter(SimulatedPresenter):
    def start(self, task, meta):
        meta["frame_rate_measured_hz"] = 75.0
        raise RuntimeError("display check failed")


def check_failed_start_and_ids(root):
    import json
    try:
        Session("unimodal_auditory", "SIM99", FailingPresenter(0), root=root, simulate=True).run()
        raise AssertionError("the session should have stopped")
    except RuntimeError:
        pass
    assert not (root / "SIM99").exists(), "a failed start must not create participant files"
    last = json.loads((root / "failed_starts.jsonl").read_text().strip().splitlines()[-1])
    assert last["participant"] == "SIM99" and last["frame_rate_measured_hz"] == 75.0
    assert design.normalize_pid("p2") == design.normalize_pid("2") == design.normalize_pid(" P02 ") == "P02"
    assert design.normalize_pid("p201") == "P201" and design.normalize_pid("sim01") == "SIM01"
    assert design.normalize_pid(" oy ") == "OY" and design.normalize_pid("ln2") == "LN2"
    first = design.participant_number("ZZ", root)                         # preview only: nothing written
    assert design.participant_number("ZZ", root) == first and not (root / "participants.json").exists()
    assert design.participant_number("ZZ", root, assign=True) == first
    assert design.participant_number("YY", root, assign=True) == first + 1 and design.participant_number("ZZ", root) == first
    assert design.block_order("YY", root) == S.BLOCK_ORDERS[first % len(S.BLOCK_ORDERS)]
    for bad in ("", "a", "P2a", "toolong", "o y"):
        try:
            design.normalize_pid(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


class QuittingPresenter(SimulatedPresenter):
    def run_trial(self, spec, sched, wave):
        self.n = getattr(self, "n", 0) + 1
        return dict(quit=True) if self.n > 3 else super().run_trial(spec, sched, wave)


class DroppingPresenter(SimulatedPresenter):
    """Every 5th trial drops a frame, which lengthens that trial's last flash by one frame."""
    def run_trial(self, spec, sched, wave):
        r = super().run_trial(spec, sched, wave)
        self.n = getattr(self, "n", 0) + 1
        if self.n % 5 == 0:
            r["n_dropped_frames"] = 1
            last = max(k for k in r["flips"] if k[1] == "off")
            r["flips"][last] += 1 / sched.frame_rate
        return r


def check_solid(tmp):
    """Only complete, full sessions of real participants (and their bias files) go to data_solid/."""
    real, solid = tmp / "data", tmp / "data_solid"

    def run(pid, presenter=None, **kw):
        return Session("unimodal_visual", pid, presenter or SimulatedPresenter(1), seed=1, root=real,
                       solid_root=solid, **kw).run()

    assert run("SIM80")                                        # complete, real: copied
    run("SIM81", practice=True)                                 # practice
    run("SIM82", autotest=True, max_trials=10)                  # autotest, trial limit
    run("SIM83", QuittingPresenter(1))                          # stopped early
    run("SIM84", simulate=True)                                 # simulated
    run("SIM85", monitor="office_test")                         # test display
    copied = sorted({f.parent.parent.name for f in solid.glob("*/*/*")})
    assert copied == ["SIM80"], copied
    assert len(list((solid / "SIM80" / "unimodal_visual").iterdir())) == 2      # csv + json
    bias_mod.save("SIM80", bias_mod.fixed({0.1: 0.1, 1.2: -0.05}, "test"), confirmed_by="test", root=real,
                  solid_root=solid)
    assert len(list((solid / "SIM80" / "bias").glob("SIM80_bias_*.json"))) == 1
    assert not storage.is_real_root(storage.SIM_ROOT) and not storage.is_real_root(storage.TEST_ROOT)
    assert storage.is_real_root(None) and storage.is_real_root(storage.DATA_ROOT)
    try:
        Session("unimodal_visual", "SIM86", SimulatedPresenter(1), root=real, max_trials=5)
        raise AssertionError("a trial limit must be refused for real sessions")
    except ValueError:
        pass


def check_autotest_report(root):
    """The autotest report: clean sessions pass; dropped frames and mistimed flashes are reported."""
    from avdur import autotest
    for task in ("unimodal_auditory", "unimodal_visual", "crossmodal"):
        assert Session(task, "SIM70", SimulatedPresenter(7), seed=7, root=root, simulate=True, autotest=True,
                       max_trials=20).run()
    text = autotest.report("SIM70", root=root)
    assert "no problems found" in text and text.count("20/20 trials, completed") == 3, text
    assert Session("unimodal_visual", "SIM71", DroppingPresenter(7), seed=7, root=root, simulate=True,
                   autotest=True, max_trials=20).run()
    text = autotest.report("SIM71", root=root)
    assert "dropped frames in 4 trials" in text and "4 visual flashes not of their intended length" in text, text
    return text


def check_combined_session(tmp):
    """Three blocks in one run: fits and decisions saved; a failed check stops when the experimenter ends."""
    from avdur import run_blocks
    listener = dict(sigma_a=lambda n: 0.7 / np.sqrt(2) * (n / 0.8) ** 0.5)     # sigma 0.7 at noise 0.8
    ok, results = run_blocks.run("P50", SimulatedPresenter(3, listener), tmp, simulate=True, background_fits=False,
                                 log=lambda *_: None)
    n_std = 1 + len(S.CROSSMODAL_STANDARDS["standards"]) if S.CROSSMODAL_STANDARDS["enabled"] else 1
    assert ok and [r["task"] for r in results] == list(S.COMBINED_SESSION_TASKS) + ["crossmodal"] * (n_std - 1)
    noise = storage.latest_noise("P50", tmp)[0]["noise_high"]
    assert 0.40 < 0.7 * (noise / 0.8) ** 0.5 < 1.25, noise           # sigma reached near the 0.7 target (99% range)
    for task in S.COMBINED_SESSION_TASKS:
        metas = storage.sessions("P50", task, tmp)
        assert len(metas) == (n_std if task == "crossmodal" else 1), (task, len(metas))
        for meta in metas:
            assert meta["screening"]["decision"] == "no warnings" and Path(meta["block_fit"]["plot"]).exists(), meta
            if task != "noise_calibration":
                assert meta["block_fit"]["conditions"], task
        if task in ("unimodal_auditory", "crossmodal"):
            assert sorted(storage.load_trials(task, "P50", tmp).audNoise.unique()) == [S.NOISE_LOW, noise], task
    from avdur import screening
    m = storage.sessions("SIM01", "main", SIM_ROOT)[0]
    r = screening.main_block_fit(m["csv"], tmp / "main_fits")
    assert set(r["slopes"]) == set(S.NOISE_LEVELS) and Path(r["plot"]).exists(), r
    assert r["slopes"][1.2] > r["slopes"][0.1], "vision weighs more when audition is noisier (simulated observer)"
    xm = storage.sessions("P50", "crossmodal", tmp)[0]["block_fit"]["conditions"]
    assert all("correction_s" in c for c in xm)
    assert run_blocks.run("P50", SimulatedPresenter(3), tmp, simulate=True, background_fits=False,
                          log=lambda *_: None) == (True, []), "completed blocks are skipped"

    sloppy = SimulatedPresenter(4, dict(sigma_a={0.1: 1.5, 1.2: 2.0}, lapse=0.4))
    sloppy.experimenter_continues = False
    ok, results = run_blocks.run("P51", sloppy, tmp, simulate=True, background_fits=False, log=lambda *_: None)
    assert not ok and len(results) == 1 and results[0]["warnings"], results       # stopped after the calibration
    (meta,) = storage.sessions("P51", "noise_calibration", tmp)
    assert meta["completed"] and meta["screening"]["decision"] == "ended by the experimenter"
    assert not storage.sessions("P51", "unimodal_auditory", tmp, completed_only=False), "no block after the end"


def check_crossmodal_standards(tmp):
    """Several standards: one session each (500 ms full design first); only 500 ms sets the correction."""
    from avdur import run_task, standards
    args = ["--task", "crossmodal", "--simulate", "--seed", "3", "--data-root", str(tmp), "--noise-high", "0.8"]
    assert run_task.main(["--pid", "SIM72"] + args) == 0
    metas = storage.sessions("SIM72", "crossmodal", tmp)
    extra = S.CROSSMODAL_STANDARDS["standards"]
    assert standards.plan("SIM72", tmp)[0][0] == S.STANDARD_DUR_S          # 500 ms runs first (start times tie here)
    assert sorted(round(m["standard_s"], 3) for m in metas) == sorted([S.STANDARD_DUR_S, *extra])
    d = storage.load_trials("crossmodal", "SIM72", tmp)
    n = d.groupby(d.standardDur.round(3)).size().to_dict()
    cfg = S.CROSSMODAL_STANDARDS["extra_design"]
    short = 2 * (len(cfg["staircases"]) * cfg["trials_per_staircase"] + 2 * cfg["lapse_trials_per_level"])
    assert n[S.STANDARD_DUR_S] == 388 and all(n[round(x, 3)] == short for x in extra), n
    assert (bias_mod.load_crossmodal("SIM72", tmp).standardDur == S.STANDARD_DUR_S).all()
    table = standards.bias_table("SIM72", tmp, n_boot=0)
    assert len(table) == 2 * len(metas) and table.standard_ms.nunique() == len(metas)
    assert standards.remaining("SIM72", tmp) == [] and run_task.main(["--pid", "SIM72"] + args) == 0   # nothing left
    assert run_task.main(["--pid", "SIM73", "--single-standard"] + args) == 0
    assert len(storage.sessions("SIM73", "crossmodal", tmp)) == 1


def check_noise_calibration(tmp):
    """The calibration finds the level where a listener's sigma is the target; later tasks use it;
    a real session without a calibration is refused."""
    from avdur import noise_calibration, run_task
    found = []
    for n_star in (0.3, 1.2):
        listener = dict(sigma_a=lambda n, n_star=n_star: 0.7 / np.sqrt(2) * (n / n_star) ** 0.6)
        pid = f"SIM6{len(found)}"
        cal = noise_calibration.NoiseCalibration(pid, SimulatedPresenter(5, listener), seed=5, root=tmp, simulate=True)
        assert cal.run() and Path(cal.calibration["file"]).exists() and Path(cal.calibration["plot"]).exists()
        found.append(cal.calibration["noise_high"])
        sigma = 0.7 * (found[-1] / n_star) ** 0.6
        assert 0.45 < sigma < 1.1, (n_star, found[-1], sigma)
        d = pd.read_csv(cal.csv_path)
        n_st = len(S.TASKS["noise_calibration"]["staircases"])
        assert len(d) == n_st * S.NOISE_CALIBRATION["trials_per_staircase"] + S.NOISE_CALIBRATION["catch_trials"]
        assert (d[d.kind == "staircase"].order == 1).mean() == 0.5
    assert found[0] < found[1], found
    assert run_task.main(["--task", "unimodal_auditory", "--pid", "SIM60", "--simulate", "--seed", "1",
                          "--data-root", str(tmp)]) == 0
    assert set(storage.load_trials("unimodal_auditory", "SIM60", tmp).audNoise) == {S.NOISE_LOW, found[0]}
    try:
        run_task.main(["--task", "crossmodal", "--pid", "OYX", "--data-root", str(tmp)])
        raise AssertionError("a real session without a noise calibration must be refused")
    except SystemExit as err:
        assert err.code == 2


def run_all(root=SIM_ROOT):
    if root.exists():
        shutil.rmtree(root)
    summary = {}
    check_staircase_rules()
    for pid, seed, obs in (("SIM01", 1, None),
                           ("SIM02", 2, dict(true_bias={0.1: -0.12, 1.2: 0.15}, interval_bias=0.8))):
        est, o = simulate_participant(pid, seed, obs, root)
        check_counts(root, pid)
        check_order_balance(root, pid)
        check_staircase_starts(root, pid)
        check_quest(root, pid)
        check_crossmodal_blocks(root, pid)
        check_randomised(root, pid)
        worst = check_main_alignment(root, pid)
        lead = check_preparation_and_answer_window(root, pid)
        check_bias(root, pid, est, o)
        summary[pid] = dict(true_bias=o["true_bias"],
                            fitted={k: v["correction_s"] for k, v in est["per_noise"].items()},
                            warnings=est["warnings"], worst_centre_offset_frames=worst, shortest_lead_s=lead)
    check_display_profiles(root)
    check_failed_start_and_ids(root)
    with tempfile.TemporaryDirectory() as tmp:          # throwaway participants, kept out of data_simulated/
        check_solid(Path(tmp))
        check_autotest_report(Path(tmp))
        check_combined_session(Path(tmp))
        check_noise_calibration(Path(tmp))
        check_crossmodal_standards(Path(tmp))
        check_entry_points(Path(tmp))
    snr = check_audio()
    return summary, snr


def check_entry_points(root):
    """Start each task file the way the launcher does (python -m ...), not by calling main()."""
    import subprocess
    for args in (["avdur.run_task", "--task", "crossmodal", "--noise-high", "1.0"],
                 ["avdur.run_task", "--task", "unimodal_visual"],
                 ["avdur.run_blocks"]):
        r = subprocess.run([sys.executable, "-m", *args, "--pid", "SIM80", "--simulate", "--max-trials", "2",
                            "--data-root", str(root)], cwd=ROOT, capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, AVDUR_NO_NOTIFY="1"))
        assert r.returncode == 0, f"{' '.join(args)} failed:\n{r.stderr[-2000:]}"


def test_pipeline():
    run_all()


if __name__ == "__main__":
    summary, snr = run_all()
    print("All checks passed.")
    for pid, s in summary.items():
        print(f"{pid}: true bias {s['true_bias']}  fitted {s['fitted']}  "
              f"max visual-centre offset {s['worst_centre_offset_frames']} frames  "
              f"shortest noise before the first event {1000 * s['shortest_lead_s']:.0f} ms  warnings: {s['warnings'] or 'none'}")
    snr, jitter, xm_gap = snr
    for noise, (v2, v1) in snr.items():
        print(f"Burst-to-floor SNR at noise {noise}: v2 {v2:.1f} dB, v1 {v1:.1f} dB")
    print(f"Per-trial burst level SD: v2 {jitter[0]:.2f} dB, v1 {jitter[1]:.2f} dB")
    print(f"Cross-modal minus main test level: v2 {xm_gap[0]:+.2f} dB, v1 {xm_gap[1]:+.2f} dB")
