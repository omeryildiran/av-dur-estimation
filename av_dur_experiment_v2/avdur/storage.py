"""Data layout, crash-safe trial writing, session metadata and participant status.

data/<PID>/<task>/<PID>_<task>[_<block>]_<YYYYmmdd-HHMMSS>.csv   one row per trial, flushed each trial
data/<PID>/<task>/<same stem>.json                              session metadata (completed flag at the end)
data/<PID>/bias/<PID>_bias_<YYYYmmdd-HHMMSS>.json               confirmed cross-modal bias corrections
data/<PID>/practice/...                                         practice runs (never analysed)

data_solid/ holds copies of the complete sessions of real participants on the lab display
(and their bias files): no practice, test, autotest, simulated or interrupted runs. The
analysis reads data_solid/; data/ keeps every run as it was recorded.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import experiment_settings as S

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
SOLID_ROOT = ROOT / "data_solid"            # copies of complete real sessions: what the analysis uses
SIM_ROOT = ROOT / "data_simulated"          # simulated runs never mix with real data
TEST_ROOT = ROOT / "data_test"              # test-display profiles and autotests
TASK_ORDER = ("noise_calibration", "unimodal_auditory", "unimodal_visual", "main")   # cross-modal dropped 28 Oct 2026


def root_for(simulate=False, monitor=None, autotest=False):
    """Data folder for a run: simulated, test (test display or autotest), or real participant data."""
    if simulate:
        return SIM_ROOT
    if autotest or S.MONITOR_PROFILES.get(monitor or S.DEFAULT_MONITOR, {}).get("testing_only"):
        return TEST_ROOT
    return DATA_ROOT


def is_real_root(root):
    return Path(root or DATA_ROOT).resolve() not in (SIM_ROOT.resolve(), TEST_ROOT.resolve())


def is_solid(meta, root):
    """A complete session of a real participant on the lab display: belongs in data_solid/."""
    return bool(meta.get("completed") and is_real_root(root)
                and not any(meta.get(k) for k in ("practice", "simulated", "autotest", "testing_only_profile",
                                                  "trial_limit", "refresh_warning_overridden")))


def copy_to_solid(paths, pid, folder, solid_root=None):
    """Copy finished files to data_solid/<PID>/<folder>/ (data/ keeps the originals)."""
    dest = Path(solid_root or SOLID_ROOT) / pid / folder
    dest.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, dest / Path(path).name)
    return dest


def timestamp():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def session_paths(pid, task, block=None, practice=False, root=None):
    folder = Path(root or DATA_ROOT) / pid / ("practice" if practice else task)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{pid}_{task}" + (f"_{block}" if block else "") + f"_{timestamp()}"
    return folder / f"{stem}.csv", folder / f"{stem}.json"


def code_version():
    """Git commit (+ uncommitted changes flag) and a checksum of every experiment source file."""
    files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "avdur").glob("*.py"))
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]
    info = {"source_sha256": digest}
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                            text=True, timeout=5).stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=ROOT, capture_output=True,
                               text=True, timeout=5).stdout.strip()          # source files only, not data
        info["git_uncommitted_changes"] = bool(dirty)
    except Exception:
        info["git_commit"] = None
    return info


def _json_keys(value):
    """Settings with every dict key as text (JSON allows no tuple keys)."""
    if isinstance(value, dict):
        return {k if isinstance(k, str) else str(k): _json_keys(v) for k, v in value.items()}
    return value


def settings_snapshot():
    return {k: _json_keys(v) for k, v in vars(S).items() if k.isupper()}


def base_metadata(pid, task, block, practice, seed, monitor, simulate, autotest=False) -> dict:
    return dict(
        participant=pid, task=task, block=block, practice=practice, seed=seed, monitor_profile=monitor,
        simulated=simulate, autotest=autotest, started=dt.datetime.now().isoformat(timespec="seconds"),
        completed=False,
        code=code_version(), python=sys.version.split()[0], platform=platform.platform(),
        psychopy=getattr(sys.modules.get("psychopy"), "__version__", None),     # None in simulations
        settings=settings_snapshot(),
    )


def log_failed_start(root, meta, error):
    """One JSON line per session that stopped before its first trial (e.g. failed display check)."""
    folder = Path(root or DATA_ROOT)
    folder.mkdir(parents=True, exist_ok=True)
    keep = ("participant", "task", "block", "practice", "autotest", "monitor_profile", "started",
            "frame_rate_nominal_hz", "frame_rate_measured_hz", "irregular_frame_share", "frame_interval_sd_ms",
            "ptb_libraries_preloaded", "audio_output_requested", "audio_devices_found")
    entry = {k: meta.get(k) for k in keep}
    entry["error"] = str(error)
    with open(folder / "failed_starts.jsonl", "a") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


class TrialWriter:
    """Appends one CSV row per trial and forces it to disk, so a crash loses at most one trial."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = None
        self._writer = None

    def write(self, row):
        if self._writer is None:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        if self._file:
            self._file.close()


# ── Reading ───────────────────────────────────────────────────────────────────

def sessions(pid=None, task=None, root=None, completed_only=True):
    """Metadata of every session (optionally filtered), sorted by start time."""
    out = []
    for meta_path in sorted(Path(root or DATA_ROOT).glob(f"{pid or '*'}/{task or '*'}/*.json")):
        if meta_path.parent.name in ("bias", "practice", "noise", "fits"):
            continue
        meta = json.loads(meta_path.read_text())
        if completed_only and not meta.get("completed"):
            continue
        meta["csv"] = str(meta_path.with_suffix(".csv"))
        meta["json"] = str(meta_path)
        out.append(meta)
    return sorted(out, key=lambda m: m["started"])


def load_trials(task, pid=None, root=None, completed_only=True):
    frames = []
    for meta in sessions(pid, task, root, completed_only):
        if Path(meta["csv"]).exists():
            d = pd.read_csv(meta["csv"])
            d["session_file"] = Path(meta["csv"]).name
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def participants(root=None):
    root = Path(root or DATA_ROOT)
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def latest_bias(pid, root=None):
    files = sorted((Path(root or DATA_ROOT) / pid / "bias").glob(f"{pid}_bias_*.json"))
    return (json.loads(files[-1].read_text()), files[-1]) if files else (None, None)


def latest_noise(pid, root=None):
    """The participant's latest noise calibration (payload, path), or (None, None)."""
    files = sorted((Path(root or DATA_ROOT) / pid / "noise").glob(f"{pid}_noise_*.json"))
    return (json.loads(files[-1].read_text()), files[-1]) if files else (None, None)


def participant_status(pid, root=None):
    """What this participant has completed, for the launcher."""
    from avdur.design import block_order

    status = {}
    for task in TASK_ORDER:
        done = sessions(pid, task, root)
        status[task] = [dict(block=m.get("block"), started=m["started"], n_trials=m.get("n_trials_done"))
                        for m in done]
    noise, noise_path = latest_noise(pid, root)
    status["noise"] = noise
    status["noise_file"] = str(noise_path) if noise_path else None
    bias, bias_path = latest_bias(pid, root)
    status["bias"] = bias
    status["bias_file"] = str(bias_path) if bias_path else None
    try:
        order = block_order(pid, root)
        done_blocks = {s["block"] for s in status["main"]}
        status["block_order"] = order
        status["next_block"] = next((b for b in order if b not in done_blocks), None)
    except ValueError:
        status["block_order"], status["next_block"] = None, None
    return status
