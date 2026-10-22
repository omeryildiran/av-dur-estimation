"""Autotest: run the whole experiment with automatic answers on the real display and sound
card, then report how every trial ran.

    python -m avdur.autotest --monitor lab_crt            # every task in full, 3 main blocks (about 2 h)
    python -m avdur.autotest --monitor lab_crt --quick    # first 30 trials of each task (about 7 min)
    python -m avdur.autotest --report-only --pid P99      # report on this ID's earlier autotest sessions

Each task runs in its own process, exactly as from the launcher. Answers come from the
simulated observer at random times (AUTOTEST_* in experiment_settings.py). Escape during a
trial stops the autotest. Data and the report go to data_test/<PID>/, never to data/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur import bias as bias_mod  # noqa: E402
from avdur import design, storage  # noqa: E402
from avdur.run_task import EXIT_STOPPED  # noqa: E402

DEFAULT_PID = "P99"


def task_command(task, pid, monitor, max_trials=None, block=None, bias_file=None, root=None, screen=None):
    cmd = [sys.executable, "-m", "avdur.run_task", "--task", task, "--pid", pid, "--monitor", monitor,
           "--autotest", "--data-root", str(root or storage.TEST_ROOT)]
    if screen is not None:
        cmd += ["--screen", str(screen)]
    if max_trials:
        cmd += ["--max-trials", str(max_trials)]
    if block:
        cmd += ["--block", block]
    if bias_file:
        cmd += ["--bias-file", str(bias_file)]
    return cmd


def bias_file(pid, quick=False, root=None):
    """Bias for autotest main blocks: fitted from full cross-modal autotests if any, else fixed values."""
    root = root or storage.TEST_ROOT
    est = None
    if not quick:
        try:
            est = bias_mod.estimate_for_participant(pid, root, n_boot=50)
        except FileNotFoundError:
            pass
    if est is None:
        est = bias_mod.fixed(S.AUTOTEST_BIAS_S, "autotest: fixed corrections (AUTOTEST_BIAS_S), not fitted")
    return bias_mod.save(pid, est, confirmed_by="autotest", root=root)


def run(pid=DEFAULT_PID, monitor=S.DEFAULT_MONITOR, quick=False, root=None, screen=None):
    """Run every task (all three main blocks) and return (report text, report file)."""
    root = Path(root or storage.TEST_ROOT)
    since = dt.datetime.now().isoformat(timespec="seconds")
    max_trials = S.AUTOTEST_QUICK_TRIALS if quick else None
    plan = [(t, None) for t in storage.TASK_ORDER if t != "main"] + [("main", b) for b in design.block_order(pid, root)]
    path = None
    for task, block in plan:
        if task == "main" and path is None:
            path = bias_file(pid, quick, root)
        print(f"[autotest] starting {task}{' ' + block if block else ''}", flush=True)
        code = subprocess.run(task_command(task, pid, monitor, max_trials, block, path, root, screen), cwd=ROOT).returncode
        if code == EXIT_STOPPED:
            print("[autotest] stopped with Escape.", flush=True)
            break
    text = report(pid, since, root, header=f"monitor profile {monitor}" + (
        f", quick: first {max_trials} trials per task" if quick else ", every trial"))
    out = root / pid / f"{pid}_autotest_report_{storage.timestamp()}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text, out


# ── report ────────────────────────────────────────────────────────────────────

def _ms(x):
    return f"{1000 * x:+.1f} ms"


def session_summary(meta, df):
    """(lines, problems) for one session: what ran, frame and sound timing, visual durations."""
    nominal = meta["frame_rate_nominal_hz"]
    measured = meta.get("frame_rate_measured_hz") or nominal
    half_frame = 0.5 / measured
    n, planned = int(meta.get("n_trials_done") or len(df)), meta.get("n_trials_planned")
    if meta.get("completed"):
        state = "completed"
    elif meta.get("quit_by_user"):
        state = "stopped with Escape"
    else:
        state = "ENDED EARLY (error: see the terminal)"
    name = meta["task"] + (f" {meta['block']}" if meta.get("block") else "")
    lines, problems = [f"{name:22s} {n}/{planned} trials, {state}"], []
    if not meta.get("completed") and not meta.get("quit_by_user"):
        problems.append(f"{name}: session ended early")
    if df.empty:
        return lines, problems

    dropped = df["n_dropped_frames"].fillna(0)
    lines.append(f"    frames: {int((dropped > 0).sum())} trials with dropped frames ({int(dropped.sum())} frames), "
                 f"longest frame {df['max_frame_interval_ms'].max():.1f} ms; "
                 f"frame-0 flip vs prediction: max {1000 * df['flip0_error_s'].abs().max():.2f} ms")
    if (dropped > 0).any():
        problems.append(f"{name}: dropped frames in {int((dropped > 0).sum())} trials")
    if df["flip0_error_s"].abs().max() > half_frame:
        problems.append(f"{name}: frame 0 flipped more than half a frame off its prediction")

    errors = []
    for label in ("Test", "Standard"):
        intended, recorded = df[f"intendedDurVisual{label}"], df[f"recordedDurVisual{label}"]
        ok = intended.notna() & recorded.notna()
        errors.append(recorded[ok] * measured - intended[ok] * nominal)      # in frames
    err = pd.concat(errors)
    if len(err):
        bad = int((err.abs() > 0.5).sum())
        lines.append(f"    visual flashes: {len(err)} timed, max error {err.abs().max():.2f} frames, "
                     f"{bad} off by more than half a frame")
        if bad:
            problems.append(f"{name}: {bad} visual flashes not of their intended length")

    if "audio_start_minus_flip0_s" in df and df["audio_start_minus_flip0_s"].notna().any():
        a = df["audio_start_minus_flip0_s"].dropna()
        xruns = int(df["audio_xruns"].max()) if df["audio_xruns"].notna().any() else 0     # since the stream opened
        clipped = int((df["audio_gain_reduction_db"].fillna(0) > 0).sum())
        late = (a + df.loc[a.index, "flip0_error_s"]).abs().max()     # sound start vs its scheduled time
        lines.append(f"    sound start vs its schedule: max {1000 * late:.1f} ms; "
                     f"sound onset minus frame-0 flip (PTB estimate): median {_ms(a.median())}, "
                     f"range {_ms(a.min())} to {_ms(a.max())}; buffer underruns: {xruns}; "
                     f"level reduced against clipping: {clipped} trials")
        if late > 0.001:
            problems.append(f"{name}: sound started up to {1000 * late:.1f} ms off its schedule (sound card latency)")
        if abs(a.median()) > 0.001 or a.abs().max() > half_frame:
            cause = " because frame 0 flipped off its prediction" if late <= 0.001 else ""
            problems.append(f"{name}: sound onset not aligned with frame 0{cause} (median {_ms(a.median())}, "
                            f"max {1000 * a.abs().max():.1f} ms)")
        if xruns:
            problems.append(f"{name}: {xruns} sound buffer underruns (audible glitches)")
    if "wall_time" in df and len(df) > 1:
        pace = pd.to_datetime(df["wall_time"]).diff().dt.total_seconds().median()
        lines.append(f"    prompt shown on {100 * df['prompt_shown'].mean():.0f}% of trials, median RT "
                     f"{df['response_rts'].median():.2f} s; median {pace:.2f} s per trial")
        if not df["prompt_shown"].all():
            problems.append(f"{name}: answers before the prompt")
    return lines, problems


def report(pid, since=None, root=None, header=""):
    """Text report on the autotest sessions of `pid` (those started at or after `since`)."""
    root = Path(root or storage.TEST_ROOT)
    metas = [m for m in storage.sessions(pid, None, root, completed_only=False)
             if m.get("autotest") and (since is None or m["started"] >= since)]
    out = [f"AUTOTEST REPORT  {pid}  {dt.datetime.now():%Y-%m-%d %H:%M}" + (f"  ({header})" if header else ""), ""]
    with_audio = [m for m in metas if m.get("audio_device")]
    if with_audio:
        m = with_audio[-1]
        out.append(f"Sound output: {m['audio_device']}, {m['audio_sample_rate_hz']:.0f} Hz, output latency "
                   f"{1000 * m['audio_output_latency_s']:.1f} ms, scheduled {m['audio_schedule_lead_frames']} frame(s) ahead")
    if metas:
        rates = ", ".join(f"{m['frame_rate_measured_hz']:.2f}" for m in metas if m.get("frame_rate_measured_hz"))
        out += [f"Display: measured {rates} Hz (profile {metas[-1]['frame_rate_nominal_hz']:g} Hz)", ""]
    problems = []
    for meta in metas:
        df = pd.read_csv(meta["csv"]) if Path(meta["csv"]).exists() else pd.DataFrame()
        lines, probs = session_summary(meta, df)
        out += lines
        problems += probs
    failed = []
    log = root / "failed_starts.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("participant") == pid and entry.get("autotest") and (since is None or entry["started"] >= since):
                failed.append(entry)
    for entry in failed:
        out.append(f"{entry['task']:22s} DID NOT START: {entry['error'].splitlines()[0]}")
        problems.append(f"{entry['task']}: did not start")
    if not metas and not failed:
        out.append("No autotest sessions found.")
    out += ["", "Result: " + ("no problems found." if not problems and metas else f"{len(problems)} problem(s):")]
    out += [f"  - {p}" for p in problems]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", default=DEFAULT_PID, help=f"participant ID for the test data (default {DEFAULT_PID})")
    ap.add_argument("--monitor", default=S.DEFAULT_MONITOR, choices=list(S.MONITOR_PROFILES))
    ap.add_argument("--quick", action="store_true", help=f"first {S.AUTOTEST_QUICK_TRIALS} trials of each task only")
    ap.add_argument("--report-only", action="store_true", help="only report on earlier autotest sessions")
    ap.add_argument("--screen", type=int, help="screen number for the window (default: from the monitor profile)")
    args = ap.parse_args(argv)
    pid = design.normalize_pid(args.pid)
    if args.report_only:
        print(report(pid))
        return 0
    text, out = run(pid, args.monitor, args.quick, screen=args.screen)
    print("\n" + text + f"\n\nSaved to {out}")
    return 0 if text.rstrip().endswith("no problems found.") else 1


if __name__ == "__main__":
    sys.exit(main())
