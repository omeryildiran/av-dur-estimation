"""Combined session: unimodal auditory, unimodal visual and cross-modal blocks in one window.

    python -m avdur.run_blocks --pid P14                     # lab display, real data
    python -m avdur.run_blocks --pid P14 --practice          # a short practice before each block
    add --simulate (no window) or --autotest [--max-trials N] (automatic answers, data_test/)

Each block is saved as its own session (as if run from its own button), so storage, the
data_solid/ copy and the notebooks are unchanged. Blocks this participant has already
completed are skipped, so a stopped session can be resumed. After each block the
psychometric functions are fitted in a background process while the participant rests;
the fit and a plot path appear on the terminal. If a check fails (SCREENING_CHECKS), the
participant sees "Please wait for the experimenter": C continues, Escape ends the session.
The fit and the decision are saved in the block's session metadata.

Exit code: 0 all blocks completed, 3 stopped (Escape or experimenter), 1 error.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur import design, notify, screening, standards, storage  # noqa: E402
from avdur.run_task import EXIT_STOPPED  # noqa: E402
from avdur.session import Session  # noqa: E402


def blocks_to_run(pid, root, rerun_all=False, several=None):
    """Tasks of the combined session still to run (cross-modal: until every planned standard is done)."""
    def done(t):
        if rerun_all:
            return False
        if t == "crossmodal":
            return not standards.remaining(pid, root, several)
        return bool(storage.sessions(pid, t, root))
    return [t for t in S.COMBINED_SESSION_TASKS if not done(t)]


def steps_to_run(pid, root, rerun_all=False, several=None):
    """[(task, standard_s, design, block label, interleaved standards)]; the cross-modal step runs every
    remaining standard interleaved in one session (avdur.standards)."""
    steps = []
    for t in blocks_to_run(pid, root, rerun_all, several):
        if t == "crossmodal":
            todo = standards.plan(pid, root, several) if rerun_all else standards.remaining(pid, root, several)
            steps += [(t, std, des, lab, more) for std, des, lab, more in todo]
        else:
            steps.append((t, None, None, None, []))
    return steps


def _fit_inline(task, csv_path, plot_dir):
    f = Future()
    try:
        f.set_result(screening.block_fit(task, csv_path, plot_dir))
    except Exception as err:            # a failed fit must not end the session
        f.set_exception(err)
    return f


def run(pid, presenter, root, monitor=S.DEFAULT_MONITOR, practice=False, simulate=False, autotest=False,
        max_trials=None, background_fits=True, open_plots=False, log=print, several=None):
    """Run the remaining blocks; returns (all completed, list of block results)."""
    steps = steps_to_run(pid, root, rerun_all=autotest, several=several)
    if not steps:
        log(f"{pid}: all blocks are already done.")
        return True, []
    cal = S.NOISE_CALIBRATION

    def n_trials(task, des, more=()):
        if task == "noise_calibration":
            return len(S.TASKS["noise_calibration"]["staircases"]) * cal["trials_per_staircase"] + cal["catch_trials"]
        return sum(len(design.build_trials(task, np.random.default_rng(0), cfg=dict(S.TASKS[task], **(d or {}))))
                   for d in [des] + [d for _, d in more])

    log(f"[{pid}] combined session: " + ", ".join(
        f"{screening.LABELS[t]}{' ' + lab if lab else ''} ({max_trials or n_trials(t, des, more)} trials)"
        for t, _, des, lab, more in steps) + (" — with practice" if practice else ""))
    todo = [t for t, *_ in steps]
    executor = (ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
                if background_fits else None)
    results, plot_dir = [], Path(root) / pid / "fits"
    try:
        practiced = set()
        for k, (task, standard_s, design_override, label, more) in enumerate(steps):
            last = k == len(todo) - 1
            common = dict(root=root, monitor=monitor, simulate=simulate, autotest=autotest, close_at_end=False,
                          verbose=True)
            if task == "noise_calibration":
                from avdur import noise_calibration
                session = noise_calibration.NoiseCalibration(pid, presenter, max_trials=max_trials, **common)
                if not session.run():
                    return False, results
                result = dict(session.calibration, task=task, conditions=[])
                log(noise_calibration.summary_text(session.calibration))
                if not simulate:
                    notify.block_done(pid, noise_calibration.summary_text(session.calibration),
                                      [session.calibration.get("plot")])
                presenter.wait_screen(f"Part {k + 1} of {len(todo)} is done.\n\nPlease take a short break.",
                                      lambda: True, 0.0 if last else S.BLOCK_REST_MIN_S)
                results.append(result)
                if not _decide(presenter, session, result, log):
                    presenter.finish(False)
                    return False, results
                continue
            noise, noise_path = storage.latest_noise(pid, root)
            levels = dict(noise_high=noise["noise_high"] if noise else None, noise_file=str(noise_path) if noise_path else None)
            if practice and task not in practiced:
                practiced.add(task)
                if not Session(task, pid, presenter, practice=True, **levels, **common).run():
                    return False, results
            extra = (dict(block=label, standard_s=standard_s, design=design_override, more_standards=more)
                     if task == "crossmodal" else {})
            session = Session(task, pid, presenter, max_trials=max_trials, **extra, **levels, **common)
            if not session.run():
                return False, results
            fit = None
            if executor:
                try:
                    fit = executor.submit(screening.block_fit, task, str(session.csv_path), str(plot_dir))
                except Exception as err:          # the fit process died: fit here instead, during the rest
                    log(f"(background fit unavailable: {err}; fitting in this process)")
                    executor = None
            fit = fit or _fit_inline(task, session.csv_path, plot_dir)
            text = ("This part of the experiment is done. Thank you!\n\nPlease wait for the experimenter." if last
                    else f"Part {k + 1} of {len(todo)} is done.\n\nPlease take a short break.")
            presenter.wait_screen(text, fit.done, 0.0 if last else S.BLOCK_REST_MIN_S)
            try:
                result = fit.result()
                log(screening.summary_text(result))
                if not simulate:
                    notify.block_done(pid, screening.summary_text(result), [result.get("plot"), result.get("stair_plot")])
            except Exception as err:
                result = dict(task=task, warnings=[f"the fit failed: {err}"], conditions=[], plot=None)
                log(f"── {screening.LABELS[task]}: the fit failed ({err}); the data are saved.")
            results.append(result)
            if not _decide(presenter, session, result, log):
                presenter.finish(False)
                return False, results
        if any(t == "crossmodal" for t, *_ in steps):
            text = standards.report(pid, root)
            if text:
                log(text)
        presenter.finish(True)
    finally:
        if executor:
            executor.shutdown(wait=False)       # (cancel_futures needs Python 3.9; the lab PC has 3.8)
    plots = [r["plot"] for r in results if r.get("plot")]
    if open_plots and plots:
        for path in plots:
            if sys.platform == "win32":
                os.startfile(path)                      # noqa: S606 (Windows: default image viewer)
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
    return True, results


def _decide(presenter, session, result, log):
    """Experimenter check when a block raised warnings; the fit and the decision go into its metadata."""
    continue_ = True
    if result["warnings"]:
        if not session.simulate:
            notify.send(f"⚠ {session.pid}: waiting for the experimenter\n" + "\n".join(result["warnings"]))
        log(">> Experimenter: the participant sees 'Please wait for the experimenter'. "
            "Press C to continue, Escape to end the session.")
        continue_ = presenter.experimenter_check()
        log(">> " + ("continued" if continue_ else "session ended") + " by the experimenter")
    session.meta["block_fit"] = {k: v for k, v in result.items() if k not in ("csv", "trace")}
    session.meta["screening"] = dict(
        warnings=result["warnings"],
        decision=("no warnings" if not result["warnings"] else
                  "continued by the experimenter" if continue_ else "ended by the experimenter"))
    session.save_meta()
    return continue_


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--monitor", default=S.DEFAULT_MONITOR, choices=list(S.MONITOR_PROFILES))
    ap.add_argument("--practice", action="store_true", help="a short practice before each block")
    ap.add_argument("--simulate", action="store_true", help="simulated observer, no PsychoPy window")
    ap.add_argument("--autotest", action="store_true",
                    help="automatic answers on the real display and sound card; saves to data_test/")
    ap.add_argument("--max-trials", type=int, help="simulate/autotest only: first N trials of each block")
    ap.add_argument("--screen", type=int, help="screen number for the window (default: from the monitor profile)")
    ap.add_argument("--no-open-plots", action="store_true", help="do not open the plots at the end (fit viewer in use)")
    ap.add_argument("--single-standard", action="store_true", help="cross-modal task at the 500 ms standard only")
    ap.add_argument("--all-standards", action="store_true", help="cross-modal task at every standard, whatever the count")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--data-root")
    args = ap.parse_args(argv)
    try:
        pid = design.normalize_pid(args.pid)
    except ValueError as err:
        ap.error(str(err))
    if args.simulate and args.autotest:
        ap.error("Choose --simulate or --autotest, not both.")
    if args.max_trials and not (args.simulate or args.autotest):
        ap.error("--max-trials is for --simulate or --autotest runs only")
    root = Path(args.data_root) if args.data_root else storage.root_for(args.simulate, args.monitor, args.autotest)
    if args.simulate:
        from avdur.presenter_sim import SimulatedPresenter
        presenter = SimulatedPresenter(seed=args.seed or 0)
    else:
        from avdur.presenter_psychopy import PsychoPyPresenter
        auto_seed = (args.seed if args.seed is not None else int(time.time())) % 2**31 if args.autotest else None
        presenter = PsychoPyPresenter(args.monitor, autotest_seed=auto_seed, screen=args.screen)
    ok, _ = run(pid, presenter, root, args.monitor, args.practice, args.simulate, args.autotest, args.max_trials,
                background_fits=not args.simulate,
                open_plots=not (args.simulate or args.autotest or args.no_open_plots),
                several=False if args.single_standard else (True if args.all_standards else None))
    return 0 if ok else EXIT_STOPPED


if __name__ == "__main__":
    code = main()
    notify.flush()                  # let Slack messages and plots finish sending (the window is closed)
    sys.exit(code)
