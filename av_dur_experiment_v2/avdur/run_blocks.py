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
from avdur import design, screening, storage  # noqa: E402
from avdur.run_task import EXIT_STOPPED  # noqa: E402
from avdur.session import Session  # noqa: E402


def blocks_to_run(pid, root, rerun_all=False):
    done = set() if rerun_all else {t for t in S.COMBINED_SESSION_TASKS if storage.sessions(pid, t, root)}
    return [t for t in S.COMBINED_SESSION_TASKS if t not in done]


def _fit_inline(task, csv_path, plot_dir):
    f = Future()
    try:
        f.set_result(screening.block_fit(task, csv_path, plot_dir))
    except Exception as err:            # a failed fit must not end the session
        f.set_exception(err)
    return f


def run(pid, presenter, root, monitor=S.DEFAULT_MONITOR, practice=False, simulate=False, autotest=False,
        max_trials=None, background_fits=True, open_plots=False, log=print):
    """Run the remaining blocks; returns (all completed, list of block results)."""
    todo = blocks_to_run(pid, root, rerun_all=autotest)
    if not todo:
        log(f"{pid}: all three blocks are already done.")
        return True, []
    n_trials = {t: len(design.build_trials(t, np.random.default_rng(0))) for t in todo}
    log(f"[{pid}] combined session: " + ", ".join(f"{screening.LABELS[t]} ({max_trials or n_trials[t]} trials)"
                                                   for t in todo) + (" — with practice" if practice else ""))
    executor = (ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
                if background_fits else None)
    results, plot_dir = [], Path(root) / pid / "fits"
    try:
        for k, task in enumerate(todo):
            last = k == len(todo) - 1
            common = dict(root=root, monitor=monitor, simulate=simulate, autotest=autotest, close_at_end=False,
                          verbose=True)
            if practice and not Session(task, pid, presenter, practice=True, **common).run():
                return False, results
            session = Session(task, pid, presenter, max_trials=max_trials, **common)
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
            except Exception as err:
                result = dict(task=task, warnings=[f"the fit failed: {err}"], conditions=[], plot=None)
                log(f"── {screening.LABELS[task]}: the fit failed ({err}); the data are saved.")
            results.append(result)
            continue_ = True
            if result["warnings"]:
                log(">> Experimenter: the participant sees 'Please wait for the experimenter'. "
                    "Press C to continue, Escape to end the session.")
                continue_ = presenter.experimenter_check()
                log(">> " + ("continued" if continue_ else "session ended") + " by the experimenter")
            session.meta["block_fit"] = {k: v for k, v in result.items() if k != "csv"}
            session.meta["screening"] = dict(
                warnings=result["warnings"],
                decision=("no warnings" if not result["warnings"] else
                          "continued by the experimenter" if continue_ else "ended by the experimenter"))
            session.save_meta()
            if not continue_:
                presenter.finish(False)
                return False, results
        presenter.finish(True)
    finally:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
    plots = [r["plot"] for r in results if r.get("plot")]
    if open_plots and plots and sys.platform == "darwin":
        subprocess.run(["open", *plots], check=False)
    return True, results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--monitor", default=S.DEFAULT_MONITOR, choices=list(S.MONITOR_PROFILES))
    ap.add_argument("--practice", action="store_true", help="a short practice before each block")
    ap.add_argument("--simulate", action="store_true", help="simulated observer, no PsychoPy window")
    ap.add_argument("--autotest", action="store_true",
                    help="automatic answers on the real display and sound card; saves to data_test/")
    ap.add_argument("--max-trials", type=int, help="simulate/autotest only: first N trials of each block")
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
        presenter = PsychoPyPresenter(args.monitor, autotest_seed=auto_seed)
    ok, _ = run(pid, presenter, root, args.monitor, args.practice, args.simulate, args.autotest, args.max_trials,
                background_fits=not args.simulate, open_plots=not (args.simulate or args.autotest))
    return 0 if ok else EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
