"""Run one task for one participant (normally started by run_experiment.py).

    python -m avdur.run_task --task crossmodal --pid P14
    python -m avdur.run_task --task main --pid P14 --block B2 --bias-file data/P14/bias/P14_bias_....json
    add --simulate to run the simulated observer instead of PsychoPy (no window)
    add --autotest for automatic answers on the real display and sound card (saves to data_test/)

Exit code: 0 completed, 3 stopped with Escape, 1 error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur import design, storage  # noqa: E402
from avdur.session import Session  # noqa: E402

EXIT_STOPPED = 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(S.TASKS))
    ap.add_argument("--pid", required=True)
    ap.add_argument("--block", choices=list(S.TASKS["main"]["blocks"]))
    ap.add_argument("--bias-file")
    ap.add_argument("--practice", action="store_true")
    ap.add_argument("--monitor", default=S.DEFAULT_MONITOR, choices=list(S.MONITOR_PROFILES))
    ap.add_argument("--simulate", action="store_true", help="simulated observer, no PsychoPy window")
    ap.add_argument("--autotest", action="store_true",
                    help="automatic answers on the real display and sound card; saves to data_test/")
    ap.add_argument("--max-trials", type=int, help="simulate/autotest only: run just the first N trials")
    ap.add_argument("--screen", type=int, help="screen number for the window (default: from the monitor profile)")
    ap.add_argument("--noise-high", type=float, help="high noise level (default: the participant's noise calibration)")
    ap.add_argument("--single-standard", action="store_true",
                    help="cross-modal: the 500 ms standard only (default: the extra standards for the first "
                         "CROSSMODAL_STANDARDS['n_participants'] participants)")
    ap.add_argument("--all-standards", action="store_true", help="cross-modal: every standard, whatever the count")
    ap.add_argument("--standard", type=float, help="cross-modal: run this one standard (s), e.g. to repeat it")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--data-root", help="default: data/ (data_simulated/ with --simulate, data_test/ with --autotest)")
    args = ap.parse_args(argv)
    try:
        args.pid = design.normalize_pid(args.pid)
    except ValueError as err:
        ap.error(str(err))
    if args.simulate and args.autotest:
        ap.error("--simulate runs without a window; --autotest runs the real window. Choose one.")
    if args.max_trials and not (args.simulate or args.autotest):
        ap.error("--max-trials is for --simulate or --autotest runs only")

    bias = json.loads(Path(args.bias_file).read_text()) if args.bias_file else None
    if args.task == "main" and bias is None:
        ap.error("--bias-file is required for the main experiment")
    if args.task == "main" and args.block is None:
        ap.error("--block is required for the main experiment")

    if args.simulate:
        from avdur.presenter_sim import SimulatedPresenter
        presenter = SimulatedPresenter(seed=args.seed or 0)
    else:
        from avdur.presenter_psychopy import PsychoPyPresenter
        auto_seed = (args.seed if args.seed is not None else int(time.time())) % 2**31 if args.autotest else None
        presenter = PsychoPyPresenter(args.monitor, autotest_seed=auto_seed, screen=args.screen)

    root = Path(args.data_root) if args.data_root else storage.root_for(args.simulate, args.monitor, args.autotest)
    common = dict(seed=args.seed, root=root, monitor=args.monitor, simulate=args.simulate, autotest=args.autotest,
                  max_trials=args.max_trials)
    if args.task == "noise_calibration":
        from avdur.noise_calibration import NoiseCalibration, summary_text
        session = NoiseCalibration(args.pid, presenter, **common)
        completed = session.run()
        if completed and session.calibration:
            print(summary_text(session.calibration), flush=True)
            if not args.simulate:
                from avdur import notify
                notify.block_done(args.pid, summary_text(session.calibration), [session.calibration.get("plot")])
        return 0 if completed else EXIT_STOPPED

    noise_high, noise_file = resolve_noise(ap, args, root)
    if args.task == "crossmodal" and not args.practice:
        return run_crossmodal(args, presenter, root, common, noise_high, noise_file)
    if args.task == "main" and noise_high is not None and str(float(noise_high)) not in bias["per_noise"]:
        ap.error(f"The bias file was measured at noise levels {sorted(bias['per_noise'])}, but this participant's "
                 f"high noise level is {noise_high}. Run the cross-modal task again at the calibrated level.")
    session = Session(args.task, args.pid, presenter, block=args.block, practice=args.practice, bias=bias,
                      bias_file=args.bias_file, noise_high=noise_high, noise_file=noise_file, **common)
    completed = session.run()
    if completed and not args.practice and not args.simulate:      # fit and plot (the window is closed by now)
        from avdur import screening
        try:
            print(screening.fit_after_session(args.task, session.csv_path, session.root / args.pid / "fits"), flush=True)
        except Exception as err:                                    # the data are saved either way
            print(f"(fit after the session failed: {err})", flush=True)
    return 0 if completed else EXIT_STOPPED


def resolve_noise(ap, args, root):
    """(high noise level, its calibration file): --noise-high, else the participant's latest calibration.
    Without one, only practice, simulated and autotest runs may use NOISE_HIGH_DEFAULT."""
    if args.task == "unimodal_visual":
        return None, None
    if args.noise_high is not None:
        return args.noise_high, None
    payload, path = storage.latest_noise(args.pid, root)
    if payload is not None:
        return payload["noise_high"], str(path)
    if args.practice or args.simulate or args.autotest:
        print(f"(no noise calibration for {args.pid}: using the default high noise {S.NOISE_HIGH_DEFAULT})", flush=True)
        return S.NOISE_HIGH_DEFAULT, None
    ap.error(f"{args.pid} has no auditory noise calibration yet. Run 'Auditory noise calibration' first "
             "(or give --noise-high).")


def run_crossmodal(args, presenter, root, common, noise_high, noise_file):
    """The cross-modal task: every remaining standard interleaved in one session (avdur.standards)."""
    from avdur import screening, standards
    if args.standard is not None:
        std = args.standard
        todo = [(std, standards.design_for(std), standards.label(std), [])]
    else:
        todo = standards.remaining(args.pid, root, several=False if args.single_standard else
                                   (True if args.all_standards else None))
    if not todo:
        print(f"{args.pid}: the cross-modal task is done at every standard. Use --standard to repeat one.", flush=True)
        return 0
    for k, (std, design_override, label, more) in enumerate(todo):
        last = k == len(todo) - 1
        print(f"[{args.pid}] cross-modal, standard(s): " + ", ".join(f"{1000 * v:.0f} ms" for v in [std] + [m for m, _ in more]),
              flush=True)
        session = Session("crossmodal", args.pid, presenter, block=label, noise_high=noise_high, noise_file=noise_file,
                          standard_s=std, design=design_override, more_standards=more, close_at_end=last, **common)
        if not session.run():
            return EXIT_STOPPED
        if not args.simulate:
            try:
                print(screening.fit_after_session("crossmodal", session.csv_path, session.root / args.pid / "fits"),
                      flush=True)
            except Exception as err:
                print(f"(fit after the session failed: {err})", flush=True)
        if not last:
            presenter.wait_screen(f"Part {k + 1} of {len(todo)} is done.\n\nPlease take a short break.",
                                  lambda: True, S.BLOCK_REST_MIN_S)
    text = standards.report(args.pid, root, n_boot=0 if args.simulate else 200)
    if text:
        print(text, flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    from avdur import notify
    notify.flush()                  # let Slack messages and plots finish sending (the window is closed)
    sys.exit(code)
