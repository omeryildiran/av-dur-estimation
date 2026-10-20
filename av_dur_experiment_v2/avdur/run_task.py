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
from avdur import design  # noqa: E402
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
        presenter = PsychoPyPresenter(args.monitor, autotest_seed=auto_seed)

    session = Session(args.task, args.pid, presenter, block=args.block, practice=args.practice, bias=bias,
                      bias_file=args.bias_file, seed=args.seed, root=Path(args.data_root) if args.data_root else None,
                      monitor=args.monitor, simulate=args.simulate, autotest=args.autotest,
                      max_trials=args.max_trials)
    return 0 if session.run() else EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
