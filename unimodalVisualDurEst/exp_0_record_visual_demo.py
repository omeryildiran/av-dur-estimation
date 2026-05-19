"""
Record a short visual-stimulus demo from the PsychoPy experiment logic.

This is intentionally separate from exp_0_executer_visual.py so the real
experiment files are not modified. It skips welcome, response, staircase, and
data-saving screens, then captures only a few stimulus trials.

Run from a PsychoPy environment:

    python exp_0_record_visual_demo.py --trials 4

The default output is:

    ../media/video_abstract/visual_demo_from_exp.mp4
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


try:
    from psychopy import core, event, monitors, visual
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PsychoPy is required to run this recorder. Run it from the same "
        "environment you use for the visual experiment."
    ) from exc


CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))

from dva_to_pix import dva_to_px
from sec2frame import frames2sec, sec2frames


MONITOR_OPTIONS = {
    "asusZenbook14": {
        "sizeIs": 1024,
        "screen_width": 30.5,
        "screen_height": 18,
        "screen_distance": 40,
    },
    "labMon": {
        "sizeIs": 1024,
        "screen_width": 28,
        "screen_height": 28,
        "screen_distance": 60,
    },
    "macAir": {
        "sizeIs": 800,
        "screen_width": 25,
        "screen_height": 20,
        "screen_distance": 40,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a few real PsychoPy visual-stimulus trials."
    )
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--standard-dur", type=float, default=0.5)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--monitor", choices=MONITOR_OPTIONS.keys(), default="macAir")
    parser.add_argument(
        "--output",
        default=str(PARENT_DIR / "media/video_abstract/visual_demo_from_exp.mp4"),
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the temporary PNG frames next to the output.",
    )
    return parser.parse_args()


def create_window(args: argparse.Namespace) -> tuple[visual.Window, dict]:
    monitor_specs = MONITOR_OPTIONS[args.monitor]
    size_is = monitor_specs["sizeIs"]

    mon = monitors.Monitor(
        args.monitor,
        width=monitor_specs["screen_width"],
        distance=monitor_specs["screen_distance"],
    )
    mon.setSizePix((size_is, size_is))

    win = visual.Window(
        size=(size_is, size_is),
        fullscr=args.fullscreen,
        monitor=mon,
        units="pix",
        color="black",
        useFBO=True,
        screen=0,
        colorSpace="rgb",
        allowGUI=False,
    )
    win.monitor.setWidth(monitor_specs["screen_width"])
    win.monitor.setDistance(monitor_specs["screen_distance"])
    win.monitor.setSizePix((size_is, size_is))
    return win, monitor_specs


def random_interval(low: float, high: float, fps: int) -> tuple[int, float]:
    frames = sec2frames(random.uniform(low, high), fps)
    return frames, frames2sec(frames, fps)


def render_trial(
    win: visual.Window,
    monitor_specs: dict,
    trial_idx: int,
    fps: int,
    standard_dur: float,
) -> dict:
    size_is = monitor_specs["sizeIs"]
    screen_height = monitor_specs["screen_height"]
    screen_distance = monitor_specs["screen_distance"]
    visual_stim_size = dva_to_px(
        size_in_deg=1.5,
        h=screen_height,
        d=screen_distance,
        r=size_is,
    )

    pre_frames, pre_dur = random_interval(0.2, 0.45, fps)
    isi_frames, isi_dur = random_interval(0.4, 0.9, fps)
    post_frames, post_dur = random_interval(0.2, 0.45, fps)
    stim1_frames = sec2frames(standard_dur, fps)
    stim2_frames = sec2frames(standard_dur, fps)

    onset1 = pre_frames
    offset1 = onset1 + stim1_frames
    onset2 = offset1 + isi_frames
    offset2 = onset2 + stim2_frames
    total_frames = offset2 + post_frames

    visual_stim = visual.Circle(
        win,
        radius=visual_stim_size,
        fillColor="black",
        lineColor="white",
        colorSpace="rgb",
        units="pix",
        pos=(0, 0),
    )
    visual_stim.lineWidth = 5
    visual_stim.color = "white"
    visual_stim.setAutoDraw(True)

    trial_clock = core.Clock()
    trial_clock.reset()

    for frame_n in range(total_frames):
        if frame_n < pre_frames:
            visual_stim.fillColor = "black"
        elif frame_n == onset1:
            visual_stim.fillColor = "white"
        elif frame_n == offset1:
            visual_stim.fillColor = "black"
        elif frame_n == onset2:
            visual_stim.fillColor = "white"
        elif frame_n == offset2:
            visual_stim.fillColor = "black"

        if event.getKeys(keyList=["escape"]):
            visual_stim.setAutoDraw(False)
            raise KeyboardInterrupt

        win.flip()
        win.getMovieFrame(buffer="front")

    visual_stim.setAutoDraw(False)
    win.flip()

    return {
        "trial": trial_idx,
        "preDur": pre_dur,
        "stim1Dur": frames2sec(stim1_frames, fps),
        "isiDur": isi_dur,
        "stim2Dur": frames2sec(stim2_frames, fps),
        "postDur": post_dur,
        "totalDur": frames2sec(total_frames, fps),
    }


def save_movie(win: visual.Window, output: Path, fps: int, keep_frames: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.with_suffix("")
    frame_dir = frame_dir.parent / f"{frame_dir.name}_frames"

    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    for frame_idx, frame in enumerate(win.movieFrames, start=1):
        frame.save(frame_dir / f"frame{frame_idx:06d}.png")
    win.movieFrames = []

    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    env = os.environ.copy()
    x265_compat = "/opt/homebrew/Cellar/x265/4.1/lib"
    if Path(x265_compat).exists():
        env["DYLD_LIBRARY_PATH"] = x265_compat

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        env=env,
    )

    if not keep_frames:
        shutil.rmtree(frame_dir)


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)

    win = None
    try:
        win, monitor_specs = create_window(args)
        trial_info = []

        for trial_idx in range(args.trials):
            trial_info.append(
                render_trial(
                    win=win,
                    monitor_specs=monitor_specs,
                    trial_idx=trial_idx + 1,
                    fps=args.fps,
                    standard_dur=args.standard_dur,
                )
            )

        save_movie(win, Path(args.output), args.fps, args.keep_frames)
        print(f"Saved: {args.output}")
        for row in trial_info:
            print(row)
    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        if win is not None:
            win.close()
        core.quit()


if __name__ == "__main__":
    main()
