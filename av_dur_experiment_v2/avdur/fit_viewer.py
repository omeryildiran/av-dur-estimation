"""Fit viewer for the experimenter's screen: shows each new fit plot of a participant as it appears.

    python -m avdur.fit_viewer --folder data/P14/fits [--x 0 --y 0 --width 900 --height 700]

The launcher opens it on a screen other than the participant's, before the experiment window,
and leaves it open. It checks the folder every 2 s and shows the newest plot. It only updates
its own window and never raises itself, so it does not take the focus from the full-screen
experiment window. Older plots: the arrow buttons.
"""

from __future__ import annotations

import argparse
import math
import tkinter as tk
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--title", default="Fits")
    ap.add_argument("--x", type=int, default=40)
    ap.add_argument("--y", type=int, default=40)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args(argv)
    folder = Path(args.folder)

    root = tk.Tk()
    root.title(args.title)
    root.geometry(f"{args.width}x{args.height}+{args.x}+{args.y}")
    caption = tk.Label(root, text=f"Waiting for the first fit in {folder} ...", font=("Arial", 12))
    caption.pack(pady=(8, 4))
    picture = tk.Label(root)
    picture.pack(fill="both", expand=True)
    nav = tk.Frame(root)
    nav.pack(pady=6)
    state = dict(files=[], index=-1, follow=True)

    def show():
        if not state["files"]:
            return
        path = state["files"][state["index"]]
        try:
            img = tk.PhotoImage(file=str(path))
        except tk.TclError:                      # file still being written: try again next poll
            return
        scale = max(1, math.ceil(img.width() / (args.width - 20)), math.ceil(img.height() / (args.height - 90)))
        if scale > 1:
            img = img.subsample(scale)
        picture.configure(image=img)
        picture.image = img
        caption.configure(text=f"{path.name}   ({state['index'] + 1} of {len(state['files'])})")

    def step(k):
        if state["files"]:
            state["index"] = min(max(state["index"] + k, 0), len(state["files"]) - 1)
            state["follow"] = state["index"] == len(state["files"]) - 1
            show()

    tk.Button(nav, text="◀ older", command=lambda: step(-1)).pack(side="left", padx=4)
    tk.Button(nav, text="newer ▶", command=lambda: step(+1)).pack(side="left", padx=4)

    def poll():
        files = sorted(folder.glob("*_fit.png"), key=lambda p: p.stat().st_mtime) if folder.exists() else []
        if files != state["files"]:
            new = len(files) > len(state["files"])
            state["files"] = files
            if state["follow"] or new:
                state["index"], state["follow"] = len(files) - 1, True
            show()
        root.after(2000, poll)

    poll()
    root.mainloop()


if __name__ == "__main__":
    main()
