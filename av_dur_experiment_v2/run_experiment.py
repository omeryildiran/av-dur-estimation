"""Experiment launcher: enter the participant ID, then press the button of the task to run.

    python run_experiment.py          # window with buttons
    python run_experiment.py --cli    # text menu (if Tk is unavailable)

Each task runs in its own process (python -m avdur.run_task ...). For the main
experiment the launcher picks the participant's next counterbalanced block, fits the
cross-modal bias correction from their cross-modal data, and asks you to confirm it;
later blocks reuse the bias file of the first block.

Autotest runs a task (or, with its own button, the whole experiment) on the real display
and sound card with automatic answers, saves to data_test/, and reports the timing.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import experiment_settings as S  # noqa: E402
from avdur import autotest as autotest_mod  # noqa: E402
from avdur import bias as bias_mod  # noqa: E402
from avdur import design, storage  # noqa: E402
from avdur.run_task import EXIT_STOPPED  # noqa: E402


# ── logic shared by the window and the text menu ──────────────────────────────

def status_text(raw_pid, root=None):
    try:
        pid = design.normalize_pid(raw_pid)
    except ValueError as err:
        return str(err)
    st = storage.participant_status(pid, root)
    tag = {storage.SIM_ROOT: "[SIMULATED DATA, data_simulated/] ",
           storage.TEST_ROOT: "[TEST RUNS: test display or autotest, data_test/] "}
    lines = [tag.get(Path(root) if root else None, "") + f"Participant {pid}" + (f"   (main block order: {' -> '.join(st['block_order'])})" if st["block_order"] else "")]
    for i, task in enumerate(storage.TASK_ORDER, 1):
        done = st[task]
        if task == "main":
            blocks = ", ".join(f"{s['block']} ({s['started'][:16].replace('T', ' ')})" for s in done) or "none"
            lines.append(f"  {i}. {S.TASKS[task]['label']}: blocks done: {blocks};  next: {st['next_block'] or 'all done'}")
        else:
            when = f"done {done[-1]['started'][:16].replace('T', ' ')} ({done[-1]['n_trials']} trials)" if done else "not done"
            lines.append(f"  {i}. {S.TASKS[task]['label']}: {when}")
        if task == "crossmodal":
            b = st["bias"]
            if b:
                vals = ", ".join(f"{float(v['correction_s']):+.3f} s (noise {n})" for n, v in sorted(b["per_noise"].items()))
                lines.append(f"     bias correction: {vals}  [{Path(st['bias_file']).name}]")
            else:
                lines.append("     bias correction: not fitted yet (fitted when the main experiment starts)")
    return "\n".join(lines)


def main_bias_file(pid, root=None):
    """Bias file already used by this participant's main blocks, if any (keeps blocks consistent)."""
    for meta in storage.sessions(pid, "main", root, completed_only=False):
        if meta.get("bias_file") and not meta.get("practice"):
            return meta["bias_file"]
    return None


def combined_command(pid, monitor, practice, simulate, autotest=False, quick=False):
    """Unimodal auditory, unimodal visual and cross-modal blocks in one window (avdur.run_blocks)."""
    cmd = [sys.executable, "-m", "avdur.run_blocks", "--pid", pid, "--monitor", monitor,
           "--data-root", str(storage.root_for(simulate, monitor, autotest))]
    cmd += ["--practice"] * practice + ["--simulate"] * simulate + ["--autotest"] * autotest
    if autotest and quick:
        cmd += ["--max-trials", str(S.AUTOTEST_QUICK_TRIALS)]
    return cmd


def task_command(task, pid, monitor, practice, simulate, block=None, bias_file=None, autotest=False, quick=False):
    cmd = [sys.executable, "-m", "avdur.run_task", "--task", task, "--pid", pid, "--monitor", monitor,
           "--data-root", str(storage.root_for(simulate, monitor, autotest))]
    if practice:
        cmd.append("--practice")
    if simulate:
        cmd.append("--simulate")
    if autotest:
        cmd.append("--autotest")
        if quick:
            cmd += ["--max-trials", str(S.AUTOTEST_QUICK_TRIALS)]
    if block:
        cmd += ["--block", block]
    if bias_file:
        cmd += ["--bias-file", str(bias_file)]
    return cmd


# ── window ────────────────────────────────────────────────────────────────────

def run_window():
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Audiovisual duration experiment v2")
    pid, monitor = tk.StringVar(), tk.StringVar(value=S.DEFAULT_MONITOR)
    block, practice, simulate = tk.StringVar(value="auto"), tk.BooleanVar(), tk.BooleanVar()
    autotest, quick = tk.BooleanVar(), tk.BooleanVar(value=True)

    frm = ttk.Frame(root, padding=16)
    frm.grid(sticky="nsew")
    ttk.Label(frm, text="Participant ID (e.g. P14)").grid(row=0, column=0, sticky="w")
    entry = ttk.Entry(frm, textvariable=pid, width=14)
    entry.grid(row=0, column=1, sticky="w")
    ttk.Label(frm, text="Monitor profile").grid(row=1, column=0, sticky="w")
    ttk.OptionMenu(frm, monitor, S.DEFAULT_MONITOR, *S.MONITOR_PROFILES,
                   command=lambda _: refresh()).grid(row=1, column=1, sticky="w")
    ttk.Label(frm, text="Main-experiment block").grid(row=2, column=0, sticky="w")
    ttk.OptionMenu(frm, block, "auto", "auto", *S.TASKS["main"]["blocks"]).grid(row=2, column=1, sticky="w")
    ttk.Checkbutton(frm, text="Practice run (easy trials with feedback)", variable=practice).grid(
        row=3, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(frm, text="Simulate (no window; saves to data_simulated/)", variable=simulate,
                    command=lambda: refresh()).grid(row=4, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(frm, text="Autotest: automatic answers on this display and sound card (saves to data_test/)",
                    variable=autotest, command=lambda: refresh()).grid(row=5, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(frm, text=f"Quick autotest: first {S.AUTOTEST_QUICK_TRIALS} trials of each task",
                    variable=quick).grid(row=6, column=0, columnspan=2, sticky="w", padx=(24, 0))

    status = tk.Text(frm, width=92, height=11, font=("Menlo", 11), state="disabled")
    status.grid(row=15, column=0, columnspan=2, pady=(12, 0))

    def refresh(*_):
        status.configure(state="normal")
        status.delete("1.0", "end")
        status.insert("end", status_text(pid.get().strip(), data_root()) if pid.get().strip() else "Enter a participant ID.")
        status.configure(state="disabled")

    def data_root():
        return storage.root_for(simulate.get(), monitor.get(), autotest.get())

    def show_text(title, text):
        win = tk.Toplevel(root)
        win.title(title)
        box = tk.Text(win, width=120, height=min(40, text.count("\n") + 2), font=("Menlo", 11))
        box.insert("end", text)
        box.configure(state="disabled")
        box.pack(fill="both", expand=True, padx=8, pady=8)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 8))

    def confirm_bias(p):
        existing = main_bias_file(p, data_root())
        if existing:
            payload = json.loads(Path(existing).read_text())
            text = bias_mod.summary_text(payload)
            return existing if messagebox.askyesno(
                "Bias correction", f"This participant's earlier main blocks used:\n\n{text}\n\n"
                                   "Use the same correction for this block?") else None
        try:
            est = bias_mod.estimate_for_participant(p, data_root())
        except FileNotFoundError as err:
            messagebox.showerror("No cross-modal data", str(err))
            return None
        ask = messagebox.askyesno if not est["warnings"] else messagebox.askokcancel
        if not ask("Confirm bias correction", bias_mod.summary_text(est) +
                   "\n\nUse these corrections for the main experiment?"):
            return None
        return str(bias_mod.save(p, est, confirmed_by="launcher", root=data_root()))

    def run(task):
        if simulate.get() and autotest.get():
            messagebox.showerror("Simulate or autotest", "Simulate runs without a window, Autotest with the real "
                                                         "window and sound card. Tick only one.")
            return
        try:
            p = design.normalize_pid(pid.get())
        except ValueError as err:
            messagebox.showerror("Participant ID", str(err))
            return
        b, bias_file = None, None
        if task == "main":
            st = storage.participant_status(p, data_root())
            b = st["next_block"] if block.get() == "auto" else block.get()
            if b is None:
                messagebox.showinfo("Main experiment", "All three blocks are done. Choose a block to run another.")
                return
            bias_file = (str(autotest_mod.bias_file(p, quick.get(), data_root())) if autotest.get()
                         else confirm_bias(p))
            if bias_file is None:
                return
        cmd = task_command(task, p, monitor.get(), practice.get(), simulate.get(), b, bias_file,
                           autotest.get(), quick.get())
        since = dt.datetime.now().isoformat(timespec="seconds")
        root.withdraw()
        try:
            code = subprocess.run(cmd, cwd=ROOT).returncode
        finally:
            root.deiconify()
            refresh()
        if code == EXIT_STOPPED:
            messagebox.showwarning("Session not completed", "The session was stopped with Escape. "
                                                            "Trials run so far are saved.")
        elif code != 0:
            messagebox.showwarning("Session not completed", "The session stopped with an error; see the terminal. "
                                                            "Trials run so far are saved.")
        if autotest.get():
            show_text("Autotest report", autotest_mod.report(p, since=since))

    def run_combined():
        if simulate.get() and autotest.get():
            messagebox.showerror("Simulate or autotest", "Tick only one of Simulate and Autotest.")
            return
        try:
            p = design.normalize_pid(pid.get())
        except ValueError as err:
            messagebox.showerror("Participant ID", str(err))
            return
        from avdur.run_blocks import blocks_to_run
        todo = blocks_to_run(p, data_root(), rerun_all=autotest.get())
        if not todo:
            messagebox.showinfo("Combined session", "Blocks 1-3 are all done for this participant.")
            return
        names = ", ".join(S.TASKS[t]["label"].split(" (")[0] for t in todo)
        if not messagebox.askokcancel("Combined session", f"Run {names} in one session"
                                      + (" (each with a practice first)" if practice.get() else "") +
                                      "?\n\nFits and checks appear in the terminal after each block."):
            return
        root.withdraw()
        try:
            code = subprocess.run(combined_command(p, monitor.get(), practice.get(), simulate.get(),
                                                   autotest.get(), quick.get()), cwd=ROOT).returncode
        finally:
            root.deiconify()
            refresh()
        if code == EXIT_STOPPED:
            messagebox.showwarning("Session not completed", "The session was stopped (Escape or the experimenter). "
                                                            "Completed blocks are saved; press the button again to "
                                                            "run the remaining blocks.")
        elif code != 0:
            messagebox.showwarning("Session not completed", "The session stopped with an error; see the terminal.")

    def run_autotest_all():
        try:
            p = design.normalize_pid(pid.get().strip() or autotest_mod.DEFAULT_PID)
        except ValueError as err:
            messagebox.showerror("Participant ID", str(err))
            return
        length = (f"the first {S.AUTOTEST_QUICK_TRIALS} trials of every task (about 7 minutes)" if quick.get()
                  else "every task in full, all three main blocks (about 2 hours)")
        if not messagebox.askokcancel("Autotest the whole experiment",
                                      f"Run {length} with automatic answers as {p} on the '{monitor.get()}' "
                                      "profile?\n\nData go to data_test/. Press Escape during a trial to stop."):
            return
        cmd = [sys.executable, "-m", "avdur.autotest", "--pid", p, "--monitor", monitor.get()]
        if quick.get():
            cmd.append("--quick")
        t0 = time.time()
        root.withdraw()
        try:
            subprocess.run(cmd, cwd=ROOT)
        finally:
            root.deiconify()
            pid.set(p)
            autotest.set(True)
            refresh()
        reports = [r for r in sorted((storage.TEST_ROOT / p).glob(f"{p}_autotest_report_*.txt"))
                   if r.stat().st_mtime >= t0]
        if reports:
            show_text(f"Autotest report: {reports[-1].name}", reports[-1].read_text())
        else:
            messagebox.showwarning("Autotest", "The autotest ended without a report; see the terminal.")

    def show_bias():
        try:
            p = design.normalize_pid(pid.get())
            est = bias_mod.estimate_for_participant(p, data_root())
            messagebox.showinfo("Cross-modal bias (not saved)", bias_mod.summary_text(est))
        except (FileNotFoundError, ValueError) as err:
            messagebox.showerror("Cross-modal bias", str(err))

    for i, task in enumerate(storage.TASK_ORDER):
        ttk.Button(frm, text=f"{i + 1}.  {S.TASKS[task]['label']}", width=48,
                   command=lambda t=task: run(t)).grid(row=7 + i, column=0, columnspan=2, pady=3)
    ttk.Button(frm, text="1-3 in one session (fits after each block)", width=48, command=run_combined).grid(
        row=11, column=0, columnspan=2, pady=(10, 3))
    ttk.Button(frm, text="Autotest the whole experiment", width=48, command=run_autotest_all).grid(
        row=12, column=0, columnspan=2, pady=3)
    ttk.Button(frm, text="Preview cross-modal bias fit", command=show_bias).grid(row=13, column=0, pady=(8, 0), sticky="w")
    ttk.Button(frm, text="Quit", command=root.destroy).grid(row=13, column=1, pady=(8, 0), sticky="e")
    entry.bind("<KeyRelease>", refresh)
    refresh()
    entry.focus()
    root.mainloop()


# ── text menu ─────────────────────────────────────────────────────────────────

def run_cli():
    p = design.normalize_pid(input("Participant ID (e.g. P14): "))
    mon = input(f"Monitor profile {list(S.MONITOR_PROFILES)} [{S.DEFAULT_MONITOR}]: ").strip() or S.DEFAULT_MONITOR
    sim = input("Simulate (no window, saves to data_simulated/)? [y/N] ").strip().lower() == "y"
    root = storage.root_for(sim, mon)
    while True:
        print("\n" + status_text(p, root))
        choice = input("\nRun 1-4 (add 'p' for practice, e.g. '4p'), 'c' blocks 1-3 in one session ('cp' with "
                       "practice), 'a' autotest everything ('aq' quick), 'b' bias preview, 'q' quit: ").strip()
        if choice == "q":
            return
        if choice in ("c", "cp"):
            subprocess.run(combined_command(p, mon, choice == "cp", sim), cwd=ROOT)
            continue
        if choice in ("a", "aq"):
            text, out = autotest_mod.run(p, mon, quick=(choice == "aq"))
            print("\n" + text + f"\n\nSaved to {out}")
            continue
        if choice == "b":
            print(bias_mod.summary_text(bias_mod.estimate_for_participant(p, root)))
            continue
        if not choice or choice[0] not in "1234":
            continue
        task = storage.TASK_ORDER[int(choice[0]) - 1]
        b, bias_file = None, None
        if task == "main":
            nxt = storage.participant_status(p, root)["next_block"]
            b = input(f"Block [{nxt}]: ").strip() or nxt
            bias_file = main_bias_file(p, root)
            if bias_file is None:
                est = bias_mod.estimate_for_participant(p, root)
                print(bias_mod.summary_text(est))
                if input("Use these corrections? [y/N] ").strip().lower() != "y":
                    continue
                bias_file = str(bias_mod.save(p, est, confirmed_by="cli", root=root))
        subprocess.run(task_command(task, p, mon, "p" in choice, sim, b, bias_file), cwd=ROOT)


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        try:
            run_window()
        except ImportError:
            print("Tk is not available; using the text menu.")
            run_cli()
