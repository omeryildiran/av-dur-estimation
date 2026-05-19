#!/usr/bin/env python3
"""Render a black-theme audiovisual duration waveform asset.

The output is intentionally caption-light so it can be used as a background or
intro section in a video abstract.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
if Path("/opt/homebrew/Cellar/x265/4.1/lib/libx265.215.dylib").exists():
    os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/Cellar/x265/4.1/lib")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle


BG = "#02030a"
GRID = "#182033"
WHITE = "#f5f7ff"
MUTED = "#9aa6c3"
AUDIO = "#00d4ff"
VISUAL = "#ff4fd8"
FUSED = "#f8d66d"


def smoothstep(x: np.ndarray | float, edge0: float, edge1: float) -> np.ndarray | float:
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def pulse_envelope(t: np.ndarray, start: float, dur: float, attack: float = 0.07, release: float = 0.11) -> np.ndarray:
    on = smoothstep(t, start, start + attack)
    off = 1.0 - smoothstep(t, start + dur - release, start + dur)
    return on * off


def make_waveform(t: np.ndarray, center_time: float, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freqs = np.array([7.5, 13.0, 19.5, 31.0, 43.0])
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    amps = np.array([0.55, 0.35, 0.23, 0.15, 0.10])
    wave = sum(a * np.sin(2 * np.pi * f * (t + 0.035 * np.sin(center_time * 0.7)) + p) for a, f, p in zip(amps, freqs, phases))
    shimmer = 0.12 * np.sin(2 * np.pi * (75 * t + 0.12 * np.sin(center_time)))
    return wave + shimmer


def build_animation(output: Path, seconds: float, fps: int, width: int, height: int) -> None:
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Subtle scanning grid.
    for y in np.linspace(0.18, 0.82, 6):
        ax.plot([0.08, 0.92], [y, y], color=GRID, lw=0.8, alpha=0.35)
    for x in np.linspace(0.10, 0.90, 9):
        ax.plot([x, x], [0.18, 0.82], color=GRID, lw=0.6, alpha=0.22)

    title = ax.text(
        0.075,
        0.89,
        "audiovisual duration conflict",
        color=WHITE,
        fontsize=30,
        weight="semibold",
        ha="left",
        va="center",
    )
    subtitle = ax.text(
        0.075,
        0.85,
        "sound and vision carry different temporal evidence",
        color=MUTED,
        fontsize=15,
        ha="left",
        va="center",
    )

    ax.text(0.075, 0.68, "auditory", color=AUDIO, fontsize=15, weight="semibold", ha="left", va="center")
    ax.text(0.075, 0.42, "visual", color=VISUAL, fontsize=15, weight="semibold", ha="left", va="center")
    ax.text(0.075, 0.24, "perceived", color=FUSED, fontsize=15, weight="semibold", ha="left", va="center")

    # Artists updated frame-by-frame.
    x = np.linspace(0.12, 0.90, 900)
    local_t = np.linspace(0, 1, x.size)

    glow_lines = []
    for lw, alpha in [(10, 0.045), (6, 0.08), (3, 0.16)]:
        line, = ax.plot(x, np.full_like(x, 0.68), color=AUDIO, lw=lw, alpha=alpha, solid_capstyle="round")
        glow_lines.append(line)
    audio_line, = ax.plot(x, np.full_like(x, 0.68), color=AUDIO, lw=1.45, alpha=0.95, solid_capstyle="round")

    fused_line, = ax.plot(x, np.full_like(x, 0.24), color=FUSED, lw=2.0, alpha=0.9, solid_capstyle="round")

    visual_base = Rectangle((0.12, 0.385), 0.78, 0.07, facecolor="#17172a", edgecolor="#2a2f45", lw=1.2, alpha=0.85)
    visual_pulse = Rectangle((0.12, 0.385), 0.18, 0.07, facecolor=VISUAL, edgecolor="none", alpha=0.65)
    visual_glow = Rectangle((0.12, 0.367), 0.18, 0.106, facecolor=VISUAL, edgecolor="none", alpha=0.10)
    ax.add_patch(visual_base)
    ax.add_patch(visual_glow)
    ax.add_patch(visual_pulse)

    audio_marker = ax.plot([], [], marker="o", ms=8, color=AUDIO, alpha=0.92)[0]
    visual_marker = ax.plot([], [], marker="o", ms=8, color=VISUAL, alpha=0.92)[0]
    fused_marker = ax.plot([], [], marker="o", ms=8, color=FUSED, alpha=0.92)[0]

    particles = [
        Circle((0.5, 0.5), 0.003, facecolor=WHITE, edgecolor="none", alpha=0.0)
        for _ in range(46)
    ]
    for particle in particles:
        ax.add_patch(particle)

    conflict_text = ax.text(0.90, 0.50, "", color=WHITE, fontsize=18, ha="right", va="center")
    bias_text = ax.text(0.90, 0.205, "", color=MUTED, fontsize=13, ha="right", va="center")

    # Four conflict examples, looped as a continuous design element.
    conflicts_ms = np.array([-250, -83, 83, 250])
    starts = np.array([0.10, 0.16, 0.22, 0.28])
    audio_dur = 0.42
    visual_durs = audio_dur + conflicts_ms / 1000 * 0.42
    visual_durs = np.clip(visual_durs, 0.24, 0.60)

    rng = np.random.default_rng(4)
    particle_seed = rng.uniform(size=(len(particles), 4))
    total_frames = int(seconds * fps)

    def frame(i: int):
        t_abs = i / fps
        cycle = (t_abs * 0.37) % len(conflicts_ms)
        idx = int(cycle)
        frac = cycle - idx
        next_idx = (idx + 1) % len(conflicts_ms)
        eased = smoothstep(frac, 0.0, 1.0)

        conflict = (1 - eased) * conflicts_ms[idx] + eased * conflicts_ms[next_idx]
        vdur = (1 - eased) * visual_durs[idx] + eased * visual_durs[next_idx]
        start = (1 - eased) * starts[idx] + eased * starts[next_idx]

        env = pulse_envelope(local_t, 0.10, audio_dur)
        carrier = make_waveform(local_t, t_abs)
        tremolo = 0.65 + 0.35 * np.sin(2 * np.pi * (0.28 * t_abs + local_t * 1.4)) ** 2
        y_audio = 0.68 + 0.095 * env * carrier * tremolo
        y_fused = 0.24 + 0.035 * np.sin(2 * np.pi * (local_t * 2.0 - t_abs * 0.33))
        y_fused += 0.085 * pulse_envelope(local_t, 0.10 + 0.18 * smoothstep(abs(conflict), 0, 250), audio_dur + 0.10 * conflict / 250) * np.sin(
            2 * np.pi * (local_t * 4.5 + t_abs * 0.45)
        )

        for k, line in enumerate(glow_lines):
            line.set_ydata(y_audio)
            line.set_alpha([0.045, 0.08, 0.16][k] * (0.75 + 0.25 * np.sin(t_abs * 1.7) ** 2))
        audio_line.set_ydata(y_audio)
        fused_line.set_ydata(y_fused)

        visual_pulse.set_x(0.12 + start * 0.78)
        visual_pulse.set_width(vdur * 0.78)
        visual_glow.set_x(0.12 + start * 0.78 - 0.012)
        visual_glow.set_width(vdur * 0.78 + 0.024)

        marker_x = 0.12 + (0.10 + audio_dur) * 0.78
        v_marker_x = 0.12 + (start + vdur) * 0.78
        f_marker_x = marker_x + (v_marker_x - marker_x) * (0.35 + 0.22 * smoothstep(abs(conflict), 0, 250))
        audio_marker.set_data([marker_x], [0.68])
        visual_marker.set_data([v_marker_x], [0.42])
        fused_marker.set_data([f_marker_x], [0.24])

        for p_idx, particle in enumerate(particles):
            px_seed, py_seed, speed_seed, phase_seed = particle_seed[p_idx]
            px = 0.08 + ((px_seed + 0.035 * t_abs * (0.4 + speed_seed)) % 0.84)
            py = 0.18 + 0.64 * py_seed + 0.015 * np.sin(t_abs * (0.9 + speed_seed) + phase_seed * 6.28)
            near_wave = np.exp(-((py - 0.68) ** 2) / 0.004) + 0.5 * np.exp(-((py - 0.42) ** 2) / 0.006)
            particle.center = (px, py)
            particle.set_alpha(0.05 + 0.18 * near_wave * (0.5 + 0.5 * np.sin(t_abs * 2.2 + phase_seed * 6.28)))

        conflict_text.set_text(f"visual conflict {conflict:+.0f} ms")
        bias_text.set_text("larger visual pull when auditory timing is noisy")

        fade_in = smoothstep(t_abs, 0.0, 1.2)
        fade_out = 1.0 - smoothstep(t_abs, seconds - 1.0, seconds)
        alpha = fade_in * fade_out
        title.set_alpha(alpha)
        subtitle.set_alpha(0.85 * alpha)

        return (
            *glow_lines,
            audio_line,
            fused_line,
            visual_pulse,
            visual_glow,
            audio_marker,
            visual_marker,
            fused_marker,
            *particles,
            conflict_text,
            bias_text,
            title,
            subtitle,
        )

    anim = FuncAnimation(fig, frame, frames=total_frames, interval=1000 / fps, blit=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=9000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    anim.save(str(output), writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="media/video_abstract/waveform_black_theme.mp4")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    build_animation(Path(args.output), args.seconds, args.fps, args.width, args.height)
    print(args.output)


if __name__ == "__main__":
    main()
