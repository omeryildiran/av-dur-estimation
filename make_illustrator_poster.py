from __future__ import annotations

from html import escape
from pathlib import Path
import shutil
import textwrap


W = 90.0
H = 42.0
MIN_FS = 40 / 72  # 40 pt in inches.

SVG_OUT = Path("poster_illustrator.svg")
TEX_OUT = Path("poster_illustrator_pdf.tex")
PDF_OUT = Path("poster_illustrator.pdf")


C = {
    "black": "#050505",
    "ink": "#111111",
    "ink2": "#242424",
    "muted": "#4a4a4a",
    "rule": "#8a8a8a",
    "soft": "#f1f1f1",
    "tint": "#fafafa",
    "white": "#ffffff",
}


def fs(size: float) -> float:
    return max(size, MIN_FS)


def clean(text: str) -> str:
    return " ".join(text.split())


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(clean(text), width=width)


def svg_rect(x: float, y: float, w: float, h: float, fill: str = C["white"],
             stroke: str | None = None, sw: float = 0.02, rx: float = 0.0) -> str:
    stroke_part = "" if stroke is None else f' stroke="{stroke}" stroke-width="{sw:.3f}"'
    rx_part = "" if rx == 0 else f' rx="{rx:.3f}"'
    return f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{fill}"{stroke_part}{rx_part}/>'


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str = C["black"], sw: float = 0.04) -> str:
    return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{sw:.3f}"/>'


def svg_text(x: float, y: float, content: str, size: float = MIN_FS, color: str = C["ink"],
             weight: int | str = 400, anchor: str = "start", style: str = "") -> str:
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" font-size="{fs(size):.3f}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}" {style}>{escape(content)}</text>'
    )


def svg_block(x: float, y: float, width_chars: int, text: str, size: float = MIN_FS,
              color: str = C["ink2"], weight: int | str = 400, leading: float = 1.23,
              bullet: bool = False) -> tuple[list[str], float]:
    out: list[str] = []
    cursor = y
    for line in wrap(text, width_chars):
        if bullet:
            out.append(svg_text(x, cursor, "•", size, color, weight))
            out.append(svg_text(x + 0.62, cursor, line, size, color, weight))
        else:
            out.append(svg_text(x, cursor, line, size, color, weight))
        cursor += fs(size) * leading
    return out, cursor


def svg_header(x: float, y: float, w: float, num: str, title: str, sub: str | None = None) -> list[str]:
    out = [
        svg_rect(x, y - 0.72, 1.45, 0.92, C["black"], rx=0.06),
        svg_text(x + 0.725, y - 0.08, num, 0.58, C["white"], 800, "middle"),
        svg_text(x + 1.80, y - 0.05, title, 0.86, C["black"], 800),
        svg_line(x, y + 0.32, x + w, y + 0.32, C["black"], 0.055),
    ]
    if sub:
        out.append(svg_text(x + w, y - 0.10, sub, 0.56, C["muted"], 400, "end", 'font-style="italic"'))
    return out


def svg_figure(x: float, y: float, w: float, h: float, href: str, label: str | None = None) -> list[str]:
    out = [
        svg_rect(x, y, w, h, C["white"], C["rule"], 0.025, 0.05),
        (
            f'<image href="{escape(href, quote=True)}" x="{x + 0.10:.3f}" y="{y + 0.10:.3f}" '
            f'width="{w - 0.20:.3f}" height="{h - 0.20:.3f}" preserveAspectRatio="xMidYMid meet"/>'
        ),
    ]
    if label:
        label_w = min(w - 0.40, 0.80 + len(label) * 0.28)
        out.extend([
            svg_rect(x + 0.25, y + 0.25, label_w, 0.85, C["black"], rx=0.04),
            svg_text(x + 0.48, y + 0.86, label, 0.56, C["white"], 800),
        ])
    return out


def svg_kpi(x: float, y: float, w: float, value: str, label: str) -> list[str]:
    return [
        svg_rect(x, y, w, 1.75, C["white"], C["rule"], 0.02, 0.04),
        svg_line(x, y, x + w, y, C["black"], 0.08),
        svg_text(x + w / 2, y + 0.74, value, 0.74, C["black"], 800, "middle"),
        svg_text(x + w / 2, y + 1.38, label, 0.56, C["muted"], 500, "middle"),
    ]


def svg_stat(x: float, y: float, w: float, label: str, value: str, detail: str) -> list[str]:
    return [
        svg_rect(x, y, w, 2.05, C["white"], C["rule"], 0.02, 0.04),
        svg_line(x, y, x, y + 2.05, C["black"], 0.10),
        svg_text(x + 0.42, y + 0.62, label.upper(), 0.56, C["muted"], 800),
        svg_text(x + 0.42, y + 1.32, value, 0.72, C["black"], 800),
        svg_text(x + 0.42, y + 1.78, detail, 0.56, C["muted"], 500),
    ]


def svg_model_card(x: float, y: float, w: float, tag: str, title: str, desc: str) -> list[str]:
    out = [
        svg_rect(x, y, w, 2.10, C["white"], C["rule"], 0.02, 0.04),
        svg_line(x, y, x, y + 2.10, C["black"], 0.10),
        svg_rect(x + 0.35, y + 0.28, 1.18, 0.72, C["black"], rx=0.04),
        svg_text(x + 0.94, y + 0.80, tag, 0.56, C["white"], 800, "middle"),
        svg_text(x + 1.85, y + 0.82, title, 0.62, C["black"], 800),
    ]
    body, _ = svg_block(x + 0.35, y + 1.45, 88, desc, 0.56, C["ink2"], 500, 1.08)
    out.extend(body[:1])
    return out


def tex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
        .replace("·", r"\textperiodcentered{}")
        .replace("−", "-")
        .replace("–", "--")
        .replace("—", "---")
    )


def color_name(color: str) -> str:
    return "poster" + color.lstrip("#").lower()


def tex_rect(x: float, y: float, w: float, h: float, fill: str = C["white"],
             draw: str | None = None, lw: float = 0.02, rounded: bool = False) -> str:
    opts = [f"fill={color_name(fill)}"]
    if draw:
        opts.extend([f"draw={color_name(draw)}", f"line width={lw:.3f}in"])
    else:
        opts.append("draw=none")
    if rounded:
        opts.append("rounded corners=0.045in")
    return rf"\path[{', '.join(opts)}] ({x:.3f},{H-y:.3f}) rectangle ({x+w:.3f},{H-y-h:.3f});"


def tex_line(x1: float, y1: float, x2: float, y2: float, color: str = C["black"], lw: float = 0.04) -> str:
    return rf"\draw[{color_name(color)}, line width={lw:.3f}in] ({x1:.3f},{H-y1:.3f}) -- ({x2:.3f},{H-y2:.3f});"


def tex_text(x: float, y: float, w: float, content: str, size: float = MIN_FS,
             color: str = C["ink"], weight: str = "", align: str = "left") -> str:
    size = fs(size)
    font = rf"\fontsize{{{size:.3f}in}}{{{size * 1.20:.3f}in}}\selectfont"
    if weight == "bold":
        font = r"\bfseries " + font
    align_cmd = {"left": r"\raggedright", "center": r"\centering", "right": r"\raggedleft"}[align]
    return (
        rf"\node[anchor=north west, text width={w:.3f}in, align={align}, inner sep=0pt] "
        rf"at ({x:.3f},{H-y:.3f}) "
        rf"{{\color{{{color_name(color)}}}{font} {align_cmd} "
        rf"\spaceskip=0.32em plus 0.10em minus 0.06em {tex_escape(content)}\par}};"
    )


def tex_header(x: float, y: float, w: float, num: str, title: str, sub: str | None = None) -> list[str]:
    out = [
        tex_rect(x, y - 0.72, 1.45, 0.92, C["black"], rounded=True),
        tex_text(x + 0.10, y - 0.55, 1.25, num, 0.58, C["white"], "bold", "center"),
        tex_text(x + 1.80, y - 0.62, w - 1.80, title, 0.86, C["black"], "bold"),
        tex_line(x, y + 0.32, x + w, y + 0.32, C["black"], 0.055),
    ]
    if sub:
        out.append(tex_text(x + w - 8.2, y - 0.48, 8.2, sub, 0.56, C["muted"], "", "right"))
    return out


def tex_fig(x: float, y: float, w: float, h: float, href: str, label: str | None = None) -> list[str]:
    out = [
        tex_rect(x, y, w, h, C["white"], C["rule"], 0.020, True),
        rf"\node[anchor=north west, inner sep=0pt] at ({x+0.12:.3f},{H-y-0.12:.3f}) "
        rf"{{\includegraphics[width={w-0.24:.3f}in,height={h-0.24:.3f}in,keepaspectratio]{{{href}}}}};",
    ]
    if label:
        label_w = min(w - 0.40, 0.80 + len(label) * 0.28)
        out.extend([
            tex_rect(x + 0.25, y + 0.25, label_w, 0.85, C["black"], rounded=True),
            tex_text(x + 0.48, y + 0.38, label_w - 0.40, label, 0.56, C["white"], "bold"),
        ])
    return out


def tex_kpi(x: float, y: float, w: float, value: str, label: str) -> list[str]:
    return [
        tex_rect(x, y, w, 1.75, C["white"], C["rule"], 0.020, True),
        tex_line(x, y, x + w, y, C["black"], 0.08),
        tex_text(x, y + 0.23, w, value, 0.74, C["black"], "bold", "center"),
        tex_text(x + 0.15, y + 1.12, w - 0.30, label, 0.56, C["muted"], "", "center"),
    ]


def tex_stat(x: float, y: float, w: float, label: str, value: str, detail: str) -> list[str]:
    return [
        tex_rect(x, y, w, 2.05, C["white"], C["rule"], 0.020, True),
        tex_line(x, y, x, y + 2.05, C["black"], 0.10),
        tex_text(x + 0.42, y + 0.28, w - 0.70, label.upper(), 0.56, C["muted"], "bold"),
        tex_text(x + 0.42, y + 0.93, w - 0.70, value, 0.72, C["black"], "bold"),
        tex_text(x + 0.42, y + 1.58, w - 0.70, detail, 0.56, C["muted"]),
    ]


def tex_model_card(x: float, y: float, w: float, tag: str, title: str, desc: str) -> list[str]:
    return [
        tex_rect(x, y, w, 2.10, C["white"], C["rule"], 0.020, True),
        tex_line(x, y, x, y + 2.10, C["black"], 0.10),
        tex_rect(x + 0.35, y + 0.28, 1.18, 0.72, C["black"], rounded=True),
        tex_text(x + 0.35, y + 0.39, 1.18, tag, 0.56, C["white"], "bold", "center"),
        tex_text(x + 1.85, y + 0.32, w - 2.15, title, 0.62, C["black"], "bold"),
        tex_text(x + 0.35, y + 1.30, w - 0.70, desc, 0.56, C["ink2"]),
    ]


def make_svg() -> None:
    m = 1.20
    gap = 0.90
    col_w = (W - 2 * m - 2 * gap) / 3
    col1 = m
    col2 = m + col_w + gap
    col3 = m + 2 * (col_w + gap)
    y0 = 8.10

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:g}in" height="{H:g}in" viewBox="0 0 {W:g} {H:g}" version="1.1">',
        '<title>Duration judgments with conflicting audiovisual cues - 90 x 42 poster</title>',
        '<style><![CDATA[text{font-family:Helvetica Neue, Helvetica, Arial, sans-serif;}]]></style>',
        svg_rect(0, 0, W, H, C["white"]),
        svg_rect(1.20, 0.70, 3.45, 3.45, C["black"], rx=0.08),
        svg_text(2.925, 3.03, "NYU", 1.20, C["white"], 800, "middle"),
        svg_text(5.35, 1.35, "New York University", 0.78, C["black"], 800),
        svg_text(5.35, 2.23, "Department of Psychology", 0.56, C["muted"], 500),
        svg_text(5.35, 2.90, "Center for Neural Science", 0.56, C["muted"], 500),
        svg_text(W / 2, 1.50, "Duration judgments with conflicting audiovisual cues", 1.32, C["black"], 800, "middle"),
        svg_text(W / 2, 2.82, "Ömer F. Yildiran, Long Ni, Michael S. Landy", 0.64, C["black"], 700, "middle"),
        svg_text(W / 2, 3.58, "Department of Psychology · Center for Neural Science · New York University", 0.56, C["muted"], 500, "middle"),
        svg_rect(W - 4.40, 0.70, 3.20, 3.20, C["white"], C["black"], 0.035),
        svg_text(W - 2.80, 2.65, "QR", 0.92, C["black"], 800, "middle"),
        svg_line(1.20, 4.75, W - 1.20, 4.75, C["black"], 0.08),
        svg_rect(1.20, 5.25, W - 2.40, 1.90, C["tint"], C["rule"], 0.02, 0.05),
        svg_rect(1.65, 5.77, 2.20, 0.88, C["black"], rx=0.04),
        svg_text(2.75, 6.43, "TL;DR", 0.58, C["white"], 800, "middle"),
        svg_text(4.45, 6.36, "Auditory duration judgments are pulled linearly toward visual duration; bias scales with auditory noise but shows no non-linear breakdown.", 0.64, C["black"], 800),
        svg_text(W - 1.65, 6.33, "11 participants · 4 tasks · 5 models · 2,156 trials/subj", 0.56, C["muted"], 700, "end"),
    ]

    # Column 1
    out.extend(svg_header(col1, y0, col_w, "01", "Question"))
    body, yy = svg_block(col1, y0 + 1.10, 82, "Does causal inference break audiovisual duration integration when AV conflicts grow large, the way it does in spatial localization?", 0.60)
    out.extend(body)
    for item in [
        "Fuse, segregate, or switch between duration cues?",
        "How does auditory reliability shape visual bias?",
        "Can behavior discriminate the computational strategies?",
    ]:
        b, yy = svg_block(col1 + 0.20, yy + 0.18, 74, item, 0.56, bullet=True)
        out.extend(b)
    out.append(svg_rect(col1, yy + 0.35, col_w, 1.80, C["soft"], C["rule"], 0.02, 0.05))
    call, _ = svg_block(col1 + 0.40, yy + 1.05, 74, "We test 7 AV-duration conflicts (±250 ms) at two auditory-noise levels and compare forced fusion, causal-inference variants, and cue switching.", 0.56, C["black"], 700)
    out.extend(call)

    y = 16.15
    out.extend(svg_header(col1, y, col_w, "02", "Stimuli & Tasks", "2-IFC · 500 ms standard"))
    out.extend(svg_figure(col1, y + 0.70, col_w, 7.10, "ms_latex/assets/figures/unimodal_tasks_timeline.pdf", "A · UNIMODAL CALIBRATION"))
    cap, _ = svg_block(col1, y + 8.35, 86, "Auditory and visual duration-discrimination tasks calibrate per-modality sensory noise.", 0.56, C["ink2"])
    out.extend(cap)
    out.extend(svg_figure(col1, y + 9.65, col_w, 6.45, "ms_latex/assets/figures/bimodal_exp_scheme.png", "B · BIMODAL AV CONFLICT"))
    cap, _ = svg_block(col1, y + 16.65, 86, "Conflict is applied only to the visual standard; observers judge which interval was longer in audition.", 0.56, C["ink2"])
    out.extend(cap)
    for i, (value, label) in enumerate([("7", "conflicts"), ("±250", "ms range"), ("2", "noise levels"), ("2,156", "trials/subj")]):
        out.extend(svg_kpi(col1 + i * (col_w / 4), 36.85, col_w / 4 - 0.25, value, label))

    # Column 2
    out.extend(svg_header(col2, y0, col_w, "03", "Cue Reliability"))
    body, _ = svg_block(col2, y0 + 1.05, 84, "Auditory reliability is highest at low noise and lowest at high noise; vision sits in between.", 0.60)
    out.extend(body)
    out.extend(svg_figure(col2, y0 + 3.25, col_w, 8.40, "ms_latex/assets/figures/psychometric_curves_publication.pdf"))
    for i, (value, label) in enumerate([("0.18", "WF low-noise"), ("0.42", "WF visual"), ("2.54", "WF high-noise")]):
        out.extend(svg_kpi(col2 + i * (col_w / 3), y0 + 12.20, col_w / 3 - 0.25, value, label))

    y = 23.05
    out.extend(svg_header(col2, y, col_w, "04", "Modality Bias", "cross-modal calibration"))
    body, _ = svg_block(col2, y + 1.05, 84, "We estimate each subject's auditory-visual bias before adding conflict, then correct the visual stimulus so nominal zero conflict is perceptually matched.", 0.56)
    out.extend(body)
    out.extend(svg_figure(col2, y + 3.20, col_w, 7.80, "ms_latex/assets/figures/psychometric_curves_crossmodal.pdf"))
    out.extend(svg_stat(col2, y + 11.55, col_w / 2 - 0.25, "Low-noise PSE shift", "-116 ms", "W = 6, p = .014"))
    out.extend(svg_stat(col2 + col_w / 2 + 0.25, y + 11.55, col_w / 2 - 0.25, "High-noise PSE shift", "-73 ms", "n.s."))

    # Column 3
    out.extend(svg_header(col3, y0, col_w, "05", "Main Result · PSE Shifts"))
    body, _ = svg_block(col3, y0 + 1.05, 84, "Visual bias grows approximately linearly with conflict and is much larger under high auditory noise.", 0.60)
    out.extend(body)
    out.extend(svg_figure(col3, y0 + 3.20, col_w, 8.80, "ms_latex/assets/figures/aggregated_mu_vs_models_sem.pdf"))
    out.extend(svg_stat(col3, y0 + 12.60, col_w / 2 - 0.25, "Low-noise slope", "0.144", "r = 0.91, p = .004"))
    out.extend(svg_stat(col3 + col_w / 2 + 0.25, y0 + 12.60, col_w / 2 - 0.25, "High-noise slope", "0.606", "r = 0.96, p < .001"))

    y = 25.40
    out.extend(svg_header(col3, y, col_w, "06", "Models & Comparison", "free sensory noise"))
    out.extend(svg_figure(col3, y + 0.75, col_w, 6.15, "ms_latex/assets/figures/modelCompFree.pdf"))
    out.extend(svg_model_card(col3, y + 7.55, col_w, "M1", "Forced fusion", "Always integrate auditory and visual cues using reliability-weighted averaging."))
    out.extend(svg_model_card(col3, y + 9.95, col_w, "M2-4", "Causal inference", "Fuse or segregate according to the posterior probability of a common cause."))
    out.extend(svg_model_card(col3, y + 12.35, col_w, "M5", "Probabilistic cue switching", "Use auditory or visual cue alone on each trial with a fixed switching probability."))
    out.append(svg_rect(col3, 40.20, col_w, 1.45, C["black"], rx=0.05))
    out.append(svg_text(col3 + 0.55, 41.12, "Take-home: linear bias, flat model evidence, and an identifiability limit.", 0.58, C["white"], 800))

    out.append("</svg>")
    SVG_OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {SVG_OUT}")


def make_tex() -> None:
    m = 1.20
    gap = 0.90
    col_w = (W - 2 * m - 2 * gap) / 3
    col1 = m
    col2 = m + col_w + gap
    col3 = m + 2 * (col_w + gap)
    y0 = 8.10
    colors = sorted({v.lstrip("#").lower() for v in C.values()})
    color_defs = [rf"\definecolor{{poster{c}}}{{HTML}}{{{c}}}" for c in colors]

    t: list[str] = [
        r"\documentclass{article}",
        rf"\usepackage[paperwidth={W:g}in,paperheight={H:g}in,margin=0in]{{geometry}}",
        r"\usepackage{tikz}",
        r"\usepackage{graphicx}",
        r"\usepackage{xcolor}",
        r"\usepackage{fontspec}",
        r"\setmainfont{Helvetica}",
        r"\setsansfont{Helvetica}",
        *color_defs,
        r"\pagestyle{empty}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1in,y=1in]",
        tex_rect(0, 0, W, H, C["white"]),
        tex_rect(1.20, 0.70, 3.45, 3.45, C["black"], rounded=True),
        tex_text(1.45, 1.35, 2.95, "NYU", 1.20, C["white"], "bold", "center"),
        tex_text(5.35, 1.02, 11.0, "New York University", 0.78, C["black"], "bold"),
        tex_text(5.35, 2.05, 11.0, "Department of Psychology", 0.56, C["muted"]),
        tex_text(5.35, 2.75, 11.0, "Center for Neural Science", 0.56, C["muted"]),
        tex_text(19.0, 0.85, 52.0, "Duration judgments with conflicting audiovisual cues", 1.30, C["black"], "bold", "center"),
        tex_text(28.0, 2.48, 34.0, "Ömer F. Yildiran, Long Ni, Michael S. Landy", 0.64, C["black"], "bold", "center"),
        tex_text(23.0, 3.28, 44.0, "Department of Psychology · Center for Neural Science · New York University", 0.56, C["muted"], "", "center"),
        tex_rect(W - 4.40, 0.70, 3.20, 3.20, C["white"], C["black"], 0.035),
        tex_text(W - 4.20, 1.35, 2.80, "QR", 0.92, C["black"], "bold", "center"),
        tex_line(1.20, 4.75, W - 1.20, 4.75, C["black"], 0.08),
        tex_rect(1.20, 5.25, W - 2.40, 1.90, C["tint"], C["rule"], 0.02, True),
        tex_rect(1.65, 5.77, 2.20, 0.88, C["black"], rounded=True),
        tex_text(1.65, 5.93, 2.20, "TL;DR", 0.58, C["white"], "bold", "center"),
        tex_text(4.45, 5.72, 65.0, "Auditory duration judgments are pulled linearly toward visual duration; bias scales with auditory noise but shows no non-linear breakdown.", 0.64, C["black"], "bold"),
        tex_text(W - 20.0, 5.88, 18.35, "11 participants · 4 tasks · 5 models · 2,156 trials/subj", 0.56, C["muted"], "bold", "right"),
    ]

    # Column 1.
    t += tex_header(col1, y0, col_w, "01", "Question")
    t.append(tex_text(col1, y0 + 0.86, col_w, "Does causal inference break audiovisual duration integration when AV conflicts grow large, the way it does in spatial localization?", 0.60, C["ink2"]))
    for i, item in enumerate([
        "• Fuse, segregate, or switch between duration cues?",
        "• How does auditory reliability shape visual bias?",
        "• Can behavior discriminate the computational strategies?",
    ]):
        t.append(tex_text(col1 + 0.30, y0 + 3.05 + i * 0.86, col_w - 0.30, item, 0.56, C["ink2"]))
    t.append(tex_rect(col1, y0 + 5.90, col_w, 1.80, C["soft"], C["rule"], 0.02, True))
    t.append(tex_text(col1 + 0.40, y0 + 6.30, col_w - 0.80, "We test 7 AV-duration conflicts (±250 ms) at two auditory-noise levels and compare forced fusion, causal-inference variants, and cue switching.", 0.56, C["black"], "bold"))

    y = 16.15
    t += tex_header(col1, y, col_w, "02", "Stimuli & Tasks", "2-IFC · 500 ms standard")
    t += tex_fig(col1, y + 0.70, col_w, 7.10, "ms_latex/assets/figures/unimodal_tasks_timeline.pdf", "A · UNIMODAL CALIBRATION")
    t.append(tex_text(col1, y + 8.25, col_w, "Auditory and visual duration-discrimination tasks calibrate per-modality sensory noise.", 0.56, C["ink2"]))
    t += tex_fig(col1, y + 9.65, col_w, 6.45, "ms_latex/assets/figures/bimodal_exp_scheme.png", "B · BIMODAL AV CONFLICT")
    t.append(tex_text(col1, y + 16.55, col_w, "Conflict is applied only to the visual standard; observers judge which interval was longer in audition.", 0.56, C["ink2"]))
    for i, (value, label) in enumerate([("7", "conflicts"), ("±250", "ms range"), ("2", "noise levels"), ("2,156", "trials/subj")]):
        t += tex_kpi(col1 + i * (col_w / 4), 36.85, col_w / 4 - 0.25, value, label)

    # Column 2.
    t += tex_header(col2, y0, col_w, "03", "Cue Reliability")
    t.append(tex_text(col2, y0 + 0.86, col_w, "Auditory reliability is highest at low noise and lowest at high noise; vision sits in between.", 0.60, C["ink2"]))
    t += tex_fig(col2, y0 + 3.25, col_w, 8.40, "ms_latex/assets/figures/psychometric_curves_publication.pdf")
    for i, (value, label) in enumerate([("0.18", "WF low-noise"), ("0.42", "WF visual"), ("2.54", "WF high-noise")]):
        t += tex_kpi(col2 + i * (col_w / 3), y0 + 12.20, col_w / 3 - 0.25, value, label)

    y = 23.05
    t += tex_header(col2, y, col_w, "04", "Modality Bias", "cross-modal calibration")
    t.append(tex_text(col2, y + 0.86, col_w, "We estimate each subject's auditory-visual bias before adding conflict, then correct the visual stimulus so nominal zero conflict is perceptually matched.", 0.56, C["ink2"]))
    t += tex_fig(col2, y + 3.20, col_w, 7.80, "ms_latex/assets/figures/psychometric_curves_crossmodal.pdf")
    t += tex_stat(col2, y + 11.55, col_w / 2 - 0.25, "Low-noise PSE shift", "-116 ms", "W = 6, p = .014")
    t += tex_stat(col2 + col_w / 2 + 0.25, y + 11.55, col_w / 2 - 0.25, "High-noise PSE shift", "-73 ms", "n.s.")

    # Column 3.
    t += tex_header(col3, y0, col_w, "05", "Main Result · PSE Shifts")
    t.append(tex_text(col3, y0 + 0.86, col_w, "Visual bias grows approximately linearly with conflict and is much larger under high auditory noise.", 0.60, C["ink2"]))
    t += tex_fig(col3, y0 + 3.20, col_w, 8.80, "ms_latex/assets/figures/aggregated_mu_vs_models_sem.pdf")
    t += tex_stat(col3, y0 + 12.60, col_w / 2 - 0.25, "Low-noise slope", "0.144", "r = 0.91, p = .004")
    t += tex_stat(col3 + col_w / 2 + 0.25, y0 + 12.60, col_w / 2 - 0.25, "High-noise slope", "0.606", "r = 0.96, p < .001")

    y = 25.40
    t += tex_header(col3, y, col_w, "06", "Models & Comparison", "free sensory noise")
    t += tex_fig(col3, y + 0.75, col_w, 6.15, "ms_latex/assets/figures/modelCompFree.pdf")
    t += tex_model_card(col3, y + 7.55, col_w, "M1", "Forced fusion", "Always integrate auditory and visual cues using reliability-weighted averaging.")
    t += tex_model_card(col3, y + 9.95, col_w, "M2-4", "Causal inference", "Fuse or segregate according to the posterior probability of a common cause.")
    t += tex_model_card(col3, y + 12.35, col_w, "M5", "Probabilistic cue switching", "Use auditory or visual cue alone on each trial with a fixed switching probability.")
    t.append(tex_rect(col3, 40.20, col_w, 1.45, C["black"], rounded=True))
    t.append(tex_text(col3 + 0.55, 40.62, col_w - 1.10, "Take-home: linear bias, flat model evidence, and an identifiability limit.", 0.58, C["white"], "bold"))

    t.extend([r"\end{tikzpicture}", r"\end{document}"])
    TEX_OUT.write_text("\n".join(t), encoding="utf-8")
    print(f"Wrote {TEX_OUT}")


def main() -> None:
    make_svg()
    make_tex()


if __name__ == "__main__":
    main()
