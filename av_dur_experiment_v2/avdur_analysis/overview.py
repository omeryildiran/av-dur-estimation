"""All-in-one overview sheets: the saved figure PNGs tiled onto one image per topic, each tile labelled with
its file name, plus every sheet as a page of one PDF. For a quick look, not for a paper (for that, draw
the panels together in matplotlib).

    python -m avdur_analysis.overview              # every sheet -> results/figures/overview/
    python -m avdur_analysis.overview raw models    # only these

Sheets read whatever is in results/figures/ now, so run the notebooks (or the figure functions) first.
SHEETS lists the files of each sheet as glob patterns, in order; a pattern with no match is skipped and
reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import data as D

FIG_ROOT = D.RESULTS / "figures"
OUT_DIR = FIG_ROOT / "overview"
SHEETS = {
    "raw": ("Raw data: unimodal, bimodal, conflict effect",
            ["raw/fig1_unimodal_all.png", "raw/fig1_unimodal_group_*.png", "raw/fig2_bimodal_all.png",
             "raw/fig2_bimodal_group.png", "raw/fig3_conflict_all_standard_pf.png",
             "raw/fig3_conflict_group_standard_pf.png"]),
    "order_sessions": ("Presentation order and sessions",
                       ["raw/fig5_order_unimodal_group.png", "raw/fig5_order_unimodal_all.png",
                        "raw/fig5_order_bimodal_group.png", "raw/fig5_order_bimodal_all.png",
                        "raw/fig6_sessions_group.png", "raw/fig6_sessions_all.png"]),
    "staircases": ("Staircases", ["raw/fig4_staircases_*_main.png", "raw/fig4_staircases_*_unimodal.png"]),
    "models": ("Models (no order): comparison and predictions",
               ["models/fig1_optimal_integration_*.png", "models/fig2_aic_heatmap.png",
                "models/fig3_delta_aic_bars.png", "models/fig4_group_delta_aic.png",
                "models/fig5_pse_models_group.png", "models/fig5_pse_models_all.png",
                "models/fig6_sigma_models_group.png", "models/fig6_sigma_models_all.png",
                "models/fig7_posterior_common_group.png", "models/fig7_posterior_common_all.png"]),
    "order_models": ("Order-aware models (memory contraction)",
                     ["order_models/fig_fit_summary.png", "order_models/fig_aic_heatmap.png",
                      "order_models/fig_predictions_shift_ms_group.png",
                      "order_models/fig_predictions_shift_ms_all.png", "order_models/fig_predictions_sigma_all.png"]),
    "recovery": ("Model and parameter recovery",
                 ["recovery/fig_model_recovery_aic.png", "recovery/fig_model_recovery_bic.png",
                  "recovery/fig_parameter_recovery.png"]),
}
TILE_W, COLS, PAD, LABEL_H, TITLE_H = 900, 2, 16, 30, 70
WIDE = 2.6          # width / height above which a figure spans the whole sheet


def _font(size):
    from PIL import ImageFont
    import matplotlib
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def files_for(name):
    """The PNGs of one sheet in order, and the patterns that matched nothing."""
    files, missing = [], []
    for pattern in SHEETS[name][1]:
        hits = sorted(FIG_ROOT.glob(pattern))
        files += [h for h in hits if h not in files]
        if not hits:
            missing.append(pattern)
    return files, missing


def contact_sheet(files, out, title="", tile_w=TILE_W, cols=COLS):
    """Tile PNGs (resized to tile_w wide) in rows of `cols`, each with its file name above it; a figure wider
    than WIDE x its height gets a full-width row, and so does a landscape figure left alone in a row. Returns the PIL image and saves it to `out` (None: do not
    save)."""
    from PIL import Image, ImageDraw
    tiles = []
    for f in files:
        im = Image.open(f).convert("RGB")
        tiles.append((Path(f), im.resize((tile_w, max(1, round(im.height * tile_w / im.width))), Image.LANCZOS)))
    full_w = cols * tile_w + (cols - 1) * PAD
    rows, row = [], []
    for f, im in tiles:                              # a wide figure (strip of panels) gets a row of its own
        if Image.open(f).width / Image.open(f).height > WIDE:
            if row:
                rows.append(row)
                row = []
            big = Image.open(f).convert("RGB")
            rows.append([(f, big.resize((full_w, max(1, round(big.height * full_w / big.width))), Image.LANCZOS))])
            continue
        row.append((f, im))
        if len(row) == cols:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    for i, r in enumerate(rows):                     # a lone landscape figure in a row: use the whole width
        if len(r) == 1 and r[0][1].width < full_w and r[0][1].width >= r[0][1].height:
            big = Image.open(r[0][0]).convert("RGB")
            rows[i] = [(r[0][0], big.resize((full_w, max(1, round(big.height * full_w / big.width))), Image.LANCZOS))]
    heights = [LABEL_H + max(im.height for _, im in r) + PAD for r in rows]
    width = cols * tile_w + (cols + 1) * PAD
    sheet = Image.new("RGB", (width, (TITLE_H if title else PAD) + sum(heights)), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((PAD, 18), title, fill="black", font=_font(34))
    y = TITLE_H if title else PAD
    for r, h in zip(rows, heights):
        for c, (f, im) in enumerate(r):
            x = PAD + c * (tile_w + PAD)
            draw.text((x, y + 4), f"{f.parent.name}/{f.name}", fill=(90, 90, 90), font=_font(18))
            sheet.paste(im, (x, y + LABEL_H))
        y += h
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)
    return sheet


def make(names=None, pdf=True, verbose=True):
    """Write results/figures/overview/<sheet>.png for every sheet (or `names`) and overview.pdf with one
    page per sheet. Returns {sheet: path}."""
    out = {}
    pages = []
    for name in names or SHEETS:
        files, missing = files_for(name)
        if verbose and missing:
            print(f"[overview] {name}: no file for {', '.join(missing)}")
        if not files:
            continue
        path = OUT_DIR / f"{name}.png"
        pages.append(contact_sheet(files, path, SHEETS[name][0]))
        out[name] = path
        if verbose:
            print(f"[overview] {path.relative_to(D.RESULTS.parent)} ({len(files)} figures)")
    if pdf and pages and names is None:
        pages[0].save(OUT_DIR / "overview.pdf", save_all=True, append_images=pages[1:], resolution=150)
        out["pdf"] = OUT_DIR / "overview.pdf"
    return out


if __name__ == "__main__":
    make(sys.argv[1:] or None)
