"""
Shared matplotlib / seaborn style configuration.

Usage at the top of any analysis notebook:
    from plot_style import *          # imports constants + sets rcParams
    # or, to also apply seaborn theme:
    from plot_style import setup_style; setup_style()

Naming conventions
------------------
All module-level configuration values follow the Python convention of
ALL_CAPS names (e.g. FONT_SIZE_LABEL, PANEL_LABELS).  This signals that
they are intended as read-only constants — notebooks should read them
freely but should not reassign them.

Panel-label capitalization
--------------------------
Panel labels (A, B, C …) are uppercase by convention in scientific figures.
``PANEL_LABELS`` therefore contains the 26 uppercase ASCII letters.
If you need lowercase labels for a specific figure, pass an explicit list::

    add_panel_labels(axes, labels=list("abcdefg"))

Labels are also rendered bold (``fontweight='bold'``) to visually
distinguish them from axis tick labels and legend text.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ── Font size constants ────────────────────────────────────────────────────────
FONT_SIZE_LABEL  = 18   # axis labels, tick labels, base font
FONT_SIZE_TITLE  = 20   # subplot titles, panel letters
FONT_SIZE_LEGEND = 18   # legend text
FONT_SIZE_SUPER  = 22   # figure suptitle  (FONT_SIZE_TITLE + 2)

# ── Figure size helpers ────────────────────────────────────────────────────────
# Pass (n_cols, n_rows) to get a figsize tuple
def figsize_scatter(n_cols=1, n_rows=1): return (6.0 * n_cols, 6.0 * n_rows)
def figsize_panel  (n_cols=1, n_rows=1): return (5.0 * n_cols, 4.0 * n_rows)
def figsize_matrix (n_cols=1, n_rows=1): return (4.0 * n_cols, 4.0 * n_rows)

# ── Panel label helper ─────────────────────────────────────────────────────────
PANEL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # uppercase; pass a custom list to add_panel_labels() for lowercase

def add_panel_labels(axes, labels=None, x=-0.15, y=1.06, fontsize=None, **kwargs):
    """
    Add panel labels (A, B, C …) to a list / array of axes.

    Labels are uppercase by default — this matches the scientific-figure
    convention used in journals such as Nature and Cell.  Pass a custom
    ``labels`` list to use lowercase letters or arbitrary strings.

    Parameters
    ----------
    axes    : iterable of Axes
    labels  : list of strings, defaults to PANEL_LABELS (['A', 'B', 'C', …]).
              Supply e.g. ``list("abcde")`` for lowercase labels.
    x, y    : position in axes-fraction coordinates.  x < 0 places the label
              to the left of the y-axis spine; y > 1 places it above the top
              spine.
    fontsize: int, defaults to FONT_SIZE_TITLE (20 pt).
    kwargs  : forwarded to ax.text().  Note that ``fontweight`` is already set
              to ``'bold'`` internally; pass ``fontweight='normal'`` to override.
    """
    if fontsize is None:
        fontsize = FONT_SIZE_TITLE
    if labels is None:
        labels = PANEL_LABELS
    for ax, label in zip(axes, labels):
        ax.text(x, y, label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                va='top',
                **kwargs)

# ── rcParams ───────────────────────────────────────────────────────────────────
_RC = {
    # Font
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size'         : FONT_SIZE_LABEL,
    'font.weight'       : 'normal',

    # Axes labels & titles
    'axes.labelsize'    : FONT_SIZE_LABEL,
    'axes.labelweight'  : 'normal',
    'axes.titlesize'    : FONT_SIZE_TITLE,
    'axes.titleweight'  : 'normal',
    'axes.linewidth'    : 1.5,

    # Tick labels
    'xtick.labelsize'   : FONT_SIZE_LABEL,
    'ytick.labelsize'   : FONT_SIZE_LABEL,

    # Tick marks
    'xtick.major.width' : 1.5,
    'ytick.major.width' : 1.5,
    'xtick.major.size'  : 6,
    'ytick.major.size'  : 6,

    # Lines & markers
    'lines.linewidth'   : 2.0,
    'lines.markersize'  : 8,

    # Legend
    'legend.fontsize'   : FONT_SIZE_LEGEND,
    'legend.frameon'    : True,
    'legend.framealpha' : 0.8,
    'legend.edgecolor'  : 'black',

    # Grid (off by default)
    'axes.grid'         : False,
    'grid.linewidth'    : 1.0,
    'grid.alpha'        : 0.3,

    # Display & saving
    'figure.dpi'        : 100,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.1,
}

def setup_style(sns_style='white', sns_context='talk'):
    """
    Apply rcParams and seaborn theme.

    Parameters
    ----------
    sns_style   : seaborn style ('white', 'whitegrid', 'ticks', …)
    sns_context : seaborn context ('talk', 'paper', 'poster', …)
    """
    sns.set_theme(context=sns_context, style=sns_style, font_scale=1.0)
    plt.rcParams.update(_RC)


# Apply immediately on import
setup_style()
