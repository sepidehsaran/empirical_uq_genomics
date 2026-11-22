"""
style.py - ONE definition of figure size and font for every plot in the pipeline.

Why this matters: the old figures were exported at wildly different figsizes (6.4x4.8 in
for the Kendall panels, 12x10 for accuracy, 40x20 and 60x30 for the SD-shift grids). The
apparent font size in the PDF is  fontsize / figsize, so the same rcParams font produced
labels from 4 pt to 12 pt once the images were scaled into a 6.5 in column.

The rule here: **export every figure at the width it will actually occupy on the page.**
Then 1 pt in the figure is 1 pt on paper, and font size is identical everywhere by
construction. `width_frac=1.0` -> full \textwidth, `0.48` -> a two-up panel, etc.

Do not use bbox_inches='tight' when saving: it silently changes the output width and
breaks the guarantee. Constrained layout is used instead.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 6.5          # TMLR \textwidth at 10pt, letter paper
DPI = 300

FS_BASE  = 9.0              # tick labels / legend body
FS_LABEL = 9.5              # axis labels
FS_TITLE = 10.0             # panel title
FS_TICK  = 8.5
FS_LEG   = 8.0

# Colour palette. One switch drives every figure:  export UQFIG_PALETTE=<name>
# Names live in palettes.py; run `python3 palettes.py` for the colour-blind safety table
# and `python3 preview_palettes.py` to see each one on the real figure.
#
# The default is verified against deuteranopia, protanopia and tritanopia. The previous
# default, seaborn's "Set2", has a worst-case separation of 5.8 across four categories,
# i.e. two of its colours are effectively identical to a deuteranope.
import os as _os                                                     # noqa: E402
try:
    import palettes as _PAL
    PALETTE_NAME = _os.environ.get("UQFIG_PALETTE", _PAL.DEFAULT)
    PALETTE = _PAL.get(PALETTE_NAME)
    # reference vs perturbed overlays: darkest and warmest of the same palette, so the
    # density plots match the box plots instead of introducing two more hues
    C_REF = PALETTE[0]
    C_ALT = _PAL.PALETTES[PALETTE_NAME].get("alt", PALETTE[2 if len(PALETTE) > 2 else 1])
except Exception:                       # palettes.py absent - keep the old behaviour
    PALETTE_NAME = "Set2"
    PALETTE = "Set2"
    C_REF, C_ALT = "#4C72B0", "#DD8452"


def apply_rc():
    plt.rcdefaults()
    matplotlib.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": FS_BASE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "axes.titleweight": "bold",
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
        "savefig.bbox": None,          # explicitly NOT 'tight'
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def new_fig(width_frac=1.0, aspect=0.62, nrows=1, ncols=1, sharex=False, sharey=False):
    """
    width_frac : fraction of \textwidth this figure will be placed at
    aspect     : height / width of the WHOLE figure
    """
    w = TEXTWIDTH_IN * float(width_frac)
    h = w * float(aspect)
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), sharex=sharex, sharey=sharey)
    return fig, ax


def save(fig, path, close=True):
    fig.savefig(path, dpi=DPI)          # no bbox_inches -> width stays exactly figsize
    if close:
        plt.close(fig)
    return path


def panel_title(ax, letter, text):
    """'a) short description' - bold letter, regular rest, one size (matches the
    convention agreed for the combined figures)."""
    if letter and text:
        ax.set_title(f"$\\bf{{{letter})}}$ {text}", loc="left", fontweight="normal")
    elif letter:
        ax.set_title(f"$\\bf{{{letter})}}$", loc="left", fontweight="normal")
    else:
        ax.set_title(text, loc="left")


def fig_suptitle(fig, letter, text):
    """Figure-level title, wrapped to the figure width so it never runs off the edge."""
    import textwrap
    w_in = fig.get_size_inches()[0]
    # ~1.9 chars per 0.1 in at FS_TITLE for DejaVu Sans
    max_chars = max(24, int(w_in * 19))
    body = f"{letter}) {text}" if letter else text
    wrapped = "\n".join(textwrap.wrap(body, max_chars)) or body
    if letter:
        first, *rest = wrapped.split("\n")
        first = first.replace(f"{letter}) ", "", 1)
        wrapped = "\n".join([f"$\\bf{{{letter})}}$ {first}"] + rest)
    fig.suptitle(wrapped, x=0.01, ha="left", fontsize=FS_TITLE, fontweight="normal")
