"""Build every combined figure directly from the data, in one matplotlib figure each.

    export UQFIG_BUNDLE=$PWD/figure_data_bundle
    python3 combine_figures.py --out runs/combined

Why this replaces combine_performance.py: that script exported nine PNG panels and
stitched them with PIL. Stitching means each panel is rendered at one size and then
pasted, so keeping type consistent takes constant care, panel letters and legends have to
be drawn by hand with ImageDraw, and the result is raster-only. Here every panel is an
axes in a single figure, drawn once at its final size, so:

  - font sizes are exactly what style.py says, by construction
  - output is vector PDF (infinite resolution in the PDF); PNG is written too, for preview
  - panel letters are matplotlib text, not pixels
  - one legend per group, positioned by the layout engine rather than pasted

Figures produced:
    COMBINED_performance   3 use cases x 3 metrics, one column per use case
    COMBINED_kendall       8 use-case x setting panels
    COMBINED_top100        8 use-case x setting panels
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_paths as CP                    # noqa: E402
import data_access as DA                     # noqa: E402
import style as S                            # noqa: E402

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# column order follows Uwe's requested reading order: simulated TF | real RBP -> GEX
USECASES = [("tf", "Simulated TF binding (DNA)"),
            ("rbp", "Real RBP binding (RNA)"),
            ("gex", "Single-cell gene expression")]
# short forms for the narrow three-column performance figure, where the full names
# either wrap (needing two reserved title lines) or collide with the next column
USECASES_SHORT = {"tf": "Simulated TF (DNA)", "rbp": "Real RBP (RNA)",
                  "gex": "Single-cell GEX"}

METRICS = [("Accuracy", "Accuracy"), ("NLL", "NLL"), ("Brier", "Brier score")]
# The extended figure adds average precision as the SECOND row, next to accuracy: both
# are threshold/ranking measures of discrimination, so they belong together, while NLL and
# the Brier score are proper scoring rules that read on the opposite scale (lower is
# better). The three-metric figure keeps its own order.
METRICS_EXT = [("Accuracy", "Accuracy"), ("AP", "Average precision"),
               ("NLL", "NLL"), ("Brier", "Brier score")]

# (task, setting) for the 8-panel agreement figures
PANELS8 = [("tf", "none"), ("tf", "mislabeled"),
           ("gex", "bal_single"), ("gex", "imbal_single"),
           ("rbp", "bal_single"), ("rbp", "imbal_single"),
           ("rbp", "bal_multi"), ("rbp", "imbal_multi")]
GROUPS8 = [("Simulated TF binding (DNA)", [0, 1]),
           ("Single-cell gene expression", [2, 3]),
           ("Real RBP binding (RNA)", [4, 5, 6, 7])]

# --- Seed-level exclusions -------------------------------------------------------
# BNN did not converge in the RBP balanced multi-label setting for three of the five
# seeds (element-wise accuracy 0.489, 0.692 and 0.418 for seeds 5, 6 and 9, against
# 0.95-0.97 for MCD and ENS on the same splits). Pooling those runs into the top-100
# overlap mixes "the two methods rank different samples" with "one of the models never
# learned the task", which is a different statement. They are dropped from the pooled
# bars for the pairs that involve BNN; the MCD-ENS pair in that panel still uses all
# five seeds, since neither of those models is affected.
# UQFIG_EXCLUDE_BNN=none  -> keep every seed (what the paper shows)
# UQFIG_EXCLUDE_BNN=rule  -> drop the BNN runs below the trivial all-zeros baseline:
#                            seeds 5 and 9 (0.489, 0.418 vs baseline 0.63)
# unset / legacy          -> the original hand-picked exclusion (5, 6, 9)
_EXC = os.environ.get("UQFIG_EXCLUDE_BNN", "legacy").lower()
EXCLUDE_SEEDS = ({} if _EXC == "none"
                 else {("rbp", "bal_multi"): {"bnn": (5, 9)}} if _EXC == "rule"
                 else {("rbp", "bal_multi"): {"bnn": (5, 6, 9)}})

SHORT = {"none": "clean", "mislabeled": "mislabeled", "no_motif": "missing motif",
         "bal_single": "bal. single", "imbal_single": "imbal. single",
         "bal_multi": "bal. multi", "imbal_multi": "imbal. multi"}


# ------------------------------------------------------------------------------ helpers
def _reserve_header_space(axes_top_row):
    """Prefix the top row's titles with blank lines so constrained layout leaves room for
    the group headers. This is the only reservation method that actually moves the axes;
    layout_engine.set(rect=...) leaves get_position() unchanged."""
    for ax in axes_top_row:
        ax.set_title("\n" + ax.get_title(), loc="left",
                     fontsize=S.FS_BASE - 0.5, fontweight="normal")


def _panel_title(ax, letter, text):
    """'a) text' with only the letter bold, at one consistent size"""
    ax.set_title(f"{letter}) {text}", loc="left", fontsize=S.FS_BASE - 0.5,
                 fontweight="normal")


def _save(fig, out, stem):
    """PDF for the manuscript (vector), PNG for eyeballing. No bbox_inches: it silently
    changes the output width, which is what made the old figures inconsistent."""
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"{stem}.{ext}")
        fig.savefig(p, dpi=S.DPI)
        paths.append(p)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  {stem}: {w:.2f} x {h:.2f} in   -> pdf + png")
    return paths


def _grid(nrow, ncol, height_frac, L=0.105, R=0.985, T=0.885, B=0.215,
          hgap=0.075, vgap=0.085):
    """Absolute axes placement. See the note in performance() - neither constrained
    layout nor subplots_adjust survives here, so every combined figure uses add_axes with
    fixed rects. Returns (fig, axes[nrow, ncol])."""
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * height_frac),
                         dpi=S.DPI)
    aw = (R - L - hgap * (ncol - 1)) / ncol
    ah = (T - B - vgap * (nrow - 1)) / nrow
    axes = np.empty((nrow, ncol), dtype=object)
    for ri in range(nrow):
        for ci in range(ncol):
            axes[ri, ci] = fig.add_axes([L + ci * (aw + hgap),
                                         T - ah - ri * (ah + vgap), aw, ah])
    return fig, axes


def _rounded_path(x0, y0, x1, y1, rx, ry):
    """Rounded rectangle as an explicit Path, with SEPARATE x and y corner radii.

    This is why the radii are separate. FancyBboxPatch's boxstyle="round,rounding_size=r"
    applies one radius in the patch's coordinate system. Drawn with transform=transFigure
    on a non-square canvas, r of the width and r of the height are different physical
    lengths (0.052 vs 0.047 in on the 6.50x5.85 Kendall figure), so the corner came out
    as a quarter-ELLIPSE and the edges appeared to bow.

    Passing rx = r/width_in and ry = r/height_in makes the corner a quarter-circle of
    radius r inches on the page, which is what "rounded corner" should mean.

    Corners are quarter-circle Bezier approximations; k = 0.5523 is the standard control
    point ratio, accurate to ~0.02% of the radius.
    """
    from matplotlib.path import Path
    k = 0.5523
    kx, ky = rx * k, ry * k
    verts = [
        (x0 + rx, y0),                                            # bottom edge, start
        (x1 - rx, y0),                                            # bottom edge, end
        (x1 - rx + kx, y0), (x1, y0 + ry - ky), (x1, y0 + ry),     # bottom-right corner
        (x1, y1 - ry),                                            # right edge
        (x1, y1 - ry + ky), (x1 - rx + kx, y1), (x1 - rx, y1),     # top-right corner
        (x0 + rx, y1),                                            # top edge
        (x0 + rx - kx, y1), (x0, y1 - ry + ky), (x0, y1 - ry),     # top-left corner
        (x0, y0 + ry),                                            # left edge
        (x0, y0 + ry - ky), (x0 + rx - kx, y0), (x0 + rx, y0),     # bottom-left corner
    ]
    codes = [Path.MOVETO, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def _fit_x_margins(fig, axes, groups, margin=0.014, pad_x=0.012):
    """Rescale the axes horizontally so the outer group boxes sit `margin` from both
    edges of the canvas, in every figure.

    Needed because the box is fitted to each panel's tight bounding box, which includes
    the y tick labels - and those differ in width between figures ("1.00" vs "100" vs
    "0.0"). With fixed axes rects the resulting outer margins came out as
    L=0.020/R=0.003 for performance, L=-0.002/R=0.007 for Kendall (i.e. the Kendall box
    was running off the left edge) and L=0.003/R=0.007 for top-100.

    Measure, then apply one affine map x' = a*x + b to every axes so the measured extent
    lands exactly on [margin, 1-margin]. The relation is linear, so a single correction
    is exact rather than iterative.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    xs0, xs1 = [], []
    for _, idxs in groups:
        for i in idxs:
            bb = axes[i].get_tightbbox(r)
            if bb is None:                 # an axes with nothing drawn (no data for that panel)
                continue
            bb = bb.transformed(inv)
            xs0.append(bb.x0 - pad_x)
            xs1.append(bb.x1 + pad_x)
    if not xs0:
        return
    lo, hi = min(xs0), max(xs1)
    if hi - lo <= 0:
        return
    a = (1.0 - 2 * margin) / (hi - lo)
    b = margin - a * lo
    for ax in np.asarray(axes).ravel():
        pos = ax.get_position()
        ax.set_position([a * pos.x0 + b, pos.y0, a * pos.width, pos.height])


def _boxed_groups(fig, axes, groups, pad_x=0.012, pad_top=0.052, pad_bot=0.012,
                  label_gap=0.008, align_rows=True, radius_in=0.055, default_mode="line"):
    """Draw a real box around each group of panels and put its label inside the top edge.

    Everything is MEASURED, not guessed: the box is fitted to the union of each panel's
    tight bounding box (axes + tick labels + axis labels + title), so the label can never
    land on top of a title. Earlier versions placed the label at a hand-tuned offset in
    axes coordinates and it collided with "a) Accuracy".

    Returns the list of Rectangles so the caller can inspect them.
    """
    from matplotlib.patches import PathPatch
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes, specs = [], []
    for name, idxs in groups:
        xs0, xs1, ys0, ys1 = [], [], [], []
        for i in idxs:
            ax = axes[i]
            bb = ax.get_tightbbox(r)
            if bb is None:                 # nothing drawn in this panel
                pos = ax.get_position()
                xs0.append(pos.x0); xs1.append(pos.x1); ys0.append(pos.y0); ys1.append(pos.y1)
                continue
            bb = bb.transformed(inv)
            xs0.append(bb.x0); xs1.append(bb.x1)
            ys0.append(bb.y0); ys1.append(bb.y1)
            # get_tightbbox does not reliably cover the title here, and the title is the
            # very thing the group label must clear - measure it explicitly.
            if ax.title.get_text():
                tb = ax.title.get_window_extent(r).transformed(inv)
                xs0.append(tb.x0); xs1.append(tb.x1)
                ys0.append(tb.y0); ys1.append(tb.y1)
        specs.append([name, min(xs0) - pad_x, max(xs1) + pad_x,
                      min(ys0) - pad_bot, max(ys1) + pad_top,
                      axes[idxs[0]].get_position().y0])

    if align_rows:
        # Boxes on the same row are given identical top and bottom edges. Without this
        # they differ in height purely because of x-tick label length: the 6 short TF
        # names occupy far less depth than "CD4+ T activated", so the GEX box hung lower
        # and the two looked mismatched even though the panels are the same size.
        # Grouped by the axes' own y0 so rows are identified geometrically.
        rows = {}
        for sp in specs:
            key = round(sp[5], 3)
            rows.setdefault(key, []).append(sp)
        for members in rows.values():
            lo = min(m[3] for m in members)
            hi = max(m[4] for m in members)
            for m in members:
                m[3], m[4] = lo, hi

    # corner radius in INCHES, converted per-axis so the arc is a true quarter-circle
    fw, fh = fig.get_size_inches()
    rx, ry = radius_in / fw, radius_in / fh

    # UQFIG_GROUP_BOX=none drops the black rectangle (it crowded the
    # panels) and keeps only the bold group label; "line" is the earlier look.
    # "light" is a thin grey outline that separates the use cases without competing
    # with the panels' own frames.
    mode = os.environ.get("UQFIG_GROUP_BOX", default_mode).lower()
    edge, lw = {"none": ("none", 0.0), "light": ("#B0B0B0", 0.6)}.get(mode, ("#000000", 1.0))
    for name, x0, x1, y0, y1, _ in specs:
        rect = PathPatch(_rounded_path(x0, y0, x1, y1, rx, ry),
                         transform=fig.transFigure, linewidth=lw,
                         edgecolor=edge, facecolor="none",
                         zorder=0, clip_on=False)
        fig.patches.append(rect)
        # label sits inside the top edge, above every measured title
        fig.text(x0 + label_gap, y1 - label_gap, name, fontsize=S.FS_LABEL,
                 fontweight="bold", va="top", ha="left")
        boxes.append(rect)
    return boxes


def _headers_constrained(fig, axes, groups):
    """Boxed group labels for the constrained-layout figures.

    A top strip is reserved first so the box outline and its label are not clipped, then
    the boxes are measured from the drawn result.
    """
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    _boxed_groups(fig, axes, groups, pad_top=0.032)


# -------------------------------------------------------------------------- performance
def performance(out):
    """3 metrics (rows) x 3 use cases (columns), one shared legend per column

    A legend per column rather than one for the figure: the three use cases have
    different dataset settings (TF clean/mislabelled, GEX 2, RBP 4), so a single legend
    cannot describe them all.
    """
    import plots_performance as PP
    dfs = {}
    for task, _ in USECASES:
        df, note = PP._collect(task)
        if df is None:
            print(f"  performance: {task} unavailable ({note})")
            return None
        dfs[task] = df

    # Axes are placed with add_axes at ABSOLUTE figure coordinates.
    #
    # Neither constrained layout nor subplots_adjust survives here: style.apply_rc sets
    # figure.constrained_layout.use=True globally, an active layout engine silently
    # discards subplots_adjust, set_layout_engine("none") leaves an incompatible
    # placeholder, and even after creating the figure inside an rc_context the axes were
    # still being repositioned between measurement and savefig - so a group box measured
    # at draw time ended up in the wrong place. Absolute rects cannot move, which makes
    # the box geometry below exactly reproducible.
    # T lowered from the 0.885 default: _boxed_groups reserves pad_top=0.052
    # above each group, and the label must clear 'a) Accuracy' beneath it
    fig, axes = _grid(len(METRICS), len(USECASES), 1.10, T=0.852)
    handles = {}
    for ci, (task, _) in enumerate(USECASES):
        df = dfs[task]
        for ri, (col, nice) in enumerate(METRICS):
            ax = axes[ri, ci]
            order = [m.upper() for m in CP.MODELS]
            if _HAS_SNS:
                sns.boxplot(data=df, x="model", y=col, hue="setting", order=order,
                            palette=S.PALETTE[:df["setting"].nunique()],
                            ax=ax, linewidth=0.7, fliersize=1.5)
                if task not in handles:
                    handles[task] = ax.get_legend_handles_labels()
                if ax.get_legend():
                    ax.get_legend().remove()
            ax.set_xlabel("")
            # the x tick labels already say MCD / BNN / ENS
            ax.set_ylabel(nice if ci == 0 else "", fontsize=S.FS_LABEL)
            ax.tick_params(labelsize=S.FS_TICK)
            _panel_title(ax, LETTERS[ri * len(USECASES) + ci], nice)

    # Column headers are one narrow column wide (~2.1 in), so the long use-case names
    # must wrap or they collide with the neighbouring column.
    # one box per column, spanning all three metric rows of that use case
    groups_perf = [(USECASES_SHORT[t], [i, i + 3, i + 6])
                   for i, (t, _) in enumerate(USECASES)]
    _fit_x_margins(fig, axes.ravel(), groups_perf)
    _boxed_groups(fig, axes.ravel(), groups_perf)
    for ci, (task, _) in enumerate(USECASES):
        if task not in handles:
            continue
        h, l = handles[task]
        # Anchored to the bottom-row AXES (bbox_to_anchor is in axes coordinates for
        # ax.legend), not to figure coordinates: a figure-coordinate anchor computed
        # from get_position() kept landing inside the panel. Negative y puts the legend
        # below the axes, and it tracks the axes if the layout ever changes.
        axes[-1, ci].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                            ncols=1, fontsize=S.FS_LEG - 0.5, frameon=False,
                            handlelength=1.1, handletextpad=0.5, borderaxespad=0.0)
    return _save(fig, out, "COMBINED_performance_all_usecases")


def _perf_df(task):
    """Long-form accuracy / NLL / Brier / AP per (setting, model, seed).

    plots_performance._collect() computes the first three; average precision is added
    here rather than there so the existing three-metric figure keeps producing byte-for-
    byte identical output. AP is the macro mean of the one-vs-rest average precision per
    class, which is the definition used in src/utils.py:274 and in plots_utility.
    """
    import metrics as M
    spec = CP.VERSIONS[task]
    comp = DA.load_comparison(task)
    if "mean" not in comp:
        return None
    rows = []
    for seed in spec["seeds"]:
        for setting in spec["settings"]:
            kw = DA.setting_to_split_args(task, setting)
            probe = DA.dict_key(task, seed, setting, CP.MODELS[0])
            n_exp = (np.asarray(comp["mean"][probe]).shape[0]
                     if probe in comp["mean"] else None)
            y = DA.load_y_test(task, spec["species"], spec["n_classes"], seed,
                               expect_n=n_exp, **kw)
            if y is None:
                continue
            for model in CP.MODELS:
                key = DA.dict_key(task, seed, setting, model)
                if key not in comp["mean"]:
                    continue
                p = np.asarray(comp["mean"][key])
                if p.shape[0] != y.shape[0]:
                    continue
                multi = kw.get("multi_label", False)
                acc = (M.accuracy_multi_label(p, y, apply_sigmoid=False, th=0.5)
                       if multi else M.accuracy_single_label(p, y))
                rows.append(dict(task=task, setting_key=setting,
                                 setting=DA.setting_label(task, setting),
                                 model=model.upper(), seed=seed, Accuracy=acc,
                                 NLL=M.calculate_nll(p, y, multi_label=multi),
                                 Brier=M.calculate_brier_score(p, y),
                                 AP=float(np.nanmean(
                                     M.average_precision_per_class(np.asarray(y), p)))))
    return pd.DataFrame(rows) if rows else None


def best_seed_map(metric="Accuracy", verbose=True):
    """Per panel, the seed on which the three methods are MOST ALIKE.

    Criterion: the seed minimising the standard deviation of the three models' `metric`.
    This is deliberately not "the seed with the highest score". The purpose of the seed
    choice here is to make the uncertainty comparison fair - the manuscript's own
    argument is that the methods are trained to comparable performance so that
    differences in uncertainty cannot be attributed to differences in accuracy. Picking
    the seed where the three are closest serves that argument directly, and it does not
    reward whichever method happened to get a lucky draw.

    Note this is still selection on the outcome, so the spread shown at the chosen seed
    is smaller than a typical seed's. It is a "best case for comparability" figure, and
    the caption has to say so.

    Returns {(task, setting): seed}.
    """
    out = {}
    if verbose:
        print(f"  shared best seed per panel, by minimum spread of {metric}")
        print(f"    {'panel':22s} {'seed':>5s} {'SD of 3 models':>15s} "
              f"{'(worst seed)':>14s}   per-seed SD")
    for task, setting in PANELS8:
        df = _perf_df(task)
        if df is None:
            continue
        sub = df[df["setting_key"] == setting]
        spread = sub.groupby("seed")[metric].std(ddof=1)
        if spread.empty:
            continue
        seed = int(spread.idxmin())
        out[(task, setting)] = seed
        if verbose:
            per = "  ".join(f"{s}:{v:.4f}" for s, v in spread.items())
            print(f"    {task + '/' + setting:22s} {seed:>5d} {spread.min():>15.4f} "
                  f"{spread.max():>14.4f}   {per}")
    return out


def performance_ext(out):
    """Extended version of performance(): the same three metrics plus average precision.

    Kept as a SEPARATE figure rather than replacing performance(): four rows of three
    panels is 8.8 in tall, which needs a dedicated float page, whereas the three-metric
    version shares a page with text. Geometry is rescaled from the three-row figure so
    the absolute margins, gaps and header space stay the same in inches.
    """
    dfs = {}
    for task, _ in USECASES:
        d = _perf_df(task)
        if d is None:
            print(f"  performance_ext: {task} unavailable")
            return None
        dfs[task] = d

    # Total height 7.40 in, not the 8.84 in a straight fourth row would cost.
    #
    # A TMLR text block is about 9 in, so at 8.84 in the figure filled the float page and
    # the caption was pushed off it. The saving comes from the FIXED costs rather than
    # from the panels: header 1.06 -> 0.95 in, legend strip 1.54 -> 1.15 in and the row
    # gap 0.61 -> 0.44 in. That leaves 0.97 in per panel row against 1.10 in before, an
    # 11% reduction, instead of the 30% that trimming the height alone would have forced.
    #
    # hgap is wider than the three-metric figure's 0.075. The AP panel on the single-cell
    # column ticks at 0.985-1.000, five characters against the usual three ("1.0"), and
    # _boxed_groups fits each box to the tight bbox INCLUDING those labels - so at the
    # default gap the single-cell box overlapped the RBP box.
    inch = float(os.environ.get("UQFIG_PERF_HEIGHT_IN", "7.30"))
    hf = inch / S.TEXTWIDTH_IN
    fig, axes = _grid(len(METRICS_EXT), len(USECASES), hf, L=0.125,
                      T=1 - 0.95 / inch, B=1.15 / inch, vgap=0.44 / inch, hgap=0.105)
    handles = {}
    for ci, (task, _) in enumerate(USECASES):
        df = dfs[task]
        for ri, (col, nice) in enumerate(METRICS_EXT):
            ax = axes[ri, ci]
            order = [m.upper() for m in CP.MODELS]
            if _HAS_SNS:
                sns.boxplot(data=df, x="model", y=col, hue="setting", order=order,
                            palette=S.PALETTE[:df["setting"].nunique()],
                            ax=ax, linewidth=0.7, fliersize=1.5)
                if task not in handles:
                    handles[task] = ax.get_legend_handles_labels()
                if ax.get_legend():
                    ax.get_legend().remove()
            ax.set_xlabel("")
            ax.set_ylabel(nice if ci == 0 else "", fontsize=S.FS_LABEL)
            ax.tick_params(labelsize=S.FS_TICK)
            _panel_title(ax, LETTERS[ri * len(USECASES) + ci], nice)

    groups = [(USECASES_SHORT[t], [i, i + 3, i + 6, i + 9])
              for i, (t, _) in enumerate(USECASES)]
    _fit_x_margins(fig, axes.ravel(), groups)
    _boxed_groups(fig, axes.ravel(), groups, default_mode="none")   # Fig. 5 is published without the box
    for ci, (task, _) in enumerate(USECASES):
        if task not in handles:
            continue
        h, l = handles[task]
        axes[-1, ci].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                            ncols=1, fontsize=S.FS_LEG - 0.5, frameon=False,
                            handlelength=1.1, handletextpad=0.5, borderaxespad=0.0)
    return _save(fig, out, "COMBINED_performance_ext_all_usecases")


# ----------------------------------------------------------------------------- agreement
def _kendall_df(task, setting, alpha=None):
    """recompute Kendall's tau per (pair, subset, seed) - same code path as the
    single-panel figures, so the numbers are identical"""
    import metrics as M
    spec = CP.VERSIONS[task]
    comp = DA.load_comparison(task)
    if "mean" not in comp:
        return None
    kw = DA.setting_to_split_args(task, setting)
    rows = []
    for seed in spec["seeds"]:
        probe = DA.dict_key(task, seed, setting, CP.MODELS[0])
        n_exp = np.asarray(comp["mean"][probe]).shape[0] if probe in comp["mean"] else None
        y = DA.load_y_test(task, spec["species"], spec["n_classes"], seed,
                           expect_n=n_exp, **kw)
        if y is None:
            continue
        for a, b in CP.PAIRS:
            ka, kb = DA.dict_key(task, seed, setting, a), DA.dict_key(task, seed, setting, b)
            if ka not in comp["mean"] or kb not in comp["mean"]:
                continue
            try:
                st = M.kendall_pair(y, comp["mean"][ka], comp["sd"][ka],
                                    comp["mean"][kb], comp["sd"][kb])
            except M.ShapeMismatch:
                continue
            rows.append(dict(pair=CP.PAIR_LABEL[(a, b)], subset="all samples",
                             tau=st["tau_all"], p=st["p_all"]))
            rows.append(dict(pair=CP.PAIR_LABEL[(a, b)], subset="correctly classified",
                             tau=st["tau_correct"], p=st["p_correct"]))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if alpha is not None:
        # Keep only tests that reject the null of no association.
        #
        # CAUTION: this is selection on the outcome. A non-significant tau is
        # non-significant precisely because it is near zero, so filtering removes the
        # near-zero values preferentially and shifts every surviving distribution away
        # from zero. The result is NOT "the true agreement" - it is the agreement
        # conditional on having detected agreement. See the note printed by
        # kendall(alpha=...) for how many points each panel loses.
        df = df[df["p"] < alpha].copy()
        if df.empty:
            return None
    return df


def _top_overlap_df(task, setting, topn=100, seed=None):
    import metrics as M
    spec = CP.VERSIONS[task]
    seed = seed or spec["seeds"][0]
    comp = DA.load_comparison(task)
    if "mean" not in comp:
        return None
    kw = DA.setting_to_split_args(task, setting)
    probe = DA.dict_key(task, seed, setting, CP.MODELS[0])
    n_exp = np.asarray(comp["mean"][probe]).shape[0] if probe in comp["mean"] else None
    y = DA.load_y_test(task, spec["species"], spec["n_classes"], seed,
                       expect_n=n_exp, **kw)
    if y is None:
        return None
    idx = {}
    for m in CP.MODELS:
        k = DA.dict_key(task, seed, setting, m)
        if k not in comp["mean"]:
            return None
        try:
            M.check_same_n("top_overlap", y, comp["mean"][k], comp["sd"][k])
        except M.ShapeMismatch:
            return None
        idx[m] = [M.topn_uncertain_per_class(y, comp["sd"][k], comp["mean"][k], c, topn)
                  for c in range(spec["n_classes"])]
    counts = {CP.PAIR_LABEL[(a, b)]:
              [M.overlap_counts(idx[a][c], idx[b][c]) for c in range(spec["n_classes"])]
              for a, b in CP.PAIRS}
    return pd.DataFrame(counts, index=spec["class_names"])


def _top_overlap_pooled(task, setting, topn=100):
    """Mean and SD of the overlap count across ALL seeds -> (mean_df, sd_df, n_seeds).

    The single-seed version above is what the published figures used, but a bar height
    taken from one seed cannot be told apart from seed noise: on RBP imbalanced
    multi-label the MCD-BNN overlap for MBNL1 is 3 at seed 7 and 24 at seed 6. Pooling
    all five and drawing the spread makes the comparison between method pairs - which is
    the point of the figure - actually readable.

    Each seed is a complete, internally consistent experiment (same seed for all three
    methods), so averaging across them is averaging over data splits and initialisations,
    exactly as the box plots in the performance and Kendall figures do.
    """
    spec = CP.VERSIONS[task]
    drop = EXCLUDE_SEEDS.get((task, setting), {})
    frames, seeds_used = [], []
    for s in spec["seeds"]:
        d = _top_overlap_df(task, setting, topn, seed=s)
        if d is not None:
            frames.append(d)
            seeds_used.append(s)
    if not frames:
        return None, None, 0

    stack = np.stack([f.values for f in frames]).astype(float)   # (seed, class, pair)

    # Blank out (seed, pair) cells for excluded models. Done per column rather than by
    # dropping whole seeds, so a pair of two unaffected methods keeps its full sample.
    if drop:
        cols = list(frames[0].columns)
        for model, bad in drop.items():
            for j, lab in enumerate(cols):
                if model.upper() not in lab.upper():
                    continue
                for i, s in enumerate(seeds_used):
                    if s in bad:
                        stack[i, :, j] = np.nan

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
        n_ok = np.sum(~np.isnan(stack), axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1)
    sd = np.where(n_ok > 1, sd, 0.0)

    if drop:
        per_pair = {c: int(n_ok[:, j].max()) for j, c in enumerate(frames[0].columns)}
        print(f"    {task}/{setting}: seeds per pair after exclusion -> {per_pair}")

    mean = pd.DataFrame(mean, index=frames[0].index, columns=frames[0].columns)
    sd = pd.DataFrame(sd, index=frames[0].index, columns=frames[0].columns)
    return mean, sd, len(frames)


def kendall(out, alpha=None, mark_alpha=None, mark_style="count"):
    # vgap is generous: the second row's group box label sits between the rows
    # T lowered and vgap raised: _boxed_groups now reserves more space above each
    # group's panels so the bold use-case label clears "a) clean" / "e) bal. single"
    fig, axes = _grid(2, 4, 0.90, L=0.085, R=0.988, T=0.858, B=0.130,
                      hgap=0.068, vgap=0.205)
    axes = axes.ravel()
    handles = None

    # Gather every panel BEFORE plotting so the shared y-limit can be derived from the
    # data. The previous hard-coded ax.set_ylim(0, 1) silently clipped 44 of the 240
    # values: Kendall's tau is negative wherever two methods disagree, which happens for
    # MCD-BNN and MCD-ENS on balanced single-label GEX and on both RBP multi-label
    # settings. Those panels showed only the few positive outliers as loose dots, with
    # the box, median and whiskers hidden below the axis.
    # mark_alpha: keep every point, but hatch the boxes where not all seeds reached
    # significance. Unlike alpha= (which deletes points and therefore biases the
    # surviving distribution away from zero) this changes nothing about the data shown -
    # it only labels which boxes are well supported.
    dfs = {p: _kendall_df(*p, alpha=alpha) for p in PANELS8}
    if alpha is not None:
        full = {p: _kendall_df(*p) for p in PANELS8}
        print(f"  significance filter p < {alpha}: points kept per panel")
        for pkey in PANELS8:
            a_, b_ = full[pkey], dfs[pkey]
            na = 0 if a_ is None else len(a_)
            nb = 0 if b_ is None else len(b_)
            mark = "" if na == nb else f"   <-- {na - nb} dropped"
            print(f"    {pkey[0]:4s} {pkey[1]:14s} {nb:3d}/{na}{mark}")
    vals = [d["tau"] for d in dfs.values() if d is not None]
    lo = min(float(v.min()) for v in vals) if vals else 0.0
    ylo = min(0.0, np.floor(lo * 20) / 20) - 0.02      # round down to a 0.05 step

    for i, (task, setting) in enumerate(PANELS8):
        ax = axes[i]
        df = dfs[(task, setting)]
        if df is None:
            ax.set_visible(False)
            continue
        if _HAS_SNS:
            # Explicit order/hue_order: when the significance filter removes every row
            # of a category, seaborn silently re-orders the remaining ones, so panel g
            # came out as MCD-ENS / BNN-ENS / MCD-BNN while its neighbours kept the
            # canonical order. Fixing both keeps all panels comparable.
            sns.boxplot(data=df, x="pair", y="tau", hue="subset",
                        order=[CP.PAIR_LABEL[pr] for pr in CP.PAIRS],
                        hue_order=["all samples", "correctly classified"],
                        palette=S.PALETTE[:2],
                        ax=ax, linewidth=0.7, fliersize=1.5)
            if handles is None:
                handles = ax.get_legend_handles_labels()
            if ax.get_legend():
                ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("Kendall's tau" if i % 4 == 0 else "", fontsize=S.FS_LABEL)
        # three "MCD-BNN" labels do not fit side by side in a quarter-width panel
        ax.set_xticks(range(len(ax.get_xticklabels())))
        ax.set_xticklabels([t.get_text().replace("-", "\n") for t in ax.get_xticklabels()],
                           fontsize=S.FS_TICK - 1.5)
        ax.tick_params(axis="y", labelsize=S.FS_TICK - 0.5)
        if mark_alpha is not None:
            # Hatching was the first attempt and is unreadable here: in the panels that
            # need marking the boxes span ~0.1 tau on an axis running to 1.0, so they
            # render about 10 px tall and the hatch lines merge into a smudge.
            #
            # The mark then sat just above each box, which put a hollow circle a few
            # pixels from the filled flier dots in panel c - close enough to read as a
            # data point. Drawing it instead at ONE fixed height, in axes coordinates,
            # fixes that: six slots per panel at identical y across all eight panels form
            # an obvious annotation row that no reader will mistake for data, and the
            # x position still ties each mark to its box.
            #
            # An asterisk is deliberately avoided: */**/*** conventionally denote
            # SIGNIFICANT results, so using one here would invert the usual meaning.
            from matplotlib.transforms import blended_transform_factory
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            pairs_o = [CP.PAIR_LABEL[pr] for pr in CP.PAIRS]
            subs_o = ["all samples", "correctly classified"]
            width = 0.8 / len(subs_o)
            for pi, pair in enumerate(pairs_o):
                for si, sub in enumerate(subs_o):
                    sel = df[(df["pair"] == pair) & (df["subset"] == sub)]
                    if sel.empty:
                        continue
                    n_sig = int((sel["p"] < mark_alpha).sum())
                    if n_sig >= len(sel):
                        continue
                    xc = pi - 0.4 + width * (si + 0.5)
                    if mark_style == "count":
                        txt, fs, rot = f"{n_sig}/{len(sel)}", S.FS_TICK - 2.0, 0
                    elif mark_style == "ns":
                        txt, fs, rot = "n.s.", S.FS_TICK - 2.5, 90
                    else:                                   # "circle" - the default
                        txt, fs, rot = "\u25cb", S.FS_TICK - 1.5, 0
                    ax.text(xc, 0.945, txt, ha="center", va="center", transform=tr,
                            fontsize=fs, color="#333333", rotation=rot, clip_on=False)
        ax.set_ylim(ylo, 1.0)
        if ylo < 0:
            # zero line, so "no agreement" is readable at a glance
            ax.axhline(0, color="#999999", linewidth=0.6, zorder=0)
        _panel_title(ax, LETTERS[i], SHORT.get(setting, setting))
    if handles:
        fig.legend(*handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncols=2,
                   fontsize=S.FS_LEG, frameon=False)
    _fit_x_margins(fig, axes, GROUPS8, pad_x=0.005)
    _boxed_groups(fig, axes, GROUPS8, pad_x=0.005)
    if mark_alpha is not None and handles:
        from matplotlib.lines import Line2D
        h, l = handles
        # The legend has to say what the glyph MEANS, not how it was computed. "n/5:
        # number of significant seeds" made the reader do the arithmetic - work out that
        # n < 5 is the condition being flagged - before the mark meant anything. The
        # circle carries one bit and the legend states that bit directly.
        glyph = {"count": r"$n/5$", "ns": "n.s."}.get(mark_style, "○")
        text = {"count": f"$n/5$: significant in $n$ of 5 seeds ($p<{mark_alpha}$)"}.get(
            mark_style, f"{glyph} not significant in some seeds ($p<{mark_alpha}$)")
        h = list(h) + [Line2D([], [], linestyle="none", marker="none")]
        l = list(l) + [text]
        for lg in list(fig.legends):
            lg.remove()
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncols=3,
                   fontsize=S.FS_LEG, frameon=False)
    stem = ("COMBINED_kendall_all_usecases"
            + ("_significant" if alpha else "")
            + ("_marked" if mark_alpha else ""))
    return _save(fig, out, stem)


def top100(out, topn=100, seed=None, tag=None):
    # extra vgap: the rotated class names under row 1 need room before row 2's box
    fig, axes = _grid(2, 4, 1.00, L=0.085, R=0.988, T=0.858, B=0.120,
                      hgap=0.068, vgap=0.285)
    axes = axes.ravel()
    handles = None
    nseed = 0
    for i, (task, setting) in enumerate(PANELS8):
        ax = axes[i]
        # seed may be None (pool every seed), an int (one seed everywhere), or a dict
        # {(task, setting): seed} from best_seed_map() - a different seed per panel, but
        # always shared by the three methods within a panel, which is what keeps the
        # overlap counts meaningful.
        s_i = seed[(task, setting)] if isinstance(seed, dict) else seed
        if s_i is None:                        # pool every seed (the default)
            df, err, nseed = _top_overlap_pooled(task, setting, topn)
        else:
            df, err = _top_overlap_df(task, setting, topn, seed=s_i), None
        if df is None:
            ax.set_visible(False)
            continue
        classes, pairs = list(df.index), list(df.columns)
        x = np.arange(len(classes))
        width = 0.8 / len(pairs)
        for k, pair in enumerate(pairs):
            h = df[pair].values
            if err is None:
                ye = None
            else:
                # Asymmetric, with the lower arm clipped at 0. An overlap count cannot be
                # negative, so a symmetric SD arm on a small mean (e.g. 6.2 +/- 13.3)
                # would draw a whisker into a region no experiment could produce - and
                # ax.set_ylim(0, ...) then silently cut it off, which is exactly what
                # test_no_clipping.py exists to catch.
                ye = np.vstack([np.minimum(err[pair].values, h), err[pair].values])
            ax.bar(x + k * width, h, width, label=pair,
                   color=S.PALETTE[k % len(S.PALETTE)], edgecolor="#333333",
                   linewidth=0.4, yerr=ye,
                   error_kw=dict(ecolor="#333333", elinewidth=0.6, capsize=1.2,
                                 capthick=0.6))
        if handles is None:
            handles = ax.get_legend_handles_labels()
        ax.set_xticks(x + width * (len(pairs) - 1) / 2)
        ax.set_xticklabels(classes, rotation=45 if len(classes) > 3 else 20,
                           ha="right", fontsize=S.FS_TICK - 1.5)
        ax.set_ylabel(f"common in top {topn}" if i % 4 == 0 else "",
                      fontsize=S.FS_LABEL)
        ax.set_ylim(0, topn * 1.08)
        ax.tick_params(axis="y", labelsize=S.FS_TICK - 0.5)
        # With a per-panel seed the reader cannot know which seed a panel used unless the
        # panel says so. The seed goes INSIDE the axes, not in the title: appending
        # "(seed 11)" to the title pushed "f) imbal. single" into "g) bal. multi" and
        # made the TF group box overlap the single-cell one, because _boxed_groups fits
        # each box to the tight bbox and the title is part of it. Text inside the axes
        # has no effect on that bbox.
        _panel_title(ax, LETTERS[i], SHORT.get(setting, setting))
        if isinstance(seed, dict):
            ax.text(0.97, 0.96, f"seed {s_i}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=S.FS_TICK - 1.5, color="#555555")
    if handles:
        fig.legend(*handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncols=3,
                   fontsize=S.FS_LEG, frameon=False)
    _fit_x_margins(fig, axes, GROUPS8, pad_x=0.005)
    _boxed_groups(fig, axes, GROUPS8, pad_x=0.005)
    if seed is None:
        print(f"    pooled over {nseed} seeds (bars = mean, error bars = SD)")
    if isinstance(seed, dict):
        suffix = f"_best{tag or 'seed'}"
    elif seed:
        suffix = f"_seed{seed}"
    else:
        suffix = ""
    stem = f"COMBINED_top{topn}_all_usecases{suffix}"
    return _save(fig, out, stem)


def _ks_df(task):
    """KS statistic per (model, comparison, class, seed) for one use case.

    Reference = the first dataset setting, perturbed = each later one, per
    plots_sensitivity.COMPARISONS. The comparison is therefore "does the SD distribution
    for this class shift when the perturbation is introduced", asked separately for each
    UQ method - NOT a comparison between methods. See the note in ks() about the text.
    """
    import metrics as M
    import plots_sensitivity as PS
    spec = CP.VERSIONS[task]
    comp = DA.load_comparison(task)
    if "sd" not in comp:
        return None
    rows = []
    for ref, alt, lab in PS.COMPARISONS.get(task, []):
        for seed in spec["seeds"]:
            for m in CP.MODELS:
                kr, ka = DA.dict_key(task, seed, ref, m), DA.dict_key(task, seed, alt, m)
                if kr not in comp["sd"] or ka not in comp["sd"]:
                    continue
                for c in range(spec["n_classes"]):
                    st, p, na, nb = M.ks_sd_shift_unpaired(
                        comp["sd"][kr], comp["mean"][kr],
                        comp["sd"][ka], comp["mean"][ka], c, method="exact")
                    if np.isnan(st):
                        continue
                    rows.append(dict(model=m.upper(), comparison=lab, ks=st, p=p,
                                     cls=spec["class_names"][c], seed=seed,
                                     setting_alt=alt))
    return pd.DataFrame(rows) if rows else None


# Panels of the combined KS figure: one per (use case, perturbation).
#
# 2x2 rather than 1x4. At four panels across, each is ~1.4 in wide and the group headers
# ("Simulated TF binding (DNA)", "Single-cell gene expression") are far wider than that,
# so all three ran into each other across the top of the figure. Two columns gives each
# panel ~2.9 in and puts the two RBP panels together on the bottom row, matching the
# grouping used by the Kendall and top-100 figures. Titles are shortened for the same
# reason - the use case is already named by the box around the panel.
KS_PANELS = [("tf", "mislabeled", "partial mislabeling"),
             ("gex", "imbal_single", "class imbalance"),
             ("rbp", "imbal_single", "imbalance, single-label"),
             ("rbp", "imbal_multi", "imbalance, multi-label")]
KS_GROUPS = [("Simulated TF binding (DNA)", [0]),
             ("Single-cell gene expression", [1]),
             ("Real RBP binding (RNA)", [2, 3])]


def ks(out, mark_alpha=0.05):
    """Combined KS figure: one panel per use case x perturbation, x = UQ method.

    Each point is one (class, seed) KS statistic. A high value means the SD distribution
    for that class moved when the perturbation was introduced, i.e. the method's
    uncertainty responds to it; a value near zero means it did not.
    """
    dfs = {}
    for task in ("tf", "gex", "rbp"):
        d = _ks_df(task)
        if d is not None:
            dfs[task] = d
    if not dfs:
        print("  ks: no data")
        return None

    fig, axes = _grid(2, 2, 0.78, L=0.095, R=0.988, T=0.860, B=0.135,
                      hgap=0.105, vgap=0.185)
    axes = axes.ravel()
    handles = None
    for i, (task, alt, lab) in enumerate(KS_PANELS):
        ax = axes[i]
        d = dfs.get(task)
        if d is None:
            ax.set_visible(False)
            continue
        sub = d[d["setting_alt"] == alt]
        if sub.empty:
            ax.set_visible(False)
            continue
        if _HAS_SNS:
            sns.boxplot(data=sub, x="model", y="ks", hue="model", legend=False,
                        order=[m.upper() for m in CP.MODELS],
                        palette=dict(zip([m.upper() for m in CP.MODELS], S.PALETTE[:3])), ax=ax, linewidth=0.7, fliersize=1.5)
        # every point, jittered, so the per-class spread is visible rather than implied
        rng = np.random.default_rng(0)
        for xi, m in enumerate([m.upper() for m in CP.MODELS]):
            v = sub[sub["model"] == m]
            if v.empty:
                continue
            ns = v["p"] >= mark_alpha
            for msk, fc, ec in ((~ns, "#333333", "#333333"), (ns, "none", "#333333")):
                if not msk.any():
                    continue
                ax.scatter(xi + rng.uniform(-0.16, 0.16, int(msk.sum())),
                           v["ks"][msk], s=4.5, facecolors=fc, edgecolors=ec,
                           linewidths=0.45, zorder=3)
        ax.set_xlabel("")
        # Every panel gets the y label, not just the first. _boxed_groups fits each box
        # to the union of its panels' tight bounding boxes, and the y label is part of
        # that - so labelling only panel a made the TF box reach further left than the
        # RBP box below it, and made the TF box wider than the single-cell box beside it.
        # With the label on all four, the TF and single-cell boxes are the same width and
        # their outer edges line up with the RBP box spanning the row underneath.
        ax.set_ylabel("KS statistic", fontsize=S.FS_LABEL)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=S.FS_TICK - 0.5)
        _panel_title(ax, LETTERS[i], lab)
    if handles is None:
        from matplotlib.lines import Line2D
        handles = ([Line2D([], [], marker="o", linestyle="none", markersize=3.2,
                           markerfacecolor="#333333", markeredgecolor="#333333"),
                    Line2D([], [], marker="o", linestyle="none", markersize=3.2,
                           markerfacecolor="none", markeredgecolor="#333333")],
                   [f"distribution shifted ($p<{mark_alpha}$)",
                    f"not shifted ($p\\geq{mark_alpha}$)"])
    fig.legend(*handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncols=2,
               fontsize=S.FS_LEG, frameon=False)
    _fit_x_margins(fig, axes, KS_GROUPS, pad_x=0.005)
    _boxed_groups(fig, axes, KS_GROUPS, pad_x=0.005)
    n_tot = sum(len(d) for d in dfs.values())
    n_sig = sum(int((d["p"] < mark_alpha).sum()) for d in dfs.values())
    print(f"    {n_sig}/{n_tot} tests reject the null of no shift at p<{mark_alpha}")
    return _save(fig, out, "COMBINED_ks_all_usecases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/combined")
    ap.add_argument("--only", nargs="*",
                    choices=["performance", "performance_ext", "kendall", "ks",
                             "top100", "top100_best"])
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the single-seed figures (top100). Default 7.")
    ap.add_argument("--mark-style", default="circle", choices=["circle", "count", "ns"],
                    help="glyph used by --mark-alpha. 'circle' just flags the boxes that "
                         "are not significant in all 5 seeds; 'count' writes n/5.")
    ap.add_argument("--mark-alpha", type=float, default=None,
                    help="Kendall figure: keep ALL points, hatch boxes where not every "
                         "seed reaches p < value. Preferred over --alpha.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Kendall figure: keep only tests with p < alpha (e.g. 0.05). "
                         "Selection on the outcome - see the docstring in _kendall_df.")
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    S.apply_rc()
    print(f"palette: {S.PALETTE_NAME}  {S.PALETTE}")
    print(f"writing to {out}")
    todo = args.only or ["performance", "performance_ext", "kendall", "ks",
                         "top100", "top100_best"]

    def _top100_best(o):
        """Both selection criteria: minimum spread of accuracy, and of average
        precision. Two figures, because the two criteria disagree on half the panels."""
        for metric, tag in (("Accuracy", "acc"), ("AP", "ap")):
            top100(o, seed=best_seed_map(metric), tag=tag)

    fns = {"performance": performance,
           "performance_ext": performance_ext,
           "kendall": (lambda o: kendall(o, alpha=args.alpha,
                                         mark_alpha=args.mark_alpha,
                                         mark_style=args.mark_style)),
           "ks": ks,
           "top100": (lambda o: top100(o, seed=args.seed)),
           "top100_best": _top100_best}
    for t in todo:
        fns[t](out)
    print("\nplace with  \\includegraphics[width=\\textwidth]{<name>.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
