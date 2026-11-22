"""Assert that no plotted data point falls outside its axes limits.

A hard-coded ax.set_ylim() is easy to write and silently destroys results: the combined
Kendall figure had ax.set_ylim(0, 1), which hid 44 of 240 values because Kendall's tau is
negative wherever two methods disagree. The boxes, medians and whiskers for three panels
were below the axis; only stray positive outliers remained visible.

This walks every artist in every combined figure and compares its data extent against the
axes limits, so the same mistake fails loudly next time.

    export UQFIG_BUNDLE=$PWD/figure_data_bundle
    python3 test_no_clipping.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S                        # noqa: E402

TOL = 1e-9


def _data_extent(ax):
    """(ymin, ymax) over every artist that carries data on this axes"""
    ys = []
    for ln in ax.lines:                       # whiskers, caps, medians, flier markers
        d = np.asarray(ln.get_ydata(), dtype=float)
        if d.size:
            ys.append(d[np.isfinite(d)])
    for p in ax.patches:                      # boxes, bars
        try:
            bb = p.get_path().get_extents(p.get_patch_transform())
        except Exception:
            continue
        # Seaborn leaves degenerate zero-size Rectangles at the origin; they carry no
        # data and would otherwise report a spurious minimum of 0 for every panel.
        if bb.width == 0 and bb.height == 0:
            continue
        ys.append(np.array([bb.y0, bb.y1], dtype=float))
    for c in ax.collections:                  # scatter / fliers drawn as collections
        try:
            off = np.asarray(c.get_offsets(), dtype=float)
        except Exception:
            continue
        if off.ndim == 2 and off.size:
            col = off[:, 1]
            ys.append(col[np.isfinite(col)])
    ys = [a for a in ys if a.size]
    if not ys:
        return None
    allv = np.concatenate(ys)
    return float(allv.min()), float(allv.max())


def check(fig, name):
    """returns list of (panel_index, ymin, ymax, lo, hi) for axes that clip data"""
    bad = []
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        ext = _data_extent(ax)
        if ext is None:
            continue
        lo, hi = ax.get_ylim()
        if ext[0] < lo - TOL or ext[1] > hi + TOL:
            bad.append((i, ext[0], ext[1], lo, hi))
    status = "CLIPS DATA" if bad else "ok"
    print(f"  {name:34s} {len(fig.axes):2d} axes   {status}")
    for i, y0, y1, lo, hi in bad:
        print(f"      panel {i}: data [{y0:+.4f}, {y1:+.4f}] "
              f"vs axis [{lo:+.4f}, {hi:+.4f}]")
    return bad


def main():
    import combine_figures as CF
    import config_paths as CP
    if not CP.BUNDLE:
        print("skipped: no data bundle (set UQFIG_BUNDLE); without data every panel is empty")
        return 0
    S.apply_rc()
    out = os.path.join(HERE, "runs", "_cliptest")
    os.makedirs(out, exist_ok=True)

    # Patch _save so the figures are checked before they are closed.
    results = {}
    orig_save = CF._save

    def spy(fig, o, stem):
        results[stem] = check(fig, stem)
        return orig_save(fig, o, stem)

    CF._save = spy
    print("checking every combined figure for clipped data\n")
    CF.performance(out)
    CF.performance_ext(out)
    CF.kendall(out)
    CF.kendall(out, mark_alpha=0.05)
    CF.ks(out)
    CF.top100(out)
    CF._save = orig_save

    n_bad = sum(len(v) for v in results.values())
    print(f"\n{len(results)} figures checked, {n_bad} clipped panel(s)")
    if n_bad:
        print("FAILED - widen the axis limits, or remove the hard-coded set_ylim")
    else:
        print("all data visible")
    return 1 if n_bad else 0



def test_script():
    """pytest entry point: the script's main() returns 0 on success"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
