"""Does the class that lost the most training data become the most uncertain?

    export UQFIG_BUNDLE=$PWD/figure_data_bundle
    python3 plots_depletion.py --out runs/depletion

The sensitivity figures show, for one seed, that the uncertainty distribution moves when
the training set is made imbalanced. They do not show whether the size of that move
tracks how much data each class actually lost, which is the claim the paper makes.

This figure plots exactly that: per class, the change in training-set size against the
median log2(SD) shift, pooling all 5 seeds, one panel per UQ method and use case. A
method that reflects imbalance in the useful way produces a downward slope -- classes
that were cut get more uncertain, classes that grew get more confident -- i.e. a
negative Spearman rho.

Two details that matter for reading it:

  * The two use cases build imbalance differently. GEX depletes the minority class
    (Erythroblast 2673 -> 587, -78%). RBP leaves its smallest class alone
    (QKI 816 -> 808, -1%) and enriches the majorities instead (MBNL1 +786%). So only
    the GEX panels test "the harshly subsampled class becomes most uncertain"; the RBP
    panels test the complementary manipulation.
  * RBP needs the test-set correction (see plots_sdshift._balance_rows): the
    aggregated imbalanced test set is 71% MBNL1, so class composition alone would move
    the distributions.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from scipy.stats import spearmanr            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_paths as CP                    # noqa: E402
import data_access as DA                     # noqa: E402
import style as S                            # noqa: E402
import plots_sensitivity as PS               # noqa: E402

NDRAW = 100          # subsample repeats for the RBP test-set correction


def _size_ratio(task):
    cd = DA.bundle_part("classdist")
    if task == "rbp":
        kb = ("rbp", (("balanced", True), ("multi_label", False)), 7)
        ki = ("rbp", (("balanced", False), ("multi_label", False)), 7)
    else:
        kb = ("gex", (("balanced", True),), 7)
        ki = ("gex", (("balanced", False),), 7)
    return (np.asarray(cd[ki]["y_train"], float)
            / np.asarray(cd[kb]["y_train"], float))


def _shifts(task, model, seed, balance_test):
    """median log2(SD) shift per class, balanced -> imbalanced"""
    spec = CP.VERSIONS[task]
    comp = DA.load_comparison(task)
    a, ma = PS._sd_for(comp, task, seed, "bal_single", model)
    b, mb = PS._sd_for(comp, task, seed, "imbal_single", model)
    if a is None or b is None:
        return None
    nc = spec["n_classes"]
    if os.environ.get("UQFIG_DEPLETION_SCORE", "all") == "pred":
        # SD at the predicted class, over the samples predicted as that class (Figure 10's
        # definition); the two settings share the balanced test set, so no re-balancing
        pa, pb = np.asarray(ma).argmax(1), np.asarray(mb).argmax(1)
        sa, sb = a[np.arange(len(pa)), pa], b[np.arange(len(pb)), pb]
        out = []
        for c in range(nc):
            za, zb = sa[(pa == c) & (sa > 0)], sb[(pb == c) & (sb > 0)]
            out.append(np.median(np.log2(zb)) - np.median(np.log2(za)) if len(za) > 2 and len(zb) > 2 else np.nan)
        return np.array(out)
    ref = [np.median(np.log2(a[:, c][a[:, c] > 0])) for c in range(nc)]
    if not balance_test:
        return np.array([np.median(np.log2(b[:, c][b[:, c] > 0])) - ref[c]
                         for c in range(nc)])
    lab = DA.bundle_part(f"labels_{task}")
    y = np.asarray(lab[("imbal_single", seed)]).argmax(1)
    k = int(np.bincount(y, minlength=nc).min())
    out = []
    for c in range(nc):
        draws = []
        for i in range(NDRAW):
            rng = np.random.default_rng(i)
            idx = np.concatenate([rng.choice(np.where(y == cc)[0], k, replace=False)
                                  for cc in range(nc)])
            z = b[idx][:, c]
            z = z[z > 0]
            draws.append(np.median(np.log2(z)))
        out.append(float(np.mean(draws)) - ref[c])
    return np.array(out)


def build(out, aspect=0.62):
    tasks = [("rbp", True, "Real RBP"), ("gex", False, "Single-cell")]
    nrow, ncol = len(tasks), len(CP.MODELS)
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * aspect), dpi=S.DPI)
    L, R, T, B, hgap, vgap = 0.105, 0.988, 0.885, 0.155, 0.058, 0.175
    aw = (R - L - hgap * (ncol - 1)) / ncol
    ah = (T - B - vgap * (nrow - 1)) / nrow

    pal = S.PALETTE if isinstance(S.PALETTE, (list, tuple)) else None
    for ri, (task, baltest, tlabel) in enumerate(tasks):
        spec = CP.VERSIONS[task]
        ratio = _size_ratio(task)
        # x is the RANK of the size ratio, not the ratio itself. Spearman's rho is a
        # rank statistic, so ranks are what it actually operates on, and evenly spaced
        # positions stop the three GEX classes with similar ratios (x1.23, x1.58,
        # x2.35) from printing on top of each other.
        order = np.argsort(ratio)
        x = np.empty(spec["n_classes"], float)
        x[order] = np.arange(spec["n_classes"], dtype=float)
        ticks = np.arange(spec["n_classes"], dtype=float)
        # four ticks share ~1.7 in of panel width, so the cell-type names have to be
        # abbreviated; the full names are in the caption
        short = {"CD4+ T activated": "CD4+ T", "CD14+ Mono": "CD14+",
                 "Erythroblast": "Eryth."}
        ticklabels = [short.get(spec["class_names"][c], spec["class_names"][c])
                      + "\n" + rf"$\times${ratio[c]:.2f}" for c in order]
        for mi, model in enumerate(CP.MODELS):
            ax = fig.add_axes([L + mi * (aw + hgap), T - ah - ri * (ah + vgap), aw, ah])
            xs, ys, rhos = [], [], []
            for seed in spec["seeds"]:
                sh = _shifts(task, model, seed, baltest)
                if sh is None:
                    continue
                xs.append(x)
                ys.append(sh)
                rhos.append(spearmanr(x, sh).correlation)
            if not xs:
                continue
            X = np.concatenate(xs)
            Y = np.concatenate(ys)
            for c in range(spec["n_classes"]):
                ax.scatter(X[c::spec["n_classes"]], Y[c::spec["n_classes"]],
                           s=14, color=pal[c % len(pal)] if pal else None,
                           alpha=0.85, linewidths=0, zorder=3)
            ax.axhline(0, color="0.6", lw=0.7, zorder=1)
            # least-squares guide line, purely to make the sign of the trend visible
            if len(np.unique(X)) > 1:
                m, q = np.polyfit(X, Y, 1)
                xx = np.linspace(X.min(), X.max(), 20)
                ax.plot(xx, m * xx + q, color="0.35", lw=1.0, ls="--", zorder=2)
            # top-left, boxed: the previous bottom-left position collided with the
            # x axis and with the lowest points in several panels
            ax.text(0.04, 0.95, rf"$\rho$ = {np.median(rhos):+.2f}",
                    transform=ax.transAxes, fontsize=S.FS_TICK - 0.5,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.75", lw=0.5))
            if ri == 0:
                ax.set_title(CP.MODEL_LABEL[model], fontsize=S.FS_LABEL - 0.5)
            if mi == 0:
                ax.set_ylabel(f"{tlabel}\nmedian $\\log_2$(SD) shift",
                              fontsize=S.FS_TICK - 0.5)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels, fontsize=S.FS_TICK - 2.5)
            ax.set_xlim(ticks.min() - 0.5, ticks.max() + 0.5)
            ax.tick_params(axis="y", labelsize=S.FS_TICK - 1.5)
            ax.tick_params(axis="x", length=2)
    fig.text(0.5, 0.035,
             "training-set class size, imbalanced relative to balanced, in rank order "
             "(left = depleted, right = enriched)",
             ha="center", va="bottom", fontsize=S.FS_LABEL - 1.0)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"DEPLETION_shift.{ext}")
        fig.savefig(p, dpi=S.DPI)
        paths.append(p)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  DEPLETION_shift: {w:.2f} x {h:.2f} in")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/depletion")
    ap.add_argument("--aspect", type=float, default=0.62)
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    S.apply_rc()
    print(f"writing to {out}")
    build(out, args.aspect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
