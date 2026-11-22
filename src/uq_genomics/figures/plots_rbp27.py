"""Figures of the 27-class RBP application, from the raw per-iteration predictions.

    export UQFIG_BUNDLE=$PWD/figure_data_bundle
    python3 plots_rbp27.py --out runs/updated

Everything written here carries an "_updated" suffix so it sits beside the previous
version rather than replacing it.

Why these two figures need the raw iterations rather than the bundle:

  RBP imbalance (Figure 10a).  The aggregated dicts evaluate the balanced setting on
  574 sequences and the imbalanced setting on a different 2411-sequence test set that
  is 71% MBNL1, so class composition alone moves the distributions. The
  `train_imbalanced_test_balanced` runs evaluate BOTH settings on the same 574
  sequences, which is the comparison the paper describes. Those runs only exist from
  v_18 on; v_17, which the bundle was built from, has no such tree. Using them removes
  the subsampling approximation entirely.

  27-class filtering (Figure 11b).  The paper defines a high-quality prediction as one
  whose uncertainty score is below 0.1. That threshold lives in probability space, but
  the bundle stores the SD of the LOGITS (range 0.72-4.58, i.e. > 0.5, so not
  probabilities). Softmax is non-linear, so SD(logits) cannot be converted; the softmax
  has to be applied to each Monte-Carlo iteration before the SD is taken. That needs
  the (n_iter, n_samples, n_class) array.

Both prediction sets are stored as logits: the last axis of a single sample sums to
about 15 for RBP and 478 for the 27-class run, not to 1.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_paths as CP                    # noqa: E402
import data_access as DA                     # noqa: E402
import style as S                            # noqa: E402
import plots_sdshift as SDS             # noqa: E402

PREDS = CP.PREDS          # raw per-pass predictions; UQFIG_PREDS, default <repo>/cluster_preds
UQ_THRESHOLD = 0.1

# The bundle stores class_names=None for the 27-class run. These are recovered from the
# published figure: its bars match y_test.sum(0) = [1119, 241, 274, 90, 432, 540, ...]
# in exactly this order, so index i is name i.
RBP27_NAMES = ["MBNL1", "P53_NONO", "PUM2", "QKI", "AGO3", "FUS", "TAF15", "ZFP36",
               "DICER1", "EIF3A", "EIF3D", "EIF3G", "SSB", "PAPD5", "CPSF4", "CPSF3",
               "RTCB", "FXR1", "NOP58", "NOP56", "FBL", "LIN28A", "LIN28B", "UPF1",
               "G35", "G45", "XPO5"]


def _softmax(a, axis=-1):
    z = a - a.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _load_iters(path):
    with open(path, "rb") as fh:
        return np.asarray(pickle.load(fh))


def _rbp_mean_sd(setting, model, seed):
    """(mean, sd) in PROBABILITY space for one RBP run, or (None, None)."""
    hits = glob.glob(os.path.join(PREDS, "rbp_v19", setting, f"3class_split{seed}",
                                  f"iter_preds_{model}_*_100.pickle"))
    if not hits:
        return None, None
    pr = _softmax(_load_iters(hits[0]))          # (n_iter, n, 3)
    return pr.mean(0), pr.std(0)


def rbp_sd_source(model, seed):
    _, sd_ref = _rbp_mean_sd("balanced", model, seed)
    _, sd_alt = _rbp_mean_sd("train_imbalanced_test_balanced", model, seed)
    return sd_ref, sd_alt


def build_rbp(out, seed, aspect):
    got = rbp_sd_source("bnn", seed)
    if got[0] is None:
        print("  rbp: no cluster predictions found")
        return
    n = got[0].shape[0]
    print(f"  rbp: both settings on the same {n}-sequence test set (seed {seed})")
    for log2, letter in ((True, "a"), (False, "a")):
        SDS.build("rbp", "bal_single", "imbal_single", "rbp_imbalance", out,
                  log2, letter, aspect, seed=seed,
                  sd_source=rbp_sd_source, suffix="_updated")


def build_app27(out, aspect=0.72, n_iter=None, delta_panel=False, extended=False):
    """Class sizes (a) and AP with/without uncertainty filtering (b).

    extended=True adds two more rows and writes APP27_filtering_extended instead:
      c) the uncertainty score per PREDICTED class, same x order as a and b

    Panel c groups by prediction rather than by true label on purpose. The filter acts
    on the SD at the predicted class before any label is known, so that is the grouping
    the threshold actually sees. It is also the stronger effect: -0.59 (p = 0.001)
    against -0.36 (p = 0.067) when grouping by true label, because a rare protein's
    missed sequences are scattered across whichever classes were predicted instead,
    which dilutes the by-label trend.
    """
    from sklearn.metrics import average_precision_score
    p = os.path.join(PREDS, "iter_preds_bnn_v_10_s_9.pickle")
    if not os.path.exists(p):
        print("  app27: prediction file not found")
        return
    raw = _load_iters(p)
    if n_iter:                       # the paper states 100 iterations
        raw = raw[:n_iter]
    pr = _softmax(raw)
    mean, sd = pr.mean(0), pr.std(0)
    b = DA.bundle_part("app27")
    y = np.asarray(b["y_test"])
    names = b.get("class_names") or (RBP27_NAMES if y.shape[1] == len(RBP27_NAMES)
                                     else [f"c{i}" for i in range(y.shape[1])])
    counts = y.sum(0).astype(int)
    print(f"  app27: {raw.shape[0]} iterations, sd max {sd.max():.3f} "
          f"(probability space), {int((sd <= UQ_THRESHOLD).all(1).sum())} samples "
          f"pass SD<{UQ_THRESHOLD} on every class")

    # "a high-quality prediction is one with an uncertainty score below 0.1 for the
    # PREDICTED class" -- one mask over samples, using the SD at argmax, not a separate
    # mask per class. Masking per class instead leaves mean AP flat (0.201 -> 0.199)
    # and does not reproduce the published figure; this definition does, to 3 decimals.
    pred = mean.argmax(1)
    keep = sd[np.arange(len(sd)), pred] <= UQ_THRESHOLD
    ap_all, ap_f = [], []
    for c in range(y.shape[1]):
        ap_all.append(average_precision_score(y[:, c], mean[:, c]))
        ap_f.append(average_precision_score(y[keep][:, c], mean[keep][:, c])
                    if y[keep][:, c].sum() > 0 else np.nan)
    ap_all, ap_f = np.array(ap_all), np.array(ap_f)
    order = np.argsort(-counts)
    print(f"     kept {int(keep.sum())} of {len(keep)} samples ({100*keep.mean():.0f}%); "
          f"AP mean {np.nanmean(ap_all):.3f} -> {np.nanmean(ap_f):.3f}, "
          f"improved in {int(np.nansum(ap_f > ap_all))} of {len(ap_f)} classes")

    nrow = 3 if (extended or delta_panel) else 2
    if delta_panel:
        aspect = aspect * 1.42          # room for the third row
    if extended:
        aspect = aspect * 1.48          # room for row c
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * aspect), dpi=S.DPI)
    # vgap has to hold a full set of rotated class labels, because every panel is
    # labelled rather than only the bottom one
    L, R, T, B, vgap = 0.098, 0.988, 0.955, 0.185, 0.215
    if delta_panel:
        L, T, B, vgap = 0.098, 0.965, 0.130, 0.150
    if extended:
        L, T, B, vgap = 0.098, 0.962, 0.135, 0.155
    ah = (T - B - vgap * (nrow - 1)) / nrow
    x = np.arange(len(order))
    pal = S.PALETTE if isinstance(S.PALETTE, (list, tuple)) else None
    c_ref = pal[0] if pal else S.C_REF
    c_alt = S.C_ALT
    c_box = pal[2] if pal and len(pal) > 2 else S.C_ALT

    ax1 = fig.add_axes([L, T - ah, R - L, ah])
    ax1.bar(x, counts[order], 0.72, color=c_ref, linewidth=0)
    ax1.set_ylabel("test-set\nclass size", fontsize=S.FS_TICK - 0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([names[i] for i in order], rotation=90,
                        fontsize=S.FS_TICK - 2.0)
    ax1.set_xlim(-0.7, len(x) - 0.3)
    ax1.tick_params(axis="y", labelsize=S.FS_TICK - 1.5)
    ax1.text(-0.058, 1.02, "a)", transform=ax1.transAxes, fontsize=S.FS_BASE,
             fontweight="bold", va="bottom", ha="left")

    ax2 = fig.add_axes([L, T - 2 * ah - vgap, R - L, ah])
    w = 0.38
    ax2.bar(x - w / 2, ap_all[order], w, label="all predictions",
            color=c_ref, linewidth=0)
    ax2.bar(x + w / 2, np.nan_to_num(ap_f[order]), w,
            label=f"uncertainty score $<${UQ_THRESHOLD}", color=c_alt, linewidth=0)
    ax2.set_ylabel("average\nprecision", fontsize=S.FS_TICK - 0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([names[i] for i in order], rotation=90,
                        fontsize=S.FS_TICK - 2.0)
    ax2.set_xlim(-0.7, len(x) - 0.3)
    ax2.tick_params(axis="y", labelsize=S.FS_TICK - 1.5)
    ax2.text(-0.058, 1.02, "b)", transform=ax2.transAxes, fontsize=S.FS_BASE,
             fontweight="bold", va="bottom", ha="left")
    ax2.legend(loc="upper right", fontsize=S.FS_LEG, frameon=False)

    if delta_panel:
        # The paired bars in b) make a change of 0.02 on a class whose AP is 0.05 look
        # like nothing, while the same change on MBNL1 is equally invisible next to a
        # bar of 0.89. Plotting the difference puts every class on the same footing.
        d = ap_f - ap_all
        ax3 = fig.add_axes([L, T - 3 * ah - 2 * vgap, R - L, ah])
        up = pal[2] if pal and len(pal) > 2 else S.C_ALT
        dn = pal[1] if pal and len(pal) > 1 else "#d62728"
        cols = [up if v >= 0 else dn for v in d[order]]
        ax3.bar(x, d[order], 0.72, color=cols, linewidth=0)
        ax3.axhline(0, color="0.35", lw=0.8)
        ax3.set_ylabel("change in AP\nafter filtering", fontsize=S.FS_TICK - 0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels([names[i] for i in order], rotation=90,
                            fontsize=S.FS_TICK - 2.0)
        ax3.set_xlim(-0.7, len(x) - 0.3)
        ax3.tick_params(axis="y", labelsize=S.FS_TICK - 1.5)
        ax3.text(-0.058, 1.02, "c)", transform=ax3.transAxes, fontsize=S.FS_BASE,
                 fontweight="bold", va="bottom", ha="left")
        nz = d[np.isfinite(d)]
        print(f"     AP change: median {np.median(nz):+.3f}, "
              f"best {nz.max():+.3f} ({names[order[int(np.nanargmax(d[order]))]]}), "
              f"worst {nz.min():+.3f} ({names[order[int(np.nanargmin(d[order]))]]})")

    if extended:
        from scipy.stats import spearmanr
        tr = np.asarray(b.get("train_counts", counts), dtype=float)
        sdp = sd[np.arange(len(sd)), pred]          # the score the threshold acts on
        _med = np.array([np.median(sdp[pred == i]) if (pred == i).sum() >= 10
                         else np.nan for i in range(y.shape[1])])
        _ok = np.isfinite(_med) & (tr > 0)
        _rho, _p = spearmanr(tr[_ok], _med[_ok])
        print(f"     per-predicted-class uncertainty vs training size: "
              f"rho={_rho:+.3f}, p={_p:.3g}, n={int(_ok.sum())} classes")

        # c) the score itself, grouped by PREDICTED class, in the same x order
        ax3 = fig.add_axes([L, T - 3 * ah - 2 * vgap, R - L, ah])
        vals = [sdp[pred == i] for i in order]
        keep_i = [j for j, v in enumerate(vals) if len(v) >= 10]
        bp = ax3.boxplot([vals[j] for j in keep_i], positions=[x[j] for j in keep_i],
                         widths=0.66, patch_artist=True, showfliers=False,
                         medianprops=dict(color="0.2", lw=0.9))
        for box in bp["boxes"]:
            box.set(facecolor=c_box, lw=0, alpha=0.9)
        # the score spans seven decades, so a linear axis shows only the largest class
        ax3.set_yscale("log")
        ax3.set_ylabel("uncertainty score\n(predicted class)", fontsize=S.FS_TICK - 0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels([names[i] for i in order], rotation=90,
                            fontsize=S.FS_TICK - 2.0)
        ax3.set_xlim(-0.7, len(x) - 0.3)
        ax3.tick_params(axis="y", labelsize=S.FS_TICK - 1.5)
        ax3.text(-0.058, 1.02, "c)", transform=ax3.transAxes, fontsize=S.FS_BASE,
                 fontweight="bold", va="bottom", ha="left")


    paths = []
    stem = ("APP27_filtering_extended" if extended else
            "APP27_filtering_delta" if delta_panel else "APP27_filtering_updated")
    for ext in ("pdf", "png"):
        q = os.path.join(out, f"{stem}.{ext}")
        fig.savefig(q, dpi=S.DPI)
        paths.append(q)
    w_in, h_in = fig.get_size_inches()
    plt.close(fig)
    print(f"  {stem}: {w_in:.2f} x {h_in:.2f} in")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/updated")
    ap.add_argument("--aspect", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--app27-iters", type=int, default=100,
                    help="the stored run has 1000; the paper states 100")
    ap.add_argument("--only", nargs="*", choices=["rbp", "app27"])
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    S.apply_rc()
    print(f"writing to {out}")
    if not args.only or "rbp" in args.only:
        build_rbp(out, args.seed, args.aspect)
    if not args.only or "app27" in args.only:
        build_app27(out, n_iter=args.app27_iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
