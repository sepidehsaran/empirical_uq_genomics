"""Uncertainty score per class on the 27-RBP task, by true label and by predicted class.

Reads the stored per-iteration predictions. No model and no GPU needed.

    python scripts/rbp27_application/uncertainty_per_class.py \
        --preds /path/to/iter_preds_bnn_v_10_s_9.pickle \
        --out   out/unc_per_class
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

RBP27 = ["MBNL1", "P53_NONO", "PUM2", "QKI", "AGO3", "FUS", "TAF15", "ZFP36",
         "DICER1", "EIF3A", "EIF3D", "EIF3G", "SSB", "PAPD5", "CPSF4", "CPSF3",
         "RTCB", "FXR1", "NOP58", "NOP56", "FBL", "LIN28A", "LIN28B", "UPF1",
         "G35", "G45", "XPO5"]


def softmax(a, axis=-1):
    z = a - a.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--bundle", default=None,
                    help="figure_data_bundle directory; else $UQFIG_BUNDLE")
    ap.add_argument("--out", default="out/unc_per_class")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--min-n", type=int, default=10,
                    help="a class needs this many samples in a group to be drawn")
    ap.add_argument("--log", action="store_true",
                    help="log scale on the y axis; the score spans several decades")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.bundle:
        os.environ["UQFIG_BUNDLE"] = args.bundle

    import pickle
    from scipy.stats import spearmanr
    from uq_genomics.figures import data_access as DA

    b = DA.bundle_part("app27")
    y = np.asarray(b["y_test"])
    counts = np.asarray(b["train_counts"], dtype=float)

    with open(args.preds, "rb") as fh:
        raw = np.asarray(pickle.load(fh))[:args.n_iter]
    pr = softmax(raw)
    mean, sd = pr.mean(0), pr.std(0)
    true, pred = y.argmax(1), mean.argmax(1)
    sd_pred = sd[np.arange(len(sd)), pred]          # the score the filter uses
    print(f"{len(true)} test sequences, {y.shape[1]} classes, {raw.shape[0]} MC iterations")

    order = np.argsort(-counts)                     # largest training class first
    groups = {}
    for tag, key in (("by true label", true), ("by predicted class", pred)):
        vals, labs, ns, sizes = [], [], [], []
        for c in order:
            v = sd_pred[key == c]
            if len(v) < args.min_n:
                continue
            vals.append(v); labs.append(RBP27[c]); ns.append(len(v)); sizes.append(counts[c])
        med = [float(np.median(v)) for v in vals]
        rho, p = spearmanr(sizes, med)
        groups[tag] = dict(vals=vals, labs=labs, n=ns, sizes=sizes, med=med,
                           rho=float(rho), p=float(p))
        print(f"\n{tag}: {len(labs)} classes with n >= {args.min_n}")
        print(f"  Spearman(training-set size, median uncertainty) = {rho:+.3f}  p = {p:.3g}")
        k = min(5, len(labs))
        big = sorted(zip(sizes, med, labs, ns), reverse=True)[:k]
        small = sorted(zip(sizes, med, labs, ns))[:k]
        print(f"  {'':4s}{'class':10s}{'train n':>9}{'test n':>8}{'median SD':>11}")
        for s, m, l, n in big:   print(f"  big {l:10s}{int(s):9d}{n:8d}{m:11.4f}")
        for s, m, l, n in small: print(f"  sml {l:10s}{int(s):9d}{n:8d}{m:11.4f}")
        r = np.median([m for _, m, _, _ in small]) / max(np.median([m for _, m, _, _ in big]), 1e-12)
        print(f"  ratio of medians, {k} smallest / {k} largest: {r:.1f}x")

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from uq_genomics.figures import style as S
        S.apply_rc(); W = S.TEXTWIDTH_IN
    except Exception:
        W = 6.5
        matplotlib.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                                    "pdf.fonttype": 42, "font.size": 9})
    GREEN, TERRA = "#66C2A5", "#C4705A"

    fig = plt.figure(figsize=(W, W * 0.78))
    axes = [fig.add_axes([0.105, 0.585, 0.875, 0.335]),
            fig.add_axes([0.105, 0.115, 0.875, 0.335])]
    for ax, (tag, g), letter, col in zip(axes, groups.items(), "ab", (GREEN, TERRA)):
        bp = ax.boxplot(g["vals"], positions=np.arange(len(g["labs"])), widths=.62,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="0.2", lw=1.0))
        for box in bp["boxes"]:
            box.set(facecolor=col, lw=0, alpha=.9)
        ax.set_xticks(np.arange(len(g["labs"])))
        ax.set_xticklabels(g["labs"], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("uncertainty score\n(SD at predicted class)", fontsize=7.5)
        if args.log:
            ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title(f"{letter}) {tag} — classes ordered by training-set size "
                     f"(largest first);  Spearman {g['rho']:+.2f}, p = {g['p']:.2g}",
                     fontsize=7.5, loc="left", pad=3)
    axes[1].set_xlabel("RBP", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"UNC_per_class.{ext}"), dpi=300)
    plt.close(fig)

    json.dump({t: {k: v for k, v in g.items() if k != "vals"} for t, g in groups.items()},
              open(os.path.join(args.out, "unc_per_class.json"), "w"), indent=1, default=float)
    print(f"\nwrote {os.path.join(args.out, 'UNC_per_class.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
