"""Confusion between RBPs that belong to the same protein complex.

Enrichment is measured against how often each class is predicted overall, so a large
class absorbing errors from everywhere does not count as specific confusion. Needs only
the stored predictions: no model, no GPU.

    python scripts/rbp27_application/complex_confusion.py \
        --preds iter_preds_bnn_v_10_s_9.pickle \
        --out   out/complex
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

RBP27 = ["MBNL1", "P53_NONO", "PUM2", "QKI", "AGO3", "FUS", "TAF15", "ZFP36",
         "DICER1", "EIF3A", "EIF3D", "EIF3G", "SSB", "PAPD5", "CPSF4", "CPSF3",
         "RTCB", "FXR1", "NOP58", "NOP56", "FBL", "LIN28A", "LIN28B", "UPF1",
         "G35", "G45", "XPO5"]

# Complexes and families represented among these 27 proteins. Membership is taken from
# the standard literature on each complex, not derived from the data.
GROUPS = {
    "eIF3 complex":   ["EIF3A", "EIF3D", "EIF3G"],
    "FET family":     ["FUS", "TAF15"],
    "CPSF complex":   ["CPSF3", "CPSF4"],
    "box C/D snoRNP": ["FBL", "NOP56", "NOP58"],
    "LIN28 paralogs": ["LIN28A", "LIN28B"],
}


def softmax(a, axis=-1):
    z = a - a.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def enrichments(true, pred, n_cls, min_errors=10):
    rate = np.bincount(pred, minlength=n_cls) / len(pred)
    out = {}
    for a in range(n_cls):
        w = (true == a) & (pred != a)
        if w.sum() < min_errors:
            continue
        for b in range(n_cls):
            if a == b:
                continue
            exp = rate[b] / (1 - rate[a]) if rate[a] < 1 else 0
            if exp > 0:
                out[(a, b)] = float((pred[w] == b).mean() / exp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--labels", default=None,
                    help="pickle with y_test; if omitted, read from the figure bundle")
    ap.add_argument("--out", default="out/complex")
    ap.add_argument("--n-iter", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import pickle
    with open(args.preds, "rb") as fh:
        raw = np.asarray(pickle.load(fh))[:args.n_iter]
    mean = softmax(raw).mean(0)
    if args.labels:
        with open(args.labels, "rb") as fh:
            y = np.asarray(pickle.load(fh)["y_test"])
    else:
        from uq_genomics.figures import data_access as DA
        y = np.asarray(DA.bundle_part("app27")["y_test"])
    true, pred = y.argmax(1), mean.argmax(1)

    idx = {n: i for i, n in enumerate(RBP27)}
    enr = enrichments(true, pred, y.shape[1])
    same = {(idx[a], idx[b]) for mem in GROUPS.values() for a in mem for b in mem if a != b}
    within = [v for k, v in enr.items() if k in same]
    between = [v for k, v in enr.items() if k not in same]
    u, p = mannwhitneyu(within, between, alternative="greater")

    print(f"same-complex pairs      n={len(within):3d}  median={np.median(within):.2f}x")
    print(f"different-complex pairs n={len(between):3d}  median={np.median(between):.2f}x")
    print(f"Mann-Whitney one-sided  p={p:.3g}\n")
    for g, mem in GROUPS.items():
        vals = [(a, b, enr[(idx[a], idx[b])]) for a in mem for b in mem
                if a != b and (idx[a], idx[b]) in enr]
        if vals:
            print(f"  {g}")
            for a, b, v in vals:
                print(f"     {a:8s} -> {b:8s} {v:5.1f}x")

    json.dump({"within": within, "between": between, "p": float(p),
               "per_pair": {f"{RBP27[a]}->{RBP27[b]}": v for (a, b), v in enr.items()}},
              open(os.path.join(args.out, "complex_confusion.json"), "w"), indent=1)

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from uq_genomics.figures import style as S
        S.apply_rc()
        pal, W = list(S.PALETTE), S.TEXTWIDTH_IN
    except Exception:
        pal, W = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"], 6.5
    # five groups, four palette colours: add a neutral fifth rather than
    # recycling, which made LIN28 indistinguishable from eIF3
    while len(pal) < len(GROUPS):
        pal = pal + ["#9E9E9E"]

    fig = plt.figure(figsize=(W, W * 0.40), dpi=300)
    axL = fig.add_axes([0.115, 0.20, 0.275, 0.72])
    axR = fig.add_axes([0.515, 0.20, 0.470, 0.72])

    # a) distributions, log scale: enrichment is a ratio
    for i, (v, lab, col) in enumerate(((between, "different\ncomplex", pal[2]),
                                       (within, "same\ncomplex", pal[0]))):
        bp = axL.boxplot([v], positions=[i], widths=.6, patch_artist=True,
                         showfliers=False, medianprops=dict(color="0.2", lw=1.1))
        bp["boxes"][0].set(facecolor=col, lw=0, alpha=.9)
        axL.scatter(np.random.default_rng(0).normal(i, .07, len(v)), v, s=4,
                    color="0.25", alpha=.35, linewidths=0, zorder=3)
    axL.axhline(1, ls="--", color="0.5", lw=.8)
    axL.set_yscale("log")
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["different\ncomplex", "same\ncomplex"])
    axL.set_ylabel("confusion enrichment\n(observed / expected)", fontsize=8)
    axL.tick_params(labelsize=7)
    axL.set_title(f"p = {p:.1e}", fontsize=8)
    axL.text(0.0, 1.06, "a)", transform=axL.transAxes, fontweight="bold", va="bottom")

    # b) the individual pairs that drive it
    pairs, vals, cols = [], [], []
    for gi, (g, mem) in enumerate(GROUPS.items()):
        for a in mem:
            for b in mem:
                if a != b and (idx[a], idx[b]) in enr:
                    pairs.append(f"{a}→{b}")
                    vals.append(enr[(idx[a], idx[b])])
                    cols.append(pal[gi % len(pal)])
    o = np.argsort(vals)
    axR.barh(np.arange(len(o)), np.array(vals)[o],
             color=[cols[i] for i in o], height=.72, linewidth=0)
    axR.axvline(1, ls="--", color="0.5", lw=.8)
    axR.set_yticks(np.arange(len(o)))
    axR.set_yticklabels([pairs[i] for i in o], fontsize=6)
    axR.set_xlabel("confusion enrichment (observed / expected)", fontsize=8)
    axR.tick_params(axis="x", labelsize=7)
    axR.text(0.0, 1.06, "b)", transform=axR.transAxes, fontweight="bold", va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=pal[i % len(pal)], lw=0)
               for i in range(len(GROUPS))]
    axR.legend(handles, list(GROUPS), fontsize=6, frameon=False, loc="lower right")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"COMPLEX_confusion.{ext}"), dpi=300)
    plt.close(fig)
    print(f"\nwrote {os.path.join(args.out, 'COMPLEX_confusion.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
