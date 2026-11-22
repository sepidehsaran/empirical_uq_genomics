"""Does the uncertainty score carry motif information once correctness is held fixed?

Fills the correctness-by-confidence 2x2, sampling within class and comparing each cell
against its own dinucleotide-shuffled floor. See docs/REPRODUCING.md for the result.

    python scripts/rbp27_application/xai_2x2_motif.py \
        --model /path/to/bnn_model_v_10_s_9 \
        --data  /path/to/dataset.pickle \
        --preds /path/to/iter_preds.pickle \
        --out   out/xai_2x2
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

CELLS = [("A", "correct",   "confident"),
         ("B", "correct",   "uncertain"),
         ("C", "incorrect", "confident"),
         ("D", "incorrect", "uncertain")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="out/xai_2x2")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="the SD filter used in the paper")
    ap.add_argument("--per-cell", type=int, default=200,
                    help="sequences per cell, drawn proportionally across classes")
    ap.add_argument("--min-per-class", type=int, default=4,
                    help="a class must have this many in EVERY cell to be used")
    ap.add_argument("--k", type=int, default=8, help="motif window width")
    ap.add_argument("--m-steps", type=int, default=50)
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import pickle
    import tensorflow as tf
    import tensorflow_probability as tfp                    # noqa: F401
    from scipy.stats import mannwhitneyu
    import xai_uncertainty as XU                            # IG, shuffles, measures

    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    x = np.asarray(d["x_test"]).astype("float32")
    y = np.asarray(d["y_test"])
    with open(args.preds, "rb") as fh:
        raw = np.asarray(pickle.load(fh))[:args.n_iter]
    pr = XU.softmax(raw)
    mean, sd = pr.mean(0), pr.std(0)
    true, pred = y.argmax(1), mean.argmax(1)
    sd_pred = sd[np.arange(len(sd)), pred]
    correct = pred == true
    confident = sd_pred < args.threshold

    masks = {"A": correct & confident, "B": correct & ~confident,
             "C": ~correct & confident, "D": ~correct & ~confident}
    print("cell sizes over the whole test set:")
    for tag, cor, con in CELLS:
        print(f"  {tag}  {cor:9s} + {con:9s}  n={int(masks[tag].sum()):5d}  "
              f"mean SD {sd_pred[masks[tag]].mean():.4f}")

    try:
        import tf_keras
        model = tf_keras.models.load_model(args.model, compile=False)
    except Exception:
        model = tf.keras.models.load_model(args.model, compile=False)
    if mean.shape[1] != y.shape[1]:
        sys.exit(f"ERROR: model outputs {mean.shape[1]} classes, data has {y.shape[1]}")
    g = XU.integrated_gradients(model, x[0], int(pred[0]), m_steps=4, mc_samples=1)
    if not np.isfinite(g).all() or np.abs(g).max() == 0:
        sys.exit("ERROR: gradients are zero or non-finite; the model did not load "
                 "for differentiation.")
    print("\nmodel loaded, gradients flow")

    # ---- classes usable for a stratified comparison: present in ALL FOUR cells ----
    usable = [c for c in range(y.shape[1])
              if all((masks[t] & (true == c)).sum() >= args.min_per_class
                     for t, _, _ in CELLS)]
    print(f"\nclasses present in all four cells: {len(usable)} of {y.shape[1]}")
    print("  " + ", ".join(RBP27[c] for c in usable))
    if not usable:
        sys.exit("No class populates all four cells; the 2x2 cannot be stratified.")

    # allocate the per-cell budget across classes, proportional to what is available
    avail = {c: min((masks[t] & (true == c)).sum() for t, _, _ in CELLS) for c in usable}
    tot = sum(avail.values())
    quota = {c: max(args.min_per_class, int(round(args.per_cell * avail[c] / tot)))
             for c in usable}

    # ---------------------------------------------------------------- measure
    # XU.concentration sums over the base axis itself, so it takes the raw (L, 4)
    # attribution map. Pre-summing collapses it to a scalar.
    res = {t: {"conc": [], "shuf": [], "cls": []} for t, _, _ in CELLS}
    for c in usable:
        for tag, _, _ in CELLS:
            pool = np.where(masks[tag] & (true == c))[0]
            take = rng.choice(pool, min(quota[c], len(pool)), replace=False)
            for i in take:
                att = XU.integrated_gradients(model, x[i], int(pred[i]),
                                              args.m_steps, mc_samples=args.mc_samples)
                res[tag]["conc"].append(float(XU.concentration(att, args.k)))
                xs = XU.dinuc_shuffle(x[i], rng)
                att_s = XU.integrated_gradients(model, xs, int(pred[i]),
                                                args.m_steps, mc_samples=args.mc_samples)
                res[tag]["shuf"].append(float(XU.concentration(att_s, args.k)))
                res[tag]["cls"].append(int(c))
        print(f"  {RBP27[c]:9s} done ({quota[c]} per cell)")

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 72)
    print("Attribution concentration per cell, with its own shuffled floor")
    print("=" * 72)
    print(f"  {'cell':26s} {'n':>5} {'concentration':>14} {'shuffled':>10} {'margin':>8}")
    summ = {}
    for tag, cor, con in CELLS:
        a = np.array(res[tag]["conc"]); s = np.array(res[tag]["shuf"])
        margin = float(np.median(a) - np.median(s))
        u, p = mannwhitneyu(a, s, alternative="greater")
        summ[tag] = dict(n=len(a), conc=float(np.median(a)), shuf=float(np.median(s)),
                         margin=margin, p_vs_shuffled=float(p))
        print(f"  {tag} {cor:9s}+{con:9s} {len(a):5d} {np.median(a):14.4f} "
              f"{np.median(s):10.4f} {margin:+8.4f}   vs shuffled p={p:.2g}")

    print("\n" + "=" * 72)
    print("THE TEST: within a row, correctness is fixed, so any difference is uncertainty")
    print("=" * 72)
    tests = {}
    for lo, hi, label in (("C", "D", "incorrect: uncertain vs confident"),
                          ("A", "B", "correct:   uncertain vs confident")):
        a = np.array(res[hi]["conc"]); b = np.array(res[lo]["conc"])
        u, p = mannwhitneyu(a, b, alternative="less")     # is the uncertain cell LOWER?
        tests[label] = dict(p=float(p), median_uncertain=float(np.median(a)),
                            median_confident=float(np.median(b)))
        verdict = ("uncertainty adds information" if p < 0.05
                   else "no effect of uncertainty once correctness is fixed")
        print(f"  {label}")
        print(f"     uncertain {np.median(a):.4f}   confident {np.median(b):.4f}   "
              f"p={p:.3g}   -> {verdict}")

    # the comparison that IS expected to work, as a positive control: if this one is
    # null too, something is wrong with the measurement rather than with the hypothesis
    a = np.array(res["A"]["conc"] + res["B"]["conc"])
    b = np.array(res["C"]["conc"] + res["D"]["conc"])
    u, p = mannwhitneyu(a, b, alternative="greater")
    print(f"\n  positive control -- correct vs incorrect (pooled over confidence)")
    print(f"     correct {np.median(a):.4f}   incorrect {np.median(b):.4f}   p={p:.3g}")
    tests["control: correct vs incorrect"] = dict(
        p=float(p), median_correct=float(np.median(a)), median_incorrect=float(np.median(b)))

    # ---------------------------------------------------------------- figure
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
    fig = plt.figure(figsize=(W, W * 0.40))
    ax = fig.add_axes([0.11, 0.22, 0.85, 0.68])
    data = [res[t]["conc"] for t, _, _ in CELLS]
    shuf = [res[t]["shuf"] for t, _, _ in CELLS]
    pos = np.arange(4)
    bp = ax.boxplot(data, positions=pos - 0.17, widths=.3, patch_artist=True,
                    showfliers=False, medianprops=dict(color="0.2", lw=1.1))
    for b_, t in zip(bp["boxes"], CELLS):
        b_.set(facecolor=GREEN if t[1] == "correct" else TERRA, lw=0, alpha=.9)
    bs = ax.boxplot(shuf, positions=pos + 0.17, widths=.3, patch_artist=True,
                    showfliers=False, medianprops=dict(color="0.2", lw=1.1))
    for b_ in bs["boxes"]:
        b_.set(facecolor="0.85", lw=0)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{t[1]}\n{t[2]}" for t in CELLS], fontsize=7)
    ax.set_ylabel(f"attribution concentration\n(best {args.k} nt window)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend([bp["boxes"][0], bs["boxes"][0]], ["observed", "dinucleotide-shuffled"],
              fontsize=7, frameon=False, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"XAI_2x2_motif.{ext}"), dpi=300)
    plt.close(fig)

    json.dump(dict(cells=summ, tests=tests, usable_classes=[RBP27[c] for c in usable],
                   threshold=args.threshold, k=args.k, raw=res),
              open(os.path.join(args.out, "xai_2x2.json"), "w"), indent=1, default=float)
    print(f"\nwrote {os.path.join(args.out, 'XAI_2x2_motif.pdf')}")
    print(f"wrote {os.path.join(args.out, 'xai_2x2.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
