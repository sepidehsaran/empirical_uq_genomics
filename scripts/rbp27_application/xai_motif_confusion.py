"""Does attribution explain confident errors, and do complex members share motifs?

Derives each class's motif from the sequences the model classifies confidently and
correctly, then scores errors against those motifs. Needs a GPU and the trained model.

    python scripts/rbp27_application/xai_motif_confusion.py \
        --model /path/to/bnn_model_v_10_s_9 \
        --data  /path/to/dataset.pickle \
        --preds /path/to/iter_preds.pickle \
        --out   out/motif_confusion
"""
import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

ALPHABET = "ACGU"
RBP27 = ["MBNL1", "P53_NONO", "PUM2", "QKI", "AGO3", "FUS", "TAF15", "ZFP36",
         "DICER1", "EIF3A", "EIF3D", "EIF3G", "SSB", "PAPD5", "CPSF4", "CPSF3",
         "RTCB", "FXR1", "NOP58", "NOP56", "FBL", "LIN28A", "LIN28B", "UPF1",
         "G35", "G45", "XPO5"]

# Complexes and families among these 27. Membership from the literature, not the data.
GROUPS = {
    "eIF3 complex":   ["EIF3A", "EIF3D", "EIF3G"],
    "FET family":     ["FUS", "TAF15"],
    "CPSF complex":   ["CPSF3", "CPSF4"],
    "box C/D snoRNP": ["FBL", "NOP56", "NOP58"],
    "LIN28 paralogs": ["LIN28A", "LIN28B"],
}
# box C/D is excluded from the motif tests by default: its specificity comes from
# snoRNA base-pairing rather than direct sequence recognition, so there is no shared
# motif for the model to conflate and it would dilute the effect. It dominates the
# within-complex errors by count (131 of 311), so this matters.
MOTIF_GROUPS = ["eIF3 complex", "FET family", "CPSF complex", "LIN28 paralogs"]


def softmax(a, axis=-1):
    z = a - a.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------ integrated grads
def integrated_gradients(model, x, target, m_steps=50, mc_samples=10, batch=64):
    """IG with a zero baseline, trapezoidal rule, averaged over posterior draws."""
    import tensorflow as tf
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    baseline = tf.zeros_like(x)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    total = tf.zeros_like(x)
    for _ in range(mc_samples):
        grads = []
        for i in range(0, m_steps + 1, batch):
            a = alphas[i:i + batch][:, None, None]
            pts = baseline[None] + a * (x - baseline)[None]
            with tf.GradientTape() as tape:
                tape.watch(pts)
                out = model(pts)
                out = tf.nn.softmax(out) if out.shape[-1] > 1 else out
                sel = out[:, target]
            grads.append(tape.gradient(sel, pts))
        g = tf.concat(grads, axis=0)
        total += (x - baseline) * tf.reduce_mean((g[:-1] + g[1:]) / 2.0, axis=0)
    return (total / mc_samples).numpy()


# --------------------------------------------------------------------- motif machinery
def best_window(att, k=8):
    """Start index of the k-nt window carrying the most absolute attribution."""
    imp = np.abs(att).sum(-1)
    return int(np.convolve(imp, np.ones(k), "valid").argmax())


def top_windows(att, k=8, n=2, min_gap=None):
    """The n highest-scoring non-overlapping windows, with their scores."""
    imp = np.abs(att).sum(-1)
    conv = np.convolve(imp, np.ones(k), "valid")
    min_gap = min_gap or k
    out, taken = [], []
    for s in np.argsort(-conv):
        if all(abs(int(s) - t) >= min_gap for t in taken):
            taken.append(int(s))
            out.append((int(s), float(conv[s])))
        if len(out) == n:
            break
    return out


def pfm_from_windows(seqs, starts, k=8):
    """Position frequency matrix built from the aligned high-attribution windows."""
    m = np.zeros((k, 4))
    for s, st in zip(seqs, starts):
        w = s[st:st + k]
        if len(w) < k:
            continue
        for j, b in enumerate(w):
            m[j, b] += 1
    tot = m.sum(1, keepdims=True)
    tot[tot == 0] = 1
    return m / tot


def pfm_similarity(p, q):
    """Similarity of two PFMs, allowing a shift, as the best mean column correlation."""
    best = -1.0
    k = len(p)
    for off in range(-(k // 2), k // 2 + 1):
        cols = []
        for i in range(k):
            j = i + off
            if 0 <= j < k:
                a, b = p[i] - 0.25, q[j] - 0.25
                d = np.linalg.norm(a) * np.linalg.norm(b)
                cols.append(float(a @ b / d) if d > 0 else 0.0)
        if cols:
            best = max(best, float(np.mean(cols)))
    return best


def score_window(window_bases, pfm):
    """Log-odds of one window under a PFM, against a uniform background."""
    lo = np.log((pfm + 0.01) / (1 + 0.04) / 0.25)
    return float(sum(lo[j, b] for j, b in enumerate(window_bases[:len(pfm)])))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="out/motif_confusion")
    ap.add_argument("--k", type=int, default=8, help="motif window width")
    ap.add_argument("--n-proto", type=int, default=120,
                    help="confident correct sequences per class used to build its motif")
    ap.add_argument("--n-err", type=int, default=80, help="errors scored per group")
    ap.add_argument("--n-unc", type=int, default=150, help="samples per uncertainty bin")
    ap.add_argument("--m-steps", type=int, default=50)
    ap.add_argument("--mc-samples", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.1, help="the SD filter in the paper")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import pickle
    import tensorflow as tf
    import tensorflow_probability as tfp                     # noqa: F401
    from scipy.stats import mannwhitneyu

    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    x = np.asarray(d["x_test"]).astype("float32")
    y = np.asarray(d["y_test"])
    with open(args.preds, "rb") as fh:
        raw = np.asarray(pickle.load(fh))[:args.n_iter]
    pr = softmax(raw)
    mean, sd = pr.mean(0), pr.std(0)
    true, pred = y.argmax(1), mean.argmax(1)
    sd_pred = sd[np.arange(len(sd)), pred]
    seqs = x.argmax(-1).astype(int)
    correct = pred == true
    confident = sd_pred < args.threshold
    print(f"{len(true)} sequences | correct {correct.sum()} | "
          f"confident {confident.sum()} | confident+wrong {(confident & ~correct).sum()}")

    try:
        import tf_keras
        model = tf_keras.models.load_model(args.model, compile=False)
    except Exception:
        model = tf.keras.models.load_model(args.model, compile=False)
    if mean.shape[1] != y.shape[1]:
        sys.exit(f"ERROR: model outputs {mean.shape[1]} classes, data has {y.shape[1]}")
    g = integrated_gradients(model, x[0], int(pred[0]), m_steps=4, mc_samples=1)
    if not np.isfinite(g).all() or np.abs(g).max() == 0:
        sys.exit("ERROR: gradients are zero or non-finite; the model did not load "
                 "for differentiation.")
    print("model loaded, gradients flow\n")

    IG = lambda i, c: integrated_gradients(model, x[i], int(c),
                                           args.m_steps, args.mc_samples)
    results = {}

    # ---------------------------------------------------------------- 1. class motifs
    # The motif the model learned for each class, from sequences it gets right AND is
    # confident about. Confidence matters: an uncertain correct prediction may be right
    # for the wrong reason, which would blur the motif.
    print("=" * 70)
    print("1. Learned motif per class (confident + correct sequences)")
    print("=" * 70)
    pfms, n_used = {}, {}
    for c in range(y.shape[1]):
        pool = np.where(correct & confident & (true == c))[0]
        if len(pool) < 20:
            continue
        take = rng.choice(pool, min(args.n_proto, len(pool)), replace=False)
        starts, ss = [], []
        for i in take:
            starts.append(best_window(IG(i, c), args.k))
            ss.append(seqs[i])
        pfms[c] = pfm_from_windows(ss, starts, args.k)
        n_used[c] = len(take)
        cons = "".join(ALPHABET[j] for j in pfms[c].argmax(1))
        ic = float((pfms[c] * np.log2((pfms[c] + 1e-9) / 0.25)).sum())
        print(f"  {RBP27[c]:9s} n={len(take):4d}  consensus={cons}  information={ic:5.2f} bits")
    print(f"\n  motifs built for {len(pfms)} of {y.shape[1]} classes")

    # --------------------------------------------- 2. within vs across class similarity
    # The premise of the whole complex story: proteins in one complex should have more
    # similar learned motifs than proteins picked at random.
    print("\n" + "=" * 70)
    print("2. Motif similarity: same complex vs different complex")
    print("=" * 70)
    idx = {n: i for i, n in enumerate(RBP27)}
    same_pairs = {tuple(sorted((idx[a], idx[b])))
                  for gname in MOTIF_GROUPS
                  for a, b in combinations(GROUPS[gname], 2)}
    within, between, rows = [], [], []
    for a, b in combinations(sorted(pfms), 2):
        s = pfm_similarity(pfms[a], pfms[b])
        grp = tuple(sorted((a, b))) in same_pairs
        (within if grp else between).append(s)
        rows.append(dict(a=RBP27[a], b=RBP27[b], sim=round(s, 4), same_complex=grp))
    if within and between:
        u, p = mannwhitneyu(within, between, alternative="greater")
        print(f"  same complex      n={len(within):4d}  median {np.median(within):+.3f}")
        print(f"  different complex n={len(between):4d}  median {np.median(between):+.3f}")
        print(f"  Mann-Whitney one-sided p = {p:.3g}")
        results["motif_similarity"] = dict(
            within=within, between=between, p=float(p), pairs=rows)
        print("\n  most similar pairs:")
        for r in sorted(rows, key=lambda r: -r["sim"])[:8]:
            print(f"    {r['a']:9s} {r['b']:9s} {r['sim']:+.3f}"
                  f"{'   <- same complex' if r['same_complex'] else ''}")

    # ------------------------------------- 3. confident errors: whose motif did it read?
    # The direct test of "it saw the other protein's motif". For each confident error,
    # take the window the model attended to and score it under the true class motif and
    # under the predicted class motif. If the predicted-class motif scores higher, the
    # attribution explains the error.
    print("\n" + "=" * 70)
    print("3. Confident errors: does the attended window match the PREDICTED class?")
    print("=" * 70)
    err = np.where(confident & ~correct)[0]
    err = np.array([i for i in err if true[i] in pfms and pred[i] in pfms])
    same_cx = np.array([tuple(sorted((true[i], pred[i]))) in same_pairs for i in err])
    print(f"  confident errors with motifs for both classes: {len(err)} "
          f"({int(same_cx.sum())} within complex)")

    out3 = {}
    for lab, mask in (("within complex", same_cx), ("other", ~same_cx)):
        pool = err[mask]
        if len(pool) < 10:
            print(f"  {lab}: only {len(pool)}, skipped")
            continue
        take = rng.choice(pool, min(args.n_err, len(pool)), replace=False)
        d_pred, d_true, wins = [], [], []
        for i in take:
            att = IG(i, pred[i])
            s = best_window(att, args.k)
            w = seqs[i][s:s + args.k]
            sp, st = score_window(w, pfms[pred[i]]), score_window(w, pfms[true[i]])
            d_pred.append(sp); d_true.append(st); wins.append(sp - st)
        frac = float(np.mean(np.array(wins) > 0))
        u, p = mannwhitneyu(d_pred, d_true, alternative="greater")
        print(f"  {lab:15s} n={len(take):4d}  window scores better under PREDICTED "
              f"class in {100*frac:.0f}% of cases  (p={p:.3g})")
        print(f"                   median log-odds  predicted {np.median(d_pred):+.2f}  "
              f"true {np.median(d_true):+.2f}")
        out3[lab] = dict(n=len(take), frac_pred_wins=frac, p=float(p),
                         d_pred=d_pred, d_true=d_true)
    results["confident_errors"] = out3

    # ------------------------------------------- 4. what does an uncertain sample look like
    # Two candidate explanations, tested against each other within class so class size
    # cannot drive the result:
    #   (a) no clear motif      -> weak attribution concentration
    #   (b) more than one motif -> two separated windows, both strong, and each
    #                              matching a DIFFERENT class's motif
    print("\n" + "=" * 70)
    print("4. Uncertain samples: no motif, or competing motifs?")
    print("=" * 70)
    conc_hi, conc_lo, comp_hi, comp_lo = [], [], [], []
    for c in sorted(pfms):
        pool = np.where(true == c)[0]
        if len(pool) < 40:
            continue
        s = sd_pred[pool]
        hi = pool[s >= np.quantile(s, 0.75)]
        lo = pool[s <= np.quantile(s, 0.25)]
        n = min(args.n_unc // max(1, len(pfms)) + 4, len(hi), len(lo))
        if n < 4:
            continue
        for grp, C, K in ((rng.choice(hi, n, False), conc_hi, comp_hi),
                          (rng.choice(lo, n, False), conc_lo, comp_lo)):
            for i in grp:
                att = IG(i, pred[i])
                imp = np.abs(att).sum(-1)
                tw = top_windows(att, args.k, n=2)
                C.append(float(sum(imp[tw[0][0]:tw[0][0] + args.k]) / (imp.sum() + 1e-12)))
                # competing evidence: second window nearly as strong as the first, and
                # the two windows preferred by different classes
                if len(tw) == 2 and tw[0][1] > 0:
                    ratio = tw[1][1] / tw[0][1]
                    w1 = seqs[i][tw[0][0]:tw[0][0] + args.k]
                    w2 = seqs[i][tw[1][0]:tw[1][0] + args.k]
                    b1 = max(pfms, key=lambda cc: score_window(w1, pfms[cc]))
                    b2 = max(pfms, key=lambda cc: score_window(w2, pfms[cc]))
                    K.append(float(ratio if b1 != b2 else 0.0))
                else:
                    K.append(0.0)
    if conc_hi and conc_lo:
        u1, p1 = mannwhitneyu(conc_hi, conc_lo, alternative="less")
        print(f"  (a) attribution concentration in the top window")
        print(f"      high-SD {np.median(conc_hi):.3f}   low-SD {np.median(conc_lo):.3f}"
              f"   (high < low? p={p1:.3g})")
        u2, p2 = mannwhitneyu(comp_hi, comp_lo, alternative="greater")
        print(f"  (b) competing-evidence score (2nd window strong AND preferred by "
              f"another class)")
        print(f"      high-SD {np.mean(comp_hi):.3f}   low-SD {np.mean(comp_lo):.3f}"
              f"   (high > low? p={p2:.3g})")
        results["uncertainty"] = dict(conc_hi=conc_hi, conc_lo=conc_lo,
                                      comp_hi=comp_hi, comp_lo=comp_lo,
                                      p_concentration=float(p1), p_competing=float(p2))

    # ------------------------------------------------------------------- null control
    # Shuffle class labels and rebuild the similarity comparison. Whatever the real
    # numbers are, this says what they would be with no structure at all.
    if "motif_similarity" in results:
        perm = rng.permutation(sorted(pfms))
        remap = dict(zip(sorted(pfms), perm))
        nw, nb = [], []
        for a, b in combinations(sorted(pfms), 2):
            s = pfm_similarity(pfms[a], pfms[b])
            (nw if tuple(sorted((remap[a], remap[b]))) in same_pairs else nb).append(s)
        if nw and nb:
            print(f"\n  null (labels permuted): same {np.median(nw):+.3f} vs "
                  f"different {np.median(nb):+.3f}")
            results["motif_similarity"]["null_within"] = nw
            results["motif_similarity"]["null_between"] = nb

    # ----------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from uq_genomics.figures import style as S
        S.apply_rc(); W, PAL = S.TEXTWIDTH_IN, list(S.PALETTE)
    except Exception:
        W, PAL = 6.5, ["#66C2A5", "#FC8D62", "#C4705A", "#8DA0CB"]
    GREEN, TERRA = "#66C2A5", "#C4705A"

    fig = plt.figure(figsize=(W, W * 0.34), dpi=300)
    axes = [fig.add_axes([0.085, 0.20, 0.24, 0.68]),
            fig.add_axes([0.415, 0.20, 0.24, 0.68]),
            fig.add_axes([0.745, 0.20, 0.24, 0.68])]

    def box(ax, groups, labels, colors, ylab, title):
        bp = ax.boxplot(groups, positions=range(len(groups)), widths=.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="0.2", lw=1.1))
        for b, c in zip(bp["boxes"], colors):
            b.set(facecolor=c, lw=0, alpha=.9)
        for i, v in enumerate(groups):
            ax.scatter(np.random.default_rng(0).normal(i, .06, len(v)), v,
                       s=3.5, color="0.25", alpha=.35, linewidths=0, zorder=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.tick_params(labelsize=6.5)

    ms = results.get("motif_similarity")
    if ms:
        box(axes[0], [ms["between"], ms["within"]],
            ["different\ncomplex", "same\ncomplex"], [PAL[2] if len(PAL) > 2 else TERRA, GREEN],
            "learned-motif similarity", "a)")
        axes[0].set_title(f"a)   p = {ms['p']:.1e}", fontsize=7.5, loc="left")
    ce = results.get("confident_errors", {})
    if "within complex" in ce:
        g = ce["within complex"]
        box(axes[1], [g["d_true"], g["d_pred"]],
            ["true\nclass", "predicted\nclass"], [GREEN, TERRA],
            "window log-odds", f"b)   p = {g['p']:.1e}")
    un = results.get("uncertainty")
    if un:
        box(axes[2], [un["conc_lo"], un["conc_hi"]],
            ["low\nuncertainty", "high\nuncertainty"], [GREEN, TERRA],
            "attribution concentration", f"c)   p = {un['p_concentration']:.1e}")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"XAI_motif_confusion.{ext}"), dpi=300)
    plt.close(fig)

    json.dump(results, open(os.path.join(args.out, "motif_confusion.json"), "w"),
              indent=1, default=float)
    print(f"\nwrote {os.path.join(args.out, 'XAI_motif_confusion.pdf')}")
    print(f"wrote {os.path.join(args.out, 'motif_confusion.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
