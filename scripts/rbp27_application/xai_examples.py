"""Example attribution maps: confident-correct, uncertain-correct, confident-wrong.

Sequences are the median of each group by attribution concentration rather than the
sharpest available, so the figure is representative rather than best-case. Needs a GPU.

    python scripts/rbp27_application/xai_examples.py \
        --model /path/to/bnn_model_v_10_s_9 \
        --data  /path/to/dataset.pickle \
        --preds /path/to/iter_preds.pickle \
        --out   out/xai_examples
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
BASES = "ACGT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="out/xai_examples")
    ap.add_argument("--classes", default="0,2,3")
    ap.add_argument("--n-per-cell", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--m-steps", type=int, default=50)
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import pickle
    import tensorflow as tf
    import tensorflow_probability as tfp                     # noqa: F401
    if args.eager:
        tf.config.run_functions_eagerly(True)
    import xai_uncertainty as XU                             # reuse IG + measures

    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    x_test = np.asarray(d["x_test"]).astype("float32")
    y_test = np.asarray(d["y_test"])
    with open(args.preds, "rb") as fh:
        raw = np.asarray(pickle.load(fh))[:args.n_iter]
    pr = XU.softmax(raw)
    mean, sd = pr.mean(0), pr.std(0)
    true, pred = y_test.argmax(1), mean.argmax(1)
    sd_pred = sd[np.arange(len(sd)), pred]
    correct = pred == true
    keep = sd_pred <= args.threshold

    try:
        import tf_keras
        model = tf_keras.models.load_model(args.model, compile=False)
    except Exception:
        model = tf.keras.models.load_model(args.model, compile=False)
    print("model loaded")

    CELLS = [("correct, confident", correct & keep),
             ("correct, unsure", correct & ~keep),
             ("wrong, confident", (~correct) & keep)]
    sel = [int(c) for c in args.classes.split(",")]
    rng = np.random.default_rng(0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from uq_genomics.figures import style as S
        S.apply_rc(); W = S.TEXTWIDTH_IN
    except Exception:
        W = 6.5

    chosen = []
    for c in sel:
        nm = RBP27[c] if c < 27 else str(c)
        picks = []
        for cell_name, mask in CELLS:
            idx = np.where(mask & (true == c))[0]
            if len(idx) == 0:
                continue
            # score a random subset, then take the MEDIAN sample, not the sharpest
            cand = idx if len(idx) <= 40 else rng.choice(idx, 40, replace=False)
            sc = []
            for i in cand:
                a = XU.integrated_gradients(model, x_test[i], int(pred[i]),
                                            args.m_steps, mc_samples=3)
                sc.append(XU.concentration(a, args.window))
            i_med = cand[int(np.argsort(sc)[len(sc) // 2])]
            picks.append((cell_name, int(i_med)))
        if not picks:
            continue

        # build the rows first: profile on the left, zoomed logo on the right
        rows = []
        for cell_name, i in picks:
            att = XU.integrated_gradients(model, x_test[i], int(pred[i]),
                                          args.m_steps, mc_samples=args.mc_samples)
            rows.append((cell_name, x_test[i], att,
                         f"SD={sd_pred[i]:.3f}   "
                         f"p({RBP27[pred[i]] if pred[i] < 27 else pred[i]})"
                         f"={mean[i, pred[i]]:.2f}   "
                         f"conc.={XU.concentration(att, args.window):.3f}"))
            chosen.append(dict(cls=int(c), cell=cell_name, idx=int(i),
                               sd=float(sd_pred[i])))
        i0 = picks[0][1]
        xs = XU.dinuc_shuffle(x_test[i0], rng)
        att_s = XU.integrated_gradients(model, xs, int(pred[i0]), args.m_steps,
                                        mc_samples=args.mc_samples)
        rows.append(("shuffled control", xs, att_s,
                     f"dinucleotide-shuffled copy of the first sequence   "
                     f"conc.={XU.concentration(att_s, args.window):.3f}"))

        n = len(rows)
        fig = plt.figure(figsize=(W, 0.92 * n + 0.55), dpi=300)
        TOP, BOT, VG = 0.93, 0.10, 0.055
        h = (TOP - BOT - VG * (n - 1)) / n
        for r, (cell_name, seq, att, title) in enumerate(rows):
            y = TOP - (r + 1) * h - r * VG
            axP = fig.add_axes([0.085, y, 0.60, h])
            p = _draw(axP, att, XU, args.window, xlabels=(r == n - 1))
            axP.set_ylabel(cell_name.replace(" ", "\n", 1), fontsize=7)
            # title sits above its own axes, so it can no longer land on the
            # tick labels of the panel above
            axP.set_title(f"{nm}   {title}", fontsize=6.5, loc="left", pad=2)
            axL = fig.add_axes([0.72, y, 0.265, h])
            _logo(axL, att, seq, (p + args.window // 2) if p is not None else 75)
            if r == 0:
                ttl = "attribution of the observed base"
                if nm in KNOWN_MOTIF:
                    ttl += f"\nknown {nm} motif: {KNOWN_MOTIF[nm]}"
                axL.set_title(ttl, fontsize=6, pad=2)
            if r == n - 1:
                axP.set_xlabel("position in the 150 nt window", fontsize=7)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.out, f"XAI_example_{nm}.{ext}"), dpi=300)
        plt.close(fig)
        print(f"  wrote XAI_example_{nm}.pdf")

    json.dump(chosen, open(os.path.join(args.out, "examples.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(args.out, 'examples.json')}")
    return 0


# Standard nucleotide colouring, matching the motif panel in the supplement.
BASE_COLOR = {"A": "#3DA02D", "C": "#2B6CB0", "G": "#E8A33D",
              "T": "#D6473E", "U": "#D6473E"}

# These are RNA sequences from PAR-CLIP, and the motif figure in the supplement is
# drawn with U. Labelling the fourth base T here would print GCTT where the reader is
# looking for GCUU, and the whole point of the panel is that the two can be compared
# by eye.
ALPHABET = "ACGU"

# Published consensus for the RBPs with a well-characterised motif, shown beside the
# logo so the comparison does not depend on the reader remembering the other figure.
KNOWN_MOTIF = {"MBNL1": "GCUU / GCUUGC", "PUM2": "UGUA", "QKI": "CUAAC"}


def _letter(ax, base, x, y, height, width=0.92):
    """Draw one nucleotide glyph scaled to `height`, anchored at (x, y)."""
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    from matplotlib.font_manager import FontProperties
    if height == 0:
        return
    tp = TextPath((0, 0), base, size=1,
                  prop=FontProperties(family="DejaVu Sans", weight="bold"))
    bb = tp.get_extents()
    if bb.width == 0 or bb.height == 0:
        return
    t = (Affine2D()
         .translate(-bb.x0, -bb.y0)
         .scale(1.0 / bb.width, 1.0 / bb.height)
         .scale(width, height)
         .translate(x + (1 - width) / 2, y))
    ax.add_patch(PathPatch(t.transform_path(tp),
                           facecolor=BASE_COLOR.get(base, "0.4"), edgecolor="none"))


def _draw(ax, att, XU, k=8, xlabels=False):
    """Attribution summed over bases, per position, with the best window shaded."""
    imp = np.abs(att).sum(-1)
    ax.bar(np.arange(len(imp)), imp, width=1.0, color="0.30", linewidth=0)
    p = None
    if len(imp) >= k:
        p = int(np.convolve(imp, np.ones(k), "valid").argmax())
        ax.axvspan(p, p + k, color="#FC8D62", alpha=0.35, zorder=0)
    ax.set_xlim(0, len(imp))
    ax.set_yticks([])
    # x tick labels only on the bottom row: with one label set per panel they
    # collided with the title of the panel below
    ax.tick_params(labelsize=6, labelbottom=xlabels)
    return p


def _logo(ax, att, seq_onehot, centre, half=9):
    """Sequence logo of the attribution around `centre`."""
    lo = max(0, centre - half)
    hi = min(att.shape[0], centre + half)
    seq = seq_onehot.argmax(-1)
    vals = att[np.arange(len(seq)), seq]          # attribution of the observed base
    seg = vals[lo:hi]
    for j, v in enumerate(seg):
        _letter(ax, ALPHABET[seq[lo + j]], j, 0 if v >= 0 else v, abs(v))
    m = np.abs(seg).max() or 1.0
    ax.set_xlim(0, len(seg))
    ax.set_ylim(min(seg.min(), 0) * 1.1 - 1e-9, max(seg.max(), 0) * 1.1 + 1e-9)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


if __name__ == "__main__":
    sys.exit(main())
