"""Attribution maps for simulated sequences carrying a second, unplanted motif.

Scans each sequence against the PWMs that generated it and marks every hit, so the
contamination is visible rather than asserted. Needs a GPU and the trained model.

    python scripts/simulated_tf/simdna_xai_figure.py \
        --model  /path/to/bnn_model \
        --data   /path/to/simulated.pickle \
        --motifs /path/to/rbp_motif_matrix_binary \
        --out    out/simdna_xai
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

ALPHABET = "ACGU"

# Class order as the SimDNA generation notebook built it. The 3-class datasets are the
# first three of this list, the 6-class ones are all six, so the same order serves both
# and the number of classes is read from the data rather than assumed.
RBP_ORDER = ["mbnl1", "pum2", "qki", "elavl1", "rbfox2", "tardbp"]

MOTIF_DIRS = {}      # filled from the data: {class index -> directory name}
CLASS_NAMES = {}     # {class index -> display name}


def set_classes(n):
    if n > len(RBP_ORDER):
        sys.exit(f"ERROR: data has {n} classes but only {len(RBP_ORDER)} RBPs are known: "
                 f"{RBP_ORDER}. Add the missing names to RBP_ORDER.")
    MOTIF_DIRS.clear(); CLASS_NAMES.clear()
    for i, name in enumerate(RBP_ORDER[:n]):
        MOTIF_DIRS[i] = name
        CLASS_NAMES[i] = name.upper()
    print(f"  {n} classes: {', '.join(CLASS_NAMES[i] for i in range(n))}")

BASE_COLOR = {"A": "#3DA02D", "C": "#2B6CB0", "G": "#E8A33D", "U": "#D6473E"}


# --------------------------------------------------------------------------- motifs
def load_pwms(root):
    """Read the binary PWM text files used to generate the data."""
    pwms = {}
    for cls, name in MOTIF_DIRS.items():
        files = sorted(glob.glob(os.path.join(root, name, "*.txt")))
        if not files:
            print(f"  warning: no PWM found for {name}")
            continue
        rows = [l.split() for l in open(files[0]).read().strip().split("\n")[1:]]
        m = np.array([[float(x) for x in r[1:5]] for r in rows])
        m = (m + 0.001) / (m + 0.001).sum(1, keepdims=True)      # pseudocount
        pwms[cls] = m
        cons = "".join(ALPHABET[i] for i in m.argmax(1))
        print(f"  {name:6s} len={m.shape[0]:2d} consensus={cons}  ({os.path.basename(files[0])})")
    return pwms


def scan(seq_idx, pwm, frac=0.80):
    """Positions where `pwm` scores above `frac` of its own maximum."""
    lo = np.log(pwm / 0.25)
    L, n = len(pwm), len(seq_idx)
    best = lo.max(1).sum()
    worst = lo.min(1).sum()
    hits = []
    for s in range(n - L + 1):
        sc = lo[np.arange(L), seq_idx[s:s + L]].sum()
        f = (sc - worst) / (best - worst)
        if f >= frac:
            hits.append((s, f))
    out = []
    for s, f in sorted(hits, key=lambda t: -t[1]):
        if all(abs(s - s2) >= L for s2, _ in out):
            out.append((s, f))
    return sorted(out)


# ------------------------------------------------------------------ integrated grads
def integrated_gradients(model, x, target, m_steps=50, mc_samples=20, batch=64):
    """IG with a zero baseline and the trapezoidal rule."""
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
        avg = tf.reduce_mean((g[:-1] + g[1:]) / 2.0, axis=0)
        total += (x - baseline) * avg
    return (total / mc_samples).numpy()


def softmax(a, axis=-1):
    z = a - a.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------------- plotting
def letter(ax, base, x, y, height, width=0.92):
    """One nucleotide glyph scaled to `height`. Font size is in points and does not
    follow the data transform, so the character has to become a path."""
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
    t = (Affine2D().translate(-bb.x0, -bb.y0)
         .scale(1.0 / bb.width, 1.0 / bb.height)
         .scale(width, height)
         .translate(x + (1 - width) / 2, y))
    ax.add_patch(PathPatch(t.transform_path(tp),
                           facecolor=BASE_COLOR.get(base, "0.4"), edgecolor="none"))


def draw_panel(ax, att, seq_idx, hits, label_cls, pred_cls, sd, prob, zoom=None):
    """Attribution logo with every detected motif boxed and named."""
    vals = att[np.arange(len(seq_idx)), seq_idx]      # attribution of the base present
    lo, hi = zoom if zoom else (0, len(seq_idx))
    seg = vals[lo:hi]
    for j, v in enumerate(seg):
        letter(ax, ALPHABET[seq_idx[lo + j]], j, 0 if v >= 0 else v, abs(v))

    span = max(abs(seg.min()), abs(seg.max())) or 1.0

    # Motif captions sit above the trace. The panel title also starts at the left edge,
    # so a motif near the start of the sequence collided with it. Reserve the leftmost
    # stretch for the title and push any caption that would land there onto a second,
    # lower line instead of overlapping.
    TITLE_GUARD = len(seg) * 0.34
    drawn = []
    for cls, s, f in hits:
        if not (lo <= s < hi):
            continue
        on_label = cls == label_cls
        x0, w = s - lo, MOTIF_LEN[cls]
        ax.axvspan(x0, x0 + w, color="#2CA089" if on_label else "#D6473E",
                   alpha=0.16, zorder=0, linewidth=0)
        cx = x0 + w / 2
        # second line if it would sit under the title, or too close to a caption
        # already placed on the top line
        low = cx < TITLE_GUARD or any(abs(cx - p) < len(seg) * 0.18 for p in drawn)
        if not low:
            drawn.append(cx)
        ax.text(cx, span * (1.02 if low else 1.16),
                f"{CLASS_NAMES[cls]}" + ("" if on_label else "  (not the label)"),
                ha="center", va="bottom", fontsize=6,
                color="#1A6B5C" if on_label else "#A32A22")

    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(0, len(seg))
    ax.set_ylim(-span * 1.15, span * 1.42)
    ax.set_yticks([])
    ax.tick_params(labelsize=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel(f"labeled {CLASS_NAMES[label_cls]}", fontsize=7)
    # Say plainly when the prediction followed the off-label motif; that is the point
    # of the figure and should not need decoding from two class names.
    tail = "  <- followed the wrong motif" if pred_cls != label_cls else ""
    ax.set_title(f"predicted {CLASS_NAMES[pred_cls]}   p={prob:.2f}   SD={sd:.3f}{tail}",
                 fontsize=6.5, loc="left", pad=2,
                 color="#A32A22" if tail else "black")


MOTIF_LEN = {}


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained BNN on the 3-class sim data")
    ap.add_argument("--data", required=True, help="ds*_balanced_singlelabel_*3class_simulated.pickle")
    ap.add_argument("--motifs", required=True, help="rbp_motif_matrix_binary directory")
    ap.add_argument("--out", default="out/simdna_xai")
    ap.add_argument("--n-panels", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=100, help="MC forward passes for the SD")
    ap.add_argument("--m-steps", type=int, default=50)
    ap.add_argument("--mc-samples", type=int, default=20)
    # 0.90 is calibrated, not guessed: on the 3-class balanced set it detects the
    # planted motif in ~100% of sequences (200/200 per class) while flagging only
    # near-exact off-label matches. At 0.85 the on-label count exceeds the number of
    # sequences, i.e. false positives have started. The script prints both rates so
    # this can be re-checked on a different dataset.
    ap.add_argument("--pwm-frac", type=float, default=0.90)
    ap.add_argument("--zoom", type=int, default=0,
                    help="if >0, crop each panel to this many nt around the hits")
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import pickle
    import tensorflow as tf
    import tensorflow_probability as tfp                          # noqa: F401
    if args.eager:
        tf.config.run_functions_eagerly(True)

    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    x_test = np.asarray(d["x_test"]).astype("float32")
    y_test = np.asarray(d["y_test"])
    label = y_test.argmax(1)

    set_classes(y_test.shape[1])
    print("motifs:")
    pwms = load_pwms(args.motifs)
    for c, m in pwms.items():
        MOTIF_LEN[c] = m.shape[0]

    print(f"\ntest set: {x_test.shape[0]} sequences, {x_test.shape[1]} nt, "
          f"{y_test.shape[1]} classes")

    try:
        import tf_keras
        model = tf_keras.models.load_model(args.model, compile=False)
    except Exception:
        model = tf.keras.models.load_model(args.model, compile=False)
    print("model loaded")

    # fail fast if gradients do not flow -- a silently non-differentiable load gives
    # an all-zero attribution map that looks like a real (null) result
    g = integrated_gradients(model, x_test[0], int(label[0]), m_steps=4, mc_samples=1)
    if not np.isfinite(g).all() or np.abs(g).max() == 0:
        sys.exit("ERROR: gradients are zero or non-finite; the model did not load for "
                 "differentiation. Check TF_USE_LEGACY_KERAS and the TFP version.")

    # --- uncertainty: softmax per iteration, then SD. The SD of logits is not the
    # --- SD of probabilities, and the threshold in the paper is on probabilities.
    print(f"running {args.n_iter} MC forward passes ...")
    it = np.stack([softmax(model(x_test, training=True).numpy())
                   for _ in range(args.n_iter)])
    mean, sd = it.mean(0), it.std(0)
    pred = mean.argmax(1)
    sd_pred = sd[np.arange(len(sd)), pred]
    p_pred = mean[np.arange(len(mean)), pred]
    print(f"  SD at predicted class: median {np.median(sd_pred):.4f}, "
          f"max {sd_pred.max():.4f}")

    # Confirm this model was trained on this dataset. A model from another run will
    # load and produce attribution maps quite happily; the maps will just be
    # meaningless. Chance is 1/n_classes.
    if mean.shape[1] != y_test.shape[1]:
        sys.exit(f"ERROR: the model outputs {mean.shape[1]} classes but the data has "
                 f"{y_test.shape[1]}. Wrong model for this dataset.")
    acc = float((pred == label).mean())
    print(f"  test accuracy: {acc:.3f}  (chance {1 / y_test.shape[1]:.3f})")
    if acc < 0.6:
        sys.exit(f"ERROR: accuracy {acc:.3f} is too low for this to be the model "
                 f"trained on this dataset. Check --model and --data match.")

    # --- find sequences carrying a motif that is not their label
    print("\nscanning for off-label motifs ...")
    seqs = x_test.argmax(-1).astype(int)
    contaminated, n_on, n_off = [], 0, 0
    for i in range(len(seqs)):
        hits = [(c, s, f) for c, m in pwms.items() for s, f in scan(seqs[i], m, args.pwm_frac)]
        on = [h for h in hits if h[0] == label[i]]
        off = [h for h in hits if h[0] != label[i]]
        n_on += len(on)
        n_off += len(off)
        if off:
            contaminated.append((i, hits))

    # Sanity check on the threshold. Every sequence had exactly one motif planted, so
    # on-label hits should be close to one per sequence. Far below means the threshold
    # is too strict and real contamination is being missed; far above means false
    # positives, and the off-label count is then inflated.
    rate = n_on / len(seqs)
    print(f"  on-label detection: {n_on} hits over {len(seqs)} sequences "
          f"({rate:.2f} per sequence)")
    if rate < 0.9:
        print("    -> threshold looks too strict; try a lower --pwm-frac")
    elif rate > 1.15:
        print("    -> false positives likely; try a higher --pwm-frac")
    print(f"  {len(contaminated)} / {len(seqs)} sequences carry a motif of another class "
          f"({100 * len(contaminated) / len(seqs):.1f}%), {n_off} off-label hits total")
    if not contaminated:
        sys.exit("No off-label motifs found. Lower --pwm-frac, or check that --motifs "
                 "points at the PWMs actually used to generate this dataset.")

    # The figure's claim is that an unplanted motif misleads the model, so prefer
    # sequences where the prediction actually went to the class of the off-label motif.
    # A contaminated sequence that was classified correctly shows the contamination but
    # not the consequence, which is the weaker picture.
    misled = []
    for i, hits in contaminated:
        off = {c for c, _, _ in hits if c != label[i]}
        if pred[i] in off:
            misled.append((i, hits))
    print(f"  of these, {len(misled)} were predicted as the off-label class")

    pool = misled if len(misled) >= args.n_panels else contaminated
    if pool is contaminated and misled:
        print(f"    (only {len(misled)} available; falling back to all contaminated)")

    # most uncertain first, then spread rather than taking the extreme tail
    pool = sorted(pool, key=lambda t: -sd_pred[t[0]])
    top = pool[:max(args.n_panels * 4, 12)]
    step = max(1, len(top) // args.n_panels)
    chosen = top[::step][:args.n_panels]

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from uq_genomics.figures import style as S
        S.apply_rc()
        W = S.TEXTWIDTH_IN
    except Exception:
        W = 6.5
        matplotlib.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                                    "pdf.fonttype": 42, "font.size": 9,
                                    "font.family": "sans-serif"})

    n = len(chosen)
    fig = plt.figure(figsize=(W, 1.15 * n + 0.45))
    TOP, BOT, VG = 0.90, 0.11, 0.10
    h = (TOP - BOT - VG * (n - 1)) / n
    records = []
    for r, (i, hits) in enumerate(chosen):
        att = integrated_gradients(model, x_test[i], int(pred[i]),
                                   args.m_steps, args.mc_samples)
        ax = fig.add_axes([0.085, TOP - (r + 1) * h - r * VG, 0.895, h])
        zoom = None
        if args.zoom > 0:
            c = int(np.mean([s for _, s, _ in hits]))
            zoom = (max(0, c - args.zoom // 2),
                    min(seqs.shape[1], c + args.zoom // 2))
        draw_panel(ax, att, seqs[i], hits, int(label[i]), int(pred[i]),
                   sd_pred[i], p_pred[i], zoom)
        if r == n - 1:
            ax.set_xlabel("position in the 150 nt window", fontsize=7)
        records.append(dict(idx=int(i), label=CLASS_NAMES[int(label[i])],
                            pred=CLASS_NAMES[int(pred[i])], sd=float(sd_pred[i]),
                            prob=float(p_pred[i]),
                            hits=[dict(cls=CLASS_NAMES[c], start=int(s),
                                       score=round(float(f), 3)) for c, s, f in hits]))
        print(f"  panel {r + 1}: sample {i}, labeled {CLASS_NAMES[int(label[i])]}, "
              f"SD={sd_pred[i]:.3f}, hits="
              f"{[(CLASS_NAMES[c], s) for c, s, _ in hits]}")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"XAI_with_UQ.{ext}"))
    plt.close(fig)

    json.dump(dict(dataset=os.path.basename(args.data),
                   model=os.path.basename(args.model.rstrip("/")),
                   n_iter=args.n_iter, m_steps=args.m_steps,
                   mc_samples=args.mc_samples, pwm_frac=args.pwm_frac,
                   n_contaminated=len(contaminated), n_test=len(seqs),
                   panels=records),
              open(os.path.join(args.out, "examples.json"), "w"), indent=1)

    print(f"\nwrote {os.path.join(args.out, 'XAI_with_UQ.pdf')}")
    print(f"wrote {os.path.join(args.out, 'examples.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
