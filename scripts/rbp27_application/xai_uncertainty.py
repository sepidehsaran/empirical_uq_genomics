"""Does the model become uncertain where the sequence motif is not sharp?"""
import argparse
import json
import os
import pickle
import sys

import numpy as np


RBP27_NAMES = ["MBNL1","P53_NONO","PUM2","QKI","AGO3","FUS","TAF15","ZFP36",
               "DICER1","EIF3A","EIF3D","EIF3G","SSB","PAPD5","CPSF4","CPSF3",
               "RTCB","FXR1","NOP58","NOP56","FBL","LIN28A","LIN28B","UPF1",
               "G35","G45","XPO5"]


# --------------------------------------------------------------------- attribution
def entropy(imp_scores, pseudocount=0.001):
    """Positional entropy of an I x 4 or N x I x 4 attribution array."""
    imp = np.abs(imp_scores).sum(axis=-1)                  # collapse the 4 bases
    imp = imp + pseudocount
    imp = imp / imp.sum(axis=-1, keepdims=True)            # distribution over positions
    return -np.sum(imp * np.log2(imp), axis=-1)


def integrated_gradients(model, x, target, m_steps=50, batch_size=32, mc_samples=1,
                         return_draws=False):
    """IG for one sequence against a zero baseline."""
    import tensorflow as tf

    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    baseline = tf.zeros_like(x)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)

    draws = []
    for _ in range(mc_samples):
        grads = []
        for i in range(0, len(alphas), batch_size):
            a = alphas[i:i + batch_size][:, None, None]
            interp = baseline[None] + a * (x[None] - baseline[None])
            with tf.GradientTape() as tape:
                tape.watch(interp)
                out = model(interp)
                out = out[:, target]
            grads.append(tape.gradient(out, interp))
        g = tf.concat(grads, axis=0)
        # trapezoidal rule over the interpolation path
        g = (g[:-1] + g[1:]) / 2.0
        draws.append((tf.reduce_mean(g, axis=0) * (x - baseline)).numpy())
    draws = np.stack(draws)
    mean = draws.mean(0)
    return (mean, draws) if return_draws else mean


# ------------------------------------------------- explanation-level uncertainty
def explanation_consistency(draws):
    """Mean pairwise cosine similarity between the per-draw attribution maps."""
    if len(draws) < 2:
        return np.nan
    v = draws.reshape(len(draws), -1)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    sims = v @ v.T
    iu = np.triu_indices(len(draws), k=1)
    return float(sims[iu].mean())


def peak_agreement(draws, k=8):
    """Fraction of draws whose sharpest k-nt window falls at the modal position."""
    if len(draws) < 2:
        return np.nan
    peaks = []
    for d in draws:
        imp = np.abs(d).sum(axis=-1)
        if len(imp) < k:
            peaks.append(0)
            continue
        peaks.append(int(np.convolve(imp, np.ones(k), mode="valid").argmax()))
    peaks = np.asarray(peaks)
    # within +/- k/2 of the modal peak counts as the same location
    vals, counts = np.unique(peaks, return_counts=True)
    modal = vals[counts.argmax()]
    return float(np.mean(np.abs(peaks - modal) <= k // 2))


def _to_onehot(seq, like):
    z = np.zeros_like(like)
    z[np.arange(len(seq)), np.asarray(seq)] = 1.0
    return z


def mono_shuffle(onehot, rng):
    """Random permutation of positions. Destroys every motif, keeps base composition."""
    seq = onehot.argmax(-1).copy()
    rng.shuffle(seq)
    return _to_onehot(seq, onehot)


def dinuc_shuffle(onehot, rng, max_tries=25):
    """Dinucleotide-preserving shuffle (Altschul & Erikson)."""
    seq = onehot.argmax(-1)
    n = len(seq)
    last = seq[-1]
    for _ in range(max_tries):
        edges = {}
        for a, b in zip(seq[:-1], seq[1:]):
            edges.setdefault(int(a), []).append(int(b))
        for k in edges:
            rng.shuffle(edges[k])
        # each vertex's final edge must reach `last` by following final edges
        ok = True
        for v in edges:
            if v == last:
                continue
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in edges:
                    ok = False
                    break
                seen.add(cur)
                cur = edges[cur][-1]
            if not ok:
                break
        if not ok:
            continue
        avail = {k: list(v) for k, v in edges.items()}
        out, cur = [int(seq[0])], int(seq[0])
        while len(out) < n and avail.get(cur):
            out.append(avail[cur].pop(0))
            cur = out[-1]
        if len(out) == n:
            return _to_onehot(out, onehot)
    return mono_shuffle(onehot, rng)


# --------------------------------------------------------------- sharpness measures
def concentration(imp_scores, k=8):
    """Fraction of total attribution inside the best contiguous k-nucleotide window."""
    imp = np.abs(imp_scores).sum(axis=-1)
    imp = imp / (imp.sum() + 1e-12)
    if len(imp) < k:
        return float(imp.sum())
    windows = np.convolve(imp, np.ones(k), mode="valid")
    return float(windows.max())


def gini(imp_scores):
    """Gini coefficient of the positional attribution. 0 = flat, 1 = one position."""
    imp = np.sort(np.abs(imp_scores).sum(axis=-1))
    n = len(imp)
    if imp.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * imp).sum()) / (n * imp.sum()) - (n + 1) / n)


# --------------------------------------------------------------------- data
def softmax(a, axis=-1):
    z = a - a.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def load_everything(args):
    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    x_test = np.asarray(d["x_test"]).astype("float32")
    y_test = np.asarray(d["y_test"])

    with open(args.preds, "rb") as fh:
        raw = np.asarray(pickle.load(fh))
    if args.n_iter:
        raw = raw[:args.n_iter]
    pr = softmax(raw)                          # stored outputs are logits
    mean, sd = pr.mean(0), pr.std(0)
    if len(mean) != len(x_test):
        raise SystemExit(f"prediction rows ({len(mean)}) != test rows ({len(x_test)}); "
                         "the prediction file and the dataset do not match")
    return x_test, y_test, mean, sd


def choose_samples(y_test, mean, sd, threshold, per_group, seed=0, classes=None):
    """A 2x2 of correctness by confidence, sampled evenly per class."""
    rng = np.random.default_rng(seed)
    true = y_test.argmax(1)
    pred = mean.argmax(1)
    correct = pred == true
    keep = sd[np.arange(len(sd)), pred] <= threshold

    cells = {"correct_kept": correct & keep,
             "correct_filtered": correct & ~keep,
             "wrong_kept": (~correct) & keep,
             "wrong_filtered": (~correct) & ~keep}
    groups = {k: [] for k in cells}
    for c in np.unique(true):
        if classes is not None and int(c) not in classes:
            continue
        for name, mask in cells.items():
            idx = np.where(mask & (true == c))[0]
            if len(idx) == 0:
                continue
            groups[name].extend(
                rng.choice(idx, min(per_group, len(idx)), replace=False).tolist())
    return {k: np.array(sorted(v)) for k, v in groups.items()}, true


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="saved BNN model directory")
    ap.add_argument("--data", required=True, help="27-class dataset pickle")
    ap.add_argument("--preds", required=True, help="iter_preds_bnn_*.pickle")
    ap.add_argument("--out", default="out/xai")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--per-group", type=int, default=60,
                    help="max samples per class per group; 60 x 27 x 2 is ~3k IG runs")
    ap.add_argument("--m-steps", type=int, default=50)
    ap.add_argument("--mc-samples", type=int, default=5,
                    help="forward passes averaged per attribution (BNN is stochastic)")
    ap.add_argument("--n-iter", type=int, default=100,
                    help="MC iterations used from the prediction file")
    ap.add_argument("--shuffle-control", action="store_true", default=True)
    ap.add_argument("--target", choices=["true", "predicted"], default="predicted",
                    help="class the attribution is computed for. 'predicted' matches "
                         "how the uncertainty score is defined (SD at the predicted "
                         "class), so sharpness and uncertainty describe the same "
                         "output; 'true' asks instead whether evidence for the correct "
                         "answer was present, which for confidently wrong predictions "
                         "is a statement about the label rather than the model. The "
                         "two are identical for correctly classified samples.")
    ap.add_argument("--second-class", action="store_true",
                    help="also attribute to the second-ranked class, to test whether "
                         "uncertain samples carry two competing motifs. Adds ~50%% to "
                         "the runtime.")
    ap.add_argument("--classes", type=str, default=None,
                    help="comma-separated class indices to restrict to, e.g. 0,2,3 "
                         "for MBNL1, PUM2 and QKI. Default: all 27.")
    ap.add_argument("--window", type=int, default=8,
                    help="width for the concentration measure; RBP motifs are 4-8 nt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eager", action="store_true",
                    help="run eagerly. Much slower on GPU, but some TFP Flipout layers "
                         "misbehave under tf.function; try this if attributions come "
                         "back all-zero or the model refuses to build.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # The models were saved in the TensorFlow SavedModel format under Keras 2. Keras 3,
    # which is the default from TF 2.16, refuses to load it: "Keras 3 only supports V3
    # .keras files and legacy H5". Its suggested TFSMLayer is inference-only and does
    # not propagate gradients, so it cannot be used for integrated gradients. Setting
    # this before importing tensorflow makes tf.keras resolve to the legacy Keras 2
    # package (tf_keras), which loads the SavedModel as a real, differentiable model.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    import tensorflow as tf
    # importing TFP registers the Flipout layer classes needed to deserialise the BNN
    import tensorflow_probability as tfp                      # noqa: F401
    if args.eager:
        tf.config.run_functions_eagerly(True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}, GPUs visible: {len(gpus)}"
          f"{' -- running on CPU, this will be slow' if not gpus else ''}")

    print("loading data ...")
    x_test, y_test, mean, sd = load_everything(args)
    sel = ([int(x) for x in args.classes.split(',')] if args.classes else None)
    groups, true = choose_samples(y_test, mean, sd, args.threshold,
                                  args.per_group, args.seed, classes=sel)
    for g, v in groups.items():
        print(f"  {g:17s} n={len(v)}")

    print("loading model ...")
    try:
        import tf_keras
        model = tf_keras.models.load_model(args.model, compile=False)
        print("  loaded with tf_keras (legacy Keras 2)")
    except Exception as e:
        print(f"  tf_keras load failed ({type(e).__name__}: {e}); trying tf.keras")
        model = tf.keras.models.load_model(args.model, compile=False)
    # a SavedModel round-trip can produce a model whose call() is not differentiable;
    # check once here rather than discovering it as all-zero attributions later
    probe = tf.convert_to_tensor(x_test[:1].astype("float32"))
    with tf.GradientTape() as tape:
        tape.watch(probe)
        g = tape.gradient(tf.reduce_sum(model(probe)), probe)
    if g is None or float(tf.reduce_max(tf.abs(g))) == 0.0:
        raise SystemExit(
            "the loaded model returns no usable gradient, so integrated gradients "
            "cannot be computed. Re-run with --eager; if that also fails the model has "
            "to be rebuilt from its architecture and weights rather than loaded from "
            "the SavedModel.")
    print(f"  gradient check ok (max |grad| = {float(tf.reduce_max(tf.abs(g))):.3e})")

    rng = np.random.default_rng(args.seed)
    pred_cls = mean.argmax(1)
    print(f"attributing with respect to the {args.target} class")
    rows = []
    for gname, idx in groups.items():
        for j, i in enumerate(idx):
            t = int(true[i] if args.target == "true" else pred_cls[i])
            att, draws = integrated_gradients(model, x_test[i], t, args.m_steps,
                                              mc_samples=args.mc_samples,
                                              return_draws=True)
            rec = dict(idx=int(i), group=gname, cls=int(true[i]), target=t,
                       entropy=float(entropy(att)),
                       conc=concentration(att, args.window),
                       gini=gini(att),
                       consistency=explanation_consistency(draws),
                       peak_agree=peak_agreement(draws, args.window),
                       sd=float(sd[i, mean[i].argmax()]),
                       p_true=float(mean[i, t]))

            # competing evidence: attribute to the SECOND-ranked class as well. On the
            # simulated data, uncertainty arose where a second protein's motif was
            # present by chance, so two class motifs competed. The real-data analogue is
            # a sample with a sharp second-class motif at a DIFFERENT position.
            if args.second_class:
                t2 = int(np.argsort(mean[i])[-2])
                att2 = integrated_gradients(model, x_test[i], t2, args.m_steps,
                                            mc_samples=max(1, args.mc_samples // 2))
                p1 = int(np.convolve(np.abs(att).sum(-1),
                                     np.ones(args.window), "valid").argmax())
                p2 = int(np.convolve(np.abs(att2).sum(-1),
                                     np.ones(args.window), "valid").argmax())
                rec.update(conc2=concentration(att2, args.window),
                           peak1=p1, peak2=p2,
                           peak_dist=int(abs(p1 - p2)))
            rows.append(rec)
            if args.shuffle_control:
                for tag, fn in (("dinuc_shuf", dinuc_shuffle),
                                ("mono_shuf", mono_shuffle)):
                    xs = fn(x_test[i], rng)
                    att_s = integrated_gradients(model, xs, t, args.m_steps,
                                                 mc_samples=args.mc_samples)
                    rows.append(dict(idx=int(i), group=tag, cls=int(true[i]), target=t,
                                     entropy=float(entropy(att_s)),
                                     conc=concentration(att_s, args.window),
                                     gini=gini(att_s),
                                     sd=float(sd[i, mean[i].argmax()]),
                                     p_true=float(mean[i, t])))
            if j % 25 == 0:
                print(f"  {gname}: {j}/{len(idx)}", flush=True)

    stem = f"entropy_{args.target}.json"
    with open(os.path.join(args.out, stem), "w") as fh:
        json.dump(rows, fh)

    # ----------------------------------------------------------------- report
    import collections
    from scipy.stats import mannwhitneyu, spearmanr
    GROUPS = ("correct_kept", "correct_filtered", "wrong_kept", "wrong_filtered",
              "dinuc_shuf", "mono_shuf")
    REAL = GROUPS[:4]

    # --- per class, and pooled after removing the class means -------------------
    # Baseline sharpness differs between RBPs, so pooling raw values across classes
    # adds between-class variance that hides any within-class effect. Standardising
    # within class before pooling removes it.
    for metric in ("conc", "gini"):
        print(f"\n=== {metric}: per class ===")
        byc = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            byc[r["cls"]][r["group"]].append(r[metric])
        for c in sorted(byc):
            cells = byc[c]
            if not (cells.get("correct_kept") and cells.get("wrong_filtered")):
                continue
            nm = RBP27_NAMES[c] if c < len(RBP27_NAMES) else f"c{c}"
            bits = "  ".join(f"{g.replace('correct','C').replace('wrong','W')}"
                             f"={np.median(cells[g]):.4f}(n={len(cells[g])})"
                             for g in REAL if cells.get(g))
            print(f"  {nm:10s} {bits}")
            # every pairwise contrast, per class. Pooling these hides real
            # heterogeneity: the confidence effect can be ~20% of the motif signal in
            # one RBP and absent in another, which averages to nothing.
            ck = cells.get("correct_kept", [])
            for other in ("correct_filtered", "wrong_kept", "wrong_filtered"):
                o = cells.get(other, [])
                if len(ck) >= 8 and len(o) >= 8:
                    _, p = mannwhitneyu(ck, o, alternative="greater")
                    print(f"             C_kept > {other:16s} p={p:.3g}"
                          f"{'  *' if p < 0.05 else ''}")
            # precondition: does this class have any motif signal to grade? If
            # confident-correct predictions do not clear the shuffled floor, no
            # within-class comparison here is interpretable.
            sh = cells.get("dinuc_shuf", [])
            if len(ck) >= 8 and len(sh) >= 8:
                _, p = mannwhitneyu(ck, sh, alternative="greater")
                signal = np.median(ck) - np.median(sh)
                print(f"             signal over shuffled: {signal:+.4f}  p={p:.3g}"
                      f"{'' if p < 0.05 else '   <-- no motif signal; exclude this class'}")

        # class-standardised pooling
        zs = collections.defaultdict(list)
        for c, cells in byc.items():
            allv = np.concatenate([v for g, v in cells.items() if g in REAL]) \
                if any(g in cells for g in REAL) else None
            if allv is None or len(allv) < 4 or allv.std() == 0:
                continue
            mu, s = allv.mean(), allv.std()
            for g in REAL:
                for v in cells.get(g, []):
                    zs[g].append((v - mu) / s)
        print(f"  --- pooled, standardised within class ---")
        for g in REAL:
            if zs.get(g):
                print(f"    {g:17s} n={len(zs[g]):4d}  median z={np.median(zs[g]):+.3f}")
        if zs.get("correct_kept") and zs.get("correct_filtered"):
            _, p = mannwhitneyu(zs["correct_kept"], zs["correct_filtered"],
                                alternative="greater")
            print(f"    confidence effect, correct only: p={p:.3g}")
        if zs.get("correct_kept") and zs.get("wrong_kept"):
            _, p = mannwhitneyu(zs["correct_kept"], zs["wrong_kept"],
                                alternative="greater")
            print(f"    correctness effect, confident only: p={p:.3g}")

    # --- the explanation-level tests, per class, correct predictions only --------
    # Restricting to correct predictions removes the correctness confound: any
    # relationship left is between the uncertainty score and the explanation, not
    # between both of them and whether the label matched.
    print("\n=== XAI vs uncertainty, WITHIN class, CORRECT predictions only ===")
    extra = [m for m in ("consistency", "peak_agree", "conc2", "peak_dist")
             if any(m in r and r[m] == r[m] for r in rows)]
    byc2 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["group"] in ("correct_kept", "correct_filtered"):
            byc2[r["cls"]][r["group"]].append(r)
    for c in sorted(byc2):
        nm = RBP27_NAMES[c] if c < len(RBP27_NAMES) else f"class {c}"
        cells = byc2[c]
        allr = cells.get("correct_kept", []) + cells.get("correct_filtered", [])
        if len(allr) < 20:
            continue
        print(f"  --- {nm} (n={len(allr)} correct) ---")
        for m in ["conc"] + extra:
            v = [(r["sd"], r[m]) for r in allr if m in r and r[m] == r[m]]
            if len(v) < 20:
                continue
            s, x = map(np.array, zip(*v))
            if np.std(x) == 0:
                continue
            rho, p = spearmanr(s, x)
            ck = [r[m] for r in cells.get("correct_kept", []) if m in r]
            cf = [r[m] for r in cells.get("correct_filtered", []) if m in r]
            extra_txt = ""
            if len(ck) >= 8 and len(cf) >= 8:
                _, pm = mannwhitneyu(ck, cf)
                extra_txt = (f"   confident={np.median(ck):.4f} "
                             f"unsure={np.median(cf):.4f} p={pm:.3g}")
            print(f"     {m:12s} rho(SD)={rho:+.3f} p={p:.3g}"
                  f"{'*' if p < 0.05 else ' '}{extra_txt}")

    # --- uncertainty as a continuous variable -----------------------------------
    print("\n=== sharpness vs uncertainty score (continuous, all real samples) ===")
    for metric in ("conc", "gini", "entropy"):
        v = [(r["sd"], r[metric], r["cls"]) for r in rows if r["group"] in REAL]
        if len(v) < 10:
            continue
        s, m, cl = map(np.array, zip(*v))
        rho, p = spearmanr(s, m)
        # within class, to remove the between-class baseline again
        rz = []
        for c in np.unique(cl):
            k = cl == c
            if k.sum() >= 8 and np.std(m[k]) > 0:
                rz.append(spearmanr(s[k], m[k]).correlation)
        print(f"  {metric:8s} pooled rho={rho:+.3f} (p={p:.3g})   "
              f"median within-class rho={np.nanmedian(rz):+.3f} over {len(rz)} classes")
    # entropy: lower is sharper. concentration and gini: higher is sharper.
    for metric, lower_is_sharper in (("conc", False), ("gini", False),
                                     ("entropy", True)):
        by = collections.defaultdict(list)
        for r in rows:
            by[r["group"]].append(r[metric])
        name = {"conc": f"attribution in best {args.window}nt window",
                "gini": "Gini of positional attribution",
                "entropy": "positional entropy (bits)"}[metric]
        print(f"\n=== {name} ===")
        for g in GROUPS:
            v = np.array(by.get(g, []))
            if len(v):
                print(f"  {g:11s} n={len(v):5d}  median={np.median(v):.4f}  "
                      f"IQR=[{np.percentile(v,25):.4f}, {np.percentile(v,75):.4f}]")
        if not (by.get("kept") and by.get("filtered")):
            continue
        alt = "less" if lower_is_sharper else "greater"
        _, p = mannwhitneyu(by["kept"], by["filtered"], alternative=alt)
        print(f"  kept sharper than filtered: p = {p:.3g}")
        # calibration: can this measure tell real sequences from shuffled ones at all?
        for ctrl in ("dinuc_shuf", "mono_shuf"):
            if by.get(ctrl):
                _, pc = mannwhitneyu(by["kept"], by[ctrl], alternative=alt)
                d = np.median(by["kept"]) - np.median(by[ctrl])
                warn = ("" if pc < 0.05 else
                        "   <-- does not separate real from shuffled; "
                        "the test above is not interpretable")
                print(f"  CALIBRATION kept vs {ctrl}: diff={d:+.4f}  p={pc:.3g}{warn}")
    print(f"\nwrote {os.path.join(args.out, stem)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
