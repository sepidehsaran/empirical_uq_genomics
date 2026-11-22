"""Sequence-level covariates against the uncertainty score.

Tests GC content, homopolymer runs, k-mer novelty, distance to the training set and
class size. Reported in the paper only as ruled out; see docs/REPRODUCING.md.

    python scripts/rbp27_application/uncertainty_covariates.py \
        --data  /path/to/dataset.pickle \
        --preds /path/to/iter_preds.pickle \
        --out   out/covariates
"""
import argparse
import collections
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr

RBP27_NAMES = ["MBNL1", "P53_NONO", "PUM2", "QKI", "AGO3", "FUS", "TAF15", "ZFP36",
               "DICER1", "EIF3A", "EIF3D", "EIF3G", "SSB", "PAPD5", "CPSF4", "CPSF3",
               "RTCB", "FXR1", "NOP58", "NOP56", "FBL", "LIN28A", "LIN28B", "UPF1",
               "G35", "G45", "XPO5"]
BASES = "ACGT"


def softmax(a, axis=-1):
    z = a - a.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------ covariates
def gc_content(seq_idx):
    return float(np.mean((seq_idx == 1) | (seq_idx == 2)))


def max_homopolymer(seq_idx):
    best = run = 1
    for i in range(1, len(seq_idx)):
        run = run + 1 if seq_idx[i] == seq_idx[i - 1] else 1
        best = max(best, run)
    return int(best)


def seq_entropy(seq_idx, k=3):
    """Shannon entropy of k-mer usage. Low means repetitive, low-complexity sequence."""
    counts = collections.Counter(tuple(seq_idx[i:i + k]) for i in range(len(seq_idx) - k + 1))
    p = np.array(list(counts.values()), float)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def kmer_table(seqs, k=6):
    c = collections.Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            c[tuple(s[i:i + k])] += 1
    total = sum(c.values())
    return c, total


def kmer_novelty(seq_idx, table, total, k=6, pseudo=0.5):
    """Mean negative log frequency of the sequence's k-mers under the training set."""
    c, n = table
    vals = []
    for i in range(len(seq_idx) - k + 1):
        f = (c.get(tuple(seq_idx[i:i + k]), 0) + pseudo) / (n + pseudo)
        vals.append(-np.log10(f))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="out/covariates")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--classes", type=str, default=None,
                    help="comma-separated class indices, e.g. 0,2,3")
    ap.add_argument("--nn-sample", type=int, default=400,
                    help="training sequences per class used for the nearest-neighbour "
                         "distance; the full comparison is O(n*m) and not worth it")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    x_test = np.asarray(d["x_test"]).argmax(-1)
    y_test = np.asarray(d["y_test"])
    x_train = np.asarray(d["x_train"]).argmax(-1)
    y_train = np.asarray(d["y_train"])

    with open(args.preds, "rb") as fh:
        raw = np.asarray(pickle.load(fh))[:args.n_iter]
    pr = softmax(raw)
    mean, sd = pr.mean(0), pr.std(0)
    true, pred = y_test.argmax(1), mean.argmax(1)
    sd_pred = sd[np.arange(len(sd)), pred]
    p_pred = mean[np.arange(len(mean)), pred]
    correct = pred == true
    train_counts = y_train.sum(0).astype(int)

    sel = [int(x) for x in args.classes.split(",")] if args.classes else list(range(y_test.shape[1]))

    print("=" * 74)
    r, p = spearmanr(p_pred, sd_pred)
    print(f"CAVEAT  SD vs predicted probability: rho={r:+.3f}. The uncertainty score is")
    print("        near-redundant with confidence, so everything below explains why the")
    print("        model is unconfident, which on this task is the same thing.")
    print("=" * 74)

    # ---- 1. class-level: training size
    print("\n[1] class training size vs median uncertainty (across classes)")
    med = np.array([np.median(sd_pred[true == c]) if (true == c).sum() else np.nan
                    for c in range(y_test.shape[1])])
    ok = ~np.isnan(med)
    rho, pv = spearmanr(train_counts[ok], med[ok])
    print(f"    rho={rho:+.3f}  p={pv:.3g}   (n={ok.sum()} classes)")
    order = np.argsort(train_counts[ok])
    print(f"    5 smallest classes: median SD={np.median(med[ok][order[:5]]):.4f}")
    print(f"    5 largest  classes: median SD={np.median(med[ok][order[-5:]]):.4f}")

    # ---- per-sequence covariates, computed within class
    print("\n[2-6] per-sequence covariates, Spearman against SD, WITHIN class")
    print("      (c = correct predictions only, to remove the correctness confound)\n")
    rows = []
    hdr = f"{'class':10s}{'n':>6s}{'GC':>9s}{'homopol':>9s}{'3mer-H':>9s}{'novelty':>9s}{'NNdist':>9s}"
    print(hdr)
    for c in sel:
        idx = np.where(true == c)[0]
        if len(idx) < 30:
            continue
        tr = x_train[y_train.argmax(1) == c]
        if len(tr) > args.nn_sample:
            tr = tr[np.random.default_rng(0).choice(len(tr), args.nn_sample, replace=False)]
        table = kmer_table(x_train[y_train.argmax(1) == c])

        feats = collections.defaultdict(list)
        for i in idx:
            s = x_test[i]
            feats["GC"].append(gc_content(s))
            feats["homopol"].append(max_homopolymer(s))
            feats["3mer-H"].append(seq_entropy(s))
            feats["novelty"].append(kmer_novelty(s, table, None))
            feats["NNdist"].append(float((tr != s).sum(1).min()) if len(tr) else np.nan)
        cmask = correct[idx]
        line = f"{RBP27_NAMES[c] if c < 27 else c:10s}{len(idx):>6d}"
        for k in ("GC", "homopol", "3mer-H", "novelty", "NNdist"):
            v = np.array(feats[k], float)
            if np.all(np.isnan(v)) or np.nanstd(v) == 0:
                line += f"{'-':>9s}"
                continue
            rho, pv = spearmanr(sd_pred[idx][cmask], v[cmask], nan_policy="omit")
            line += f"{rho:+8.2f}{'*' if pv < 0.05 else ' '}"
            rows.append(dict(cls=int(c), feature=k, rho=float(rho), p=float(pv),
                             n=int(cmask.sum())))
        print(line)
    print("\n  * p < 0.05. Positive rho = more of this feature goes with MORE uncertainty.")

    # ---- 6. confusability
    print("\n[6] where does the probability mass go when the model is unsure?")
    for c in sel:
        idx = np.where((true == c) & ~correct)[0]
        if len(idx) < 10:
            continue
        conf = collections.Counter(pred[idx])
        top = conf.most_common(3)
        names = ", ".join(f"{RBP27_NAMES[k] if k < 27 else k} {100*v/len(idx):.0f}%"
                          for k, v in top)
        print(f"    {RBP27_NAMES[c] if c < 27 else c:10s} misclassified as: {names}")

    with open(os.path.join(args.out, "covariates.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote {os.path.join(args.out, 'covariates.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
