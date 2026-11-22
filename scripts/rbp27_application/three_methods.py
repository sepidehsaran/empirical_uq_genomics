"""Uncertainty filtering on the 27-class RBP task for all three methods.

For each method: accuracy, macro AP over the 27 classes with all predictions and with the
filter of Section 4.4 (SD at the predicted class, probability space, below 0.1), the fraction
kept, the AP-coverage curve, the Spearman correlation between training-set size and the
median score per predicted class, and the per-class coverage. Same definitions as
app27_selective.py and plots_rbp27.build_app27, applied to each method's 100 passes.

Inputs: the BNN passes of the paper (iter_preds_bnn_v_10_s_9.pickle, logits -> softmax per
pass) and the MCD / ENS passes from results_v23 (probabilities).

    python3 scripts/rbp27_application/three_methods.py --bnn cluster_preds/iter_preds_bnn_v_10_s_9.pickle \
        --v23 cluster_preds/results_v23 --bundle figure_data_bundle_v20 \
        --out out/tables
"""
import argparse, glob, gzip, os, pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

THRESH = 0.1
NAMES = ['MBNL1', 'P53_NONO', 'PUM2', 'QKI', 'AGO3', 'FUS', 'TAF15', 'ZFP36', 'DICER1', 'EIF3A', 'EIF3D', 'EIF3G', 'SSB',
         'PAPD5', 'CPSF4', 'CPSF3', 'RTCB', 'FXR1', 'NOP58', 'NOP56', 'FBL', 'LIN28A', 'LIN28B', 'UPF1', 'G35', 'G45', 'XPO5']


def softmax(a):
    z = a - a.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def macro_ap(y, m, keep):
    aps = []
    for c in range(y.shape[1]):
        if y[keep, c].sum() > 0:
            aps.append(average_precision_score(y[keep, c], m[keep, c]))
    return float(np.mean(aps))


def analyse(name, it, y, train_counts):
    mean, sd = it.mean(0), it.std(0); pred = mean.argmax(1); score = sd[np.arange(len(pred)), pred]
    acc = float((pred == y.argmax(1)).mean())
    keep = score < THRESH; n = len(pred)
    ap_all = macro_ap(y, mean, np.ones(n, bool)); ap_f = macro_ap(y, mean, keep)
    # per-class AP, all vs filtered, as Figure 12b
    per_all = np.array([average_precision_score(y[:, c], mean[:, c]) for c in range(27)])
    per_f = np.array([average_precision_score(y[keep, c], mean[keep, c]) if y[keep, c].sum() > 0 else np.nan for c in range(27)])
    improved = int(np.nansum(per_f > per_all))
    # AP-coverage curve
    order = np.argsort(score); curve = []
    for cov in np.arange(0.1, 1.0001, 0.05):
        k = max(1, int(round(cov * n))); kp = np.zeros(n, bool); kp[order[:k]] = True
        curve.append((float(cov), macro_ap(y, mean, kp)))
    # score per predicted class vs training size
    med = np.array([np.median(score[pred == c]) if (pred == c).sum() > 0 else np.nan for c in range(27)])
    ok = np.isfinite(med); rho, p = stats.spearmanr(train_counts[ok], med[ok])
    # per-class coverage of positives
    tc = y.argmax(1); cov_cls = np.array([keep[tc == c].mean() for c in range(27)])
    small = np.argsort(train_counts)[:5]; large = np.argsort(train_counts)[-5:]
    return dict(method=name, accuracy=acc, kept=float(keep.mean()), ap_all=ap_all, ap_filtered=ap_f, classes_improved=improved,
                rho_size_score=float(rho), rho_p=float(p), median_score=float(np.median(score)), zero_score=float((score == 0).mean()),
                cov_min=float(cov_cls.min()), cov_max=float(cov_cls.max()), cov_5_smallest=float(cov_cls[small].mean()),
                cov_5_largest=float(cov_cls[large].mean()), curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bnn", required=True); ap.add_argument("--v23", required=True)
    ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    b = pickle.load(gzip.open(os.path.join(a.bundle, "app27.pkl.gz")))
    y = np.asarray(b["y_test"]); train_counts = np.asarray(b["train_counts"])
    runs = {"BNN (paper)": softmax(np.asarray(pickle.load(open(a.bnn, "rb")), dtype=np.float32)[:100])}
    sub = "rbp/real_human/train_imbalanced_test_balanced/single_label/27class_split9"
    for m in ("bnn", "mcd", "ens"):
        f = glob.glob(os.path.join(a.v23, sub, f"iter_preds_{m}_v_23_s_9_*.pickle"))
        if f:
            it = np.asarray(pickle.load(open(f[0], "rb")), dtype=np.float32)
            runs[m.upper() + (" (retrained)" if m == "bnn" else "")] = softmax(it) if m == "bnn" else it
        else:
            print(f"  {m}: no passes found under {a.v23}/{sub}")
    rows = [analyse(k, v, y, train_counts) for k, v in runs.items()]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "curve"} for r in rows])
    pd.set_option("display.width", 220); print(df.round(3).to_string())
    df.to_csv(os.path.join(a.out, "three_methods.csv"), index=False)
    cv = pd.DataFrame([dict(method=r["method"], coverage=c, macro_ap=v) for r in rows for c, v in r["curve"]])
    cv.to_csv(os.path.join(a.out, "app27_three_methods_curve.csv"), index=False)
    print("\nAP-coverage (macro AP at coverage):")
    print(cv.pivot(index="coverage", columns="method", values="macro_ap").round(3).to_string())


if __name__ == "__main__":
    main()
