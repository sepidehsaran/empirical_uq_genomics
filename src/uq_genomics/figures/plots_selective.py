"""
plots_selective.py - selective prediction on the 27-class RBP task (supplement):
macro average precision against coverage as the uncertainty threshold is swept, with
bootstrap confidence intervals, and where the paper's threshold of 0.1 falls on that
curve. Writes APP27_ap_coverage.pdf, app27_ap_coverage.csv and app27_selective_summary.csv.

Input: the raw per-pass BNN predictions (``iter_preds_bnn_v_10_s_9.pickle`` under
UQFIG_PREDS; first 100 passes, softmax per pass) and the labels from the ``app27``
bundle part. The score is the SD of the predicted class's probability.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as CP            # noqa: E402
import data_access as DA             # noqa: E402
import style as S                    # noqa: E402

THRESH = 0.1
N_BOOT = 200
PRED_FILE = "iter_preds_bnn_v_10_s_9.pickle"


def softmax(a):
    z = a - a.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def macro_ap(y, m, keep, w=None):
    """Macro AP over classes with at least one positive among the kept samples. `w` are
    per-sample integer weights (bootstrap multiplicities); the weighted AP is computed
    with one sort per class, which is what makes the bootstrap affordable."""
    idx = np.where(keep)[0]
    ww = np.ones(len(idx)) if w is None else w[idx]
    aps = []
    for c in range(y.shape[1]):
        yc, sc = y[idx, c], m[idx, c]
        if (yc * ww).sum() == 0:
            continue
        o = np.argsort(-sc, kind="stable")
        yo, wo = yc[o] * ww[o], ww[o]
        tp = np.cumsum(yo)
        n_seen = np.cumsum(wo)
        prec = np.where(n_seen > 0, tp / np.maximum(n_seen, 1e-12), 0.0)
        aps.append(float((prec * yo).sum() / yo.sum()))
    return float(np.mean(aps)) if aps else np.nan


def load(preds=None, n_iter=100):
    p = os.path.join(preds or CP.PREDS, PRED_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found (set UQFIG_PREDS; see docs/REPRODUCING.md)")
    with open(p, "rb") as fh:
        raw = np.asarray(pickle.load(fh))[:n_iter]
    y = np.asarray(DA.bundle_part("app27")["y_test"]).astype(int)
    pr = softmax(raw)
    return y, pr.mean(0), pr.std(0)


def sweep(y, mean, sd, n_boot=N_BOOT, seed=0):
    pred = mean.argmax(1)
    score = sd[np.arange(len(pred)), pred]
    n = len(y)
    covs = np.linspace(0.1, 1.0, 37)
    order = np.argsort(score)
    rng = np.random.default_rng(seed)
    rows = []
    for cov in covs:
        k = max(int(round(cov * n)), 50)
        keep = np.zeros(n, bool)
        keep[order[:k]] = True
        thr = float(score[order[k - 1]])
        ap = macro_ap(y, mean, keep)
        idx = np.where(keep)[0]
        boots = []
        for _ in range(n_boot):
            w = np.zeros(n)
            w[idx] = rng.multinomial(len(idx), np.ones(len(idx)) / len(idx))
            boots.append(macro_ap(y, mean, keep, w))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append(dict(coverage=cov, threshold=thr, macro_ap=ap, ci_lo=lo, ci_hi=hi, n_kept=k))
    d = pd.DataFrame(rows)

    keep01 = score <= THRESH
    ap01 = macro_ap(y, mean, keep01)
    cov01 = float(keep01.mean())
    ap_all = macro_ap(y, mean, np.ones(n, bool))
    gain = d.macro_ap - ap_all
    half = d[gain >= 0.5 * gain.max()].coverage.max()
    summary = dict(ap_all=ap_all, ap_at_0p1=ap01, coverage_at_0p1=cov01,
                   max_gain=float(gain.max()), coverage_at_half_gain=float(half),
                   score_q95_correct=float(np.quantile(score[pred == y.argmax(1)], 0.95)))
    return d, summary


def build(out, preds=None, n_iter=100, n_boot=N_BOOT):
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    y, mean, sd = load(preds, n_iter)
    d, summary = sweep(y, mean, sd, n_boot)
    d.to_csv(os.path.join(out, "app27_ap_coverage.csv"), index=False)
    pd.Series(summary).to_csv(os.path.join(out, "app27_selective_summary.csv"))
    print(pd.Series(summary).round(3).to_string())

    fig, ax = plt.subplots(figsize=(S.TEXTWIDTH_IN * 0.6, S.TEXTWIDTH_IN * 0.42), dpi=S.DPI)
    c = S.PALETTE[1]
    ax.fill_between(d.coverage, d.ci_lo, d.ci_hi, color=c, alpha=0.25, lw=0, label="95% bootstrap CI")
    ax.plot(d.coverage, d.macro_ap, color=c, lw=1.4, label="macro AP of retained predictions")
    ax.axhline(summary["ap_all"], color="0.4", lw=0.8, ls="--", label="all predictions")
    ax.plot([summary["coverage_at_0p1"]], [summary["ap_at_0p1"]], marker="o", ms=5, color="#333333", lw=0,
            label=f"threshold {THRESH} (coverage {100 * summary['coverage_at_0p1']:.0f}%)")
    ax.set_xlabel("coverage (fraction of test sequences retained)", fontsize=S.FS_LABEL - 0.5)
    ax.set_ylabel("macro average precision", fontsize=S.FS_LABEL - 0.5)
    ax.set_xlim(0.1, 1.0)
    ax.tick_params(labelsize=S.FS_TICK - 0.5)
    ax.legend(fontsize=S.FS_LEG - 1, frameon=False, loc="upper right")
    fig.tight_layout()
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"APP27_ap_coverage.{ext}")
        fig.savefig(p, dpi=S.DPI)
        paths.append(p)
    plt.close(fig)
    print("->", paths[0])
    return d, summary, paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=CP.OUT_DIR)
    ap.add_argument("--preds", default=None)
    a = ap.parse_args()
    build(a.out, preds=a.preds)
