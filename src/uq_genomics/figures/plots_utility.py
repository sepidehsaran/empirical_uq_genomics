"""
plots_utility.py - the practical-impact use case: filtering the 27-RBP predictions by
uncertainty (average precision per class, and PR curves before/after filtering).

Provenance: notebooks/06_applications/27rbp_utility/*.ipynb. Those notebooks read the
27-class predictions produced by a BNN run. This module needs either
  * APP27_PRED_DIR  - a folder with the saved 27-class predictive distribution, or
  * MODELS_ROOT     - the trained 27-class BNN, to run inference here.
Neither was found locally, so this module reports precisely what it needs.

Expected contents of APP27_PRED_DIR (any one of):
  iter_preds_bnn_v_<ver>_s_<seed>_100.pickle   (n_iter, n_samples, 27)
  or a pickle holding {'mean': (n,27), 'sd': (n,27), 'y_true': (n,27)}
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
import config_paths as CP
import data_access as DA
import style as S
import metrics as M

UQ_THRESHOLD = 0.1          # "high-quality prediction" definition used in the paper


def _load_predictions():
    b = DA.bundle_part("app27")                      # laptop path
    if b:
        return dict(mean=b["mean"], sd=b["sd"]), ""
    if not CP.APP27_PRED_DIR or not os.path.isdir(CP.APP27_PRED_DIR):
        return None, ("APP27_PRED_DIR is empty - point it at the folder holding the "
                      "27-class predictions (iter_preds_bnn_*.pickle)")
    cands = sorted(glob.glob(os.path.join(CP.APP27_PRED_DIR, "*.pickle")) +
                   glob.glob(os.path.join(CP.APP27_PRED_DIR, "*.pkl")))
    if not cands:
        return None, f"no pickles inside APP27_PRED_DIR ({CP.APP27_PRED_DIR})"
    with open(cands[0], "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and "mean" in obj:
        return obj, ""
    arr = np.asarray(obj)
    if arr.ndim == 3:
        return dict(mean=arr.mean(axis=0), sd=arr.std(axis=0)), ""
    return None, f"unrecognised structure in {os.path.basename(cands[0])}"


def _y_true(n_classes):
    b = DA.bundle_part("app27")
    if b and b.get("y_test") is not None:
        return np.asarray(b["y_test"])
    if not CP.APP27_DATASET or not os.path.exists(CP.APP27_DATASET):
        return None
    with open(CP.APP27_DATASET, "rb") as fh:
        d = pickle.load(fh)
    return np.asarray(d.get("y_test")) if "y_test" in d else None


def run(out):
    res = []
    pred, err = _load_predictions()
    if pred is None:
        return [dict(id="utility_27rbp", status="skip", reason=err)]
    mean, sd = np.asarray(pred["mean"]), np.asarray(pred["sd"])
    y = pred.get("y_true")
    if y is None:
        y = _y_true(mean.shape[1])
    if y is None:
        return [dict(id="utility_27rbp", status="skip",
                     reason="ground-truth labels for the 27-class test set not found "
                            "(APP27_DATASET)")]
    y = np.asarray(y)
    keep = sd <= UQ_THRESHOLD

    # ---- AP per class, filtered vs unfiltered
    ap_all = M.average_precision_per_class(y, mean)
    ap_f = []
    for c in range(y.shape[1]):
        m = keep[:, c]
        ap_f.append(M.average_precision_per_class(y[m][:, [c]], mean[m][:, [c]])[0]
                    if m.sum() > 10 else np.nan)
    ap_f = np.array(ap_f)
    order = np.argsort(-y.sum(axis=0))
    fig, ax = S.new_fig(1.0, aspect=0.42)
    x = np.arange(len(order)); w = 0.4
    ax.bar(x - w / 2, ap_all[order], w, label="all predictions", color=S.C_REF)
    ax.bar(x + w / 2, ap_f[order], w, label=f"SD < {UQ_THRESHOLD}", color=S.C_ALT)
    ax.set_xticks(x); ax.set_xticklabels([f"c{i}" for i in order], rotation=90)
    ax.set_ylabel("average precision"); ax.set_xlabel("RBP class (by class size)")
    ax.legend(fontsize=S.FS_LEG)
    S.fig_suptitle(fig, None, "Uncertainty filtering improves precision (27-RBP)")
    res.append(dict(id="utility_27rbp_ap", status="ok",
                    path=S.save(fig, os.path.join(out, "utility_27rbp_ap.png"))))
    pd.DataFrame(dict(cls=order, ap_all=ap_all[order], ap_filtered=ap_f[order])) \
        .to_csv(os.path.join(out, "utility_27rbp_ap_values.csv"), index=False)

    # ---- PR curves for the 5 largest classes
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        res.append(dict(id="utility_27rbp_pr", status="skip", reason="sklearn not installed"))
        return res
    top5 = order[:5]
    fig, axes = S.new_fig(0.48, aspect=0.85, ncols=2)
    for ax, (title, mask) in zip(axes, [("unfiltered", None), (f"SD < {UQ_THRESHOLD}", keep)]):
        for c in top5:
            m = np.ones(len(y), bool) if mask is None else mask[:, c]
            if m.sum() < 10:
                continue
            p, r, _ = precision_recall_curve(y[m][:, c], mean[m][:, c])
            apv = average_precision_score(y[m][:, c], mean[m][:, c])
            ax.plot(r, p, lw=1.1, label=f"c{c} (AP={apv:.2f})")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title(title, fontsize=S.FS_LABEL)
        ax.legend(fontsize=S.FS_LEG - 1.5, loc="lower left")
    S.fig_suptitle(fig, None, "PR curves before and after uncertainty filtering (27-RBP)")
    res.append(dict(id="utility_27rbp_pr", status="ok",
                    path=S.save(fig, os.path.join(out, "utility_27rbp_pr.png"))))
    return res


def tasks(out):
    return [("27-RBP filtering", lambda: run(out))]
