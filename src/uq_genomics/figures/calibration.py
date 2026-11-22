"""
calibration.py - expected calibration error and reliability diagrams (supplement).

Single-label settings only: ECE on the confidence of the predicted class is defined for
softmax output. Uses the predictive means in the figure bundle; 15 equal-width bins,
pooled over the five seeds. Writes RELIABILITY_all_settings.pdf and calibration_ece.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as CP                # noqa: E402
import data_access as DA                 # noqa: E402
import style as S                        # noqa: E402

SETTINGS = [("tf", "none", "TF clean"), ("tf", "mislabeled", "TF mislabeled"),
            ("rbp", "bal_single", "RBP balanced"), ("rbp", "imbal_single", "RBP imbalanced"),
            ("gex", "bal_single", "single-cell balanced"), ("gex", "imbal_single", "single-cell imbalanced")]
NB = 15


def ece_bins(conf, correct, nb=NB):
    edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, nb - 1)
    acc_b, conf_b, n_b = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    for b in range(nb):
        m = idx == b
        n_b[b] = m.sum()
        if n_b[b]:
            acc_b[b], conf_b[b] = correct[m].mean(), conf[m].mean()
    ece = float(np.sum(n_b / n_b.sum() * np.abs(acc_b - conf_b)))
    return ece, acc_b, conf_b, n_b


def compute(settings=SETTINGS):
    """per (setting, model, seed) ECE table and the binned curves"""
    rows, curves = [], {}
    for task, setting, label in settings:
        comp = DA.load_comparison(task)
        for m in CP.MODELS:
            for seed in CP.VERSIONS[task]["seeds"]:
                mean = comp["mean"].get(DA.dict_key(task, seed, setting, m))
                y = DA.load_y_test(task, CP.VERSIONS[task]["species"], CP.VERSIONS[task]["n_classes"], seed,
                                   **DA.setting_to_split_args(task, setting),
                                   expect_n=None if mean is None else np.asarray(mean).shape[0])
                if mean is None or y is None or np.asarray(mean).shape[0] != np.asarray(y).shape[0]:
                    continue
                mean = np.asarray(mean)
                y = np.asarray(y).argmax(1)
                pred = mean.argmax(1)
                conf = mean.max(1)
                correct = (pred == y).astype(float)
                ece, acc_b, conf_b, n_b = ece_bins(conf, correct)
                rows.append(dict(task=task, setting=setting, label=label, model=m.upper(), seed=seed, ece=ece,
                                 mean_conf=float(conf.mean()), acc=float(correct.mean())))
                curves.setdefault((label, m.upper()), []).append((acc_b, conf_b, n_b))
    return pd.DataFrame(rows), curves


def build(out, settings=SETTINGS):
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    d, curves = compute(settings)
    d.to_csv(os.path.join(out, "calibration_ece.csv"), index=False)
    g = d.groupby(["label", "model"]).agg(ece_mean=("ece", "mean"), ece_sd=("ece", "std"),
                                           conf=("mean_conf", "mean"), acc=("acc", "mean")).round(3)
    print(g.to_string())

    # reliability diagrams: 2 x 3 panels, one per setting, three methods per panel
    fig, axes = plt.subplots(2, 3, figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * 0.66), dpi=S.DPI)
    colors = {"MCD": S.PALETTE[0], "BNN": S.PALETTE[1], "ENS": S.PALETTE[2]}
    for ax, (task, setting, label) in zip(axes.ravel(), settings):
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
        for m in ["MCD", "BNN", "ENS"]:
            cs = curves.get((label, m), [])
            if not cs:
                continue
            acc = np.array([c[0] for c in cs])
            conf = np.array([c[1] for c in cs])
            n = np.array([c[2] for c in cs])
            w = n.sum(0)
            ok = w > 0
            a = np.where(ok, (acc * n).sum(0) / np.maximum(w, 1), np.nan)
            c = np.where(ok, (conf * n).sum(0) / np.maximum(w, 1), np.nan)
            ax.plot(c[ok], a[ok], marker="o", ms=2.5, lw=1.0, color=colors[m], label=m)
        ax.set_title(label, fontsize=S.FS_LABEL - 0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence", fontsize=S.FS_TICK)
        ax.set_ylabel("accuracy", fontsize=S.FS_TICK)
        ax.tick_params(labelsize=S.FS_TICK - 1)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=S.FS_LEG,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"RELIABILITY_all_settings.{ext}")
        fig.savefig(p, dpi=S.DPI)
        paths.append(p)
    plt.close(fig)
    print("figure ->", paths[0])
    return d, paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=CP.OUT_DIR)
    build(ap.parse_args().out)
