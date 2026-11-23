"""Label-noise sweep on the simulated TF data: 15%, 30% and 50% of the JUN and MEF2A labels
swapped. Per rate and method: accuracy on the clean test set, and the shift of the median
log2 SD from the clean setting for the swapped classes and for the four untouched classes,
mean over five seeds (seeds with a positive shift), over all test samples (the reading of
Figure 9 and Table S3).

Inputs: clean and 50% passes from the released cluster_preds/ tree (tf_v18/, the bundle's), 15% / 30% runs from results_v22.

    python3 scripts/sensitivity/label_noise_sweep.py --clean cluster_preds \
        --v22 cluster_preds/results_v22 \
        --out out/tables
"""
import argparse, glob, os, pickle
import numpy as np
import pandas as pd

SEEDS = [7, 5, 11, 6, 9]
MODELS = ["mcd", "bnn", "ens"]
BASE = "tf/simulated_human/balanced/single_label"
SWAPPED, UNTOUCHED = [2, 3], [0, 1, 4, 5]          # JUN, MEF2A ; CTCF, HNF4A, MYC, TAL1


def softmax(a):
    z = a - a.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def load(pattern, model):
    f = sorted(glob.glob(pattern))
    if not f:
        return None
    it = np.asarray(pickle.load(open(f[0], "rb")), dtype=np.float32)
    return softmax(it) if model == "bnn" else it


def med_log2(sd, c):
    z = sd[:, c][sd[:, c] > 0]
    return np.log2(np.median(z)) if len(z) > 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="released cluster_preds/ tree (clean TF passes under tf_v18/.../no_noise)")
    ap.add_argument("--v22", required=True, help="results_v22 tree (15% and 30%)")
    ap.add_argument("--datasets", default="datasets", help="clean TF splits, for the test labels")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rates = [("15%", a.v22, "train_mislabeled15_test_clean", "22"), ("30%", a.v22, "train_mislabeled30_test_clean", "22"),
             ("50%", None, "train_mislabeled_test_clean", "20")]
    rows = []
    for model in MODELS:
        for s in SEEDS:
            y = np.asarray(pickle.load(open(os.path.join(a.datasets, BASE, "no_noise", f"6class_split{s}.pickle"), "rb"))["y_test"])
            clean = load(os.path.join(a.clean, "tf_v18/balanced/single_label/no_noise", f"6class_split{s}", f"iter_preds_{model}_*.pickle"), model)
            if clean is None:
                continue
            sd0 = clean.std(0)
            for rate, root, sub, ver in rates:
                pat = (os.path.join(a.clean, "tf_v18/balanced/single_label", sub, f"6class_split{s}", f"iter_preds_{model}_*.pickle") if root is None
                       else os.path.join(root, BASE, sub, f"6class_split{s}", f"iter_preds_{model}_v_{ver}_s_{s}_*.pickle"))
                it = load(pat, model)
                if it is None:
                    continue
                sd = it.std(0); acc = float((it.mean(0).argmax(1) == y.argmax(1)).mean())
                sw = np.nanmean([med_log2(sd, c) - med_log2(sd0, c) for c in SWAPPED])
                un = np.nanmean([med_log2(sd, c) - med_log2(sd0, c) for c in UNTOUCHED])
                rows.append(dict(model=model.upper(), rate=rate, seed=s, accuracy=acc, shift_swapped=sw, shift_untouched=un))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "label_noise_sweep.csv"), index=False)
    g = df.groupby(["model", "rate"]).agg(acc=("accuracy", "mean"), sw=("shift_swapped", "mean"), sw_up=("shift_swapped", lambda v: int((v > 0).sum())),
                                          un=("shift_untouched", "mean"), un_up=("shift_untouched", lambda v: int((v > 0).sum())), n=("seed", "count"))
    pd.set_option("display.width", 200); print(g.round(2).to_string())
    lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"swapped & method & accuracy & shift, swapped classes & shift, untouched classes \\", r"\midrule"]
    for rate in ("15%", "30%", "50%"):
        for i, model in enumerate(["MCD", "BNN", "ENS"]):
            r = g.loc[(model, rate)]
            lines.append(f"{rate.replace('%', chr(92) + '%') if i == 0 else ''} & {model} & ${r.acc:.2f}$ & ${r.sw:+.1f}$ ({int(r.sw_up)}/{int(r.n)}) & ${r.un:+.1f}$ ({int(r.un_up)}/{int(r.n)}) \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    open(os.path.join(a.out, "label_noise_sweep.tex"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
