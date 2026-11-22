"""Consolidated MC-dropout placement/rate ablation (Table S2): TF, RBP and single-cell, four
inference configurations each, on the retrained (v20) models. Inference-only passes from
scripts/mcd_inference/mcd_inference.py (run_mcd_ablation_v20.sbatch, run_mcd_ablation_gex.sbatch).

Per configuration and setting: accuracy of the mean prediction, median SD at the predicted
class (mean over seeds), fraction of test samples with SD exactly zero, and the shift of the
perturbed class(es) from the reference to the perturbed setting (median log2 SD at the
predicted class, each setting's own model; mean over seeds, seeds with a positive shift):
JUN/MEF2A for TF label noise, QKI for RBP imbalance, Erythroblast for single-cell imbalance.

    python3 scripts/mcd_inference/ablation_table.py --preds cluster_preds \
        --datasets datasets --datasets-v2 datasets_v2 --out out/tables
"""
import argparse, glob, os, pickle
import numpy as np
import pandas as pd

SEEDS = [7, 5, 11, 6, 9]
TASKS = {  # task: (ablation dir, reference setting dir, perturbed setting dir, datasets root key per setting, n_classes, perturbed classes, config names)
    "tf": dict(root="mcd_ablation_v20", ref="tf/simulated_human/balanced/single_label/no_noise",
               alt="tf/simulated_human/balanced/single_label/train_mislabeled_test_clean", k=6, cls=[2, 3],
               variants=[("baseline", "last layer, 0.25"), ("all_layers", "all layers, 0.25 (used)"),
                         ("rate05", "last layer, 0.5"), ("both", "all layers, 0.5")]),
    "rbp": dict(root="mcd_ablation_v20", ref="rbp/real_human/balanced/single_label",
                alt="rbp/real_human/train_imbalanced_test_balanced/single_label", k=3, cls=[2],
                variants=[("baseline", "last layer, 0.25"), ("all_layers", "all layers, 0.25 (used)"),
                          ("rate05", "last layer, 0.5"), ("both", "all layers, 0.5")]),
    "gex": dict(root="mcd_ablation_gex", ref="gex/real_human/balanced/single_label",
                alt="gex/real_human/train_imbalanced_test_balanced/single_label", k=4, cls=[2],
                variants=[("last_layer", "last layer, 0.3"), ("baseline", "both layers, 0.3 (used)"),
                          ("last015", "last layer, 0.15"), ("rate015", "both layers, 0.15")]),
}
LABEL = {"tf": "simulated TF, clean $\\to$ mislabeled (JUN, MEF2A)", "rbp": "real RBP, balanced $\\to$ imbalanced (QKI)",
         "gex": "single-cell, balanced $\\to$ imbalanced (Erythroblast)"}


def load(preds, root, variant, setting, k, seed):
    f = glob.glob(os.path.join(preds, root, variant, setting, f"{k}class_split{seed}", "iter_preds_mcd_v_20_s_*.pickle"))
    return np.asarray(pickle.load(open(f[0], "rb")), dtype=np.float32) if f else None


def labels(dsroot_ref, dsroot_v2, setting, k, seed):
    root = dsroot_ref if ("no_noise" in setting or "/balanced/" in setting) else dsroot_v2
    d = pickle.load(open(os.path.join(root, setting, f"{k}class_split{seed}.pickle"), "rb"))
    return np.asarray(d["y_test"])


def stats(it, y):
    mean, sd = it.mean(0), it.std(0); pred = mean.argmax(1); sdp = sd[np.arange(len(pred)), pred]
    return float((pred == y.argmax(1)).mean()), float(np.median(sdp)), float((sdp == 0).mean()), pred, sdp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--datasets", required=True)
    ap.add_argument("--datasets-v2", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = []
    for task, T in TASKS.items():
        for vkey, vname in T["variants"]:
            acc = {"ref": [], "alt": []}; med = {"ref": [], "alt": []}; zero = {"ref": [], "alt": []}; shifts = []
            for s in SEEDS:
                st = {}
                for key in ("ref", "alt"):
                    it = load(a.preds, T["root"], vkey, T[key], T["k"], s)
                    if it is None:
                        continue
                    y = labels(a.datasets, a.datasets_v2, T[key], T["k"], s)
                    ac, m, z, pred, sdp = stats(it, y); acc[key].append(ac); med[key].append(m); zero[key].append(z); st[key] = (pred, sdp)
                if "ref" in st and "alt" in st:
                    vals = []
                    for c in T["cls"]:
                        pr, sr = st["ref"]; pa, sa = st["alt"]
                        r, b = sr[pr == c], sa[pa == c]; r, b = r[r > 0], b[b > 0]
                        if len(r) > 2 and len(b) > 2:
                            vals.append(np.log2(np.median(b)) - np.log2(np.median(r)))
                    if vals:
                        shifts.append(float(np.mean(vals)))
            rows.append(dict(task=task, config=vname,
                             acc_ref=np.mean(acc["ref"]), acc_alt=np.mean(acc["alt"]),
                             sd_ref=np.mean(med["ref"]), sd_ref_min=np.min(med["ref"]), sd_ref_max=np.max(med["ref"]),
                             sd_alt=np.mean(med["alt"]), zero_ref=100 * np.mean(zero["ref"]), zero_alt=100 * np.mean(zero["alt"]),
                             shift=np.mean(shifts), shift_up=int(sum(v > 0 for v in shifts)), n=len(shifts)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "ablation_table.csv"), index=False)
    pd.set_option("display.width", 200); print(df.round(4).to_string())

    def sci(v):
        if v >= 0.01:
            return f"${v:.2f}$"
        e = int(np.floor(np.log10(v))); m = v / 10 ** e
        return f"${m:.1f}\\times10^{{{e}}}$"
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"dropout at inference & \multicolumn{2}{c}{accuracy} & \multicolumn{2}{c}{median SD at predicted class} & zero SD & shift of the perturbed class \\",
             r" & ref. & pert. & ref. & pert. & ref. / pert. & $\log_2$ (seeds up) \\", r"\midrule"]
    for task in TASKS:
        lines.append(r"\multicolumn{7}{l}{\textit{" + LABEL[task] + r"}} \\")
        for _, r in df[df.task == task].iterrows():
            lines.append(f"{r.config} & ${r.acc_ref:.3f}$ & ${r.acc_alt:.3f}$ & {sci(r.sd_ref)} & {sci(r.sd_alt)} & "
                         f"${r.zero_ref:.1f}\\%$ / ${r.zero_alt:.1f}\\%$ & ${r['shift']:+.1f}$ ({r.shift_up}/{r.n}) \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    open(os.path.join(a.out, "ablation_table.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
