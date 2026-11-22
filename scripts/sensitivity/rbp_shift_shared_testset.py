"""RBP balanced -> imbalanced SD shift on the SHARED balanced test set (v19 raw predictions).
The v17 bundle compares different test sets (574 vs 2411 sequences); this removes that confound."""
import os, glob, pickle, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root
PREDS = os.path.join(os.environ.get("UQFIG_PREDS", os.path.join(ROOT, "cluster_preds")), "rbp_v19")
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))
CLS = ["MBNL1", "PUM2", "QKI"]; SEEDS = [7, 5, 11, 6, 9]; MODELS = ["mcd", "bnn", "ens"]
def softmax(a): z = a - a.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def load(setting, seed, model):
    p = glob.glob(os.path.join(PREDS, setting, f"3class_split{seed}", f"iter_preds_{model}_*.pickle"))
    if not p: return None
    raw = np.asarray(pickle.load(open(p[0], "rb")))
    return softmax(raw) if model == "bnn" else raw
rows = []
for model in MODELS:
    for seed in SEEDS:
        a = load("balanced", seed, model); b = load("train_imbalanced_test_balanced", seed, model)
        if a is None or b is None or a.shape[1] != b.shape[1]: continue
        sd_a, sd_b = a.std(0), b.std(0); pred_a = a.mean(0).argmax(1)
        for c, name in enumerate(CLS):
            sel = pred_a == c                       # samples the reference model predicts as c
            x, y = sd_a[sel, c], sd_b[sel, c]
            d = np.log2(np.clip(y, 1e-12, None)) - np.log2(np.clip(x, 1e-12, None))
            try: w_p = stats.wilcoxon(d).pvalue
            except ValueError: w_p = np.nan
            rows.append(dict(model=model.upper(), seed=seed, cls=name, n=int(sel.sum()),
                             median_log2_ratio=float(np.median(d)), frac_increase=float((d > 0).mean()),
                             wilcoxon_p=w_p, frac_zero_sd_ref=float((x == 0).mean())))
df = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); df.to_csv(os.path.join(OUT, "rbp_shift_shared_testset.csv"), index=False)
pd.set_option("display.width", 200)
g = df.groupby(["cls", "model"]).agg(mean_med_log2=("median_log2_ratio", "mean"), sd_over_seeds=("median_log2_ratio", "std"),
                                      seeds_up=("median_log2_ratio", lambda s: int((s > 0).sum())), n_seeds=("seed", "count"),
                                      frac_inc=("frac_increase", "mean")).round(2)
print("RBP balanced -> imbalanced on the SHARED 574-seq test set; paired per-sample log2(SD ratio), median per seed, then over seeds")
print("training set: MBNL1 x8.9, PUM2 x2.5, QKI unchanged"); print(g.unstack("model"))
