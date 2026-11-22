"""Inputs for tables_sensitivity.py that were previously computed ad hoc:
  sdshift_paperdef_allsamples.csv        median log2(SD) shift per class, SD over all test samples
  paired_signedrank_tf_gex.csv           paired per-sample log2 SD difference + Wilcoxon signed-rank
  rbp_shift_shared_testset_paperdef.csv  RBP shared-test-set shift, paper definition (raw v19 passes)

    UQFIG_BUNDLE=... UQFIG_PREDS=... UQFIG_OUT=... python3 scripts/sensitivity/sdshift_table_inputs.py
"""
import glob, os, pickle, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "uq_genomics", "figures"))
import data_access as DA, config_paths as CP, plots_sensitivity as PS  # noqa: E402
PREDS = os.environ.get("UQFIG_PREDS", os.path.join(ROOT, "cluster_preds"))
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables")); os.makedirs(OUT, exist_ok=True)


def med(a, c):
    z = a[:, c][a[:, c] > 0]; return np.median(np.log2(z)) if len(z) > 2 else np.nan


rows, prows = [], []
for task, ref, alt in [("tf", "none", "mislabeled"), ("gex", "bal_single", "imbal_single"), ("rbp", "bal_single", "imbal_single")]:
    spec = CP.VERSIONS[task]; comp = DA.load_comparison(task)
    for m in CP.MODELS:
        for s in spec["seeds"]:
            a, _ = PS._sd_for(comp, task, s, ref, m); b, _ = PS._sd_for(comp, task, s, alt, m)
            if a is None or b is None:
                continue
            for c in range(spec["n_classes"]):
                rows.append(dict(task=task, model=m.upper(), seed=s, cls=spec["class_names"][c], sh=med(b, c) - med(a, c), shift=med(b, c) - med(a, c),
                                 zero_ref=float((a[:, c] == 0).mean()), zero_alt=float((b[:, c] == 0).mean())))
                if task != "rbp" and a.shape[0] == b.shape[0]:     # paired: same test set
                    x, y = a[:, c], b[:, c]; ok = (x > 0) & (y > 0); d = np.log2(y[ok]) - np.log2(x[ok])
                    try:
                        p = stats.wilcoxon(d).pvalue
                    except ValueError:
                        p = np.nan
                    prows.append(dict(task=task, model=m.upper(), seed=s, cls=spec["class_names"][c], n=int(ok.sum()), n_total=len(x),
                                      median_paired_diff=float(np.median(d)) if len(d) else np.nan, frac_up=float((d > 0).mean()) if len(d) else np.nan, p=p))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "sdshift_paperdef_allsamples.csv"), index=False)
pd.DataFrame(prows).to_csv(os.path.join(OUT, "paired_signedrank_tf_gex.csv"), index=False)

P = os.path.join(PREDS, "rbp_v19"); CLS = ["MBNL1", "PUM2", "QKI"]
def sm(a): z = a - a.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def load(st, s, m):
    p = glob.glob(f"{P}/{st}/3class_split{s}/iter_preds_{m}_*.pickle")
    if not p: return None
    r = np.asarray(pickle.load(open(p[0], "rb"))); return sm(r) if m == "bnn" else r
rrows = []
for m in ["mcd", "bnn", "ens"]:
    for s in [7, 5, 11, 6, 9]:
        a = load("balanced", s, m); b = load("train_imbalanced_test_balanced", s, m)
        if a is None or b is None: continue
        sa, sb = a.std(0), b.std(0)
        for c, n in enumerate(CLS):
            rrows.append(dict(model=m.upper(), seed=s, cls=n, sh=med(sb, c) - med(sa, c), zero_ref=float((sa[:, c] == 0).mean())))
pd.DataFrame(rrows).to_csv(os.path.join(OUT, "rbp_shift_shared_testset_paperdef.csv"), index=False)

d = pd.DataFrame(rows); pd.set_option("display.width", 200)
for task in ["tf", "gex", "rbp"]:
    t = d[d.task == task]
    print(f"\n=== {task}: median log2(SD) shift, all test samples, mean over seeds [n seeds up]")
    piv = t.pivot_table(index="cls", columns="model", values="shift", aggfunc="mean").round(2)
    up = t.assign(u=(t["shift"] > 0).astype(int)).pivot_table(index="cls", columns="model", values="u", aggfunc="sum")
    print(pd.concat({"mean": piv, "n_up": up}, axis=1).to_string())
    if task == "tf":
        for m in ["BNN", "ENS", "MCD"]:
            x = t[t.model == m].pivot(index="seed", columns="cls", values="shift")
            cor = x[["JUN", "MEF2A"]].mean(1); unt = x[[c for c in x.columns if c not in ("JUN", "MEF2A")]].mean(1)
            print(f"  {m}: corrupted {cor.mean():.2f}  untouched {unt.mean():.2f}  corrupted>untouched in {(cor > unt).sum()}/5 seeds")
print("\nMCD zero fraction (all-sample SD), mean:"); print(d[d.model == "MCD"].groupby("task")[["zero_ref", "zero_alt"]].mean().round(3))
pr = pd.DataFrame(prows)
for task in ["tf", "gex"]:
    t = pr[pr.task == task]
    g = t.groupby(["cls", "model"]).agg(med=("median_paired_diff", "mean"), frac_up=("frac_up", "mean"), sig=("p", lambda s: int((s < 0.05).sum()))).round(2)
    print(f"\n=== {task}: paired per-sample log2(SD) diff"); print(g.unstack("model").to_string())
