"""Does the reference -> perturbed uncertainty shift survive on test samples the perturbed
model never saw? Splits each composite test set into seen (in the perturbed model's
train+valid) and unseen, and recomputes the per-class median log2 SD shift at the
predicted class on each half, per method and seed.

    UQFIG_PREDS=<raw passes> UQFIG_DATASETS=<datasets root> python3 scripts/sensitivity/leak_seen_unseen.py
"""
import glob, hashlib, os, pickle, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
PREDS = os.environ.get("UQFIG_PREDS", os.path.join(ROOT, "cluster_preds"))
DS = os.environ.get("UQFIG_DATASETS", os.path.join(os.path.dirname(ROOT), "datasets"))
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))

CASES = {  # task_dir: (ref preds glob dir, alt preds glob dir, alt dataset dir, classes)
    "tf_v18": ("balanced/single_label/no_noise", "balanced/single_label/train_mislabeled_test_clean",
               "tf/simulated_human/balanced/single_label/train_mislabeled_test_clean", ["CTCF", "HNF4A", "JUN", "MEF2A", "MYC", "TAL1"]),
    "rbp_v19": ("balanced", "train_imbalanced_test_balanced", "rbp/real_human/train_imbalanced_test_balanced/single_label", ["MBNL1", "PUM2", "QKI"]),
    "gex_v19": ("balanced/single_label", "train_imbalanced_test_balanced/single_label", "gex/real_human/train_imbalanced_test_balanced/single_label", ["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]),
}


def rh(r):
    a = np.ascontiguousarray(r)
    a = a.astype(np.int8) if a.dtype.kind in "biu" or (a.dtype.kind == "f" and np.all(np.mod(a, 1) == 0)) else a.astype(np.float32)
    return hashlib.md5(a.tobytes()).hexdigest()


def softmax(a):
    z = a - a.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def load_preds(path, model):
    raw = np.asarray(pickle.load(open(path, "rb")))
    return softmax(raw) if model == "bnn" else raw


def sd_pred(p):
    sd, mean = p.std(0), p.mean(0); pred = mean.argmax(1)
    return sd[np.arange(len(pred)), pred], pred


def main():
    rows = []
    for task, (rdir, adir, dsdir, names) in CASES.items():
        for seed in (7, 5, 11, 6, 9):
            dsp = glob.glob(os.path.join(DS, dsdir, f"*class_split{seed}.pickle"))
            if not dsp:
                print(f"{task} seed {seed}: dataset not found"); continue
            d = pickle.load(open(dsp[0], "rb"))
            seen_h = set(rh(r) for r in np.asarray(d["x_train"])) | set(rh(r) for r in np.asarray(d["x_valid"]))
            seen = np.array([rh(r) in seen_h for r in np.asarray(d["x_test"])])
            y = np.asarray(d["y_test"]).argmax(1)
            for model in ("mcd", "bnn", "ens"):
                fr = glob.glob(os.path.join(PREDS, task, rdir, f"*split{seed}", f"iter_preds_{model}_*.pickle"))
                fa = glob.glob(os.path.join(PREDS, task, adir, f"*split{seed}", f"iter_preds_{model}_*.pickle"))
                if not fr or not fa:
                    continue
                sr, pr = sd_pred(load_preds(fr[0], model)); sa, pa = sd_pred(load_preds(fa[0], model))
                assert len(sr) == len(sa) == len(seen), (len(sr), len(sa), len(seen))
                for c, cname in enumerate(names):
                    ref = sr[(pr == c) & (sr > 0)]
                    for part, mask in (("seen", seen), ("unseen", ~seen), ("all", np.ones_like(seen))):
                        alt = sa[(pa == c) & mask & (sa > 0)]
                        if len(ref) < 3 or len(alt) < 3:
                            continue
                        rows.append(dict(task=task, model=model.upper(), seed=seed, cls=cname, part=part, n_alt=len(alt),
                                         shift=float(np.log2(np.median(alt)) - np.log2(np.median(ref)))))
            print(f"{task} seed {seed}: test {len(seen)}, seen {seen.sum()} ({100*seen.mean():.0f}%), unseen {(~seen).sum()}; "
                  f"unseen per class {np.bincount(y[~seen], minlength=len(names)).tolist()}")
    df = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); df.to_csv(os.path.join(OUT, "leak_seen_unseen_shift.csv"), index=False)
    pd.set_option("display.width", 200)
    for task in df.task.unique():
        t = df[df.task == task]
        print(f"\n=== {task}: median log2 shift at the predicted class, mean over seeds (seeds with shift>0 / n)")
        for part in ("all", "seen", "unseen"):
            s = t[t.part == part]
            piv = s.pivot_table(index="cls", columns="model", values="shift", aggfunc="mean").round(2)
            up = s.assign(u=(s["shift"] > 0).astype(int)).groupby(["cls", "model"]).agg(u=("u", "sum"), n=("u", "size"))
            lab = up.apply(lambda r: f"{int(r.u)}/{int(r.n)}", axis=1).unstack()
            print(f"  -- {part}"); print(pd.concat({"mean": piv, "up": lab}, axis=1).to_string())


if __name__ == "__main__":
    main()
