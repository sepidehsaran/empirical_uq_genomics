"""Leave-donor-out splits for the single-cell task.

One fold per seed. The test set is every cell of the four selected types from ONE held-out
sample (site x donor); train and valid are drawn from the other twelve samples, class-balanced
as in the paper's balanced setting (n per type = min(4500, cells available)), split 75/25.
Held-out samples cover all four sites: seed 7 -> s1d1, 5 -> s2d1, 11 -> s4d1, 6 -> s3d10, 9 -> s3d3.

Writes <out>/gex/real_human/balanced/single_label/4class_split<seed>.pickle (same relative
path as the balanced setting, so the training code runs unchanged with path_datasets=<out>)
plus <out>/leave_donor_out_folds.csv.

    python3 scripts/splits/build_leave_donor_out.py --data datasets --out datasets_ldo
"""
import argparse, os, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

TYPES = ["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]
FOLDS = {7: "s1d1", 5: "s2d1", 11: "s4d1", 6: "s3d10", 9: "s3d3"}


def onehot(lab):
    y = np.zeros((len(lab), len(TYPES)), dtype=np.float32)
    for i, t in enumerate(lab):
        y[i, TYPES.index(t)] = 1
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets")
    ap.add_argument("--out", default="datasets_ldo")
    ap.add_argument("--n-per-type", type=int, default=4500)
    a = ap.parse_args()
    obs = pd.read_csv(os.path.join(a.data, "adata_obs.csv"), index_col=0)
    sub = obs[obs.cell_type.isin(TYPES)].reset_index(drop=True)
    emb = pickle.load(open(os.path.join(a.data, "gex", "liam_embeddings", "liam_seed_0_allsd_df_gex_embedding_subset.pkl"), "rb"))
    assert len(emb) == len(sub) and (emb.cell_type.values == sub.cell_type.values).all()
    X = emb.drop(columns=["cell_type"]).to_numpy(dtype=np.float32)
    lab = sub.cell_type.values; sample = sub["sample"].values
    rows = []
    d_out = os.path.join(a.out, "gex", "real_human", "balanced", "single_label"); os.makedirs(d_out, exist_ok=True)
    for seed, held in FOLDS.items():
        rng = np.random.default_rng(seed)
        test = np.where(sample == held)[0]
        rest = np.where(sample != held)[0]
        n = min(a.n_per_type, min((lab[rest] == t).sum() for t in TYPES))
        pick = np.concatenate([rng.choice(rest[lab[rest] == t], n, replace=False) for t in TYPES])
        tr, va = train_test_split(pick, test_size=0.25, random_state=seed)
        d = dict(x_train=X[tr], y_train=onehot(lab[tr]), x_valid=X[va], y_valid=onehot(lab[va]),
                 x_test=X[test], y_test=onehot(lab[test]), held_out_sample=held, test_index=test)
        pickle.dump(d, open(os.path.join(d_out, f"4class_split{seed}.pickle"), "wb"), protocol=4)
        counts = {t: int((lab[test] == t).sum()) for t in TYPES}
        rows.append(dict(seed=seed, held_out=held, site=held[:2], n_per_type_train=n, n_train=len(tr), n_valid=len(va), n_test=len(test), **counts))
        print(f"seed {seed}: held out {held}: test {len(test)} {counts}; train {len(tr)} valid {len(va)} ({n} per type before the split)")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "leave_donor_out_folds.csv"), index=False)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
