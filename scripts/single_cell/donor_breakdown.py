"""Per-donor and per-site evaluation of the existing single-cell models.

The GEX splits are random over cells. Every test cell is matched back to the full 28,214-cell
Liam embedding by exact row match, and from there to adata.obs (sample = site x donor), so
accuracy and the uncertainty score of each method can be broken down by donor and by site
without any new inference. Also writes the per-class / per-site / per-donor count table.

Needs:  datasets/adata_obs.csv  (adata.obs exported from the h5ad)
        datasets/gex/liam_embeddings/liam_seed_0_allsd_df_gex_embedding_subset.pkl
        datasets/gex/real_human/{balanced,imbalanced}/single_label/4class_split{seed}.pickle
        UQFIG_BUNDLE (predictive means/SDs)

    python3 scripts/single_cell/donor_breakdown.py --data ../datasets
"""
import argparse, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "uq_genomics", "figures"))
import config_paths as CP        # noqa: E402
import data_access as DA         # noqa: E402

TYPES = ["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("UQFIG_PATH_DATASETS_GEX", ""), help="dataset root (UQFIG_PATH_DATASETS_GEX)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    obs = pd.read_csv(os.path.join(a.data, "adata_obs.csv"), index_col=0)
    sub = obs[obs.cell_type.isin(TYPES)].reset_index()
    emb = pickle.load(open(os.path.join(a.data, "gex", "liam_embeddings",
                                        "liam_seed_0_allsd_df_gex_embedding_subset.pkl"), "rb"))
    assert len(emb) == len(sub) and (emb.cell_type.values == sub.cell_type.values).all(), \
        "embedding rows do not line up with the obs subset"
    X = emb.drop(columns=["cell_type"]).to_numpy(dtype=np.float32).round(5)
    key = {tuple(r): i for i, r in enumerate(X)}

    # ---- count table: per class x site, per class x donor
    ct_site = pd.crosstab(sub.cell_type, sub.site, margins=True)
    ct_donor = pd.crosstab(sub.cell_type, sub["sample"], margins=True)
    ct_site.to_csv(os.path.join(OUT, "gex_counts_class_by_site.csv"))
    ct_donor.to_csv(os.path.join(OUT, "gex_counts_class_by_sample.csv"))

    comp = DA.load_comparison("gex")
    rows = []
    for setting, sdir in [("bal_single", "balanced"), ("imbal_single", "imbalanced")]:
        for seed in CP.VERSIONS["gex"]["seeds"]:
            d = pickle.load(open(os.path.join(a.data, "gex", "real_human", sdir, "single_label",
                                              f"4class_split{seed}.pickle"), "rb"))
            probe = comp["mean"].get(DA.dict_key("gex", seed, setting, CP.MODELS[0]))
            if probe is not None and np.asarray(probe).shape[0] != len(d["y_test"]):
                # the imbalanced models were evaluated on the balanced split's test set
                # (train_imbalanced_test_balanced); use that split's x_test / y_test
                d = pickle.load(open(os.path.join(a.data, "gex", "real_human", "balanced", "single_label",
                                                  f"4class_split{seed}.pickle"), "rb"))
            xt = np.asarray(d["x_test"], dtype=np.float32).round(5)
            idx = np.array([key[tuple(r)] for r in xt])        # KeyError = mapping broke
            meta = sub.iloc[idx]
            y = np.asarray(d["y_test"]).argmax(1)
            for model in CP.MODELS:
                k = DA.dict_key("gex", seed, setting, model)
                mean = comp["mean"].get(k); sd = comp["sd"].get(k)
                if mean is None:
                    continue
                mean, sd = np.asarray(mean), np.asarray(sd)
                assert mean.shape[0] == len(y)
                pred = mean.argmax(1)
                score = sd[np.arange(len(pred)), pred]
                if seed == 7:
                    print(f"  check {setting} {model} seed 7: accuracy {float((pred == y).mean()):.4f}")
                df = pd.DataFrame(dict(setting=setting, seed=seed, model=model.upper(),
                                       sample=meta["sample"].values, donor=meta.donor.values,
                                       site=meta.site.values, cell_type=meta.cell_type.values,
                                       correct=(pred == y), score=score))
                rows.append(df)
    cells = pd.concat(rows)
    cells.to_csv(os.path.join(OUT, "gex_test_cells_with_metadata.csv"), index=False)

    def agg(by):
        g = (cells.groupby(["setting", "model", by, "seed"])
                  .agg(acc=("correct", "mean"), med_score=("score", "median"), n=("correct", "size"))
                  .reset_index())
        return (g.groupby(["setting", "model", by])
                 .agg(acc_mean=("acc", "mean"), acc_min=("acc", "min"), acc_max=("acc", "max"),
                      med_score=("med_score", "median"), n_per_seed=("n", "mean")).reset_index())
    by_site = agg("site"); by_donor = agg("sample")
    by_site.to_csv(os.path.join(OUT, "gex_accuracy_by_site.csv"), index=False)
    by_donor.to_csv(os.path.join(OUT, "gex_accuracy_by_sample.csv"), index=False)

    pd.set_option("display.width", 220)
    print("test cells matched exactly:", len(cells) // (len(CP.MODELS)), "rows per model-setting-seed block")
    for setting in ["bal_single", "imbal_single"]:
        print(f"\n=== {setting}: accuracy by SITE, mean over seeds (min-max)")
        t = by_site[by_site.setting == setting].pivot(index="site", columns="model", values="acc_mean").round(3)
        print(t.to_string())
        print(f"\n=== {setting}: accuracy by SAMPLE (site x donor), mean over seeds")
        t = by_donor[by_donor.setting == setting].pivot(index="sample", columns="model", values="acc_mean").round(3)
        print(t.to_string())
        spread = by_donor[by_donor.setting == setting].groupby("model").acc_mean.agg(["min", "max"])
        print("  range of per-sample accuracy across the 13 samples:"); print(spread.round(3).to_string())


if __name__ == "__main__":
    main()
