"""Leave-donor-out results (single-cell): accuracy of each method on the held-out sample and
the uncertainty score per class, against the random-split models.

Inputs: results_ldo/gex/real_human/balanced/single_label/4class_split<seed>/iter_preds_<model>_v_21_s_<seed>_100.pickle
        datasets_ldo/gex/real_human/balanced/single_label/4class_split<seed>.pickle (y_test, held_out_sample)
        the bundle (random-split balanced models, for the comparison)

    UQFIG_BUNDLE=... python3 scripts/single_cell/leave_donor_out.py --results <results_ldo> --datasets <datasets_ldo>
Prints the table and writes leave_donor_out_summary.csv + a LaTeX table body to stdout (after %%TEX).
"""
import argparse, glob, os, pickle, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "uq_genomics", "figures"))
import config_paths as CP, data_access as DA   # noqa: E402
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))
TYPES = ["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]


def softmax(a):
    z = a - a.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.environ.get("UQFIG_PREDS", os.path.join(ROOT, "cluster_preds")), "results_ldo"))
    ap.add_argument("--datasets", default=os.environ.get("UQFIG_PATH_DATASETS_LDO", ""))
    ap.add_argument("--version", default="21")
    a = ap.parse_args()
    comp = DA.load_comparison("gex")
    rows = []
    for seed in CP.VERSIONS["gex"]["seeds"]:
        d = pickle.load(open(os.path.join(a.datasets, "gex/real_human/balanced/single_label", f"4class_split{seed}.pickle"), "rb"))
        y = np.asarray(d["y_test"]).argmax(1); held = d.get("held_out_sample", "?")
        for model in CP.MODELS:
            f = glob.glob(os.path.join(a.results, "gex/real_human/balanced/single_label", f"4class_split{seed}", f"iter_preds_{model}_v_{a.version}_s_{seed}_*.pickle"))
            if not f:
                print(f"missing: seed {seed} {model}"); continue
            it = np.asarray(pickle.load(open(f[0], "rb")), dtype=np.float32)
            if model == "bnn":
                it = softmax(it)
            mean, sd = it.mean(0), it.std(0); pred = mean.argmax(1); sdp = sd[np.arange(len(pred)), pred]
            # random-split reference: same seed's balanced model on its own (random) test set
            k = DA.dict_key("gex", seed, "bal_single", model)
            rm, rs = np.asarray(comp["mean"][k]), np.asarray(comp["sd"][k]); rp = rm.argmax(1)
            ry = DA.load_y_test("gex", "real_human", 4, seed, expect_n=rm.shape[0], balanced=True, multi_label=False, noise="none")
            ry = np.asarray(ry).argmax(1); rsdp = rs[np.arange(len(rp)), rp]
            row = dict(seed=seed, held_out=held, model=model.upper(), n_test=len(y),
                       acc_ldo=float((pred == y).mean()), acc_random=float((rp == ry).mean()),
                       median_sd_ldo=float(np.median(sdp)), median_sd_random=float(np.median(rsdp)))
            for c, t in enumerate(TYPES):
                row[f"acc_{t}"] = float((pred[y == c] == c).mean()) if (y == c).any() else np.nan
                row[f"sd_ldo_{t}"] = float(np.median(sdp[pred == c])) if (pred == c).any() else np.nan
                row[f"sd_rand_{t}"] = float(np.median(rsdp[rp == c])) if (rp == c).any() else np.nan
            rows.append(row)
    df = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); df.to_csv(os.path.join(OUT, "leave_donor_out_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df[["seed", "held_out", "model", "n_test", "acc_ldo", "acc_random", "median_sd_ldo", "median_sd_random"]].round(3).to_string())
    g = df.groupby("model")[["acc_ldo", "acc_random", "median_sd_ldo", "median_sd_random"]].agg(["mean", "min", "max"]).round(3)
    print(g.to_string())
    # LaTeX body: one row per held-out sample x method
    print("%%TEX")
    for seed in CP.VERSIONS["gex"]["seeds"]:
        t = df[df.seed == seed]
        if t.empty: continue
        held = t.held_out.iloc[0]; n = int(t.n_test.iloc[0])
        cells = " & ".join(f"{t[t.model == m].acc_ldo.iloc[0]:.3f} ({t[t.model == m].acc_random.iloc[0]:.3f})" for m in ("MCD", "BNN", "ENS") if (t.model == m).any())
        print(f"{held} & {n} & {cells} \\\\")


if __name__ == "__main__":
    main()
