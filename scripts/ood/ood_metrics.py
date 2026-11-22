"""OOD-detection metrics for the coronavirus experiment.

Positive class = viral window (out-of-distribution), negative = human test sequence
(in-distribution); score = uncertainty. The bundle stores only the SD for the viral windows,
so the score that can be computed for BOTH sides is the maximum SD over classes. The
manuscript figure uses SD at the predicted class for the human side and max SD for the viral
side; both variants are reported, they differ little.

    UQFIG_BUNDLE=... python3 scripts/ood/ood_metrics.py
"""
import gzip, os, pickle, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUNDLE = os.environ.get("UQFIG_BUNDLE", os.path.join(ROOT, "figure_data_bundle"))
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))
MODELS = ["mcd", "bnn", "ens"]


def tpr_at_fpr(y, s, fpr_target=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def main():
    os.makedirs(OUT, exist_ok=True)
    v = pickle.load(gzip.open(os.path.join(BUNDLE, "virus.pkl.gz"), "rb"))
    seeds = sorted({k[0] for k in v})
    viruses = list(next(iter(v.values()))["per_virus"].keys())
    rows = []
    for seed in seeds:
        for m in MODELS:
            e = v.get((seed, m))
            if e is None:
                continue
            sd_in, mean_in = np.asarray(e["sd_in"]), np.asarray(e["mean_in"])
            s_in_max = sd_in.max(1)
            s_in_pred = sd_in[np.arange(len(sd_in)), mean_in.argmax(1)]
            thr95 = np.percentile(s_in_pred, 95)
            for virus in viruses + ["ALL"]:
                if virus == "ALL":
                    s_v = np.concatenate([np.asarray(e["per_virus"][x]).max(1) for x in viruses])
                else:
                    s_v = np.asarray(e["per_virus"][virus]).max(1)
                for variant, s_h in [("max_sd_both", s_in_max), ("manuscript_mixed", s_in_pred)]:
                    y = np.r_[np.zeros(len(s_h)), np.ones(len(s_v))]
                    s = np.r_[s_h, s_v]
                    rows.append(dict(seed=seed, model=m.upper(), virus=virus, variant=variant,
                                     auroc=roc_auc_score(y, s), tpr_at_5fpr=tpr_at_fpr(y, s),
                                     frac_flagged_95=float((s_v > thr95).mean()),
                                     n_in=len(s_h), n_virus=len(s_v)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "ood_virus_metrics.csv"), index=False)
    pd.set_option("display.width", 200)
    for variant in ["max_sd_both", "manuscript_mixed"]:
        t = d[(d.variant == variant) & (d.virus == "ALL")]
        g = t.groupby("model")[["auroc", "tpr_at_5fpr", "frac_flagged_95"]].agg(["mean", "min", "max"]).round(3)
        print(f"\n=== all eight genomes pooled, score variant: {variant}  (mean / min / max over {t.seed.nunique()} seeds)")
        print(g.to_string())
    t = d[(d.variant == "max_sd_both") & (d.virus != "ALL")]
    print("\n=== AUROC per genome (max-SD score), mean over seeds")
    print(t.pivot_table(index="virus", columns="model", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
