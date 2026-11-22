"""Predictive entropy and mutual information from the raw per-pass predictions.

For T passes with class probabilities p_t:
  predictive entropy   H[ mean_t p_t ]             total uncertainty
  expected entropy     mean_t H[ p_t ]             aleatoric part
  mutual information   H[mean p] - mean H[p]       epistemic part (disagreement between passes)

Runs on every <task>/<setting>/<split>/iter_preds_<model>_*.pickle found under UQFIG_PREDS:
    rbp_v19/{balanced,train_imbalanced_test_balanced}/3class_split<seed>/
    tf_v18/{...}/6class_split<seed>/      (once copied off the cluster)
    gex_v19/{...}/4class_split<seed>/
BNN outputs are logits (softmax applied per pass); MCD and ENS already carry softmax. For
multi-label (sigmoid) runs the per-class Bernoulli entropy is summed instead.

Reports, per task: (i) Kendall tau between MI / entropy and the SD score, so the reader knows
whether the paper's conclusions depend on the choice of score, and (ii) the same
reference -> perturbed shift table as Table S1, now for MI and entropy.

    UQFIG_PREDS=... python3 scripts/uncertainty_scores/entropy_mi.py
"""
import glob, os, pickle, re
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PREDS = os.environ.get("UQFIG_PREDS", os.path.join(ROOT, "cluster_preds"))
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))

PAIRS = {  # task_dir: (reference setting dir, perturbed setting dir, class names)
    "rbp_v19": ("balanced", "train_imbalanced_test_balanced", ["MBNL1", "PUM2", "QKI"]),
    "tf_v18": ("no_noise", "train_mislabeled_test_clean", ["CTCF", "HNF4A", "JUN", "MEF2A", "MYC", "TAL1"]),
    "gex_v19": ("balanced", "train_imbalanced_test_balanced", ["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]),
}
MODELS = ["mcd", "bnn", "ens"]


def softmax(a):
    z = a - a.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def entropies(p, multi_label=False):
    """p: (T, n, K) probabilities -> (pred_entropy, expected_entropy, MI), each (n,)"""
    eps = 1e-12
    if multi_label:
        h = lambda q: -(q * np.log(q + eps) + (1 - q) * np.log(1 - q + eps)).sum(-1)
    else:
        h = lambda q: -(q * np.log(q + eps)).sum(-1)
    pe = h(p.mean(0)); ee = h(p).mean(0)
    return pe, ee, pe - ee


def load(path, model):
    raw = np.asarray(pickle.load(open(path, "rb")))
    multi = "multi" in path
    if model == "bnn":
        raw = 1 / (1 + np.exp(-raw)) if multi else softmax(raw)
    return raw, multi


def find(task_dir, setting_dir, model):
    out = {}
    # the setting directory may sit at any depth (tf: balanced/single_label/<setting>/...)
    for p in glob.glob(os.path.join(PREDS, task_dir, "**", setting_dir, "**", f"iter_preds_{model}_*.pickle"), recursive=True):
        m = re.search(r"split(\d+)", p)
        if m:
            out[int(m.group(1))] = p
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 220)
    agree_rows, shift_rows = [], []
    for task_dir, (ref, alt, names) in PAIRS.items():
        if not os.path.isdir(os.path.join(PREDS, task_dir)):
            print(f"[{task_dir}] not found under {PREDS} -- skipped"); continue
        for model in MODELS:
            fr, fa = find(task_dir, ref, model), find(task_dir, alt, model)
            for seed in sorted(set(fr) & set(fa)):
                pr, multi = load(fr[seed], model); pa, _ = load(fa[seed], model)
                for name, p in [("ref", pr), ("alt", pa)]:
                    pe, ee, mi = entropies(p, multi)
                    mean, sd = p.mean(0), p.std(0); pred = mean.argmax(1)
                    sdp = sd[np.arange(len(pred)), pred]
                    agree_rows.append(dict(task=task_dir, setting=name, model=model.upper(), seed=seed,
                                           tau_sd_mi=stats.kendalltau(sdp, mi)[0],
                                           tau_sd_entropy=stats.kendalltau(sdp, pe)[0],
                                           tau_mi_entropy=stats.kendalltau(mi, pe)[0],
                                           frac_mi_zero=float((mi <= 1e-9).mean())))
                # shift per predicted class, median log2 ratio, as in Table S1 (predicted-class rows)
                for score_name, (sr, sa) in {"MI": (entropies(pr, multi)[2], entropies(pa, multi)[2]),
                                             "entropy": (entropies(pr, multi)[0], entropies(pa, multi)[0]),
                                             "SD": (pr.std(0)[np.arange(pr.shape[1]), pr.mean(0).argmax(1)],
                                                    pa.std(0)[np.arange(pa.shape[1]), pa.mean(0).argmax(1)])}.items():
                    pred_r, pred_a = pr.mean(0).argmax(1), pa.mean(0).argmax(1)
                    for c, cname in enumerate(names):
                        a, b = sr[pred_r == c], sa[pred_a == c]
                        a, b = a[a > 0], b[b > 0]
                        if len(a) < 3 or len(b) < 3:
                            continue
                        shift_rows.append(dict(task=task_dir, model=model.upper(), seed=seed, score=score_name, cls=cname,
                                               shift=float(np.log2(np.median(b)) - np.log2(np.median(a)))))
    A, Sh = pd.DataFrame(agree_rows), pd.DataFrame(shift_rows)
    A.to_csv(os.path.join(OUT, "entropy_mi_agreement.csv"), index=False)
    Sh.to_csv(os.path.join(OUT, "entropy_mi_shift.csv"), index=False)
    for task in A.task.unique():
        print(f"\n=== {task}: rank agreement with the SD score (Kendall tau, mean over seeds and settings)")
        print(A[A.task == task].groupby("model")[["tau_sd_mi", "tau_sd_entropy", "tau_mi_entropy", "frac_mi_zero"]].mean().round(3).to_string())
        t = Sh[Sh.task == task]
        print(f"\n=== {task}: reference -> perturbed shift at the predicted class, mean over seeds (n seeds up)")
        for score in ["SD", "MI", "entropy"]:
            s = t[t.score == score]
            piv = s.pivot_table(index="cls", columns="model", values="shift", aggfunc="mean").round(2)
            up = s.assign(u=(s["shift"] > 0).astype(int)).pivot_table(index="cls", columns="model", values="u", aggfunc="sum")
            print(f"  -- {score}"); print(pd.concat({"mean": piv, "n_up": up}, axis=1).to_string())


if __name__ == "__main__":
    main()
