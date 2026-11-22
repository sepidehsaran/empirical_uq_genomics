"""Build the figure bundle for the retrained composite settings (v20: test sets held out).

Takes a copy of the corrected bundle (MC dropout active) and replaces, for the four
composite settings, the comparison entries of all three methods, the labels, and the
class-distribution counts with the v20 runs:

    tf   mislabeled      <- results_v20/tf/simulated_human/balanced/single_label/train_mislabeled_test_clean
    rbp  imbal_single    <- results_v20/rbp/real_human/train_imbalanced_test_balanced/single_label
    rbp  imbal_multi     <- results_v20/rbp/real_human/train_imbalanced_test_balanced/multi_label
    gex  imbal_single    <- results_v20/gex/real_human/train_imbalanced_test_balanced/single_label

Keys keep the bundle's version numbers (v_18 tf, v_17 rbp, v_19 gex) so the figure pipeline
needs no change. BNN passes are logits: softmax (single-label) or sigmoid (multi-label) is
applied before mean/SD, as the aggregation notebook did. ENS files hold 100 rows = 10 members
x 10 identical passes, used as they are (same as before). Also writes a raw-pass tree in the
cluster_preds layout for the re-analysis scripts.

    python3 make_bundle.py --results cluster_preds/results_v20 --datasets ../../datasets_v2 \
        --bundle-in figure_data_bundle_mcdfix --bundle-out figure_data_bundle_v20 --raw cluster_preds/merged_v20
"""
import argparse, glob, gzip, os, pickle, shutil
import numpy as np

SEEDS = [7, 5, 11, 6, 9]
SETTINGS = [  # task, bundle setting, version, n_classes, results subdir, dataset root key, datasets subdir, raw subdir, multi
    # reference settings (original splits, retrained with the same protocol)
    ("tf", "none", 18, 6, "tf/simulated_human/balanced/single_label/no_noise", "ref", "tf/simulated_human/balanced/single_label/no_noise", "tf_v18/balanced/single_label/no_noise", False),
    ("rbp", "bal_single", 17, 3, "rbp/real_human/balanced/single_label", "ref", "rbp/real_human/balanced/single_label", "rbp_v19/balanced", False),
    ("rbp", "bal_multi", 17, 3, "rbp/real_human/balanced/multi_label", "ref", "rbp/real_human/balanced/multi_label", "rbp_v19/balanced_multi", True),
    ("gex", "bal_single", 19, 4, "gex/real_human/balanced/single_label", "ref", "gex/real_human/balanced/single_label", "gex_v19/balanced/single_label", False),
    # perturbed settings (test sets held out before perturbation)
    ("tf", "mislabeled", 18, 6, "tf/simulated_human/balanced/single_label/train_mislabeled_test_clean", "v2", "tf/simulated_human/balanced/single_label/train_mislabeled_test_clean", "tf_v18/balanced/single_label/train_mislabeled_test_clean", False),
    ("rbp", "imbal_single", 17, 3, "rbp/real_human/train_imbalanced_test_balanced/single_label", "v2", "rbp/real_human/train_imbalanced_test_balanced/single_label", "rbp_v19/train_imbalanced_test_balanced", False),
    ("rbp", "imbal_multi", 17, 3, "rbp/real_human/train_imbalanced_test_balanced/multi_label", "v2", "rbp/real_human/train_imbalanced_test_balanced/multi_label", "rbp_v19/train_imbalanced_test_balanced_multi", True),
    ("gex", "imbal_single", 19, 4, "gex/real_human/train_imbalanced_test_balanced/single_label", "v2", "gex/real_human/train_imbalanced_test_balanced/single_label", "gex_v19/train_imbalanced_test_balanced/single_label", False),
]
CLASSDIST_KEY = {("tf", "mislabeled"): None,  # tf classdist is keyed by noise 'none' only in the export; skip
                 ("tf", "none"): ("tf", (("balanced", True), ("noise", "none"))),
                 ("rbp", "bal_single"): ("rbp", (("balanced", True), ("multi_label", False))),
                 ("rbp", "bal_multi"): ("rbp", (("balanced", True), ("multi_label", True))),
                 ("gex", "bal_single"): ("gex", (("balanced", True),)),
                 ("rbp", "imbal_single"): ("rbp", (("balanced", False), ("multi_label", False))),
                 ("rbp", "imbal_multi"): ("rbp", (("balanced", False), ("multi_label", True))),
                 ("gex", "imbal_single"): ("gex", (("balanced", False),))}


def softmax(a):
    z = a - a.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def rgz(p):
    with gzip.open(p, "rb") as fh:
        return pickle.load(fh)


def wgz(obj, p):
    tmp = p + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        pickle.dump(obj, fh, protocol=4)
    os.replace(tmp, p)


def topn(y, it, c, n):
    idx = np.where(y.argmax(1) == c)[0]
    sub = it[:, idx]; sd = sub.std(0); pred = sub.mean(0).argmax(1)
    return (-sd[np.arange(len(idx)), pred]).argsort()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--datasets", required=True, help="perturbed splits with held-out test sets (datasets_v2)")
    ap.add_argument("--datasets-ref", required=True, help="original splits (datasets)")
    ap.add_argument("--inference", default=None, help="dir with virus_v20.pkl / sensitivity_v20.pkl from export_inference.py")
    ap.add_argument("--bundle-in", default="figure_data_bundle_mcdfix")
    ap.add_argument("--bundle-out", default="figure_data_bundle_v20")
    ap.add_argument("--raw", default="cluster_preds/merged_v20")
    ap.add_argument("--version", default="20")
    a = ap.parse_args()
    os.makedirs(a.bundle_out, exist_ok=True)
    for f in glob.glob(os.path.join(a.bundle_in, "*")):
        shutil.copy2(f, os.path.join(a.bundle_out, os.path.basename(f)))
    comps = {t: rgz(os.path.join(a.bundle_out, f"comparison_{t}.pkl.gz")) for t in ("tf", "rbp", "gex")}
    labels = {t: rgz(os.path.join(a.bundle_out, f"labels_{t}.pkl.gz")) for t in ("tf", "rbp", "gex")}
    classdist = rgz(os.path.join(a.bundle_out, "classdist.pkl.gz"))
    missing = []
    for task, setting, ver, k, rsub, droot, dsub, rawsub, multi in SETTINGS:
        dsroot = a.datasets_ref if droot == "ref" else a.datasets
        comp = comps[task]
        for seed in SEEDS:
            d = pickle.load(open(os.path.join(dsroot, dsub, f"{k}class_split{seed}.pickle"), "rb"))
            y = np.asarray(d["y_test"])
            labels[task][(setting, seed)] = y.astype(np.float32)
            ck = CLASSDIST_KEY[(task, setting)]
            if ck is not None:
                classdist[(ck[0], ck[1], seed)] = {p: np.asarray(d[p]).sum(axis=0).astype(int).tolist() for p in ("y_train", "y_valid", "y_test")} | \
                    {"_n": {p: int(len(d[p])) for p in ("y_train", "y_valid", "y_test")}}
            for model in ("mcd", "bnn", "ens"):
                files = glob.glob(os.path.join(a.results, rsub, f"{k}class_split{seed}", f"iter_preds_{model}_v_{a.version}_s_{seed}_*.pickle"))
                if not files:
                    missing.append(f"{task}/{setting}/s{seed}/{model}"); continue
                it = np.asarray(pickle.load(open(files[0], "rb")), dtype=np.float32)
                if model == "bnn":
                    it = (1 / (1 + np.exp(-it))) if multi else softmax(it)
                assert it.shape[1] == len(y), (it.shape, y.shape)
                key = f"v_{ver}_s_{seed}_{setting}_{model}"
                comp["mean"][key] = it.mean(0); comp["sd"][key] = it.std(0)
                for name, n in (("top50", 50), ("top100", 100)):
                    if name in comp:
                        comp[name][key] = {c: topn(y, it, c, n) for c in range(k)}
                if "top10p" in comp:
                    comp["top10p"][key] = {c: topn(y, it, c, int(y.sum(0)[c] / 10)) for c in range(k)}
                raw_dir = os.path.join(a.raw, rawsub, f"{k}class_split{seed}")
                os.makedirs(raw_dir, exist_ok=True)
                pickle.dump(it if model != "bnn" else np.asarray(pickle.load(open(files[0], "rb")), dtype=np.float32),
                            open(os.path.join(raw_dir, f"iter_preds_{model}_v_{ver if task != 'rbp' else 19}_s_{seed}_{it.shape[0]}.pickle"), "wb"), protocol=4)
                sdp = comp["sd"][key][np.arange(len(y)), comp["mean"][key].argmax(1)]
                acc = float((comp["mean"][key].argmax(1) == y.argmax(1)).mean()) if not multi else float(((comp["mean"][key] > 0.5) == y).mean())
                print(f"  {key:34s} n={len(y):5d} acc {acc:.3f}  median SD@pred {np.median(sdp):.2e}", flush=True)
        comp["_v20"] = f"{setting} replaced by v{a.version} runs (test held out)"
    for t in ("tf", "rbp", "gex"):
        wgz(comps[t], os.path.join(a.bundle_out, f"comparison_{t}.pkl.gz"))
        wgz(labels[t], os.path.join(a.bundle_out, f"labels_{t}.pkl.gz"))
    wgz(classdist, os.path.join(a.bundle_out, "classdist.pkl.gz"))
    if a.inference:
        sp = os.path.join(a.inference, "sensitivity_v20.pkl")
        vfiles = sorted(glob.glob(os.path.join(a.inference, "virus_v20_s*.pkl")))
        if vfiles:
            virus = {}
            for f in vfiles:
                virus.update(pickle.load(open(f, "rb")))
            wgz(virus, os.path.join(a.bundle_out, "virus.pkl.gz")); print(f"virus section replaced ({len(vfiles)} seeds)")
        if os.path.exists(sp):
            wgz(pickle.load(open(sp, "rb")), os.path.join(a.bundle_out, "sensitivity.pkl.gz")); print("sensitivity section replaced")
    # raw tree: add the reference-setting passes next to the replaced ones so the re-analysis scripts find pairs
    for src, dst in (("cluster_preds/merged_mcdfix/tf_v18/balanced/single_label/no_noise", "tf_v18/balanced/single_label/no_noise"),
                     ("cluster_preds/merged_mcdfix/rbp_v19/balanced", "rbp_v19/balanced"),
                     ("cluster_preds/merged_mcdfix/gex_v19/balanced/single_label", "gex_v19/balanced/single_label")):
        if os.path.isdir(src):
            for f in glob.glob(os.path.join(src, "*", "*.pickle")):
                rel = os.path.relpath(f, src); out = os.path.join(a.raw, dst, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if not os.path.exists(out):
                    shutil.copy2(f, out)
    print("missing:", missing if missing else "none")
    print(f"done -> {a.bundle_out}, raw passes in {a.raw}")


if __name__ == "__main__":
    main()
