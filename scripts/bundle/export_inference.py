"""Inference-only exports that depend on the retrained REFERENCE models (v20):

  virus        coronavirus OOD: per (seed, model) the in-distribution SD/mean on the balanced RBP
               test set and the SD on each of the eight genomes          -> <out>/virus_v20.pkl
  sensitivity  TF missing-motif: per (seed, model) SD/mean of the clean-trained TF model on the
               no-motif background set                                    -> <out>/sensitivity_v20.pkl

Uses the figure pipeline's own loaders (UQFIG_* variables, see docs/REPRODUCING.md), pointed at models_v20 / version 20.
MC-dropout passes are run with dropout active (model(x, training=True)); BNN via its weights
file (tf_keras); ENS members deterministic. GPU recommended for the genomes.

    unset UQFIG_BUNDLE
    python3 export_inference.py --models /path/to/models_v20 --out inference_v20 [--skip-virus]
Run it as a batch job; the genome inference takes hours.
"""
import argparse, gc, os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "uq_genomics", "figures"))
import config_paths as CP        # noqa: E402
import data_access as DA         # noqa: E402
import plots_virus as PV         # noqa: E402

N_ITER = 100


def passes(model, x, n_iter, training, batch=4096):
    """n_iter forward passes, called directly on large batches (no predict(): it is slow in eager
    mode and its graph compilation is what aborted on the V100 nodes)."""
    out = []
    for _ in range(n_iter):
        out.append(np.concatenate([np.asarray(model(x[i:i + batch], training=training)) for i in range(0, len(x), batch)]))
    return np.stack(out)


_MODEL_CACHE = {}


def _load(task, seed, setting, model_type):
    key = (task, seed, setting, model_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    ns, cfg = DA.try_import_src()
    if ns is None:
        raise RuntimeError("src/ not importable: set UQFIG_SRC_DIR / UQFIG_BASE_CONFIG")
    spec = CP.VERSIONS[task]; kw = DA.setting_to_split_args(task, setting)
    cfg["general"]["path_models"] = CP.models_root_for(task) + os.sep
    cfg["general"]["random_seed"] = str(seed); cfg["general"]["version"] = str(spec["version"])
    cfg["input_data_selection"]["classification_task"] = task
    cfg["input_data_selection"]["data_type_species"] = spec["species"]
    cfg["input_data_selection"]["n_classes"] = str(spec["n_classes"])
    cfg["input_data_selection"]["balanced_dataset"] = str(kw["balanced"])
    cfg["input_data_selection"]["multi_label"] = str(kw["multi_label"])
    cfg["input_data_selection"]["noise"] = kw["noise"]
    cfg["model_selection"]["model_type"] = model_type
    utils = ns["utils"]
    m = list(utils.load_ensemble_model(cfg, None).values()) if model_type == "ens" else utils.load_model(cfg, None)
    _MODEL_CACHE.clear()          # keep one model set in memory
    _MODEL_CACHE[key] = m
    return m


def infer(task, seed, setting, model_type, x):
    """100 passes (ENS: 10 members x 10 identical passes, as stored for the paper); probabilities."""
    import tensorflow as tf
    mdl = _load(task, seed, setting, model_type)
    if model_type == "ens":
        reps = N_ITER // len(mdl)
        return np.concatenate([np.repeat(passes(m, x, 1, False), reps, axis=0) for m in mdl])
    if model_type == "mcd":
        return passes(mdl, x, N_ITER, True)
    return tf.nn.softmax(passes(mdl, x, N_ITER, False), axis=-1).numpy()   # bnn: Flipout samples every call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="models_v20 root (all three tasks under it)")
    ap.add_argument("--version", type=int, default=20)
    ap.add_argument("--out", default="inference_v20")
    ap.add_argument("--skip-virus", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="virus: only these seeds (one output file per seed)")
    a = ap.parse_args()
    os.environ.pop("TF_USE_LEGACY_KERAS", None)
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    import tensorflow as tf
    tf.config.optimizer.set_jit(False)
    os.makedirs(a.out, exist_ok=True)
    for t in ("rbp", "tf", "gex"):
        CP.VERSIONS[t]["version"] = a.version
    CP.MODELS_ROOT_RBP = CP.MODELS_ROOT_TF = CP.MODELS_ROOT_GEX = a.models.rstrip("/")

    if not a.skip_sensitivity:
        print("== TF missing-motif (clean-trained v20 models)")
        x = DA.load_tf_nomotif_testset()
        sens = {}
        for seed in CP.VERSIONS["tf"]["seeds"]:
            for m in CP.MODELS:
                it = infer("tf", seed, "none", m, x)
                sens[("tf", "no_motif", seed, m)] = (it.std(0).astype(np.float32), it.mean(0).astype(np.float32))
                sdp = it.std(0)[np.arange(it.shape[1]), it.mean(0).argmax(1)]
                print(f"  s{seed} {m}: median SD@pred {np.median(sdp):.2e}", flush=True)
                del it; gc.collect()
        pickle.dump(sens, open(os.path.join(a.out, "sensitivity_v20.pkl"), "wb"), protocol=4)

    if not a.skip_virus:
        print("== virus (balanced single-label RBP v20 models)")
        found, missing = PV._virus_inventory()
        print(f"  genomes: {len(found)} found" + (f", missing {missing}" if missing else ""))
        for seed in (a.seeds or CP.VERSIONS["rbp"]["seeds"]):
            t0 = time.time()
            dtest = DA.load_split("rbp", "real_human", 3, seed, balanced=True, multi_label=False, keys=("x_test",))["x_test"]
            out = {}
            for m in CP.MODELS:
                it = infer("rbp", seed, "bal_single", m, dtest)
                out[(seed, m)] = {"sd_in": it.std(0).astype(np.float32), "mean_in": it.mean(0).astype(np.float32), "per_virus": {}}
                del it
            # same window subset for every method, as in plots_virus._collect: one rng per seed, genomes in order
            rng = np.random.default_rng(0)
            for v in CP.VIRUSES:
                x = DA.load_virus(v)
                if x is None:
                    continue
                if PV.MAX_VIRUS_WINDOWS and len(x) > PV.MAX_VIRUS_WINDOWS:
                    keep = rng.choice(len(x), PV.MAX_VIRUS_WINDOWS, replace=False); keep.sort(); x = x[keep]
                for m in CP.MODELS:
                    itv = infer("rbp", seed, "bal_single", m, x)
                    out[(seed, m)]["per_virus"][v] = itv.std(0).astype(np.float32)
                    del itv
                del x; gc.collect()
                print(f"  s{seed} {v}: done ({time.time()-t0:.0f}s)", flush=True)
            pickle.dump(out, open(os.path.join(a.out, f"virus_v20_s{seed}.pkl"), "wb"), protocol=4)
            print(f"  seed {seed} written ({time.time()-t0:.0f}s)", flush=True)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
