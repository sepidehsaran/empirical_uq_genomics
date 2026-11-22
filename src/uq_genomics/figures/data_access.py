"""
data_access.py - loading datasets, aggregated predictive distributions, and (optionally)
the user's own src/ library for model loading + inference.

Dataset layout replicates utils.read_splitted_dataset() exactly:
   <root>/<task>/<species>/<balance>/<label>[/<noise>]/<n>class_split<seed>.pickle
so the same pickles the paper used are read, with the same names.
"""
import gzip
import os
import pickle
import numpy as np
import config_paths as CP


# ------------------------------------------------------------------------- data bundle
# export_bundle.py writes a portable copy of everything the figures need. When
# UQFIG_BUNDLE points at one, the loaders below read from it, so the whole pipeline runs
# on a laptop. The cluster tree still takes precedence when it is actually reachable,
# which keeps cluster runs authoritative and makes the bundle a pure fallback.
_BUNDLE_CACHE = {}


def bundle_part(name):
    """Load one section of the bundle, or None. Cached; safe to call in a loop."""
    if not CP.BUNDLE:
        return None
    if name in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[name]
    path = os.path.join(CP.BUNDLE, name + ".pkl.gz")
    obj = None
    if os.path.exists(path):
        try:
            with gzip.open(path, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:                     # corrupt/partial copy
            print(f"  ! bundle section '{name}' unreadable: {type(exc).__name__}: {exc}")
    _BUNDLE_CACHE[name] = obj
    return obj


def bundle_status():
    """(present, missing) section names - used by run_all.py --check"""
    if not CP.BUNDLE:
        return [], []
    want = ["comparison_rbp", "comparison_tf", "comparison_gex", "labels_rbp",
            "labels_tf", "labels_gex", "classdist", "umap", "sensitivity", "virus",
            "app27"]
    present = [w for w in want
               if os.path.exists(os.path.join(CP.BUNDLE, w + ".pkl.gz"))]
    return present, [w for w in want if w not in present]


# ------------------------------------------------------------------ dataset setting name
def setting_dirs(task, species, balanced=True, multi_label=False, noise="none"):
    """
    Candidate directories for one setting, in preference order.

    RBP has THREE imbalanced-related trees on disk:
        balanced/                          test set 574   (single-label)
        imbalanced/                        test set 2411  <- imbalanced test set
        train_imbalanced_test_balanced/    test set 574   <- balanced test set
    utils.read_splitted_dataset() maps balanced_dataset=False onto the third, but the
    aggregated dicts in results_tf_paper were computed against the SECOND. Both are
    returned so the caller can pick the one whose row count matches the dict.
    """
    if task == "rbp":
        lab = "multi_label" if multi_label else "single_label"
        if balanced:
            return [os.path.join(task, species, "balanced", lab)]
        return [os.path.join(task, species, "imbalanced", lab),
                os.path.join(task, species, "train_imbalanced_test_balanced", lab)]
    if task == "gex":
        if balanced:
            return [os.path.join(task, species, "balanced", "single_label")]
        return [os.path.join(task, species, "imbalanced", "single_label"),
                os.path.join(task, species, "train_imbalanced_test_balanced",
                             "single_label")]
    return [setting_dir(task, species, balanced, multi_label, noise)]


def setting_dir(task, species, balanced=True, multi_label=False, noise="none"):
    if task == "rbp":
        bal = "balanced" if balanced else "train_imbalanced_test_balanced"
        lab = "multi_label" if multi_label else "single_label"
        return os.path.join(task, species, bal, lab)
    if task == "tf":
        bal = "balanced" if balanced else "imbalanced"
        lab = "multi_label" if multi_label else "single_label"
        noise_dir = {"none": "no_noise",
                     "mislabeled": "train_mislabeled_test_clean"}.get(noise, "no_motif")
        return os.path.join(task, species, bal, lab, noise_dir)
    if task == "gex":
        bal = "balanced" if balanced else "train_imbalanced_test_balanced"
        return os.path.join(task, species, bal, "single_label")
    raise ValueError(task)


def _dataset_root(task):
    return CP.PATH_DATASETS_GEX if task == "gex" else CP.PATH_DATASETS_SEQ


def split_paths(task, species, n_classes, seed, balanced=True, multi_label=False,
                noise="none"):
    """all candidate pickle paths for this setting, in preference order"""
    root = _dataset_root(task)
    if not root:
        return []
    return [os.path.join(root, d, f"{n_classes}class_split{seed}.pickle")
            for d in setting_dirs(task, species, balanced, multi_label, noise)]


def split_path(task, species, n_classes, seed, balanced=True, multi_label=False,
               noise="none"):
    cands = split_paths(task, species, n_classes, seed, balanced, multi_label, noise)
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0] if cands else ""


def load_split(task, species, n_classes, seed, balanced=True, multi_label=False,
               noise="none", keys=None):
    """
    returns dict with x_train,y_train,x_valid,y_valid,x_test,y_test (or None).
    Pass keys=('x_test','y_test') to keep only what you need - the x arrays are large
    (a TF train split is ~40 MB) and holding them all is what makes long runs die.
    """
    p = ""
    for c in split_paths(task, species, n_classes, seed, balanced, multi_label, noise):
        if os.path.exists(c):
            p = c
            break
    if not p:
        return None
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    wanted = keys or ("x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test")
    out = {k: d[k] for k in wanted if k in d}
    del d
    return out


_Y_CACHE = {}


def load_y_test(task, species, n_classes, seed, balanced=True, multi_label=False,
                noise="none", expect_n=None):
    """
    Labels for one setting. Cached (they are small and read by almost every figure).

    expect_n : if given, and several candidate splits exist for this setting, the one
               with that many test rows wins. This is what disambiguates the RBP
               imbalanced case, where 'imbalanced/' (2411) and
               'train_imbalanced_test_balanced/' (574) both exist and the aggregated
               dicts were built against the former.
    """
    k = (task, species, n_classes, seed, balanced, multi_label, noise, expect_n)
    if k in _Y_CACHE:
        return _Y_CACHE[k]
    def _from_bundle():
        """y_test out of the bundle, keyed by (setting, seed)"""
        lab = bundle_part(f"labels_{task}")
        if not lab:
            return None
        for (setting, s), y in lab.items():
            if s != seed:
                continue
            args = setting_to_split_args(task, setting)
            if (args.get("balanced", True) == balanced
                    and args.get("multi_label", False) == multi_label
                    and args.get("noise", "none") == noise):
                return np.asarray(y)
        return None

    # Collect every candidate, then choose. Taking the first that merely EXISTS is
    # wrong: RBP imbalanced has two directories with different row counts (2411 and
    # 574), and a bundle may be present alongside a stale local split. Whichever
    # candidate matches the aggregated entry's row count is the right one, wherever
    # it came from.
    cands = []
    for c in split_paths(task, species, n_classes, seed, balanced, multi_label, noise):
        if not os.path.exists(c):
            continue
        with open(c, "rb") as fh:
            cands.append(np.asarray(pickle.load(fh)["y_test"]))
        if expect_n is not None and cands[-1].shape[0] == expect_n:
            break                          # exact match, stop reading pickles
    b = _from_bundle()
    if b is not None:
        cands.append(b)

    best = None
    if expect_n is not None:
        best = next((y for y in cands if y.shape[0] == expect_n), None)
    if best is None:
        best = cands[0] if cands else None  # no expectation, or nothing matched
    _Y_CACHE[k] = best
    return best


# ------------------------------------------------- aggregated predictive distributions
# Base names. The actual files are NOT consistently named across the three trees:
#   rbp (results_tf_paper)    -> dict_iter_mean.pickle
#   tf  (results_tf_paper_2)  -> dict_iter_mean.pickle   (+ dict_iter_mean_v18.pickle)
#   gex (results_tf_paper_3)  -> dict_iter_mean_v19.pickle   <- NO unsuffixed file
# so each entry is resolved by glob, preferring the suffix that matches the use case's
# version, then the unsuffixed name, then whatever is newest.
DICT_STEMS = {
    "mean":  "dict_iter_mean",
    "sd":    "dict_iter_sd",
    "top50": "dict_idx_most_uncertain_samples_top_50_count",
    "top100":"dict_idx_most_uncertain_samples_top_100_count",
    "top10p":"dict_idx_most_uncertain_samples_top_10_percent",
}
DICT_FILES = {k: v + ".pickle" for k, v in DICT_STEMS.items()}


def _resolve_dict_file(directory, stem, version):
    """
    Pick the right file for `stem` in `directory`.
    Preference: <stem>_v<version>.pickle  >  <stem>.pickle  >  newest <stem>*.pickle
    """
    import glob as _glob
    exact_ver = os.path.join(directory, f"{stem}_v{version}.pickle")
    if os.path.exists(exact_ver):
        return exact_ver
    plain = os.path.join(directory, f"{stem}.pickle")
    if os.path.exists(plain):
        return plain
    cands = sorted(_glob.glob(os.path.join(directory, f"{stem}*.pickle")),
                   key=os.path.getmtime, reverse=True)
    return cands[0] if cands else ""


_COMP_CACHE = {}


def clear_caches():
    """drop cached labels and aggregated dicts (called between stages)"""
    _Y_CACHE.clear()
    _COMP_CACHE.clear()
    _BUNDLE_CACHE.clear()


def load_comparison(task):
    """
    Load the aggregated dicts written by
    2024_figures_section_3_1_agreement.ipynb cell 40 (and its RBP/GEX twins).
    Returns {'mean':..,'sd':..,'top100':..} or {} when RESULTS_ROOT is unset/missing.
    """
    spec = CP.VERSIONS[task]
    d = CP.comparison_dir(spec["task"], spec["species"])
    if not d or not os.path.isdir(d):
        b = bundle_part(f"comparison_{task}")        # laptop path
        return b if b else {}
    ck = (task, d)
    if ck in _COMP_CACHE:
        return _COMP_CACHE[ck]
    version = spec["version"]
    out = {}
    for k, stem in DICT_STEMS.items():
        p = _resolve_dict_file(d, stem, version)
        if p:
            with open(p, "rb") as fh:
                out[k] = pickle.load(fh)
            out.setdefault("_files", {})[k] = os.path.basename(p)
    _COMP_CACHE[ck] = out
    return out


def dict_key(task, seed, setting, model):
    """
    Key convention in the aggregated dicts. NOTE the two conventions:
      RBP / GEX : v_<ver>_s_<seed>_<bal|imbal>_<single|multi>_<model>
      TF        : v_<ver>_s_<seed>_<none|mislabeled>_<model>       (noise, not balance)
    """
    v = CP.VERSIONS[task]["version"]
    return f"v_{v}_s_{seed}_{setting}_{model}"


def setting_to_split_args(task, setting):
    """map a dict 'setting' string onto load_split() kwargs"""
    if task == "tf":
        return dict(balanced=True, multi_label=False, noise=setting)
    bal = not setting.startswith("imbal")
    multi = setting.endswith("multi")
    return dict(balanced=bal, multi_label=multi, noise="none")


def setting_label(task, setting):
    if task == "tf":
        # Parallel to the RBP/GEX labels: every TF condition is balanced, so say so
        # explicitly rather than leaving the reader to infer it from the other columns.
        return {"none": "balanced, clean", "mislabeled": "balanced, mislabeled",
                "no_motif": "balanced, missing motif"}.get(setting, setting)
    return {"bal_single": "balanced, single-label",
            "imbal_single": "imbalanced, single-label",
            "bal_multi": "balanced, multi-label",
            "imbal_multi": "imbalanced, multi-label"}.get(setting, setting)


def load_tf_nomotif_testset():
    """standalone background-sequence test array for the TF missing-motif condition"""
    if not CP.TF_NOMOTIF_TESTSET or not os.path.exists(CP.TF_NOMOTIF_TESTSET):
        return None
    with open(CP.TF_NOMOTIF_TESTSET, "rb") as fh:
        d = pickle.load(fh)
    if isinstance(d, dict):
        for k in ("x_test_no_motif", "x_test", "x"):
            if k in d:
                return np.asarray(d[k])
        return None
    return np.asarray(d)


# ----------------------------------------------------------------------------- virus
def load_virus(name):
    if not CP.VIRUS_DIR:
        return None
    p = os.path.join(CP.VIRUS_DIR, f"{name}_testdata.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------- optional: their src/ for inference
_SRC_READY = None


def try_import_src():
    """
    Make the user's own src/ importable and build a configparser matching what
    load_model / load_ensemble_model / infer_probabilistic_model expect.
    Returns (module_namespace, config) or (None, None).
    """
    global _SRC_READY
    if _SRC_READY is not None:
        return _SRC_READY
    if not CP.SRC_DIR or not CP.BASE_CONFIG or not (
            CP.MODELS_ROOT or CP.MODELS_ROOT_RBP or CP.MODELS_ROOT_TF or CP.MODELS_ROOT_GEX):
        _SRC_READY = (None, None)
        return _SRC_READY
    import sys, configparser
    sys.path.insert(0, CP.SRC_DIR)
    try:
        import importlib
        ns = {}
        for m in ("utils", "models", "eval_functions"):
            ns[m] = importlib.import_module(m)
        cfg = configparser.ConfigParser()
        cfg.read([CP.BASE_CONFIG])
        cfg["general"]["path_datasets"] = (CP.PATH_DATASETS_SEQ or "") + os.sep
        cfg["general"]["path_models"] = (CP.models_root_for("rbp") or CP.MODELS_ROOT) + os.sep
        cfg["general"]["path_results"] = (CP.RESULTS_ROOT or CP.OUT_DIR) + os.sep
        cfg["general"]["verbose"] = "False"
        _SRC_READY = (ns, cfg)
    except Exception as exc:                                    # TF missing, etc.
        print(f"    [src import failed: {exc}]")
        _SRC_READY = (None, None)
    return _SRC_READY


def infer_with_models(task, species, n_classes, seed, setting, model_type, x,
                      n_iter=100, n_iter_ens=10):
    """
    Run MC inference with a PRETRAINED model, exactly as the notebooks do
    (ens: concatenate n_iter_ens passes per member; bnn: apply softmax afterwards).
    Returns an array (n_iter, n_samples, n_classes) or None if unavailable.
    """
    ns, cfg = try_import_src()
    if ns is None:
        return None
    import numpy as _np
    kw = setting_to_split_args(task, setting)
    # per-use-case checkpoint root (v_17 rbp / v_18 tf / v_19 gex live in different trees)
    mroot = CP.models_root_for(task)
    if mroot:
        cfg["general"]["path_models"] = mroot + os.sep
    cfg["general"]["random_seed"] = str(seed)
    cfg["general"]["version"] = str(CP.VERSIONS[task]["version"])
    cfg["input_data_selection"]["classification_task"] = task
    cfg["input_data_selection"]["data_type_species"] = species
    cfg["input_data_selection"]["n_classes"] = str(n_classes)
    cfg["input_data_selection"]["balanced_dataset"] = str(kw["balanced"])
    cfg["input_data_selection"]["multi_label"] = str(kw["multi_label"])
    cfg["input_data_selection"]["noise"] = kw["noise"]
    cfg["model_selection"]["model_type"] = model_type
    utils, ev = ns["utils"], ns["eval_functions"]
    try:
        if model_type == "ens":
            members = utils.load_ensemble_model(cfg, None)
            parts = [ev.infer_probabilistic_model(m, x, n_iter_ens)
                     for m in members.values()]
            return _np.concatenate(parts)
        mdl = utils.load_model(cfg, None)
        out = ev.infer_probabilistic_model(mdl, x, n_iter)
        if model_type == "bnn":
            import tensorflow as tf
            out = tf.nn.softmax(out, axis=-1).numpy()
        return out
    except Exception as exc:
        print(f"    [inference failed for {task}/{setting}/{model_type} s{seed}: {exc}]")
        return None
