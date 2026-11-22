"""
config_paths.py - every input and output location the figure pipeline uses, in one place.

Nothing here is machine-specific. Every location is read from an environment variable
with the prefix ``UQFIG_``; the only defaults are relative to this repository:

    UQFIG_BUNDLE   figure_data_bundle/   released summary data (all notebooks except 06)
    UQFIG_PREDS    cluster_preds/        raw per-pass predictions (notebook 06, Fig 10a)
    UQFIG_OUT_DIR  out/                  where figures are written

The remaining variables point at the raw experiment trees (datasets, trained models,
per-run results). They are needed only to rebuild the bundle after retraining; with the
released bundle in place they can stay unset. ``docs/REPRODUCING.md`` lists them.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def _env(name, default=""):
    """UQFIG_<name>, or the default; a value that points nowhere is kept so that the
    report() below can say what is missing rather than silently falling back"""
    return os.environ.get("UQFIG_" + name, "") or default


def _existing(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return ""


# ----------------------------------------------------------------------------------
# 1. released data
# ----------------------------------------------------------------------------------
# Summary bundle produced by export_bundle.py. When present, data_access serves
# predictions, labels, UMAP coordinates and the inference-derived arrays from here, so
# every bundle-backed figure regenerates without the experiment trees.
BUNDLE = _existing(_env("BUNDLE", os.path.join(REPO_ROOT, "figure_data_bundle")))

# Raw per-pass predictions (n_iter, n_samples, n_classes). Needed for the figures whose
# uncertainty score is defined in probability space and cannot be recovered from a
# stored summary: notebook 06 and the RBP panel of the class-imbalance figure.
PREDS = _env("PREDS", os.path.join(REPO_ROOT, "cluster_preds"))

# ----------------------------------------------------------------------------------
# 2. experiment trees (optional; for rebuilding the bundle after retraining)
# ----------------------------------------------------------------------------------
# The training code, imported verbatim for model loading and inference.
SRC_DIR = _env("SRC_DIR", os.path.join(REPO_ROOT, "src", "uq_genomics", "experiments"))
BASE_CONFIG = _env("BASE_CONFIG", os.path.join(REPO_ROOT, "config", "config.ini"))

# Datasets, in the layout utils.read_splitted_dataset() builds:
#   <PATH_DATASETS>/<task>/<species>/<balanced|...>/<single_label|...>[/<noise>]/<n>class_split<seed>.pickle
PATH_DATASETS_SEQ = _env("PATH_DATASETS_SEQ")      # rbp + tf splits
PATH_DATASETS_GEX = _env("PATH_DATASETS_GEX", PATH_DATASETS_SEQ)

# The "missing motif" condition is not a train/val/test split but a standalone test
# array of background sequences without a planted motif (key 'x_test_no_motif'),
# evaluated with the model trained on the clean balanced data.
TF_NOMOTIF_TESTSET = _env("TF_NOMOTIF_TESTSET") or _existing(
    os.path.join(PATH_DATASETS_SEQ, "tf", "tfdstest_set_no_motif_150_2000_2024.pickle") if PATH_DATASETS_SEQ else "",
    os.path.join(PATH_DATASETS_SEQ, "tf", "test_set_no_motif_150_2000.pickle") if PATH_DATASETS_SEQ else "")

# One-hot encoded coronavirus genomes (OOD experiment) and the single-cell inputs.
VIRUS_DIR = _env("VIRUS_DIR") or (os.path.join(PATH_DATASETS_SEQ, "rbp", "real_virus") if PATH_DATASETS_SEQ else "")
ADATA_H5AD = _env("ADATA_H5AD") or (os.path.join(PATH_DATASETS_GEX, "adata.h5ad") if PATH_DATASETS_GEX else "")
LIAM_EMB_DIR = _env("LIAM_EMB_DIR") or (os.path.join(PATH_DATASETS_GEX, "gex", "liam_embeddings") if PATH_DATASETS_GEX else "")

# Per-run results: aggregated predictive distributions (dict_iter_mean / _sd / top-N)
# under <root>/<task>/<species>/comparison_results/. The three use cases may live in
# three different roots; RESULTS_ROOT is the fallback when everything is in one tree.
RESULTS_ROOT = _env("RESULTS_ROOT")
RESULTS_ROOT_RBP = _env("RESULTS_ROOT_RBP")
RESULTS_ROOT_TF = _env("RESULTS_ROOT_TF")
RESULTS_ROOT_GEX = _env("RESULTS_ROOT_GEX")


def results_root_for(task):
    """per-use-case results root, falling back to the shared one"""
    return {"rbp": RESULTS_ROOT_RBP, "tf": RESULTS_ROOT_TF, "gex": RESULTS_ROOT_GEX}.get(task, "") or RESULTS_ROOT


def comparison_dir(task, species):
    """<root>/<task>/<species>/comparison_results (the singular spelling is tolerated)"""
    root = results_root_for(task)
    if not root:
        return ""
    for name in ("comparison_results", "comparison_result"):
        cand = os.path.join(root, task, species, name)
        if os.path.isdir(cand):
            return cand
    return os.path.join(root, task, species, "comparison_results")


# Trained checkpoints; needed only for figures that re-run inference.
MODELS_ROOT = _env("MODELS_ROOT")
MODELS_ROOT_RBP = _env("MODELS_ROOT_RBP")
MODELS_ROOT_TF = _env("MODELS_ROOT_TF")
MODELS_ROOT_GEX = _env("MODELS_ROOT_GEX")


def models_root_for(task):
    return {"rbp": MODELS_ROOT_RBP, "tf": MODELS_ROOT_TF, "gex": MODELS_ROOT_GEX}.get(task, "") or MODELS_ROOT


# 27-RBP utility application: the single-label 27-class dataset and prediction pickles.
APP27_PRED_DIR = _env("APP27_PRED_DIR", PREDS)
APP27_DATASET = _env("APP27_DATASET") or _existing(
    os.path.join(PATH_DATASETS_SEQ, "real_imbalanced_singlelabel_27classlow_nonoise.pickle") if PATH_DATASETS_SEQ else "")

# ----------------------------------------------------------------------------------
# 3. figure sources that are images, not data (motif logos, framework overview).
#    They are reused as-is; the directory defaults to the manuscript's figures/ folder
#    if it has been copied next to this repository, and can be set explicitly.
# ----------------------------------------------------------------------------------
FIGDIR_EXISTING = _env("FIGDIR_EXISTING") or _existing(os.path.join(REPO_ROOT, "figures"))
# Cached UMAP coordinates for Fig. 2 (the full 69,249-cell embedding and the four selected
# types). They ship with the released bundle; the manuscript's figure folder is the
# fallback. Computing them from adata.h5ad (~6 GB) is possible but slow.
UMAP_COORDS_4CT = _env("UMAP_COORDS_4CT") or _existing(
    os.path.join(BUNDLE, "umap_liam_4_cell_types_coords.pkl") if BUNDLE else "",
    os.path.join(FIGDIR_EXISTING, "2024_gex_application", "umap_liam_4_cell_types_coords.pkl") if FIGDIR_EXISTING else "")
UMAP_COORDS_FULL = _env("UMAP_COORDS_FULL") or _existing(
    os.path.join(BUNDLE, "_full_xumap_labels.pkl") if BUNDLE else "",
    os.path.join(FIGDIR_EXISTING, "2024_gex_application", "_full_xumap_labels.pkl") if FIGDIR_EXISTING else "")

PASSTHROUGH = {
    "motifs_tf_rbp":      "rbp+tf_motifs_samples.png",
    "motifs_tf":          "2024_tf_simulated/tf_motifs_pwm_jasper.png",
    "motifs_rbp":         "2024_rbp_real/rbp_motifs_sample_deepripe.png",
    "framework_overview": "overal_experiment_overview_page_1k_res_swap.png",
    "xai_with_uq":        "XAI with UQ.png",
}

# ----------------------------------------------------------------------------------
# 4. output
# ----------------------------------------------------------------------------------
OUT_DIR = _env("OUT_DIR", os.path.join(REPO_ROOT, "out"))
LOG_DIR = os.path.join(OUT_DIR, "_logs")


def set_out_dir(path):
    """point the pipeline at a different output folder (used by run_all.py --out)"""
    global OUT_DIR, LOG_DIR
    OUT_DIR = os.path.abspath(os.path.expanduser(path))
    LOG_DIR = os.path.join(OUT_DIR, "_logs")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    return OUT_DIR


# ----------------------------------------------------------------------------------
# 5. experiment constants
# ----------------------------------------------------------------------------------
# per-use-case aggregation: (task, species, version, seeds, settings, n_classes, class_names)
VERSIONS = {
    "rbp": dict(task="rbp", species="real_human", version=17, n_classes=3,
                seeds=[7, 5, 11, 6, 9],
                settings=["bal_single", "imbal_single", "bal_multi", "imbal_multi"],
                class_names=["MBNL1", "PUM2", "QKI"]),
    "tf":  dict(task="tf",  species="simulated_human", version=18, n_classes=6,
                seeds=[7, 5, 11, 6, 9],
                settings=["none", "mislabeled"],          # NB: TF keys use the noise name
                class_names=["CTCF", "HNF4A", "JUN", "MEF2A", "MYC", "TAL1"]),
    "gex": dict(task="gex", species="real_human", version=19, n_classes=4,
                seeds=[7, 5, 11, 6, 9],
                settings=["bal_single", "imbal_single"],
                class_names=["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]),
}

MODELS = ["mcd", "bnn", "ens"]
# Short forms everywhere in the figures, matching the captions and body text.
MODEL_LABEL = {"mcd": "MCD", "bnn": "BNN", "ens": "ENS", "det": "DET"}
PAIRS = [("mcd", "bnn"), ("mcd", "ens"), ("bnn", "ens")]
PAIR_LABEL = {("mcd", "bnn"): "MCD-BNN", ("mcd", "ens"): "MCD-ENS", ("bnn", "ens"): "BNN-ENS"}

VIRUSES = ["SARS_CoV_1", "NC_045512", "wuhCor1", "MERS", "HKU1", "OC43", "NL63", "CoV229E"]
VIRUS_LABEL = {"SARS_CoV_1": "SARS-CoV-1", "NC_045512": "SARS-CoV-2 (NC_045512)",
               "wuhCor1": "SARS-CoV-2 (wuhCor1)", "MERS": "MERS-CoV", "HKU1": "HCoV-HKU1",
               "OC43": "HCoV-OC43", "NL63": "HCoV-NL63", "CoV229E": "HCoV-229E"}


def report():
    """(label, path, needed_for) for everything, with '' meaning NOT SET / NOT FOUND."""
    def found(p):
        return p if p and os.path.exists(p) else ""
    return [
        ("BUNDLE (UQFIG_BUNDLE)",       found(BUNDLE),             "every bundle-backed figure (notebooks 01-05, 07)"),
        ("PREDS (UQFIG_PREDS)",         found(PREDS),              "notebook 06; RBP panel of the class-imbalance figure"),
        ("FIGDIR_EXISTING",             found(FIGDIR_EXISTING),    "motif + overview passthrough panels"),
        ("UMAP_COORDS_FULL",            found(UMAP_COORDS_FULL),   "full-dataset UMAP (cached coords)"),
        ("UMAP_COORDS_4CT",             found(UMAP_COORDS_4CT),    "4-cell-type UMAP (cached coords)"),
        ("SRC_DIR",                     found(SRC_DIR),            "model loading + inference (rebuild only)"),
        ("BASE_CONFIG",                 found(BASE_CONFIG),        "model loading + inference (rebuild only)"),
        ("PATH_DATASETS_SEQ",           found(PATH_DATASETS_SEQ),  "RBP + TF splits (rebuild only)"),
        ("PATH_DATASETS_GEX",           found(PATH_DATASETS_GEX),  "single-cell splits (rebuild only)"),
        ("VIRUS_DIR",                   found(VIRUS_DIR),          "virus OOD inference (rebuild only)"),
        ("TF_NOMOTIF_TESTSET",          found(TF_NOMOTIF_TESTSET), "TF missing-motif inference (rebuild only)"),
        ("ADATA_H5AD",                  found(ADATA_H5AD),         "UMAP only if the cached coords are absent"),
        ("LIAM_EMB_DIR",                found(LIAM_EMB_DIR),       "single-cell embedding source (rebuild only)"),
        ("RESULTS_ROOT_RBP",            found(results_root_for("rbp")), "RBP per-run results (rebuild only)"),
        ("RESULTS_ROOT_TF",             found(results_root_for("tf")),  "TF per-run results (rebuild only)"),
        ("RESULTS_ROOT_GEX",            found(results_root_for("gex")), "single-cell per-run results (rebuild only)"),
        ("MODELS_ROOT_RBP",             found(models_root_for("rbp")),  "virus OOD inference (rebuild only)"),
        ("MODELS_ROOT_TF",              found(models_root_for("tf")),   "TF missing-motif inference (rebuild only)"),
        ("APP27_PRED_DIR",              found(APP27_PRED_DIR),     "27-RBP AP / PR-curve figures"),
    ]
