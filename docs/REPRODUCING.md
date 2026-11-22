# Reproducing the paper

This document maps every figure and table of the paper to the code that produces it,
says which data each one needs, and lists the environment variables the pipeline reads.
The step-by-step instructions for the common case (regenerate the figures from the
released data) are in the [README](../README.md).

## Figure map

Figure numbers refer to the published version. "Bundle" is `figure_data_bundle/`
(summary data, ~22 MB); "raw" is `cluster_preds/` (per-pass predictions, several
hundred MB). Output files are written to `out/`.

### Main text

| Figure | Content | Notebook / script | Needs | Output file |
|---|---|---|---|---|
| 1 | TF and RBP class distributions | `01_datasets` | bundle | `dataset_classdist_tf_*.png`, `dataset_classdist_rbp_*.png` |
| 2 | single-cell UMAP | `01_datasets` | bundle (cached UMAP coordinates) | `UMAP_liam_embedding_v2.pdf` |
| 3 | single-cell class distribution | `01_datasets` | bundle | `dataset_classdist_gex_*.png` |
| 4 | framework overview | drawn by hand | – | – |
| 5 | predictive performance | `02_performance` | bundle | `COMBINED_performance_ext_all_usecases.pdf` |
| 6 | top-100 overlap | `03_agreement` | bundle | `COMBINED_top100_all_usecases.pdf` |
| 7 | Kendall's tau | `03_agreement` | bundle | `COMBINED_kendall_all_usecases.pdf` |
| 8 | KS test | `03_agreement` | bundle | `COMBINED_ks_all_usecases.pdf` |
| 9 | TF label noise | `04_sensitivity` | bundle | `SDSHIFT_tf_mislabeled_pooled_log2.pdf` |
| 10 | class imbalance, RBP (a) and single-cell (b) | `04_sensitivity` | bundle; **raw** for panel a | `SDSHIFT_rbp_imbalance_predclass_pooled_log2.pdf`, `SDSHIFT_gex_imbalance_predclass_pooled_log2.pdf` |
| 11 | coronavirus OOD | `05_ood_virus` | bundle | `VIRUS_sdshift_pooled.pdf` |
| 12 | uncertainty filtering, 27-class RBP | `06_utility_filtering` | bundle + **raw** | `APP27_filtering_extended.pdf` |

### Supplement

| Figure / table | Content | Notebook / script | Needs | Output file |
|---|---|---|---|---|
| S1 | RBP and TF sequence logos | `scripts/simulated_tf/motif_logos_figure.py` | PWM files | `MOTIFS_rbp_tf.pdf` |
| S2 | SimDNA off-label motifs | `scripts/simulated_tf/simdna_xai_figure.py` † | simulated data + model | `XAI_with_UQ.pdf` |
| S3 | PR curves before/after filtering | `src/uq_genomics/experiments` (`run_mode = generate_evaluation`) | model | `*_pr_curve.png` |
| S4, S5 | attribution examples PUM2 / MBNL1, QKI | `scripts/rbp27_application/xai_examples.py` † | raw + model | `XAI_example_*.pdf` |
| S6 | confusion within protein complexes | `scripts/rbp27_application/complex_confusion.py` | raw | `COMPLEX_confusion.pdf` |
| S7 | AP against coverage | `06_utility_filtering` | bundle + raw | `APP27_ap_coverage.pdf` |
| S8 | TF missing motif | `04_sensitivity` | bundle | `SDSHIFT_tf_no_motif_pooled_log2.pdf` |
| S9 | TF sensitivity, raw scale | `04_sensitivity` | bundle | `SDSHIFT_tf_*_pooled_raw.pdf` |
| S10 | class imbalance, raw scale | `04_sensitivity` | bundle; raw for RBP | `SDSHIFT_*_imbalance_predclass_pooled_raw.pdf` |
| S11 | class size vs uncertainty shift | `04_sensitivity` | bundle | `DEPLETION_shift.pdf` |
| S12 | single-cell test cells coloured by uncertainty | `scripts/single_cell/umap_uncertainty.py` | bundle + single-cell splits + cached UMAP | `UMAP_test_uncertainty_bal_single_seed7.pdf` |
| S13 | reliability diagrams | `07_calibration` | bundle | `RELIABILITY_all_settings.pdf` |
| S14 | MNIST uncertain samples | `src/uq_genomics/experiments` (MNIST task) | MNIST + model | – |
| Table S1 | per-class coverage of the filter | `scripts/sensitivity/cross_seed_tables.py`, `scripts/supp_tables/tables_rbp27_ensemble.py` | bundle + raw | LaTeX |
| Table S2 | MC-dropout inference configurations | `scripts/mcd_inference/mcd_inference.py` †, `scripts/mcd_inference/ablation_table.py` | models | LaTeX |
| Tables S3, S4 | cross-seed shift tables | `scripts/sensitivity/cross_seed_tables.py seeds`, `scripts/sensitivity/sdshift_table_inputs.py`, `scripts/sensitivity/rbp_shift_shared_testset.py`, `scripts/supp_tables/tables_sensitivity.py` | bundle + raw | LaTeX |
| Table S5 | ensemble-size / number-of-passes | `scripts/sensitivity/cross_seed_tables.py ensemble`, `scripts/supp_tables/tables_rbp27_ensemble.py` | raw | LaTeX |
| Tables S6–S8 | leave-donor-out, per-sample counts and accuracy | `scripts/single_cell/leave_donor_out.py`, `scripts/single_cell/donor_breakdown.py`, `scripts/supp_tables/tables_single_cell.py` | retrained LDO models | LaTeX |
| Table S9 | OOD detection metrics | `scripts/ood/ood_metrics.py`, `scripts/supp_tables/tables_ood_calibration_entropy.py` | bundle | LaTeX |
| Table S10 | expected calibration error | `07_calibration` | bundle | `calibration_ece.csv` |
| Table S11 | entropy / mutual information | `scripts/uncertainty_scores/entropy_mi.py`, `scripts/supp_tables/tables_ood_calibration_entropy.py` | raw | LaTeX |
| Table S12 | label-noise rate sweep | `scripts/sensitivity/label_noise_sweep.py` | retrained sweep models | LaTeX |

† needs a GPU and the trained model. Each GPU script has a matching `run_*.sbatch`
wrapper in the same directory; all of them source `scripts/training/activate_env.sh`.

`scripts/make_all_figures.sh` runs every notebook step in one go, without Jupyter.

## Getting the data

Three artifacts are released separately from the code because of their size. Place
them (or point the environment variables at them) as follows.

| Artifact | Size | Location | Variable | Needed for |
|---|---|---|---|---|
| `figure_data_bundle/` | ~22 MB | repo root | `UQFIG_BUNDLE` | every notebook |
| `cluster_preds/` | ~725 MB | repo root | `UQFIG_PREDS` | notebook 06; Fig. 10a; several supplement tables |
| input datasets | several GB | anywhere | `UQFIG_PATH_DATASETS_SEQ`, `UQFIG_PATH_DATASETS_GEX` | retraining; Fig. S12 |

Download link: see the **Data** section of the README.

The bundle holds, per (task, setting, seed, method), the predictive mean and standard
deviation on the test set, the test labels, the class counts of every split, the
coronavirus inference, the 27-class predictions and the two cached UMAP coordinate files
(`_full_xumap_labels.pkl`, `umap_liam_4_cell_types_coords.pkl`).

`cluster_preds/` holds the raw `(n_iter, n_samples, n_classes)` arrays:
`iter_preds_bnn_v_10_s_9.pickle` (27-class task, shape `(1000, 6785, 27)`) and
`rbp_v19/<setting>/3class_split<seed>/iter_preds_<method>_*.pickle` (RBP balanced and
imbalanced training, both evaluated on the shared balanced test set).

### Rebuilding the bundle after retraining

```bash
export PYTHONPATH=$PWD/src
export UQFIG_RESULTS_ROOT_RBP=... UQFIG_RESULTS_ROOT_TF=... UQFIG_RESULTS_ROOT_GEX=...
export UQFIG_PATH_DATASETS_SEQ=... UQFIG_PATH_DATASETS_GEX=...
python -m uq_genomics.figures.export_bundle --out figure_data_bundle
```

`scripts/bundle/make_bundle.py` assembles the comparison entries, labels and class
counts of every setting with retrained runs; `scripts/bundle/export_inference.py`
runs the coronavirus and missing-motif inference with retrained models.
`notebooks/00_check_environment` (or `python -m uq_genomics.figures.run_all --check`)
prints which location each figure is missing.

## Environment variables

All optional. Defaults are relative to the repository root.

| Variable | Default | Purpose |
|---|---|---|
| `UQFIG_BUNDLE` | `figure_data_bundle/` | released summary data |
| `UQFIG_PREDS` | `cluster_preds/` | raw per-pass predictions |
| `UQFIG_OUT_DIR` | `out/` | where figures are written |
| `UQFIG_PALETTE` | `set2_terracotta` | colour palette (see below) |
| `UQFIG_GROUP_BOX` | per figure | `line`, `light` or `none`: frame around each use-case group in the combined figures |
| `UQFIG_PATH_DATASETS_SEQ`, `UQFIG_PATH_DATASETS_GEX` | – | dataset split pickles |
| `UQFIG_RESULTS_ROOT[_RBP|_TF|_GEX]` | – | per-run results trees |
| `UQFIG_MODELS_ROOT[_RBP|_TF|_GEX]` | – | trained checkpoints |
| `UQFIG_VIRUS_DIR`, `UQFIG_TF_NOMOTIF_TESTSET`, `UQFIG_ADATA_H5AD`, `UQFIG_LIAM_EMB_DIR` | derived from the dataset roots | inputs for inference-only figures |
| `UQFIG_UMAP_COORDS_FULL`, `UQFIG_UMAP_COORDS_4CT` | in the bundle | cached UMAP coordinates (Fig. 2, S12) |
| `UQFIG_FIGDIR_EXISTING` | `figures/` | hand-made panels copied as-is (motif images, overview) |
| `UQ_DATA`, `UQ_MODELS`, `UQ_RESULTS`, `UQ_LR` | – | used by the `*.sbatch` training wrappers |
| `NEPTUNE_API_TOKEN` | – | optional experiment tracking; never put it in a config file |

### The palette

Every figure carrying a reference/perturbed pair uses `set2_terracotta`, the default
(`UQFIG_PALETTE` selects another). The earlier
default paired Set2 green with Set2 blue-violet, which sit 5.8 apart in CIE76 under
deuteranopia — indistinguishable to roughly 8% of male readers, and on overlapping
density plots that means the two conditions cannot be separated at all. Terracotta
moves the worst case to 49.5 and the greyscale gap from 0.090 to 0.201. Run
`python src/uq_genomics/figures/palettes.py` for the full colour-blindness table.

Method labels are MCD, BNN and ENS everywhere, set once in `config_paths.MODEL_LABEL`.

## Three things worth knowing before you read the numbers

### The two use cases construct imbalance differently

The single-cell setting **depletes** its minority class: Erythroblast drops from 2673 to
587 training cells, a 78% reduction, while the other three classes grow. The RBP setting
leaves its smallest class essentially untouched — QKI goes from 825 to 808 sequences —
and creates the imbalance by **enriching** the majority classes instead, MBNL1 by 786%.

So only the single-cell experiment tests whether the harshly subsampled class becomes
the most uncertain one. Notebook 01 prints both tables.

### The test sets of the perturbed settings

Every perturbed setting (TF mislabeled, RBP imbalanced, single-cell imbalanced) is
evaluated on the test set of its reference setting, which is held out before the
perturbed training and validation sets are drawn
(`scripts/splits/build_perturbed_splits.py`). The RBP balanced and imbalanced
settings therefore share one 574-sequence test set and the single-cell settings one
3,600-cell test set, so the shift of a class's uncertainty distribution between
settings reflects the training set only.

### MC-dropout at inference

MC-dropout's uncertainty score depends on which dropout layers are active at inference
and at what rate. All MCD results use every dropout layer of the trained network at
its training rate, obtained by calling the model in training mode
(`model(x, training=True)`); `model.predict()` ignores the dropout flag under Keras 3
and returns the deterministic prediction. With dropout in the last layer only, the
score is near-degenerate (median SD 2e-5 on clean TF data, 4–8% of samples with exactly
zero SD); `scripts/mcd_inference/mcd_inference.py` reproduces the comparison of
configurations in the supplement (Table S2). With all layers active, the fraction of
zero-SD samples is below 1% in every setting.

### Uncertainty scores are computed in probability space

The threshold used in the filtering experiment — an uncertainty score below 0.1 — is
defined on probabilities. The stored BNN outputs are **logits**. Softmax is non-linear,
so the standard deviation of the logits cannot be converted; the softmax has to be
applied to each Monte-Carlo pass before taking the standard deviation.

This is why notebook 06 and the RBP panel of Fig. 10 need the raw
`(n_iter, n_samples, n_classes)` arrays rather than a stored summary, and why the filter
is applied on the SD of the *predicted* class across samples rather than per class.
Filtering per class instead leaves mean average precision flat at 0.20 and does not
reproduce the published figure.

## Analyses beyond the notebooks

`scripts/` has one directory per experiment (listed in the README). The scripts below
support the supplement figures on the 27-class RBP task and the simulated TF data. None of them can run from
`figure_data_bundle/` alone; each states what it needs when it starts.

| Script | Needs | GPU | Runtime |
|---|---|---|---|
| `simulated_tf/motif_logos_figure.py` | PWM files only | no | seconds |
| `rbp27_application/uncertainty_per_class.py` | raw predictions | no | seconds |
| `rbp27_application/complex_confusion.py` | raw predictions | no | seconds |
| `rbp27_application/uncertainty_covariates.py` | raw predictions + dataset | no | minutes |
| `rbp27_application/xai_examples.py` | + trained model | **yes** | ~2 min |
| `simulated_tf/simdna_xai_figure.py` | simulated data + its model | **yes** | ~5 min |
| `rbp27_application/xai_uncertainty.py` | + trained model | **yes** | ~1 h |
| `rbp27_application/xai_2x2_motif.py` | + trained model | **yes** | ~40 min |
| `rbp27_application/xai_motif_confusion.py` | + trained model | **yes** | ~2 h |

**`complex_confusion.py`** — tests whether misclassifications concentrate between RBPs
of the same protein complex (Fig. S6).

```bash
python scripts/rbp27_application/complex_confusion.py --preds cluster_preds/iter_preds_bnn_v_10_s_9.pickle --out out/complex
```

**`uncertainty_covariates.py`** — tests sequence-level explanations for high
uncertainty: GC content, homopolymer runs, k-mer novelty, distance to the training set,
and class size. Reported in the paper only as ruled out; the script is kept so the
exclusion can be checked.

**`xai_uncertainty.py`** — integrated gradients against the uncertainty score, with
dinucleotide- and mononucleotide-shuffled controls and three sharpness measures. The
sharpness-versus-uncertainty results were negative and are not in the paper; the
motif-versus-error result is.

**`xai_examples.py`** — the example attribution maps (Figs. S4, S5).

**`motif_logos_figure.py`** — the RBP and TF sequence-logo panel (Fig. S1), drawn from
the position weight matrices that generated the simulated data. Letter heights are
information content.

**`uncertainty_per_class.py`** — the uncertainty score per class, grouped by true label
and by predicted class.

**`simdna_xai_figure.py`** — attribution maps for simulated sequences that carry a
second, unplanted motif (Fig. S2).

**`xai_2x2_motif.py`**, **`xai_motif_confusion.py`** — whether the uncertainty score
carries motif information once correctness is held fixed, and whether attribution
explains confident errors.

### Negative results kept in the repository

The paper reports two findings from these analyses and states that the excluded
alternatives are available here. They are:

- **Motif sharpness does not track the uncertainty score** among correct predictions
  (Spearman −0.11, −0.08 and +0.03 for MBNL1, PUM2 and QKI).
- **Sequence composition does not explain uncertainty.** Across 27 classes and five
  covariates, significant correlations are at the rate expected by chance and none
  exceeds |rho| = 0.25.
- **Explanation instability across the posterior** correlates weakly with the score
  (−0.18 and −0.26) but the measure is saturated: mean pairwise cosine similarity
  between attribution maps is above 0.998 everywhere.
- **Uncertainty carries no motif information beyond correctness.** In the
  correctness-by-confidence 2x2, the uncertain-and-incorrect cell is the only one not
  significantly above its own dinucleotide-shuffled floor when pooled, but the effect
  does not survive per class: it holds in 11 of 18 classes, against 9 expected by
  chance, and a paired test across classes gives p = 0.15. Absent motif explains being
  *wrong*, not being *uncertain*.

## Tests

```bash
pip install pytest
pytest tests/
```

- `test_bundle_roundtrip` — builds a synthetic bundle and runs every bundle-backed
  figure from it (no data download needed)
- `test_metrics_match_src` — the figure metrics agree with the training code's own
  definitions
- `test_no_clipping` — no figure silently crops data outside its axis limits (runs
  only when a bundle is present)

## Retraining

Needs a GPU. The paper used an NVIDIA A40, Python 3.10, CUDA 12.3, cuDNN 8.9,
TensorFlow 2.16 and TensorFlow Probability 0.24.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt "tensorflow[and-cuda]==2.16.1"
cp config/config.example.ini config/config.ini      # then edit the three path_* keys
```

Drop the `tensorflow[and-cuda]` extra for a CPU-only install. The CUDA and cuDNN
libraries come from the `nvidia-*` wheels that extra pulls in, so no system CUDA module
is required — see `scripts/training/activate_env.sh`, which also sets `TF_USE_LEGACY_KERAS=1`
(TensorFlow Probability has not been ported to Keras 3) and puts the venv's `nvidia-*`
wheel directories ahead of any system CUDA on `LD_LIBRARY_PATH`. If the import fails
with `GLIBCXX_3.4.30 not found`, load a newer GCC runtime before submitting.

`config.ini` selects the task, the dataset setting and the model type. `run_mode`
switches between `training`, `generate_evaluation`, `generate_figures` and `comparison`.

Each experiment is repeated with 5 random seeds — 7, 5, 6, 9, 11 — which control both
the train/validation/test split and library-level randomness. All three methods share
the seed within a repeat, so every comparison is between models trained on identical
data.

All models of the controlled comparison were trained with one protocol, implemented in
`scripts/training/train.py` and the SLURM array scripts next to it
(`run_train_perturbed.sbatch` for the perturbed settings on the splits with
held-out test sets, `run_train_reference.sbatch` for the reference settings,
`run_train_leave_donor_out.sbatch`, `run_train_label_noise_sweep.sbatch` and
`run_train_rbp27*.sbatch` for the corresponding experiments). BNN: full batch,
one optimizer step per epoch, at most 1000 epochs, early stopping on the validation
loss with patience 50 (single-label) or 100 (multi-label), learning rate 0.005. MCD and
ENS: mini-batches of 128, at most 50 epochs, patience 5 (ENS on simulated TF data: 100
epochs, patience 10), learning rate 0.002. Adam throughout. MC-dropout inference calls
the model in training mode so that every dropout layer is active.

The `*.sbatch` files are written for a SLURM cluster; the partition, time and memory
lines at their top and the `UQ_DATA` / `UQ_MODELS` / `UQ_RESULTS` variables are the
only site-specific parts.

The perturbed splits with held-out test sets come from `scripts/splits/build_perturbed_splits.py`, the
leave-donor-out folds from `scripts/splits/build_leave_donor_out.py` and the
label-noise sweep splits from `scripts/splits/build_label_noise_sweep.py`.
`scripts/splits/audit_splits.py` checks every split for leakage between train,
validation and test.
