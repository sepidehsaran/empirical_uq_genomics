# Uncertainty-aware deep learning for genomics

Code, notebooks and released model outputs for

> **Uncertainty-Aware Deep Learning for Genomics Applications: Insights from an Empirical Study**
> Sepideh Saran, Mahsa Ghanbari, Uwe Ohler
> *Transactions on Machine Learning Research*, 2026 — [OpenReview](https://openreview.net/forum?id=YL7aleiVvY)

The paper compares three uncertainty quantification (UQ) methods for deep learning —
Bayesian neural networks (BNN), Monte Carlo dropout (MCD) and deep ensembles (ENS) — on
three genomics tasks: transcription-factor binding on simulated DNA, RNA-binding-protein
binding on real RNA, and cell-type classification from single-cell expression. Each
method is tested against one source of uncertainty at a time (label noise, missing
sequence motifs, class imbalance, out-of-distribution data), and the uncertainty scores
are then used to filter predictions on a 27-class RBP task.

## Repository structure:

```
src/uq_genomics/
  experiments/          model definitions, training, evaluation, attribution   (TensorFlow)
  figures/              every figure and table of the paper, from released data (no TensorFlow)
notebooks/              one notebook per figure group, 00-07
scripts/                one directory per experiment
  splits/               dataset splits: perturbed settings, label-noise sweep, leave-donor-out, 27-class task
  training/             the training protocol and its SLURM wrappers, one per experiment
  mcd_inference/        MC-dropout inference with dropout active; placement/rate ablation (Table S2)
  sensitivity/          cross-seed shift tables, paired tests, KS correction, label-noise sweep (Tables S3-S5, S12)
  single_cell/          leave-donor-out, per-donor breakdown, uncertainty on the UMAP (Tables S6-S8, Fig S12)
  ood/                  OOD-detection metrics on the coronavirus genomes (Table S9)
  uncertainty_scores/   entropy and mutual information as alternative scores (Table S11)
  rbp27_application/    27-class RBP task: all-method filtering, complex confusion, attribution (Figs S4-S6)
  simulated_tf/         sequence logos and off-label-motif attribution for the simulated data (Figs S1, S2)
  supp_tables/          LaTeX for the supplement tables, from the CSVs the scripts above write
  bundle/               building the released data bundle from trained models
tests/                  consistency checks that run without any data download
config/                 config.example.ini for retraining
docs/REPRODUCING.md     figure-by-figure map, data layout, environment variables
```

This repository can be used for the following purposes:

| | needs | time |
|---|---|---|
| **Regenerate the figures** of the paper from the released model outputs | Python 3.10+, ~200 MB RAM | minutes |
| **Rerun the analyses** on your own model outputs | + your prediction files | minutes |
| **Retrain the models** from scratch | + TensorFlow 2.16 and a GPU | days |


## Data

The model outputs are released separately from the code because of their size:

| Artifact | Size | Needed for |
|---|---|---|
| `figure_data_bundle/` | ~22 MB | every figure except the two below |
| `cluster_preds/` | ~725 MB | Fig. 10a, Fig. 12, Fig. S7 and several supplement tables |
| input datasets | several GB | retraining only |

Download: [data bundles](https://zenodo.org/records/24450552)

Unpack `figure_data_bundle/` and, if you want the raw-prediction figures,
`cluster_preds/` into the repository root. Other locations work through the
`UQFIG_BUNDLE` and `UQFIG_PREDS` environment variables.

## Installation

```bash
git clone https://github.com/sepidehsaran/empirical_uq_genomics.git
cd empirical_uq_genomics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-figures.txt      # numpy, scipy, pandas, scikit-learn, matplotlib, seaborn
pip install jupyterlab                       # to run the notebooks
```

`requirements-figures.txt` is enough for everything in the table above except
retraining. For retraining, see [Retraining](#retraining).

Alternatively, with Docker and nothing else installed:

```bash
docker build -t uq-genomics .
docker run --rm -v "$PWD/figure_data_bundle:/data:ro" -v "$PWD/cluster_preds:/preds:ro" \
    -v "$PWD/out:/out" uq-genomics
```

## Regenerating the results, step by step

1. **Install** as above and download the data into the repository root.

2. **Check the environment.** Open `notebooks/00_check_environment.ipynb` (or run
   `PYTHONPATH=src python -c "from uq_genomics.figures import config_paths as C; [print(*r) for r in C.report()]"`).
   It lists every input the figures need and whether it was found. The first two lines
   (`BUNDLE`, `PREDS`) are the ones that matter for the figures; the rest are for
   rebuilding the data from retrained models.

3. **Run the notebooks in order.** Each one starts with the same setup cell, writes its
   figures to `out/` under the file names used in the paper, and displays them inline.

   | Notebook | Paper figures | Needs |
   |---|---|---|
   | `01_datasets` | 1, 2, 3 | bundle |
   | `02_performance` | 5 | bundle |
   | `03_agreement` | 6, 7, 8 | bundle |
   | `04_sensitivity` | 9, 10, S8–S11 | bundle; `cluster_preds/` for Fig. 10a |
   | `05_ood_virus` | 11 | bundle |
   | `06_utility_filtering` | 12, S7 | bundle + `cluster_preds/` |
   | `07_calibration` | S13, Table S10 | bundle |

   Or run them all at once without Jupyter: `./scripts/make_all_figures.sh`.

4. **Compare with the paper.** `docs/REPRODUCING.md` has the full figure-by-figure
   map, including the supplement figures and tables that come from scripts rather than
   notebooks, and three notes on how the numbers should be read (how the two use cases
   construct imbalance, which test sets the perturbed settings use, and why some scores
   must be computed in probability space from the raw passes).

5. **Run the tests** (optional, no data needed): `pip install pytest && pytest tests/`.

## Retraining

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt "tensorflow[and-cuda]==2.16.1"   # drop the extra for CPU
cp config/config.example.ini config/config.ini                          # edit the three path_* keys
cd src/uq_genomics/experiments && python main_run.py
```

`config.ini` is git-ignored and selects the task, dataset setting, model type and
`run_mode` (`training`, `generate_evaluation`, `generate_figures`, `comparison`).
`scripts/training/train.py` and the `scripts/training/run_train_*.sbatch` wrappers implement the
training protocol of the paper for all settings and seeds; the training details, the
environment used, and the step from trained models to the figure bundle are in
`docs/REPRODUCING.md`.

Experiment tracking with Neptune is optional. Set `NEPTUNE_API_TOKEN` in your shell to
enable it; the code runs identically without it, and no token belongs in any file of
this repository.

## Citation

```bibtex
@article{saran2026uq,
  title   = {Uncertainty-Aware Deep Learning for Genomics Applications: Insights from an Empirical Study},
  author  = {Saran, Sepideh and Ghanbari, Mahsa and Ohler, Uwe},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=YL7aleiVvY}
}
```

Machine-readable metadata is in `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). Released data and models: CC BY 4.0 (Zenodo record: 10.5281/zenodo.24450552).
The paper itself is published by TMLR under CC BY 4.0.
