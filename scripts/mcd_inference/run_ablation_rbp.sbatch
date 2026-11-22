#!/bin/bash
#SBATCH --job-name=mcd_abl_v20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mcd_abl_v20_%j.out
#SBATCH --error=logs/mcd_abl_v20_%j.err

# MC-dropout placement/rate ablation on the retrained sequence models (models_v20), inference
# only: TF clean and mislabeled, RBP balanced and imbalanced (single-label), four configurations
# each (last layer 0.25, all layers 0.25 = used, last layer 0.5, all layers 0.5). Together with
# run_mcd_ablation_gex.sbatch this gives one table from one model set.
# Writes <OUT>/<variant>/<dataset_setting>/iter_preds_mcd_v_20_s_<seed>_100.pickle; minutes.
#
#   mkdir -p logs && sbatch scripts/mcd_inference/run_ablation_rbp.sbatch
# Afterwards, on the laptop:
#   rsync -avP <user>@<cluster-host>:<OUT>/ cluster_preds/mcd_ablation_v20/

set -euo pipefail
mkdir -p logs

D=${UQ_DATA:-/path/to/data}
MODELS=$D/models_v20
OUT=${UQ_OUT:-$D/results_mcd_ablation_v20}
SEEDS="7 5 11 6 9"
CFG=config/config_mcd_abl_v20.ini
SCRIPT=scripts/mcd_inference/mcd_inference.py

if [ -f "$HOME/tfp_env.sh" ]; then
    source "$HOME/tfp_env.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate full_py310_tfp
fi
unset TF_USE_LEGACY_KERAS      # MCD checkpoints are Keras 3 files
nvidia-smi -L || echo "no GPU visible"

set_cfg () {   # set_cfg <datasets_root>: reference settings live in datasets/, perturbed ones in datasets_v2/
    cp config/config.ini "$CFG"
    sed -i "s#^path_models *=.*#path_models = $MODELS/#; s#^path_datasets *=.*#path_datasets = $1/#; s#^version *=.*#version = 20#" "$CFG"
    grep -n '^path_\|^version' "$CFG"
}

echo "===== TF clean"
set_cfg "$D/datasets"
python -u $SCRIPT --config "$CFG" --task tf --species simulated_human --n-classes 6 --noise none --seeds $SEEDS --out "$OUT"
echo "===== TF mislabeled"
set_cfg "$D/datasets_v2"
python -u $SCRIPT --config "$CFG" --task tf --species simulated_human --n-classes 6 --noise mislabeled --seeds $SEEDS --out "$OUT"
echo "===== RBP balanced"
set_cfg "$D/datasets"
python -u $SCRIPT --config "$CFG" --task rbp --species real_human --n-classes 3 --balanced 1 --seeds $SEEDS --out "$OUT"
echo "===== RBP imbalanced"
set_cfg "$D/datasets_v2"
python -u $SCRIPT --config "$CFG" --task rbp --species real_human --n-classes 3 --balanced 0 --seeds $SEEDS --out "$OUT"
echo "done -> $OUT"
