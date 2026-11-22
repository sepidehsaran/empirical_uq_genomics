#!/bin/bash
#SBATCH --job-name=mcd_abl_gex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mcd_abl_gex_%j.out
#SBATCH --error=logs/mcd_abl_gex_%j.err

# MC-dropout placement/rate ablation on the single-cell classifier (retrained v20 models,
# balanced and imbalanced), inference only on the existing weights: both layers at 0.3 (used
# in the paper), last layer only at 0.3, both layers at 0.15, last layer only at 0.15.
# Writes <OUT>/<variant>/<dataset_setting>/iter_preds_mcd_v_20_s_<seed>_100.pickle; minutes.
#
#   mkdir -p logs && sbatch scripts/mcd_inference/run_ablation_gex.sbatch
# Afterwards, on the laptop:
#   rsync -avP <user>@<cluster-host>:<OUT>/ cluster_preds/mcd_ablation_gex/

set -euo pipefail
mkdir -p logs

D=${UQ_DATA:-/path/to/data}
MODELS=$D/models_v20
# balanced split lives in the original tree, the imbalanced split in datasets_v2
OUT=${UQ_OUT:-$D/results_mcd_ablation_gex}
SEEDS="7 5 11 6 9"
CFG=config/config_mcd_abl_gex.ini

if [ -f "$HOME/tfp_env.sh" ]; then
    source "$HOME/tfp_env.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate full_py310_tfp
fi
unset TF_USE_LEGACY_KERAS      # MCD checkpoints are Keras 3 files
nvidia-smi -L || echo "no GPU visible"

for bal in 1 0; do
    if [ "$bal" = 1 ]; then DATASETS=$D/datasets; else DATASETS=$D/datasets_v2; fi
    # a private copy of the config so the shared config.ini is untouched
    cp config/config.ini "$CFG"
    sed -i "s#^path_models *=.*#path_models = $MODELS/#; s#^path_datasets *=.*#path_datasets = $DATASETS/#; s#^version *=.*#version = 20#" "$CFG"
    grep -n '^path_\|^version' "$CFG"
    python -u scripts/mcd_inference/mcd_inference.py --config "$CFG" --task gex --species real_human \
        --n-classes 4 --balanced $bal --seeds $SEEDS --out "$OUT"
done
echo "done -> $OUT"
