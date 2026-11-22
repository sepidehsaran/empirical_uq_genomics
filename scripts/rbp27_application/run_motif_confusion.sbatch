#!/bin/bash
#SBATCH --job-name=motif_conf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/motif_conf_%j.out
#SBATCH --error=logs/motif_conf_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------- edit these
MODEL=/path/to/uncertainty_models/app_rbp_real_imbalanced_singlelabel_27classlow_nonoise/bnn_model_v_10_s_9
DATA=/path/to/real_imbalanced_singlelabel_27classlow_nonoise.pickle
PREDS=/path/to/iter_preds_bnn_v_10_s_9.pickle
OUT=out/motif_confusion
# ---------------------------------------------------------------------------

source scripts/training/activate_env.sh

nvidia-smi || echo "warning: no GPU visible"

python -u scripts/rbp27_application/xai_motif_confusion.py \
    --model "$MODEL" --data "$DATA" --preds "$PREDS" --out "$OUT" \
    --n-proto 120 --n-err 80 --n-unc 150 --mc-samples 10

echo "done -> $OUT/XAI_motif_confusion.pdf"
