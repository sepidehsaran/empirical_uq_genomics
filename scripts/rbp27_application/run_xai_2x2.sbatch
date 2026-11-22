#!/bin/bash
#SBATCH --job-name=xai2x2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/xai2x2_%j.out
#SBATCH --error=logs/xai2x2_%j.err

set -euo pipefail
mkdir -p logs

# --------------------------------------------------- same three paths as the last run
MODEL=/path/to/uncertainty_models/app_rbp_real_imbalanced_singlelabel_27classlow_nonoise/bnn_model_v_10_s_9
DATA=/path/to/real_imbalanced_singlelabel_27classlow_nonoise.pickle
PREDS=/path/to/iter_preds_bnn_v_10_s_9.pickle
OUT=out/xai_2x2
# -------------------------------------------------------------------------------------

source scripts/training/activate_env.sh

nvidia-smi || echo "warning: no GPU visible"

python -u scripts/rbp27_application/xai_2x2_motif.py \
    --model "$MODEL" --data "$DATA" --preds "$PREDS" --out "$OUT" \
    --per-cell 200 --mc-samples 20

echo "done -> $OUT/XAI_2x2_motif.pdf"
