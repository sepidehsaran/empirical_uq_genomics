#!/bin/bash
#SBATCH --job-name=simdna_xai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/simdna_xai_%j.out
#SBATCH --error=logs/simdna_xai_%j.err
#
# Regenerate the SimDNA XAI supplement figure.
#
#   sbatch scripts/simulated_tf/run_simdna_xai.sbatch
#
# Do not add -w with a node list: naming several nodes requests ALL of them, so the
# job pends until every one is free. Let the scheduler pick.

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------- edit these three
MODEL=/path/to/bnn_model_ds5
DATA=/path/to/ds5_balanced_singlelabel_1rep_3class_simulated.pickle
MOTIFS=/path/to/rbp_motif_matrix_binary
OUT=out/simdna_xai
# ----------------------------------------------------------------------------------

source scripts/training/activate_env.sh

nvidia-smi || echo "warning: no GPU visible"

python scripts/simulated_tf/simdna_xai_figure.py \
    --model   "$MODEL" \
    --data    "$DATA" \
    --motifs  "$MOTIFS" \
    --out     "$OUT" \
    --n-panels 3 \
    --n-iter 100 \
    --m-steps 50 \
    --mc-samples 20

echo "done -> $OUT/XAI_with_UQ.pdf"
