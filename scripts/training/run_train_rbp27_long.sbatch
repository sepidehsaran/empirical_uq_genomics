#!/bin/bash
#SBATCH --job-name=uq_app27L
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/uq_app27L_%A_%a.out
#SBATCH --error=logs/uq_app27L_%A_%a.err

# 27-class BNN with a longer step cap (2000 full-batch steps, patience 50, lr 0.005) and a
# second initialization seed, to check whether the class-size result of Section 4.4 depends on
# the step cap. Same dataset file for both seeds (the split is fixed; the seed changes the
# initialization). Writes version 24 under models_v24 / results_v24.
#
#   python3 scripts/splits/prep_rbp27_dataset.py --src <app27 pickle> --datasets $UQ_DATA/datasets_v2 --seed 3
#   mkdir -p logs && sbatch scripts/training/run_train_rbp27_long.sbatch

set -euo pipefail
mkdir -p logs

D=${UQ_DATA:-/path/to/data}
DATASETS=$D/datasets_v2
MODELS=${UQ_MODELS:-$D/models_v24}
RESULTS=${UQ_RESULTS:-$D/results_v24}
CFG=config/config.ini
export UQ_BNN_EPOCHS=2000

SEEDS=(9 3)
seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}

if [ -f "$HOME/tfp_env.sh" ]; then
    source "$HOME/tfp_env.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate full_py310_tfp
fi
nvidia-smi -L || echo "no GPU visible"
echo "task $SLURM_ARRAY_TASK_ID -> rbp_app27 / bnn / seed $seed / $UQ_BNN_EPOCHS steps"

python -u scripts/training/train.py --setting rbp_app27 --model bnn --seed "$seed" \
    --config "$CFG" --datasets "$DATASETS" --models "$MODELS" --results "$RESULTS" --version 24
