#!/bin/bash
#SBATCH --job-name=uq_app27
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/uq_app27_%A_%a.out
#SBATCH --error=logs/uq_app27_%A_%a.err

# MC-dropout (all layers active at inference) and a 10-member ensemble on the 27-class RBP
# application dataset, seed 9, same protocol as the controlled comparison, so the uncertainty
# filter of Section 4.4 can be run for all three methods. One task per model; the BNN (index 2,
# full-batch, 1000 steps) is the slow one:  sbatch --array=2 scripts/training/run_train_rbp27.sbatch
#
#   python3 scripts/data/prep_rbp27_dataset.py --src <app27 pickle> --datasets $UQ_DATA/datasets_v2 --seed 9
#   mkdir -p logs && sbatch scripts/training/run_train_rbp27.sbatch

set -euo pipefail
mkdir -p logs

D=${UQ_DATA:-/path/to/data}
DATASETS=$D/datasets_v2
MODELS=${UQ_MODELS:-$D/models_v23}
RESULTS=${UQ_RESULTS:-$D/results_v23}
CFG=config/config.ini
SEED=9

MODELS_LS=(mcd ens bnn)
model=${MODELS_LS[$SLURM_ARRAY_TASK_ID]}

if [ -f "$HOME/tfp_env.sh" ]; then
    source "$HOME/tfp_env.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate full_py310_tfp
fi
nvidia-smi -L || echo "no GPU visible"
echo "task $SLURM_ARRAY_TASK_ID -> rbp_app27 / $model / seed $SEED"

python -u scripts/training/train.py --setting rbp_app27 --model "$model" --seed "$SEED" \
    --config "$CFG" --datasets "$DATASETS" --models "$MODELS" --results "$RESULTS" --version 23
