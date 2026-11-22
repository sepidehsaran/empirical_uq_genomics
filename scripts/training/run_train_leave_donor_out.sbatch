#!/bin/bash
#SBATCH --job-name=uq_ldo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-14%8
#SBATCH --output=logs/uq_ldo_%A_%a.out
#SBATCH --error=logs/uq_ldo_%A_%a.err

# Leave-donor-out on the single-cell data: train MCD / BNN / ENS on 12 samples (balanced),
# test on the held-out sample; one array task = one (model, fold). Data from
# scripts/data/build_leave_donor_out.py, rsynced to $DATASETS.
#
#   mkdir -p logs && sbatch scripts/training/run_train_leave_donor_out.sbatch
# Index = seed_idx*3 + model_idx (models mcd, bnn, ens; seeds 7 5 11 6 9).

set -euo pipefail
mkdir -p logs

D=${UQ_DATA:-/path/to/data}
DATASETS=$D/datasets_ldo
MODELS=${UQ_MODELS:-$D/models_ldo}
RESULTS=${UQ_RESULTS:-$D/results_ldo}
# learning rate: driver default (0.005 for BNN, 0.002 otherwise) unless UQ_LR is set
CFG=config/config.ini

MODELS_LS=(mcd bnn ens)
SEEDS=(7 5 11 6 9)
i=${SLURM_ARRAY_TASK_ID}
model=${MODELS_LS[$(( i % 3 ))]}
seed=${SEEDS[$(( i / 3 ))]}

if [ -f "$HOME/tfp_env.sh" ]; then
    source "$HOME/tfp_env.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate full_py310_tfp
fi
nvidia-smi -L || echo "no GPU visible"
echo "lr ${UQ_LR:-default} -> $RESULTS"; echo "task $i -> gex_bal (leave-donor-out) / $model / seed $seed"

python -u scripts/training/train.py --setting gex_bal --model "$model" --seed "$seed" \
    --config "$CFG" --datasets "$DATASETS" --models "$MODELS" --results "$RESULTS" --version 21
