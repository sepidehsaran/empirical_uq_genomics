#!/bin/bash
#SBATCH --job-name=uq_v20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=0-59%12
#SBATCH --output=logs/uq_v20_%A_%a.out
#SBATCH --error=logs/uq_v20_%A_%a.err

# Train MCD / BNN / ENS on the perturbed settings (datasets_v2: test sets held out before perturbation)
# and write their 100 inference passes. One array task = one (setting, model, seed).
#
#   mkdir -p logs && sbatch scripts/training/run_train_perturbed.sbatch
#   squeue -u $USER | head; tail -f logs/uq_v20_<jobid>_0.out
#
# Re-run a subset:   sbatch --array=3,17,42 scripts/training/run_train_perturbed.sbatch
# Index = seed_idx*12 + model_idx*4 + setting_idx (see below), so e.g. all BNN jobs:
#   sbatch --array=4-7,16-19,28-31,40-43,52-55 scripts/training/run_train_perturbed.sbatch

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------- edit these
D=${UQ_DATA:-/path/to/data}
DATASETS=$D/datasets_v2
MODELS=${UQ_MODELS:-$D/models_v20}
RESULTS=${UQ_RESULTS:-$D/results_v20}
# learning rate: driver default (0.005 for BNN, 0.002 otherwise) unless UQ_LR is set
CFG=config/config.ini
# ---------------------------------------------------------------------------

SETTINGS=(tf_mislabeled rbp_imbal_single rbp_imbal_multi gex_imbal)
MODELS_LS=(mcd bnn ens)
SEEDS=(7 5 11 6 9)
i=${SLURM_ARRAY_TASK_ID}
setting=${SETTINGS[$(( i % 4 ))]}
model=${MODELS_LS[$(( (i / 4) % 3 ))]}
seed=${SEEDS[$(( i / 12 ))]}

if [ -f "$HOME/tfp_env.sh" ]; then
    source "$HOME/tfp_env.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate full_py310_tfp
fi
nvidia-smi -L || echo "no GPU visible"
echo "lr ${UQ_LR:-default} -> $RESULTS"; echo "task $i -> $setting / $model / seed $seed"

python -u scripts/training/train.py --setting "$setting" --model "$model" --seed "$seed" \
    --config "$CFG" --datasets "$DATASETS" --models "$MODELS" --results "$RESULTS" --version 20
