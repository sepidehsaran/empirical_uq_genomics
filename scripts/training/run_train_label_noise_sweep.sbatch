#!/bin/bash
#SBATCH --job-name=uq_noise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=0-29%12
#SBATCH --output=logs/uq_noise_%A_%a.out
#SBATCH --error=logs/uq_noise_%A_%a.err

# Label-noise sweep on the simulated TF data: 15% and 30% of the JUN and MEF2A labels swapped
# (the 50% setting is the existing tf_mislabeled run, results_v20). Same protocol as
# run_train_perturbed.sbatch; one array task = one (fraction, model, seed).
#
#   python3 scripts/splits/build_label_noise_sweep.py --datasets $D/datasets --out $D/datasets_v2
#   mkdir -p logs && sbatch scripts/training/run_train_label_noise_sweep.sbatch
# Index = seed_idx*6 + frac_idx*3 + model_idx (fractions 15 30; models mcd bnn ens; seeds 7 5 11 6 9).

set -euo pipefail
mkdir -p logs

D=${UQ_DATA:-/path/to/data}
DATASETS=$D/datasets_v2
MODELS=${UQ_MODELS:-$D/models_v22}
RESULTS=${UQ_RESULTS:-$D/results_v22}
CFG=config/config.ini

SETTINGS=(tf_mislabeled15 tf_mislabeled30)
MODELS_LS=(mcd bnn ens)
SEEDS=(7 5 11 6 9)
i=${SLURM_ARRAY_TASK_ID}
model=${MODELS_LS[$(( i % 3 ))]}
setting=${SETTINGS[$(( (i / 3) % 2 ))]}
seed=${SEEDS[$(( i / 6 ))]}

if [ -f "$HOME/tfp_env.sh" ]; then
    source "$HOME/tfp_env.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate full_py310_tfp
fi
nvidia-smi -L || echo "no GPU visible"
echo "lr ${UQ_LR:-default} -> $RESULTS"; echo "task $i -> $setting / $model / seed $seed"

python -u scripts/training/train.py --setting "$setting" --model "$model" --seed "$seed" \
    --config "$CFG" --datasets "$DATASETS" --models "$MODELS" --results "$RESULTS" --version 22
