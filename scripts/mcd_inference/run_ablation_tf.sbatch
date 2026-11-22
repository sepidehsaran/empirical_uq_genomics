#!/bin/bash
#SBATCH --job-name=mcd_fixed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mcd_fixed_%j.out
#SBATCH --error=logs/mcd_fixed_%j.err

# Corrected MC-dropout inference (model(x, training=True)) for every MCD setting in the
# paper, plus the dropout-placement/rate ablation on the sequence tasks. Inference only,
# on the existing checkpoints. Writes to $OUT/<variant>/<dataset_setting>/iter_preds_mcd_*.pickle
# ("baseline" = the configuration the paper describes).
#
#   sbatch scripts/mcd_inference/run_ablation_tf.sbatch
#
# Needs config/config.ini with path_datasets set; path_models and version are set per task
# below. Drop the two GPU lines to run on a CPU partition; it is minutes either way.

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------- edit these
D=${UQ_DATA:-/path/to/data}
OUT=$D/results_mcd_training_true
SEEDS="7 5 11 6 9"
# ---------------------------------------------------------------------------

if [ -f .venv/bin/activate ]; then
    source scripts/training/activate_env.sh
elif [ -f "$HOME/tfp_env.sh" ]; then
    source "$HOME/tfp_env.sh"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate full_py310_tfp
fi
# NOT TF_USE_LEGACY_KERAS: the MCD checkpoints are Keras 3 files.
unset TF_USE_LEGACY_KERAS
nvidia-smi || echo "no GPU visible, running on CPU"

CFG=config/config.ini
SCRIPT=scripts/mcd_inference/mcd_inference.py
set_cfg () {   # set_cfg <models_tree> <version>
    sed -i "s#^path_models *=.*#path_models = $D/$1/#; s#^version *=.*#version = $2#" "$CFG"
    grep -n '^path_\|^version' "$CFG"
}

echo "===== RBP (models_tf_paper_2, v18)"
set_cfg models_tf_paper_2 18
for bal in 1 0; do
    python -u $SCRIPT --config $CFG --task rbp --species real_human --n-classes 3 --balanced $bal \
        --seeds $SEEDS --out "$OUT"
done

echo "===== TF (models_tf_paper_2, v18)"
for noise in none mislabeled; do
    python -u $SCRIPT --config $CFG --task tf --species simulated_human --n-classes 6 --noise $noise \
        --seeds $SEEDS --out "$OUT"
done

echo "===== GEX (v19): locating the models tree"
GEX_MODEL=$(find "$D" -maxdepth 7 -path "*gex/real_human/balanced/single_label/4class_split7/mcd_model_v_19_s_7.keras" 2>/dev/null | head -1)
if [ -z "$GEX_MODEL" ]; then
    echo "no GEX v19 MCD checkpoint found under $D -- set the tree by hand"; exit 1
fi
GEX_TREE=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$GEX_MODEL")")")")")")")
echo "GEX models tree: $GEX_TREE ($GEX_MODEL)"
set_cfg "$GEX_TREE" 19
for bal in 1 0; do
    python -u $SCRIPT --config $CFG --task gex --species real_human --n-classes 4 --balanced $bal \
        --seeds $SEEDS --out "$OUT"
done

echo "done -> $OUT"
echo "copy to the laptop with:"
echo "  rsync -avP <user>@<cluster-host>:$OUT/ cluster_preds/mcd_training_true/"
