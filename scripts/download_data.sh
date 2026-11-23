#!/usr/bin/env bash
# Download the released data from Zenodo (doi:10.5281/zenodo.24450552) and unpack it
# where the notebooks expect it.
#
#   ./scripts/download_data.sh            # figure_data_bundle only (27 MB): every notebook figure
#   ./scripts/download_data.sh --preds    # + raw_predictions (2.0 GB): Fig 10a, 12, S7, supplement tables
#   ./scripts/download_data.sh --all      # + datasets (181 MB) and models (1.8 GB): retraining, attribution
#
# Files land in the repository root: figure_data_bundle/, cluster_preds/, datasets*/, models*/.
# Re-running skips archives that are already present; checksums are verified against SHA256SUMS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD="https://zenodo.org/records/24450552/files"
cd "$ROOT"

want=(figure_data_bundle)
for arg in "$@"; do
  case "$arg" in
    --preds) want+=(raw_predictions) ;;
    --all)   want+=(raw_predictions datasets models models_rbp27_bnn) ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

fetch() {  # fetch <file>
  if [ -f "$1" ]; then echo "have $1"; else echo "get  $1"; curl -fL --retry 3 -o "$1" "$RECORD/$1?download=1"; fi
}

mkdir -p .zenodo && cd .zenodo
fetch SHA256SUMS
for name in "${want[@]}"; do fetch "$name.tar.gz"; done
grep -E "$(printf '%s.tar.gz|' "${want[@]}" | sed 's/|$//')" SHA256SUMS | sha256sum -c -
cd "$ROOT"

# unpack: the bundle and raw predictions straight into the root; datasets and models
# into datasets_release/ and models_release/ so nothing collides with your own trees
for name in "${want[@]}"; do
  case "$name" in
    figure_data_bundle|raw_predictions) tar -xzf ".zenodo/$name.tar.gz" ;;
    datasets)         mkdir -p datasets_release && tar -xzf .zenodo/datasets.tar.gz -C datasets_release ;;
    models)           mkdir -p models_release   && tar -xzf .zenodo/models.tar.gz   -C models_release ;;
    models_rbp27_bnn) mkdir -p models_release   && tar -xzf .zenodo/models_rbp27_bnn.tar.gz -C models_release ;;
  esac
done
echo
echo "done. figure data in $ROOT/figure_data_bundle"
[ -d cluster_preds ] && echo "raw predictions in $ROOT/cluster_preds (and results_*/, inference_v20/)"
[ -d datasets_release ] && echo "datasets in $ROOT/datasets_release  -> export UQFIG_PATH_DATASETS_SEQ=$ROOT/datasets_release/datasets"
[ -d models_release ] && echo "models in $ROOT/models_release"
exit 0
