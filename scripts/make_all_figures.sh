#!/usr/bin/env bash
# Regenerate every figure of the paper into out/ (or the directory given as $1).
# Same calls as the notebooks, without Jupyter.
#
#   ./scripts/make_all_figures.sh [output_dir]
#
# Inputs: the released summary bundle (UQFIG_BUNDLE, default figure_data_bundle/) and,
# for Fig. 10a, Fig. 12 and Fig. S7, the raw per-pass predictions (UQFIG_PREDS, default
# cluster_preds/). Figures whose inputs are absent are skipped with a message.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/out}"
export PYTHONPATH="$ROOT/src"
export UQFIG_BUNDLE="${UQFIG_BUNDLE:-$ROOT/figure_data_bundle}"
export UQFIG_PREDS="${UQFIG_PREDS:-$ROOT/cluster_preds}"
export MPLBACKEND=Agg
mkdir -p "$OUT"
if [ ! -d "$UQFIG_BUNDLE" ]; then
  echo "data bundle not found at $UQFIG_BUNDLE - see docs/REPRODUCING.md" >&2
  exit 1
fi
echo "bundle: $UQFIG_BUNDLE"
echo "preds : $UQFIG_PREDS"
echo "output: $OUT"
python - "$OUT" <<'PY'
import os, sys
out = sys.argv[1]
from uq_genomics.figures import style as S
S.apply_rc()

def step(name, fn):
    print(f"\n== {name}")
    try:
        fn()
    except FileNotFoundError as exc:
        print(f"   skipped: {exc}")

from uq_genomics.figures import plots_datasets as PD, plots_umap as PU
step("Fig 1, 3: class distributions", lambda: PD.run(out))
step("Fig 2: single-cell UMAP", lambda: PU.build(out))

from uq_genomics.figures import combine_figures as CF
step("Fig 5: performance", lambda: CF.performance_ext(out))
step("Fig 6-8: agreement", lambda: (CF.top100(out), CF.kendall(out), CF.ks(out)))

from uq_genomics.figures import pooled as P, plots_depletion as DEP
step("Fig 9, S8: TF label noise / missing motif", lambda: P.build_tf(out, log2=True))
step("Fig 10: class imbalance", lambda: P.build_imbalance(out, log2=True))
step("Fig S9, S10: raw scale", lambda: (P.build_tf(out, log2=False), P.build_imbalance(out, log2=False)))
step("Fig S11: depletion", lambda: DEP.build(out))

from uq_genomics.figures import plots_sdshift as SDS
step("Fig 11: coronavirus OOD", lambda: SDS.build_virus(out, seed='all'))

from uq_genomics.figures import plots_rbp27 as CFG, plots_selective as PSEL
step("Fig 12: uncertainty filtering (27-class RBP)", lambda: CFG.build_app27(out, n_iter=100, extended=True))
step("Fig S7: AP against coverage", lambda: PSEL.build(out))

from uq_genomics.figures import calibration as CAL
step("Fig S13, Table S10: calibration", lambda: CAL.build(out))
print("\ndone ->", out)
PY
