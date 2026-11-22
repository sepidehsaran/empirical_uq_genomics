"""Figure generation. Reads the released data bundle, writes vector PDFs.

Point UQFIG_BUNDLE at the bundle directory before importing anything here:

    export UQFIG_BUNDLE=/path/to/figure_data_bundle

The modules in this directory import each other by bare name (``import config_paths``
rather than ``from . import config_paths``), because they are also runnable as
standalone scripts:

    python src/uq_genomics/figures/plots_sdshift.py --out runs/sdshift

Putting this directory on sys.path keeps both entry points working -- ``python -m
uq_genomics.figures.run_all`` and direct execution -- without rewriting the imports in
every module. Rewriting them to be relative would break standalone execution, which is
how the scripts document themselves.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
