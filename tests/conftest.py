"""The figure modules import each other by bare name (see src/uq_genomics/figures/__init__.py),
so the tests put that directory on sys.path, as the notebooks do through the package."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "src", "uq_genomics", "figures")
for p in (FIG, os.path.join(os.path.dirname(HERE), "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
