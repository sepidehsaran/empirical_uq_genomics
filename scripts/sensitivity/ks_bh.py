"""Benjamini-Hochberg correction over all KS tests in Figure 8 (every class x seed x method x
perturbation). Reports how many rejections survive at FDR 0.05 next to the uncorrected count.

    UQFIG_BUNDLE=... UQFIG_EXCLUDE_BNN=none python3 scripts/sensitivity/ks_bh.py
"""
import os, sys
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "uq_genomics", "figures"))
import combine_figures as C   # noqa: E402
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))

frames = []
for task in ("tf", "gex", "rbp"):
    d = C._ks_df(task)
    if d is not None:
        frames.append(d.assign(task=task))
d = pd.concat(frames, ignore_index=True)
d["reject_raw"] = d.p < 0.05
d["p_bh"] = multipletests(d.p.values, alpha=0.05, method="fdr_bh")[1]
d["reject_bh"] = d.p_bh < 0.05
os.makedirs(OUT, exist_ok=True); d.to_csv(os.path.join(OUT, "ks_tests_bh.csv"), index=False)
print(f"all tests: {len(d)}  rejected raw {int(d.reject_raw.sum())}  rejected BH(0.05) {int(d.reject_bh.sum())}")
print(d.groupby(["task", "setting_alt", "model"])[["reject_raw", "reject_bh"]].sum().astype(int).to_string())
