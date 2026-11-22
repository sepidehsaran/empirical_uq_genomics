"""Supplement section: per-donor / per-site evaluation of the single-cell models + count table."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))

site = pd.read_csv(os.path.join(R, "gex_counts_class_by_site.csv"), index_col=0)
samp = pd.read_csv(os.path.join(R, "gex_counts_class_by_sample.csv"), index_col=0)
acc = pd.read_csv(os.path.join(R, "gex_accuracy_by_sample.csv"))
M = ["MCD", "BNN", "ENS"]

def esc(x): return str(x).replace("+", "$+$").replace("_", r"\_")
count_rows = "\n".join(esc(idx) + " & " + " & ".join(str(int(v)) for v in samp.loc[idx]) + r" \\" for idx in samp.index if idx != "All")
count_rows += "\n\\midrule\n" + "All & " + " & ".join(str(int(v)) for v in samp.loc["All"]) + r" \\"
samples = [c for c in samp.columns if c != "All"]

def acc_row(s, setting):
    t = acc[(acc.setting == setting) & (acc["sample"] == s)]
    return " & ".join(f"{t[t.model == m].acc_mean.values[0]:.3f}" if len(t[t.model == m]) else "--" for m in M)
acc_rows = "\n".join(esc(s) + " & " + acc_row(s, "bal_single") + " & " + acc_row(s, "imbal_single") + r" \\" for s in samples)
rng = {st: {m: (acc[(acc.setting == st) & (acc.model == m)].acc_mean.min(), acc[(acc.setting == st) & (acc.model == m)].acc_mean.max()) for m in M} for st in ["bal_single", "imbal_single"]}
r = rng

tex = r"""
\subsection{{Evaluation per donor and per site (single-cell data)}}\label{app_b:donor}
{The train, validation and test splits of the single-cell data are random over cells, so cells from the same donor and site appear on both sides of a split. The embedding is computed by Liam, which is trained to remove the site and donor batch effect while retaining biological variation~\citep{liam}, and the comparison of UQ methods is conditional on that embedding. To check whether the random split inflates the reported accuracies, we traced every test cell back to its sample of origin (Table~\ref{supp_tab:donor_counts}; the 28,214 cells of the four selected types come from 13 samples, 10 donors at 4 sites) and recomputed the accuracy of each method per sample (Table~\ref{supp_tab:donor_acc}). Accuracy is uniform across samples: between """ + f"{r['bal_single']['MCD'][0]:.2f}" + r""" and """ + f"{r['bal_single']['MCD'][1]:.2f}" + r""" for MCD and """ + f"{r['bal_single']['ENS'][0]:.2f}" + r"""--""" + f"{r['bal_single']['ENS'][1]:.2f}" + r""" for ENS on the balanced data, and """ + f"{r['imbal_single']['MCD'][0]:.2f}" + r"""--""" + f"{r['imbal_single']['MCD'][1]:.2f}" + r""" and """ + f"{r['imbal_single']['ENS'][0]:.2f}" + r"""--""" + f"{r['imbal_single']['ENS'][1]:.2f}" + r""" on the imbalanced data, with no sample or site standing out. The BNN's range is wider (""" + f"{r['bal_single']['BNN'][0]:.2f}" + r"""--""" + f"{r['bal_single']['BNN'][1]:.2f}" + r""" balanced, """ + f"{r['imbal_single']['BNN'][0]:.2f}" + r"""--""" + f"{r['imbal_single']['BNN'][1]:.2f}" + r""" imbalanced), which reflects the seeds in which it reached lower accuracy (Section~\ref{sec_4_1:perf}) and is lowest on the sample with the most Erythroblasts, the class the imbalance depletes. A leave-donor-out evaluation, in which whole donors are held out of training, is the stricter test and is planned for the next revision.}

\begin{table}[h]
\centering\small
\caption{{Number of cells of the four selected cell types per sample (site $\times$ donor) in the single-cell dataset.}}
\label{supp_tab:donor_counts}
\begin{tabular}{l""" + "r" * len(samp.columns) + r"""}
\toprule
cell type & """ + " & ".join(esc(c) for c in samp.columns) + r""" \\
\midrule
""" + count_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering\small
\caption{{Accuracy of each UQ method per sample on the single-cell test set, mean over the five random seeds, for the balanced and the imbalanced training setting. Both settings are evaluated on the same balanced test set.}}
\label{supp_tab:donor_acc}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{balanced} & \multicolumn{3}{c}{imbalanced} \\
sample & MCD & BNN & ENS & MCD & BNN & ENS \\
\midrule
""" + acc_rows + r"""
\bottomrule
\end{tabular}
\end{table}

"""
sys.stdout.write(tex)
