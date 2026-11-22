"""Supplement text + tables for (i) per-class coverage under the 0.1 filter on the 27-RBP
task and (ii) ensemble-size / number-of-passes sensitivity. Reads out/tables/*.csv."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))
NAMES = ["MBNL1", "P53_NONO", "PUM2", "QKI", "AGO3", "FUS", "TAF15", "ZFP36", "DICER1", "EIF3A",
         "EIF3D", "EIF3G", "SSB", "PAPD5", "CPSF4", "CPSF3", "RTCB", "FXR1", "NOP58", "NOP56",
         "FBL", "LIN28A", "LIN28B", "UPF1", "G35", "G45", "XPO5"]          # plots_rbp27.RBP27_NAMES

cov = pd.read_csv(os.path.join(R, "app27_per_class_coverage.csv"))
cov["name"] = [NAMES[int(c[1:])] if str(c).startswith("c") else c for c in cov.cls]
cov = cov.sort_values("train", ascending=False)
tot_cov = cov.kept.sum() / cov.n_pos.sum()
US = "\\_"
rows = "\n".join(f"{r.name.replace('_', US)} & {int(r.train)} & {int(r.n_pos)} & {100 * r.coverage:.0f}\\% & {r.ap_all:.2f} & {r.ap_filtered:.2f} \\\\"
                 for r in cov.itertuples())
big5, small5 = cov.head(5).coverage.mean(), cov.tail(5).coverage.mean()
from scipy.stats import spearmanr
rho, p = spearmanr(cov.train, cov.coverage)

e = pd.read_csv(os.path.join(R, "ensemble_size_sensitivity.csv"))
t = e[e.check.str.startswith("tau_vs_full")].copy()
t["k"] = t.check.str.extract(r"at_(\d+)_").astype(int)
g = t.groupby(["model", "k"]).value.agg(["mean", "min"])
def row(m, ks):
    return " & ".join(f"{g.loc[(m, k), 'mean']:.2f} ({g.loc[(m, k), 'min']:.2f})" if (m, k) in g.index else "--" for k in ks)

cov_tex = r"""
\paragraph{{Per-class coverage of the filter.}}
{Filtering by an uncertainty threshold keeps a different fraction of each class, and a filter that discards rare classes first would amplify missed-detection bias for exactly the proteins with the fewest binding sites. Table~\ref{supp_tab:coverage} gives, for each of the 27 RBPs, the number of positive test sequences, the fraction retained by the threshold of $0.1$ on the predicted class, and the average precision before and after filtering. The filter retains """ + f"{100 * tot_cov:.0f}" + r"""\% of all positive sequences; per class the retained fraction ranges from """ + f"{100 * cov.coverage.min():.0f}" + r"""\% to """ + f"{100 * cov.coverage.max():.0f}" + r"""\%, with a mean of """ + f"{100 * small5:.0f}" + r"""\% for the five smallest training classes against """ + f"{100 * big5:.0f}" + r"""\% for the five largest. No class is removed entirely, and the rank correlation between training-set size and retained fraction is $\rho = """ + f"{rho:+.2f}" + r"""$ ($p = """ + f"{p:.2f}" + r"""$, $n = 27$). The filter therefore does discard proportionally more of the rare classes, but mildly: the difference between the extreme groups is """ + f"{100 * (big5 - small5):.0f}" + r""" percentage points of coverage. In applications where missed detections in rare classes are costly, we recommend setting the threshold per class, or reporting the retained fraction per class next to any aggregate gain, as done here.}

\begin{table}[h]
\centering\small
\caption{{Per-class coverage of the uncertainty filter on the 27-class RBP task (BNN, seed 9). Classes are ordered by training-set size. Coverage is the fraction of positive test sequences whose uncertainty score at the predicted class is below $0.1$; AP is the average precision before and after filtering.}}
\label{supp_tab:coverage}
\begin{tabular}{lrrrrr}
\toprule
RBP & train & test positives & coverage & AP all & AP filtered \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
"""

ens_tex = r"""
\subsection{{Number of ensemble members and inference passes}}\label{app_b:ens_size}
{The ensemble provides $10$ independent predictions per sample, against $100$ correlated inference passes for BNN and MCD. To see how many members or passes the uncertainty ranking needs, we subsampled the raw per-pass predictions of the RBP models (five seeds, balanced and imbalanced single-label settings), recomputed the standard deviation of the predicted class from $k$ members or passes, and measured Kendall's tau between that ranking and the one obtained from the full set. Table~\ref{supp_tab:ens_size} gives the mean over settings and seeds, and in parentheses the minimum. The ENS ranking is already stable with five members ($\tau = """ + f"{g.loc[('ENS', 5), 'mean']:.2f}" + r"""$); BNN reaches a comparable stability with $20$ to $50$ passes; MCD's ranking is the least stable, reaching $\tau = """ + f"{g.loc[('MCD', 50), 'mean']:.2f}" + r"""$ only at $50$ passes, because a large part of its scores are at or near zero and their order is decided by floating-point noise (Section~\ref{app_b:mcd_zero}). The standard deviation over the ten ENS members is identical to that over the $100$ stored rows, so the correction of the ENS description in Section~\ref{app_a:impl} changes no reported number.}

\begin{table}[h]
\centering\small
\caption{{Kendall's tau between the uncertainty ranking from $k$ ensemble members (ENS) or $k$ inference passes (BNN, MCD) and the ranking from the full set; mean over five seeds and two RBP settings, minimum in parentheses.}}
\label{supp_tab:ens_size}
\begin{tabular}{lccccc}
\toprule
 & $k=2$ & $k=3$ & $k=5$ & $k=10$ & \\
\midrule
ENS (members) & """ + row("ENS", [2, 3, 5, 10]) + r""" & \\
\midrule
 & $k=5$ & $k=10$ & $k=20$ & $k=50$ & $k=100$ \\
\midrule
BNN (passes) & """ + row("BNN", [5, 10, 20, 50, 100]) + r""" \\
MCD (passes) & """ + row("MCD", [5, 10, 20, 50, 100]) + r""" \\
\bottomrule
\end{tabular}
\end{table}

"""
sys.stdout.write("%%COVERAGE\n" + cov_tex + "\n%%ENS\n" + ens_tex)
