"""Build the supplementary LaTeX section with the cross-seed tables from out/tables/*.csv.

    python3 scripts/supp_tables/tables_sensitivity.py > /tmp/seed_shift_section.tex

Every number comes from a CSV written by cross_seed_tables.py, rbp_shift_shared_testset.py
or the paired-test / paper-definition scripts, so the tables can be regenerated.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))

pdef = pd.read_csv(os.path.join(R, "sdshift_paperdef_allsamples.csv"))
rbp1 = pd.read_csv(os.path.join(R, "rbp_shift_shared_testset_paperdef.csv"))
rbp2 = pd.read_csv(os.path.join(R, "rbp_shift_shared_testset.csv"))
pair = pd.read_csv(os.path.join(R, "paired_signedrank_tf_gex.csv"))
mz = pd.read_csv(os.path.join(R, "mcd_zero_variance_and_tau.csv"))
M = ["MCD", "BNN", "ENS"]


def cell(g, col):
    v = g[col].dropna().values
    if len(v) == 0:
        return "--"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"${v.mean():+.1f} \\pm {sd:.1f}$ ({int((v > 0).sum())}/{len(v)})"


def shift_rows(df, col, order):
    return "\n".join(c.replace("+", "$+$") + " & " + " & ".join(cell(df[(df.cls == c) & (df.model == m)], col) for m in M) + r" \\"
                     for c in order)


def pcell(g):
    return f"${g.median_paired_diff.mean():+.1f}$ / {100 * g.frac_up.mean():.0f}\\% ({int((g.p < 0.05).sum())}/5)"


def pair_rows(task, order):
    t = pair[pair.task == task]
    return "\n".join(c.replace("+", "$+$") + " & " + " & ".join(pcell(t[(t.cls == c) & (t.model == m)]) for m in M) + r" \\"
                     for c in order)


TF = ["JUN", "MEF2A", "CTCF", "HNF4A", "MYC", "TAL1"]
GEX = ["Erythroblast", "CD4+ T activated", "CD14+ Mono", "NK"]
RBP = ["MBNL1", "PUM2", "QKI"]

lab = {("tf", "none"): "TF clean", ("tf", "mislabeled"): "TF mislabeled",
       ("gex", "bal_single"): "single-cell balanced", ("gex", "imbal_single"): "single-cell imbalanced",
       ("rbp", "bal_single"): "RBP balanced single-label", ("rbp", "imbal_single"): "RBP imbalanced single-label",
       ("rbp", "bal_multi"): "RBP balanced multi-label", ("rbp", "imbal_multi"): "RBP imbalanced multi-label"}
m = mz[mz.model == "MCD"].dropna(subset=["frac_sd_zero_true_class"])
mrows = []
for k, l in lab.items():
    g = m[(m.task == k[0]) & (m.setting == k[1])]
    if len(g):
        mrows.append(f"{l} & {100 * g.frac_sd_zero_true_class.mean():.0f}\\% & "
                     f"{100 * g.frac_sd_zero_true_class.min():.0f}--{100 * g.frac_sd_zero_true_class.max():.0f}\\% & "
                     f"{100 * g.frac_sd_zero_all_classes.mean():.0f}\\% \\\\")

tex = r"""
\subsection{{Shift of the uncertainty scores across random seeds}}\label{app_b:seed_shift}
{The sensitivity figures in the main text show the uncertainty-score distributions for one seed. Tables~\ref{supp_tab:seed_shift} and~\ref{supp_tab:paired} give the shift for all five seeds. For each class and seed we take the median $\log_2$(SD) of that class's probability over all test samples with non-zero SD, in the reference and in the perturbed setting, and report the difference, averaged over seeds with its standard deviation; the number of seeds in which the shift is positive is given in parentheses. With five seeds a sign test cannot reach $p<0.05$ (its smallest attainable two-sided $p$ is $0.0625$), so we report the consistency of the sign rather than a $p$-value. For the RBP data the balanced and imbalanced settings in the main text were evaluated on different test sets (574 and 2,411 sequences, the latter 71\% MBNL1); the table instead uses runs that evaluate both settings on the same 574-sequence balanced test set, and gives the shift both over all test samples and restricted to the samples each model predicts as the class.}

\begin{table}[h]
\centering\small
\caption{{Median $\log_2$(SD) shift per class from the reference to the perturbed setting, mean $\pm$ sd over five seeds (seeds with a positive shift). Top: simulated TF, clean to $30\%$ mislabeled; JUN and MEF2A are the swapped classes. Middle: single-cell, balanced to imbalanced; Erythroblast is the depleted class. Bottom: real RBP, balanced to imbalanced on the shared balanced test set, over all test samples and over the samples predicted as the class.}}
\label{supp_tab:seed_shift}
\begin{tabular}{lccc}
\toprule
 & MCD & BNN & ENS \\
\midrule
\multicolumn{4}{l}{\textit{Simulated TF, label noise}} \\
""" + shift_rows(pdef[pdef.task == "tf"], "sh", TF) + r"""
\midrule
\multicolumn{4}{l}{\textit{Single-cell, class imbalance}} \\
""" + shift_rows(pdef[pdef.task == "gex"], "sh", GEX) + r"""
\midrule
\multicolumn{4}{l}{\textit{Real RBP, class imbalance, shared test set, all samples}} \\
""" + shift_rows(rbp1, "sh", RBP) + r"""
\midrule
\multicolumn{4}{l}{\textit{Real RBP, class imbalance, shared test set, predicted class}} \\
""" + shift_rows(rbp2, "median_log2_ratio", RBP) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering\small
\caption{{Paired comparison on the settings that share a test set. For each test sample, the $\log_2$(SD) of the class's probability is compared between the reference and the perturbed setting. Cells give the median paired difference averaged over seeds, the percentage of samples whose uncertainty increased, and the number of seeds in which a Wilcoxon signed-rank test on the paired differences rejects the null at $p<0.05$. With 1,800 and 3,600 paired samples the test is sensitive to small shifts, so the direction and the percentage are the informative quantities.}}
\label{supp_tab:paired}
\begin{tabular}{lccc}
\toprule
 & MCD & BNN & ENS \\
\midrule
\multicolumn{4}{l}{\textit{Simulated TF, clean to mislabeled (1,800 paired samples)}} \\
""" + pair_rows("tf", TF) + r"""
\midrule
\multicolumn{4}{l}{\textit{Single-cell, balanced to imbalanced (3,600 paired samples)}} \\
""" + pair_rows("gex", GEX) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering\small
\caption{{Fraction of test samples for which MC-dropout's uncertainty score is exactly zero, per setting: at the true class (mean and range over five seeds) and for all classes at once. Kendall's tau in Figure~\ref{fig:kendall} uses the tie-corrected $\tau_b$ and remains valid, but for MC-dropout it is computed over a shrinking effective sample whose size changes between settings.}}
\label{supp_tab:mcd_zero}
\begin{tabular}{lccc}
\toprule
Setting & zero at true class & range over seeds & zero for all classes \\
\midrule
""" + "\n".join(mrows) + r"""
\bottomrule
\end{tabular}
\end{table}

"""
sys.stdout.write(tex)
