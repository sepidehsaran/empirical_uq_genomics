"""Supplement sections for OOD-detection metrics, calibration (ECE), and entropy / MI."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))
M = ["MCD", "BNN", "ENS"]

# ---------------------------------------------------------------- OOD
o = pd.read_csv(os.path.join(R, "ood_virus_metrics.csv"))
o = o[o.variant == "max_sd_both"]
pool = o[o.virus == "ALL"].groupby("model")[["auroc", "tpr_at_5fpr", "frac_flagged_95"]].agg(["mean", "min", "max"])
def c3(m, col): v = pool.loc[m, col]; return f"{v['mean']:.2f} ({v['min']:.2f}--{v['max']:.2f})"
ood_rows = "\n".join(f"{m} & {c3(m,'auroc')} & {c3(m,'tpr_at_5fpr')} & {c3(m,'frac_flagged_95')} \\\\" for m in M)
pv = o[o.virus != "ALL"].pivot_table(index="virus", columns="model", values="auroc")
short = {"NC_045512": "SARS-CoV-2 (NC\\_045512)", "wuhCor1": "SARS-CoV-2 (wuhCor1)", "SARS_CoV_1": "SARS-CoV-1",
         "CoV229E": "229E"}
virus_rows = "\n".join(f"{short.get(v, v)} & " + " & ".join(f"{pv.loc[v, m]:.2f}" for m in M) + r" \\" for v in pv.index)
A = pool.loc["BNN", "auroc"]["mean"], pool.loc["ENS", "auroc"]["mean"], pool.loc["MCD", "auroc"]["mean"]
T = pool.loc["BNN", "tpr_at_5fpr"]["mean"], pool.loc["ENS", "tpr_at_5fpr"]["mean"], pool.loc["MCD", "tpr_at_5fpr"]["mean"]

ood_tex = r"""
\paragraph{{Out-of-distribution detection metrics.}}
{To separate the size of the response from its usefulness, we treat the viral windows as the positive class and the human test sequences as the negative class and use the uncertainty score as the detection score. Table~\ref{supp_tab:ood} gives the area under the ROC curve and the true-positive rate at $5\%$ false-positive rate, pooled over all eight genomes and averaged over the five seeds, with the minimum and maximum across seeds. BNN separates viral from human sequences best (AUROC """ + f"{A[0]:.2f}" + r"""), ENS next (""" + f"{A[1]:.2f}" + r""") and MCD worst (""" + f"{A[2]:.2f}" + r"""); at $5\%$ false positives BNN recovers """ + f"{100*T[0]:.0f}" + r"""\% of viral windows against """ + f"{100*T[1]:.0f}" + r"""\% for ENS and """ + f"{100*T[2]:.0f}" + r"""\% for MCD. The ordering is the same for every genome and every seed. The score for the viral windows is the maximum SD over classes, since no prediction is stored for them; using it for the human side as well, or the SD at the predicted class as in Figure~\ref{fig:shift_virus}, changes the numbers by at most $0.01$.}

\begin{table}[h]
\centering\small
\caption{{Out-of-distribution detection on the coronavirus genomes: AUROC, true-positive rate at $5\%$ false-positive rate, and the fraction of viral windows above the $95$th percentile of the human test scores. Mean over five seeds, range in parentheses; all eight genomes pooled.}}
\label{supp_tab:ood}
\begin{tabular}{lccc}
\toprule
 & AUROC & TPR at 5\% FPR & flagged at 95th pct. \\
\midrule
""" + ood_rows + r"""
\midrule
\multicolumn{4}{l}{\textit{AUROC per genome (mean over seeds)}} \\
 & MCD & BNN & ENS \\
""" + virus_rows + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# ---------------------------------------------------------------- calibration
e = pd.read_csv(os.path.join(R, "calibration_ece.csv"))
g = e.groupby(["label", "model"]).agg(ece=("ece", "mean"), sd=("ece", "std"), conf=("mean_conf", "mean"), acc=("acc", "mean"))
order = ["TF clean", "TF mislabeled", "RBP balanced", "RBP imbalanced", "single-cell balanced", "single-cell imbalanced"]
def ec(l, m): r = g.loc[(l, m)]; return f"{r['ece']:.3f} $\\pm$ {r['sd']:.3f} ({r['conf']:.2f} / {r['acc']:.2f})"
ece_rows = "\n".join(f"{l} & " + " & ".join(ec(l, m) for m in M) + r" \\" for l in order)
ens_sc = g.loc[("single-cell balanced", "ENS")]
cal_tex = r"""
\subsection{{Calibration}}\label{app_b:calibration}
{NLL and the Brier score are proper scoring rules and confound calibration with accuracy. Table~\ref{supp_tab:ece} reports the expected calibration error (ECE, 15 equal-width bins on the confidence of the predicted class) for the single-label settings, and Figure~\ref{supp_fig:reliability} the reliability diagrams. On the sequence data all three methods are well calibrated (ECE $0.01$--$0.04$), and BNN is as well calibrated as the others or better. On the single-cell data the deep ensemble is strongly under-confident: its mean confidence is """ + f"{ens_sc['conf']:.2f}" + r""" at an accuracy of """ + f"{ens_sc['acc']:.2f}" + r""" (ECE """ + f"{ens_sc['ece']:.2f}" + r"""), because its ten members disagree on cells they all classify correctly. This is the same effect that gives ENS the highest NLL and Brier score on that task in Figure~\ref{fig:pred_acc}, and it is the starting point from which ENS ``becomes more confident'' when trained on imbalanced data. BNN's calibration on single-cell data varies with the seed (ECE $0.04 \pm 0.03$ and $0.08 \pm 0.06$), following its accuracy.}

\begin{table}[h]
\centering\small
\caption{{Expected calibration error, mean $\pm$ sd over five seeds, with mean confidence / accuracy in parentheses. Single-label settings; 15 bins.}}
\label{supp_tab:ece}
\begin{tabular}{lccc}
\toprule
setting & MCD & BNN & ENS \\
\midrule
""" + ece_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/combined_usecases/RELIABILITY_all_settings.pdf}
\caption{{\textbf{Reliability diagrams.} Accuracy against confidence of the predicted class, 15 bins, pooled over five seeds, for the six single-label settings. The dashed line is perfect calibration; the ECE in the legend is the mean over seeds.}}
\label{supp_fig:reliability}
\end{figure}
"""

# ---------------------------------------------------------------- entropy / MI
ag = pd.read_csv(os.path.join(R, "entropy_mi_agreement.csv"))
sh = pd.read_csv(os.path.join(R, "entropy_mi_shift.csv"))
tasks_done = sorted(ag.task.unique())
agr = ag.groupby(["task", "model"])[["tau_sd_mi", "tau_sd_entropy", "frac_mi_zero"]].mean()
tname = {"rbp_v19": "RBP, class imbalance", "tf_v18": "TF, label noise", "gex_v19": "single-cell, class imbalance"}
agr_rows = "\n".join(f"{tname.get(t,t)} & {m} & {agr.loc[(t,m),'tau_sd_mi']:.2f} & {agr.loc[(t,m),'tau_sd_entropy']:.2f} & {100*agr.loc[(t,m),'frac_mi_zero']:.0f}\\% \\\\"
                     for t in tasks_done for m in M)
def srow(t, cls, score):
    s = sh[(sh.task == t) & (sh.cls == cls) & (sh.score == score)]
    cells = []
    for m in M:
        v = s[s.model == m]["shift"].values
        cells.append(f"${v.mean():+.1f}$ ({int((v>0).sum())}/{len(v)})" if len(v) else "--")
    return " & ".join(cells)
shift_rows = ""
for t in tasks_done:
    classes = list(dict.fromkeys(sh[sh.task == t].cls))
    shift_rows += r"\multicolumn{7}{l}{\textit{" + tname.get(t, t) + r"}} \\" + "\n"
    for cls in classes:
        shift_rows += cls.replace("+", "$+$") + " & " + srow(t, cls, "SD") + " & " + srow(t, cls, "MI") + r" \\" + "\n"
pending = [t for t in ["tf_v18", "gex_v19"] if t not in tasks_done]
names = {"tf_v18": "TF", "gex_v19": "single-cell"}
pend_txt = ""   # all three data sets are in the table; nothing pending
mi_tex = r"""
\subsection{{Predictive entropy and mutual information}}\label{app_b:entropy_mi}
{All analyses in the main text use the standard deviation of the predicted probability across passes as the uncertainty score. From the same passes one can also compute the predictive entropy $H[\bar p]$ (total uncertainty), the expected entropy $\mathbb{E}_t H[p_t]$ (aleatoric part) and their difference, the mutual information between prediction and parameters (epistemic part, i.e.\ the disagreement between passes). Table~\ref{supp_tab:entropy_mi} compares them with the SD score on the raw per-pass predictions. For BNN and ENS, mutual information and the SD rank the test samples nearly identically (Kendall $\tau$ between """ + f"{agr.loc[[(t, m) for t in tasks_done for m in ('BNN', 'ENS')], 'tau_sd_mi'].min():.2f}" + r""" and """ + f"{agr.loc[[(t, m) for t in tasks_done for m in ('BNN', 'ENS')], 'tau_sd_mi'].max():.2f}" + r""") and give the same reference-to-perturbed shift per class, so none of the conclusions depends on which of the three scores is used. For MC-dropout the mutual information is exactly zero for most samples, for the same reason its SD is zero (Section~\ref{app_b:mcd_zero}).""" + pend_txt + r"""}

\begin{table}[h]
\centering\small
\caption{{Top: rank agreement between the SD score and mutual information (MI) or predictive entropy (Kendall $\tau$, mean over seeds and settings), and the fraction of test samples with zero MI. Bottom: median shift ($\log_2$) at the predicted class from the reference to the perturbed setting, for the SD score and for MI; mean over five seeds (seeds with a positive shift).}}
\label{supp_tab:entropy_mi}
\begin{tabular}{llccc}
\toprule
task & method & $\tau$(SD, MI) & $\tau$(SD, entropy) & MI $=0$ \\
\midrule
""" + agr_rows + r"""
\bottomrule
\end{tabular}

\vspace{0.8em}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{SD score} & \multicolumn{3}{c}{mutual information} \\
class & MCD & BNN & ENS & MCD & BNN & ENS \\
\midrule
""" + shift_rows + r"""\bottomrule
\end{tabular}
\end{table}
"""
sys.stdout.write("%%OOD\n" + ood_tex + "\n%%CAL\n" + cal_tex + "\n%%MI\n" + mi_tex)
