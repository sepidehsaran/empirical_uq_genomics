"""
plots_agreement.py - agreement between UQ methods:
   Kendall's tau, top-N most-uncertain overlap, KS test, Pearson/Spearman.

Reproduces 2024_figures_section_3_1_agreement.ipynb (cells 7-8 for the overlap bars,
21-22 and 43-46 for Kendall) and cell 51 of section_3_3 for the KS test, and EXTENDS
all four to the simulated TF use case.

For TF the top-N index dicts already exist on the cluster (written by cell 40), but this
module recomputes the overlap from dict_iter_mean/_sd so it works for every use case with
one code path and does not depend on which index dicts happen to be present.
"""
import os
import itertools
import numpy as np
import pandas as pd
import config_paths as CP
import data_access as DA
import metrics as M
import style as S

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


def _need(task):
    comp = DA.load_comparison(task)
    if "mean" not in comp or "sd" not in comp:
        return None, "aggregated dict_iter_mean/_sd not found (set RESULTS_ROOT)"
    return comp, ""


# --------------------------------------------------------------------- Kendall's tau
def kendall(out, task):
    spec = CP.VERSIONS[task]
    comp, err = _need(task)
    res = []
    if comp is None:
        return [dict(id=f"agree_kendall_{task}_{s}", status="skip", reason=err)
                for s in spec["settings"]]
    for setting in spec["settings"]:
        fid = f"agree_kendall_{task}_{setting}"
        kw = DA.setting_to_split_args(task, setting)
        rows = []
        bad = []
        for seed in spec["seeds"]:
            # the dict entry decides which candidate split is the right one
            probe = DA.dict_key(task, seed, setting, CP.MODELS[0])
            n_exp = (np.asarray(comp["mean"][probe]).shape[0]
                     if probe in comp.get("mean", {}) else None)
            y = DA.load_y_test(task, spec["species"], spec["n_classes"], seed,
                               expect_n=n_exp, **kw)
            if y is None:
                continue
            for a, b in CP.PAIRS:
                ka, kb = DA.dict_key(task, seed, setting, a), DA.dict_key(task, seed, setting, b)
                if ka not in comp["mean"] or kb not in comp["mean"]:
                    continue
                try:
                    st = M.kendall_pair(y, comp["mean"][ka], comp["sd"][ka],
                                        comp["mean"][kb], comp["sd"][kb])
                except M.ShapeMismatch as exc:
                    bad.append(f"s{seed} {CP.PAIR_LABEL[(a, b)]}: {exc}")
                    continue
                rows.append(dict(pair=CP.PAIR_LABEL[(a, b)], subset="all samples",
                                 tau=st["tau_all"], seed=seed))
                rows.append(dict(pair=CP.PAIR_LABEL[(a, b)], subset="correctly classified",
                                 tau=st["tau_correct"], seed=seed))
        if not rows:
            why = ("all combinations had mismatched test-set sizes: " + bad[0]
                   if bad else "no matching keys / datasets for this setting")
            res.append(dict(id=fid, status="skip", reason=why))
            continue
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, f"{fid}_values.csv"), index=False)
        fig, ax = S.new_fig(0.48, aspect=0.80)
        if _HAS_SNS:
            sns.boxplot(data=df, x="pair", y="tau", hue="subset", palette=S.PALETTE,
                        ax=ax, linewidth=0.8, fliersize=2)
            ax.legend(title=None, loc="best", fontsize=S.FS_LEG)
        ax.set_xlabel("UQ method pair"); ax.set_ylabel("Kendall's tau")
        S.fig_suptitle(fig, None,
                       f"Agreement (Kendall's tau) - {task.upper()}, "
                       f"{DA.setting_label(task, setting)}")
        note = f"{len(bad)} pair(s) skipped: size mismatch" if bad else ""
        res.append(dict(id=fid, status="ok", note=note,
                        path=S.save(fig, os.path.join(out, f"{fid}.png"))))
    return res


# ------------------------------------------------------------ top-N uncertain overlap
def top_overlap(out, task, topn=100, seed=None):
    spec = CP.VERSIONS[task]
    seed = seed or spec["seeds"][0]
    comp, err = _need(task)
    res = []
    if comp is None:
        return [dict(id=f"agree_top{topn}_{task}_{s}", status="skip", reason=err)
                for s in spec["settings"]]
    for setting in spec["settings"]:
        fid = f"agree_top{topn}_{task}_{setting}"
        kw = DA.setting_to_split_args(task, setting)
        probe = DA.dict_key(task, seed, setting, CP.MODELS[0])
        n_exp = (np.asarray(comp["mean"][probe]).shape[0]
                 if probe in comp.get("mean", {}) else None)
        y = DA.load_y_test(task, spec["species"], spec["n_classes"], seed,
                           expect_n=n_exp, **kw)
        if y is None:
            res.append(dict(id=fid, status="skip", reason=f"dataset seed {seed} not found"))
            continue
        idx = {}
        problem = ""
        for m in CP.MODELS:
            k = DA.dict_key(task, seed, setting, m)
            if k not in comp["mean"]:
                problem = f"missing key for {m}"
                break
            try:
                M.check_same_n("top_overlap", y, comp["mean"][k], comp["sd"][k])
            except M.ShapeMismatch as exc:
                problem = str(exc)
                break
            idx[m] = [M.topn_uncertain_per_class(y, comp["sd"][k], comp["mean"][k], c, topn)
                      for c in range(spec["n_classes"])]
        if problem:
            res.append(dict(id=fid, status="skip", reason=problem))
            continue
        counts = {CP.PAIR_LABEL[(a, b)]:
                  [M.overlap_counts(idx[a][c], idx[b][c]) for c in range(spec["n_classes"])]
                  for a, b in CP.PAIRS}
        pd.DataFrame(counts, index=spec["class_names"]).to_csv(
            os.path.join(out, f"{fid}_values.csv"))
        fig, ax = S.new_fig(0.48, aspect=0.80)
        x = np.arange(spec["n_classes"]); width = 0.26
        for i, (lab, vals) in enumerate(counts.items()):
            bars = ax.bar(x + i * width, vals, width, label=lab)
            ax.bar_label(bars, fmt="%d", padding=2, fontsize=S.FS_LEG - 1)
        ax.set_xticks(x + width)
        ax.set_xticklabels(spec["class_names"], rotation=20, ha="right")
        ax.set_ylabel(f"common samples in top {topn}")
        ax.set_ylim(0, topn * 1.15)
        ax.legend(loc="upper left", ncols=3, fontsize=S.FS_LEG)
        S.fig_suptitle(fig, None,
                       f"Overlap of the {topn} most uncertain samples - {task.upper()}, "
                       f"{DA.setting_label(task, setting)} (seed {seed})")
        note = "NEW - did not exist before." if task == "tf" else ""
        res.append(dict(id=fid, status="ok", note=note,
                        path=S.save(fig, os.path.join(out, f"{fid}.png"))))
    return res


# ------------------------------------------------------------------------- KS test
def ks_test(out, task):
    """
    KS statistic between the reference and perturbed SD distributions.
    reference = first setting, perturbed = each subsequent setting
    (RBP/GEX: balanced vs imbalanced;  TF: clean vs mislabelled - a different question,
    so it must not be pooled with the RBP/GEX panels without saying so in the caption.)
    """
    spec = CP.VERSIONS[task]
    comp, err = _need(task)
    if comp is None:
        return [dict(id=f"agree_ks_{task}", status="skip", reason=err)]
    ref = spec["settings"][0]
    alts = [s for s in spec["settings"][1:]]
    if not alts:
        return [dict(id=f"agree_ks_{task}", status="skip",
                     reason="only one dataset setting available")]
    rows = []
    for seed in spec["seeds"]:
        for alt in alts:
            for m in CP.MODELS:
                kr, ka = DA.dict_key(task, seed, ref, m), DA.dict_key(task, seed, alt, m)
                if kr not in comp["sd"] or ka not in comp["sd"]:
                    continue
                # No equal-shape guard. RBP balanced has 574 test samples and imbalanced
                # 2411, and the old guard skipped the whole RBP figure because of it -
                # even though the KS test is defined for independent samples of
                # different sizes, as the paper's own methods section states.
                for c in range(spec["n_classes"]):
                    st, p, _, _ = M.ks_sd_shift_unpaired(
                        comp["sd"][kr], comp["mean"][kr],
                        comp["sd"][ka], comp["mean"][ka], c, method="exact")
                    if not np.isnan(st):
                        rows.append(dict(model=m.upper(), comparison=
                                         f"{DA.setting_label(task, ref)} vs "
                                         f"{DA.setting_label(task, alt)}",
                                         **{"KS statistic": st, "p": p},
                                         cls=spec["class_names"][c], seed=seed))
    if not rows:
        return [dict(id=f"agree_ks_{task}", status="skip", reason="no comparable SD pairs")]
    df = pd.DataFrame(rows)
    fid = f"agree_ks_{task}"
    df.to_csv(os.path.join(out, f"{fid}_values.csv"), index=False)
    fig, ax = S.new_fig(0.48, aspect=0.80)
    if _HAS_SNS:
        sns.boxplot(data=df, x="model", y="KS statistic", hue="comparison",
                    palette=S.PALETTE, ax=ax, linewidth=0.8, fliersize=2)
        ax.legend(title=None, loc="best", fontsize=S.FS_LEG - 1)
    ax.set_xlabel("UQ method")
    n_rej = int((df["p"] < 0.05).sum())
    S.fig_suptitle(fig, None, f"KS test on SD distributions - {task.upper()} "
                              f"({n_rej}/{len(df)} reject at p<0.05)")
    note = "NEW - did not exist before." if task == "tf" else ""
    return [dict(id=fid, status="ok", note=note,
                 path=S.save(fig, os.path.join(out, f"{fid}.png")))]


# ------------------------------------------------------------- Pearson / Spearman
def pearson_spearman(out, task):
    spec = CP.VERSIONS[task]
    comp, err = _need(task)
    if comp is None:
        return [dict(id=f"agree_corr_{task}_{w}", status="skip", reason=err)
                for w in ("means", "sds")]
    res = []
    for what, src in (("means", "mean"), ("sds", "sd")):
        rows = []
        for seed in spec["seeds"]:
            for setting in spec["settings"]:
                for a, b in CP.PAIRS:
                    ka = DA.dict_key(task, seed, setting, a)
                    kb = DA.dict_key(task, seed, setting, b)
                    if ka not in comp[src] or kb not in comp[src]:
                        continue
                    A, B = np.asarray(comp[src][ka]), np.asarray(comp[src][kb])
                    if A.shape != B.shape:
                        continue
                    st = M.pearson_spearman(A, B)
                    rows.append(dict(pair=CP.PAIR_LABEL[(a, b)],
                                     setting=DA.setting_label(task, setting),
                                     Pearson=st["pearson"], Spearman=st["spearman"],
                                     seed=seed))
        fid = f"agree_corr_{task}_{what}"
        if not rows:
            res.append(dict(id=fid, status="skip", reason="no comparable pairs"))
            continue
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, f"{fid}_values.csv"), index=False)
        long = df.melt(id_vars=["pair", "setting", "seed"],
                       value_vars=["Pearson", "Spearman"],
                       var_name="coefficient", value_name="r")
        fig, ax = S.new_fig(0.48, aspect=0.80)
        if _HAS_SNS:
            sns.boxplot(data=long, x="pair", y="r", hue="coefficient",
                        palette=S.PALETTE, ax=ax, linewidth=0.8, fliersize=2)
            ax.legend(title=None, loc="best", fontsize=S.FS_LEG)
        ax.set_xlabel("UQ method pair"); ax.set_ylabel("correlation")
        S.fig_suptitle(fig, None, f"Correlation of predictive {what} - {task.upper()}")
        res.append(dict(id=fid, status="ok",
                        path=S.save(fig, os.path.join(out, f"{fid}.png"))))
    return res


def run(out):
    r = []
    for task in ("rbp", "gex", "tf"):
        r += kendall(out, task)
        r += top_overlap(out, task, topn=100)
        r += top_overlap(out, task, topn=50)
        r += ks_test(out, task)
        r += pearson_spearman(out, task)
    return r


def tasks(out):
    t = []
    for task in ("rbp", "gex", "tf"):
        t.append((f"Kendall {task}",          lambda tt=task: kendall(out, tt)))
        t.append((f"top-100 overlap {task}",  lambda tt=task: top_overlap(out, tt, 100)))
        t.append((f"top-50 overlap {task}",   lambda tt=task: top_overlap(out, tt, 50)))
        t.append((f"KS test {task}",          lambda tt=task: ks_test(out, tt)))
        t.append((f"Pearson/Spearman {task}", lambda tt=task: pearson_spearman(out, tt)))
    return t
