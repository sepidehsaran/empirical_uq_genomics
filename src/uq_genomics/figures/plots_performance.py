"""
plots_performance.py - predictive performance: accuracy, NLL, Brier.

Reproduces 2024_figures_section_3_3.ipynb cells 62-66 (RBP) and its GEX twin, and
EXTENDS the same code to the simulated TF use case, which was never plotted.

Inputs: dict_iter_mean (aggregated predictive means) + y_test from the local dataset
pickles. No model inference, no GPU. Everything here is unblocked as soon as
RESULTS_ROOT points at a copy of the comparison_results folders.
"""
import os
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


def _collect(task):
    """long-form dataframe of accuracy / NLL / Brier per (setting, model, seed)"""
    spec = CP.VERSIONS[task]
    comp = DA.load_comparison(task)
    if "mean" not in comp:
        return None, "aggregated dict_iter_mean.pickle not found (set RESULTS_ROOT)"
    dmean = comp["mean"]
    rows, missing, mismatch = [], 0, []
    for seed in spec["seeds"]:
        for setting in spec["settings"]:
            kw = DA.setting_to_split_args(task, setting)
            probe = DA.dict_key(task, seed, setting, CP.MODELS[0])
            n_exp = np.asarray(dmean[probe]).shape[0] if probe in dmean else None
            y = DA.load_y_test(task, spec["species"], spec["n_classes"], seed,
                               expect_n=n_exp, **kw)
            if y is None:
                missing += 1
                continue
            for model in CP.MODELS:
                key = DA.dict_key(task, seed, setting, model)
                if key not in dmean:
                    missing += 1
                    continue
                preds = np.asarray(dmean[key])
                if preds.shape[0] != y.shape[0]:
                    missing += 1
                    mismatch.append(f"{key}: {preds.shape[0]} vs y {y.shape[0]}")
                    continue
                multi = kw.get("multi_label", False)
                acc = (M.accuracy_multi_label(preds, y, apply_sigmoid=False, th=0.5)
                       if multi else M.accuracy_single_label(preds, y))
                rows.append(dict(setting=DA.setting_label(task, setting),
                                 model=model.upper(), seed=seed,
                                 Accuracy=acc,
                                 NLL=M.calculate_nll(preds, y, multi_label=multi),
                                 Brier=M.calculate_brier_score(preds, y)))
    if not rows:
        return None, "no matching keys in dict_iter_mean (check version/seed/setting)"
    return pd.DataFrame(rows), (f"{missing} combination(s) unavailable" if missing else "")


# Panel width as a fraction of \textwidth. 0.48 suits a two-up figure; set 0.31 when
# the panels will be combined three-across, so each is exported at the width it will
# actually occupy and the font stays 9 pt instead of shrinking on assembly.
#   export UQFIG_WIDTH_PERF=0.31
_WIDTH = float(os.environ.get("UQFIG_WIDTH_PERF", "0.48"))


# Legend placement. The default keeps the legend inside the axes, which overlaps the
# boxes when the panel is small. 'outside' puts it under the axes instead, and
# 'none' omits it entirely so a combined figure can draw ONE shared legend per column.
#   export UQFIG_LEGEND_PERF=outside|inside|none
_LEGEND = os.environ.get("UQFIG_LEGEND_PERF", "outside").lower()


def _box(df, value, title, path, width_frac=None, legend=None):
    width_frac = _WIDTH if width_frac is None else width_frac
    legend = _LEGEND if legend is None else legend
    # 'outside lower center' makes constrained layout RESERVE space for the legend;
    # a plain fig.legend with bbox_to_anchor just draws on top of the x-axis label.
    aspect = 0.86 + (0.12 if legend == "outside" else 0.0)
    fig, ax = S.new_fig(width_frac, aspect=aspect)
    if _HAS_SNS:
        sns.boxplot(data=df, x="model", y=value, hue="setting",
                    palette=S.PALETTE, ax=ax, linewidth=0.8, fliersize=2)
        h, l = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        if legend == "outside":
            ncol = 2 if len(l) > 2 else len(l)
            try:
                fig.legend(h, l, loc="outside lower center", ncols=ncol,
                           fontsize=S.FS_LEG - (1.0 if width_frac < 0.4 else 0),
                           frameon=False)
            except ValueError:          # matplotlib < 3.6
                fig.legend(h, l, loc="lower center", ncols=ncol,
                           fontsize=S.FS_LEG - (1.0 if width_frac < 0.4 else 0),
                           frameon=False, bbox_to_anchor=(0.5, -0.02))
        elif legend == "inside":
            ax.legend(title=None, loc="best",
                      fontsize=S.FS_LEG - (1.0 if width_frac < 0.4 else 0))
        # legend == "none": handles are dropped; the caller supplies one
    else:
        groups = sorted(df["model"].unique())
        ax.boxplot([df.loc[df.model == g, value] for g in groups], labels=groups)
    # The x tick labels (MCD / BNN / ENS) already say what the axis is, and in a
    # combined figure the label would repeat once per panel. Set
    # UQFIG_XLABEL_PERF=1 to put it back.
    ax.set_xlabel("UQ method" if os.environ.get("UQFIG_XLABEL_PERF") else "")
    ax.set_ylabel(value)
    S.fig_suptitle(fig, None, title)
    return S.save(fig, path)


def legend_strip(df, path, width_frac=1.0, value="Accuracy"):
    """
    Standalone legend for a column of the combined figure.

    The handles are taken from a REAL boxplot drawn on a throwaway figure, so the
    colours and order are guaranteed identical to the panels. Building patches by
    hand re-derives the palette and silently gets it wrong when the number of
    categories differs (RBP has 4 settings, TF has 2).
    """
    import matplotlib.pyplot as plt
    if not _HAS_SNS:
        return ""
    tmp = plt.figure(figsize=(4, 3))
    ax_t = tmp.add_subplot(111)
    sns.boxplot(data=df, x="model", y=value, hue="setting",
                palette=S.PALETTE, ax=ax_t)
    handles, labels = ax_t.get_legend_handles_labels()
    plt.close(tmp)

    n = len(labels)
    # Fit check rather than a fixed 2 columns: at ~2 in wide, two columns of
    # "imbalanced, single-label" get clipped. Estimate the rendered width of the
    # widest label (0.5 em per char at FS_LEG) plus the colour handle.
    w_in = S.TEXTWIDTH_IN * width_frac
    char_in = 0.5 * S.FS_LEG / 72.0
    widest_in = max(len(l) for l in labels) * char_in + 0.34
    ncol = 2 if (n > 2 and widest_in * 2 <= w_in * 0.95) else (1 if n > 2 else n)
    if n == 2 and widest_in * 2 > w_in * 0.95:
        ncol = 1
    nrow = (n + ncol - 1) // ncol
    # height must fit the rows; too short and constrained layout collapses the axes
    # and clips the text (that is what produced the broken first version)
    h_in = 0.16 + 0.20 * nrow
    fig = plt.figure(figsize=(w_in, h_in))
    fig.legend(handles, labels, loc="center", ncols=ncol,
               fontsize=S.FS_LEG, frameon=False, handlelength=1.6,
               columnspacing=1.4, handletextpad=0.6)
    fig.savefig(path, dpi=S.DPI)
    plt.close(fig)
    return path


def run(out):
    res = []
    for task in ("rbp", "gex", "tf"):
        label = {"rbp": "real RBP binding", "gex": "single-cell GEX",
                 "tf": "simulated TF binding"}[task]
        df, note = _collect(task)
        if df is None:
            for metric in ("acc", "nll", "brier"):
                res.append(dict(id=f"perf_{task}_{metric}", status="skip", reason=note))
            continue
        df.to_csv(os.path.join(out, f"perf_{task}_values.csv"), index=False)
        for metric, col in (("acc", "Accuracy"), ("nll", "NLL"), ("brier", "Brier")):
            fid = f"perf_{task}_{metric}"
            p = _box(df, col, f"{col} - {label} (5 seeds)",
                     os.path.join(out, f"{fid}.png"))
            res.append(dict(id=fid, status="ok", path=p,
                            note=("NEW - did not exist before. " if task == "tf" else "") + note))
    return res


def tasks(out):
    return [(f"performance {t}", (lambda tt=t: _one(out, tt))) for t in ("rbp", "gex", "tf")]


def _one(out, task):
    label = {"rbp": "real RBP binding", "gex": "single-cell GEX",
             "tf": "simulated TF binding"}[task]
    df, note = _collect(task)
    if df is None:
        return [dict(id=f"perf_{task}_{m}", status="skip", reason=note)
                for m in ("acc", "nll", "brier")]
    df.to_csv(os.path.join(out, f"perf_{task}_values.csv"), index=False)
    res = []
    for metric, col in (("acc", "Accuracy"), ("nll", "NLL"), ("brier", "Brier")):
        fid = f"perf_{task}_{metric}"
        p = _box(df, col, f"{col} - {label} (5 seeds)", os.path.join(out, f"{fid}.png"))
        res.append(dict(id=fid, status="ok", path=p,
                        note=("NEW - did not exist before. " if task == "tf" else "") + note))
    fid = f"perf_{task}_legend"
    res.append(dict(id=fid, status="ok",
                    note="shared legend strip for the combined figure",
                    path=legend_strip(df, os.path.join(out, f"{fid}.png"),
                                      width_frac=1.0)))
    return res
