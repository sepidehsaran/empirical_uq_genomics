"""
plots_sensitivity.py - sensitivity of the uncertainty scores to ONE controlled source
of uncertainty: the SD-shift figures.

Reproduces 2024_figures_section_3_3.ipynb cells 80/91 (which used figsize=(60,30) and
figsize=(40,20) - the cause of the 4 pt fonts) with the shared style, and adds:
  * a compact "examples" variant showing only the diagnostic class per use case,
  * the TF class-imbalance comparison, which was never run (needs models, see notes).

Rows are classes, columns are UQ methods; blue = reference, orange = perturbed.
Optionally also emits the log2(SD) version, as in the manuscript.
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


# which comparison is meaningful per use case:  (reference_setting, perturbed_setting)
COMPARISONS = {
    "rbp": [("bal_single", "imbal_single", "class imbalance, single-label"),
            ("bal_multi",  "imbal_multi",  "class imbalance, multi-label")],
    "gex": [("bal_single", "imbal_single", "class imbalance")],
    "tf":  [("none", "mislabeled", "partial mislabeling"),
            ("none", "no_motif",   "missing motif")],
}

# the class the manuscript text singles out, per use case
DIAGNOSTIC_CLASS = {"rbp": "QKI", "gex": "Erythroblast", "tf": "JUN"}


def _density(ax, values, color, label, log2=False):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if log2:
        v = v[v > 0]
        v = np.log2(v)
    if len(v) < 5:
        return False
    if _HAS_SNS:
        sns.kdeplot(x=v, ax=ax, color=color, fill=True, alpha=0.35,
                    linewidth=1.0, label=label, warn_singular=False)
    else:
        ax.hist(v, bins=40, color=color, alpha=0.45, density=True, label=label)
    return True


def _sd_for(comp, task, seed, setting, model):
    """
    SD/mean for one (setting, model). Prefers the aggregated dicts; falls back to
    running the pretrained model directly.

    The fallback matters for TF 'no_motif': the aggregation loop
    (2024_figures_section_3_1_agreement.ipynb cell 40) only covers
    ls_noise = ['mislabeled', 'none'], so the missing-motif condition never entered the
    dicts - the published Fig 14b was made by direct inference in
    2024_figures_section_3_3.ipynb cells 88-91. With MODELS_ROOT set this reproduces it.
    """
    k = DA.dict_key(task, seed, setting, model)
    if k in comp.get("sd", {}):
        return np.asarray(comp["sd"][k]), np.asarray(comp["mean"][k])
    # laptop path: export_bundle stored the inference-only entries (notably tf/no_motif)
    sens = DA.bundle_part("sensitivity")
    if sens and (task, setting, seed, model) in sens:
        sd, mean = sens[(task, setting, seed, model)]
        return np.asarray(sd), np.asarray(mean)
    spec = CP.VERSIONS[task]
    if task == "tf" and setting == "no_motif":
        # standalone background-sequence test array, evaluated with the model trained on
        # the CLEAN balanced data (there is no no-motif training split)
        x = DA.load_tf_nomotif_testset()
        if x is None:
            return None, None
        it = DA.infer_with_models("tf", spec["species"], spec["n_classes"], seed,
                                  "none", model, x)
        if it is None:
            return None, None
        return it.std(axis=0), it.mean(axis=0)
    kw = DA.setting_to_split_args(task, setting)
    d = DA.load_split(task, spec["species"], spec["n_classes"], seed,
                      keys=("x_test",), **kw)
    if d is None:
        return None, None
    it = DA.infer_with_models(task, spec["species"], spec["n_classes"], seed, setting,
                              model, d["x_test"])
    if it is None:
        return None, None
    return it.std(axis=0), it.mean(axis=0)


def sd_shift_grid(out, task, seed=None, log2=False, examples_only=False):
    """full class x method grid (or just the diagnostic class when examples_only)"""
    spec = CP.VERSIONS[task]
    seed = seed or spec["seeds"][0]
    comp = DA.load_comparison(task)
    res = []
    if "sd" not in comp:
        for ref, alt, lab in COMPARISONS[task]:
            res.append(dict(id=f"sens_sdshift_{task}_{alt}{'_log2' if log2 else ''}"
                               f"{'_example' if examples_only else ''}",
                            status="skip",
                            reason="aggregated dict_iter_sd not found (set RESULTS_ROOT)"))
        return res

    for ref, alt, lab in COMPARISONS[task]:
        fid = (f"sens_sdshift_{task}_{alt}" + ("_log2" if log2 else "")
               + ("_example" if examples_only else ""))
        classes = list(range(spec["n_classes"]))
        if examples_only:
            want = DIAGNOSTIC_CLASS.get(task)
            if want in spec["class_names"]:
                classes = [spec["class_names"].index(want)]
        panels, missing = {}, False
        for m in CP.MODELS:
            sd_r, mn_r = _sd_for(comp, task, seed, ref, m)
            sd_a, mn_a = _sd_for(comp, task, seed, alt, m)
            if sd_r is None or sd_a is None:
                missing = True
                break
            panels[m] = (sd_r, mn_r, sd_a, mn_a)
        if missing:
            res.append(dict(id=fid, status="skip",
                            reason=f"no SDs for '{ref}' vs '{alt}' (seed {seed}): not in "
                                   f"the aggregated dicts and no models to infer with. "
                                   f"TF 'no_motif' is never aggregated - it needs "
                                   f"MODELS_ROOT (see MISSING_INPUTS)."))
            continue

        nrow, ncol = len(classes), len(CP.MODELS)
        # keep each panel ~2.1in wide so fonts land at the same size as elsewhere
        aspect = (nrow * 0.72) / ncol
        fig, axes = S.new_fig(1.0, aspect=max(0.28, aspect), nrows=nrow, ncols=ncol,
                              sharex=False)
        axes = np.atleast_2d(axes)
        if axes.shape != (nrow, ncol):
            axes = axes.reshape(nrow, ncol)
        drew = False
        for ri, ci_ in enumerate(classes):
            for mi, m in enumerate(CP.MODELS):
                ax = axes[ri, mi]
                sd_r, mn_r, sd_a, mn_a = panels[m]
                ok1 = _density(ax, sd_r[:, ci_], S.C_REF,
                               DA.setting_label(task, ref), log2)
                ok2 = _density(ax, sd_a[:, ci_], S.C_ALT,
                               DA.setting_label(task, alt), log2)
                drew = drew or ok1 or ok2
                if ri == 0:
                    ax.set_title(CP.MODEL_LABEL[m], fontsize=S.FS_LABEL)
                if mi == 0:
                    ax.set_ylabel(f"{spec['class_names'][ci_]}\ndensity")
                else:
                    ax.set_ylabel("")
                if ri == nrow - 1:
                    ax.set_xlabel("log2(SD)" if log2 else "SD")
                ax.set_yticks([])
        if not drew:
            res.append(dict(id=fid, status="skip", reason="SD arrays empty/degenerate"))
            continue
        h, l = axes[0, 0].get_legend_handles_labels()
        if h:
            # "outside lower center" reserves a strip; plain "lower center" with a
            # bbox_to_anchor floats the legend on top of the x-axis labels.
            try:
                fig.legend(h, l, loc="outside lower center", ncols=2,
                           fontsize=S.FS_LEG, frameon=False)
            except ValueError:                      # older matplotlib
                fig.legend(h, l, loc="lower center", ncols=2, fontsize=S.FS_LEG,
                           frameon=False, bbox_to_anchor=(0.5, -0.04))
        S.fig_suptitle(fig, None,
                       f"Sensitivity of uncertainty to {lab} - {task.upper()}"
                       + (" [log2]" if log2 else "")
                       + (f" - {spec['class_names'][classes[0]]} only" if examples_only else "")
                       + f" (seed {seed})")
        note = ""
        if task == "tf" and alt == "no_motif":
            note = "TF missing-motif; note the manuscript shows this as Fig 14b."
        res.append(dict(id=fid, status="ok", note=note,
                        path=S.save(fig, os.path.join(out, f"{fid}.png"))))
    return res


def tf_imbalance_from_models(out, seed=None):
    """
    The one genuinely missing experiment: TF class imbalance.
    The imbalanced TF dataset EXISTS locally
    (tf/simulated_human/imbalanced/single_label/no_noise/6class_split<seed>.pickle)
    but no models were ever trained for it - run_several_uq_experiments.ipynb cell 7 has
    ls_balance = [True] only. If MODELS_ROOT contains imbalanced-TF checkpoints this
    function will produce the figure; otherwise it reports what must be trained.
    """
    spec = CP.VERSIONS["tf"]
    seed = seed or spec["seeds"][0]
    fid = "sens_sdshift_tf_imbalance_NEW"
    have_bal = DA.load_split("tf", "simulated_human", 6, seed, balanced=True,
                             noise="none", keys=("x_test",))
    have_imb = DA.load_split("tf", "simulated_human", 6, seed, balanced=False,
                             noise="none", keys=("x_test",))
    if have_imb is None:
        return [dict(id=fid, status="skip", reason="imbalanced TF dataset not found")]
    if not CP.MODELS_ROOT:
        return [dict(id=fid, status="skip",
                     reason="MODELS_ROOT empty: needs models trained on the IMBALANCED TF "
                            "dataset (never trained; set ls_balance=[False] in "
                            "run_several_uq_experiments.ipynb cell 7, then re-aggregate)")]
    panels = {}
    for m in CP.MODELS:
        it_b = DA.infer_with_models("tf", "simulated_human", 6, seed, "none", m,
                                    have_bal["x_test"])
        it_i = DA.infer_with_models("tf", "simulated_human", 6, seed, "none", m,
                                    have_imb["x_test"])
        if it_b is None or it_i is None:
            return [dict(id=fid, status="skip",
                         reason=f"inference unavailable for TF imbalance ({m}); "
                                f"models for the imbalanced TF setting are missing")]
        panels[m] = (it_b.std(axis=0), it_b.mean(axis=0),
                     it_i.std(axis=0), it_i.mean(axis=0))
    nrow, ncol = spec["n_classes"], len(CP.MODELS)
    fig, axes = S.new_fig(1.0, aspect=(nrow * 0.72) / ncol, nrows=nrow, ncols=ncol)
    axes = np.atleast_2d(axes)
    for ri in range(nrow):
        for mi, m in enumerate(CP.MODELS):
            ax = axes[ri, mi]
            sd_b, _, sd_i, _ = panels[m]
            _density(ax, sd_b[:, ri], S.C_REF, "balanced")
            _density(ax, sd_i[:, ri], S.C_ALT, "imbalanced")
            if ri == 0:
                ax.set_title(CP.MODEL_LABEL[m], fontsize=S.FS_LABEL)
            ax.set_ylabel(f"{spec['class_names'][ri]}\ndensity" if mi == 0 else "")
            if ri == nrow - 1:
                ax.set_xlabel("SD")
            ax.set_yticks([])
    h, l = axes[0, 0].get_legend_handles_labels()
    try:
        fig.legend(h, l, loc="outside lower center", ncols=2, fontsize=S.FS_LEG,
                   frameon=False)
    except ValueError:
        fig.legend(h, l, loc="lower center", ncols=2, fontsize=S.FS_LEG, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
    S.fig_suptitle(fig, None, f"Sensitivity to class imbalance - simulated TF (seed {seed}) [NEW]")
    return [dict(id=fid, status="ok", note="NEW experiment",
                 path=S.save(fig, os.path.join(out, f"{fid}.png")))]


def run(out):
    r = []
    for task in ("tf", "rbp", "gex"):
        r += sd_shift_grid(out, task, log2=False)
        r += sd_shift_grid(out, task, log2=True)
        r += sd_shift_grid(out, task, log2=False, examples_only=True)
    r += tf_imbalance_from_models(out)
    return r


def tasks(out):
    t = []
    for task in ("tf", "rbp", "gex"):
        t.append((f"SD-shift {task}",          lambda tt=task: sd_shift_grid(out, tt)))
        t.append((f"SD-shift {task} log2",     lambda tt=task: sd_shift_grid(out, tt, log2=True)))
        t.append((f"SD-shift {task} example",  lambda tt=task: sd_shift_grid(out, tt, examples_only=True)))
    t.append(("TF imbalance [NEW]", lambda: tf_imbalance_from_models(out)))
    return t
