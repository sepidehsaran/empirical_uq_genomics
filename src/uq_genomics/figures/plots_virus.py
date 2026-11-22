"""
plots_virus.py - EXTRA use case: RBP model applied to coronavirus genomes as an
out-of-distribution / distribution-shift test.

Provenance: 2024_figures_section_3_3.ipynb cells 75-81 loaded a single encoded virus
test set (SARS_CoV_1_testdata.pkl) and ran the RBP models over it, then compared the SD
distribution against the in-distribution test set. Only 'ens' and 'mcd' were run there.

What is extended here: all EIGHT encoded coronavirus genomes are present locally
(SARS-CoV-1, SARS-CoV-2 x2 assemblies, MERS, HKU1, OC43, NL63, 229E), so this produces
  1. per-virus SD-shift densities (in-distribution vs virus), per UQ method
  2. a summary of the SD shift across all eight viruses (one panel per method)
  3. a KS-statistic heatmap: virus x UQ method
which turns a single anecdotal OOD panel into a graded distribution-shift experiment.

This module needs MODELS_ROOT (the trained RBP checkpoints); the virus data itself is
already local.
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

REF_SETTING = "bal_single"        # RBP models trained on balanced single-label

# Each encoded genome is ~30 000 sliding windows (~140 MB). For a density/KS comparison
# that is far more than needed, and holding several at once exhausts memory. Windows are
# subsampled with a fixed seed for reproducibility; set to 0 to use all of them.
MAX_VIRUS_WINDOWS = 12000
RBP = CP.VERSIONS["rbp"]


def _virus_inventory():
    vb = DA.bundle_part("virus")
    if vb:
        names = sorted({v for entry in vb.values() for v in entry["per_virus"]})
        return names, [v for v in CP.VIRUSES if v not in names]
    found, missing = [], []
    for v in CP.VIRUSES:
        arr = DA.load_virus(v)
        (found if arr is not None else missing).append(v)
    return found, missing


def _collect(seed):
    """
    returns dict[model] -> (sd_in, mean_in, {virus: sd_virus}) or None.

    Memory note: each encoded genome is ~30 000 x 150 x 4 (about 140 MB as float64).
    Genomes are therefore loaded ONE at a time, used for all three UQ methods, and
    freed immediately - loading them per-model gets the process OOM-killed.
    """
    import gc
    # laptop path: the bundle holds the post-inference SDs, so no genomes or GPU needed
    vb = DA.bundle_part("virus")
    if vb:
        got = {m: (v["sd_in"], v["mean_in"], v["per_virus"])
               for (s_, m), v in vb.items() if s_ == seed}
        if got:
            return got, ""
        seeds_have = sorted({s_ for s_, _ in vb})
        return None, f"bundle has virus data for seeds {seeds_have}, not {seed}"

    comp = DA.load_comparison("rbp")
    d = DA.load_split("rbp", "real_human", 3, seed, balanced=True, multi_label=False,
                      keys=("x_test",))
    if d is None:
        return None, "RBP test split not found"

    # in-distribution reference, per model
    ref = {}
    for m in CP.MODELS:
        key = DA.dict_key("rbp", seed, REF_SETTING, m)
        if "sd" in comp and key in comp["sd"]:
            ref[m] = (np.asarray(comp["sd"][key]), np.asarray(comp["mean"][key]))
        else:
            it = DA.infer_with_models("rbp", "real_human", 3, seed, REF_SETTING, m,
                                      d["x_test"])
            if it is None:
                return None, ("needs either RESULTS_ROOT (aggregated SDs) or MODELS_ROOT "
                              "(to run inference) for the in-distribution reference")
            ref[m] = (it.std(axis=0), it.mean(axis=0))

    per_virus = {m: {} for m in CP.MODELS}
    rng = np.random.default_rng(0)
    for v in CP.VIRUSES:
        x = DA.load_virus(v)
        if x is None:
            continue
        if MAX_VIRUS_WINDOWS and len(x) > MAX_VIRUS_WINDOWS:
            keep = rng.choice(len(x), MAX_VIRUS_WINDOWS, replace=False)
            keep.sort()
            x = x[keep]
        for m in CP.MODELS:
            it = DA.infer_with_models("rbp", "real_human", 3, seed, REF_SETTING, m, x)
            if it is None:
                del x
                gc.collect()
                return None, ("MODELS_ROOT empty or checkpoints missing: virus inference "
                              "needs the trained RBP models")
            per_virus[m][v] = it.std(axis=0)
            del it
        del x
        gc.collect()

    return {m: (ref[m][0], ref[m][1], per_virus[m]) for m in CP.MODELS}, ""


def run(out, seed=None):
    seed = seed or RBP["seeds"][0]
    res = []
    found, missing = _virus_inventory()
    if not found:
        return [dict(id="virus_all", status="skip",
                     reason="no encoded virus test sets found (VIRUS_DIR)")]
    data, err = _collect(seed)
    if data is None:
        # still report what IS available so the gap is unambiguous
        return [dict(id="virus_sdshift", status="skip",
                     reason=f"{err}. Encoded genomes present locally: {', '.join(found)}"
                            + (f"; absent: {', '.join(missing)}" if missing else ""))]

    # ---- 1. per-virus density panels (rows = virus, cols = method)
    viruses = [v for v in CP.VIRUSES if v in data[CP.MODELS[0]][2]]
    nrow, ncol = len(viruses), len(CP.MODELS)
    fig, axes = S.new_fig(1.0, aspect=max(0.3, (nrow * 0.62) / ncol), nrows=nrow, ncols=ncol)
    axes = np.atleast_2d(axes)
    for ri, v in enumerate(viruses):
        for mi, m in enumerate(CP.MODELS):
            ax = axes[ri, mi]
            sd_in, mean_in, per_v = data[m]
            pred_cls = mean_in.argmax(axis=1)
            ref_vals = sd_in[np.arange(len(pred_cls)), pred_cls]
            vs = per_v[v]
            vcls = vs.argmax(axis=1)
            alt_vals = vs[np.arange(len(vcls)), vcls]
            if _HAS_SNS:
                sns.kdeplot(x=ref_vals, ax=ax, color=S.C_REF, fill=True, alpha=0.35,
                            linewidth=1.0, label="human test set", warn_singular=False)
                sns.kdeplot(x=alt_vals, ax=ax, color=S.C_ALT, fill=True, alpha=0.35,
                            linewidth=1.0, label="virus genome", warn_singular=False)
            if ri == 0:
                ax.set_title(CP.MODEL_LABEL[m], fontsize=S.FS_LABEL)
            ax.set_ylabel(f"{CP.VIRUS_LABEL[v]}\ndensity" if mi == 0 else "")
            if ri == nrow - 1:
                ax.set_xlabel("SD of the predicted class")
            ax.set_yticks([])
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncols=2, fontsize=S.FS_LEG)
    S.fig_suptitle(fig, None, f"Distribution shift: RBP model on coronavirus genomes (seed {seed}) [NEW]")
    res.append(dict(id="virus_sdshift_all", status="ok", note="NEW - 8 viruses",
                    path=S.save(fig, os.path.join(out, "virus_sdshift_all.png"))))

    # ---- 2. KS statistic heatmap virus x method
    rows = []
    for m in CP.MODELS:
        sd_in, mean_in, per_v = data[m]
        pred_cls = mean_in.argmax(axis=1)
        ref_vals = sd_in[np.arange(len(pred_cls)), pred_cls]
        for v in viruses:
            vs = per_v[v]
            vcls = vs.argmax(axis=1)
            alt_vals = vs[np.arange(len(vcls)), vcls]
            from scipy import stats as _st
            st, p = _st.ks_2samp(ref_vals, alt_vals, alternative="two-sided")
            rows.append(dict(model=CP.MODEL_LABEL[m], virus=CP.VIRUS_LABEL[v],
                             ks=float(st), p=float(p),
                             median_shift=float(np.median(alt_vals) - np.median(ref_vals))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "virus_ks_values.csv"), index=False)
    piv = df.pivot(index="virus", columns="model", values="ks")
    fig, ax = S.new_fig(0.62, aspect=0.80)
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=S.FS_LEG - 1)
    fig.colorbar(im, ax=ax, label="KS statistic")
    S.fig_suptitle(fig, None, "OOD magnitude per virus (KS vs human test set) [NEW]")
    res.append(dict(id="virus_ks_heatmap", status="ok", note="NEW",
                    path=S.save(fig, os.path.join(out, "virus_ks_heatmap.png"))))

    # ---- 3. median SD shift summary
    fig, ax = S.new_fig(0.62, aspect=0.70)
    if _HAS_SNS:
        sns.barplot(data=df, x="median_shift", y="virus", hue="model",
                    palette=S.PALETTE, ax=ax)
        ax.legend(title=None, fontsize=S.FS_LEG)
    ax.set_xlabel("median SD(virus) - median SD(human test set)")
    ax.set_ylabel("")
    S.fig_suptitle(fig, None, "Uncertainty increase under distribution shift [NEW]")
    res.append(dict(id="virus_median_shift", status="ok", note="NEW",
                    path=S.save(fig, os.path.join(out, "virus_median_shift.png"))))
    if missing:
        res.append(dict(id="virus_inventory", status="skip",
                        reason=f"encoded test set absent for: {', '.join(missing)}"))
    return res


def tasks(out):
    return [("coronavirus OOD [NEW]", lambda: run(out))]
