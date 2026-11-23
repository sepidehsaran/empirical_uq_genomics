"""Manuscript-ready SD-shift figures, one file per (use case, perturbation, scale).

    export UQFIG_BUNDLE=$PWD/figure_data_bundle
    python3 plots_sdshift.py --out runs/sdshift

Why this exists rather than plots_sensitivity.sd_shift_grid():

  - Vector PDF. The figures currently in the manuscript are 6000x3000 PNGs.
  - Sized to be STACKED. Figure 9 places two of these in one float, so each has to fit
    in half a page; sd_shift_grid produces a 9.4 in tall figure, which cannot stack.
  - No suptitle. The manuscript caption already says what the figure is, and the baked-in
    "- TF [log2] (seed 7)" title duplicated it.
  - Panel letter instead, so \\ref{...}a and b resolve to something visible.

On the log scale. For the simulated TF data the raw-SD version is unreadable: MC-dropout's
standard deviations run to about 1e-6, so all six classes render as a single spike at the
origin and nothing can be compared. That is not a plotting choice, it is the saturation
documented in VERIFY_mcd_zero_variance.md showing through. log2 spreads the distribution
over ~60 units and makes the shift visible for every method.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_paths as CP                    # noqa: E402
import data_access as DA                     # noqa: E402
import style as S                            # noqa: E402
import plots_sensitivity as PS               # noqa: E402

# (task, ref, alt, short label, stem, letter_log2, letter_raw)
#
# The panel letter depends on which float the file ends up in, not on the scale, so it is
# stated per output rather than derived. Current layout:
#   main Fig 9   = tf_mislabeled_log2 alone           -> no letter
#   main Fig 10  = rbp_imbalance_log2 + gex_imbalance_log2   -> a, b
#   suppl        = tf_no_motif_log2 alone             -> no letter
#   suppl        = tf_mislabeled_raw + tf_no_motif_raw      -> a, b
#   suppl        = rbp_imbalance_raw + gex_imbalance_raw    -> a, b
# None means "draw no letter"; a lone panel labelled (b) is worse than none.
JOBS = [
    ("tf", "none", "mislabeled", "partial mislabeling", "tf_mislabeled", None, "a"),
    ("tf", "none", "no_motif", "missing motif", "tf_no_motif", None, "b"),
    ("rbp", "bal_single", "imbal_single", "class imbalance", "rbp_imbalance", "a", "a"),
    ("gex", "bal_single", "imbal_single", "class imbalance", "gex_imbalance", "b", "b"),
]


def _wrap_label(s, width=11):
    """Break a long class name across two lines at a space.

    Row labels used to be drawn vertically (matplotlib's default ylabel rotation). At
    four rows in a 3.9 in figure each row is under 0.8 in tall, so "CD4+ T activated"
    and "CD14+ Mono" were taller than their own panel and printed over each other down
    the left edge. Horizontal labels do not have that failure mode, but they do need to
    be short, hence the wrap.
    """
    if len(s) <= width or " " not in s:
        return s
    words, lines, cur = s.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _balance_rows(task, setting, seed, n_classes, rng_seed=0):
    """Row indices that make `setting`'s test set have equal class counts.

    Why this is needed for RBP. Three trees exist on disk for the imbalanced setting:
    `balanced/` (test 574, classes 192/188/194), `imbalanced/` (test 2411, classes
    1701/534/176) and `train_imbalanced_test_balanced/` (test 574, byte-identical to
    the balanced tree's test set). The intended comparison -- and the one the
    supplement describes -- is balanced vs train_imbalanced_test_balanced, i.e. the
    SAME test set with only the training set differing. The aggregated dicts were
    instead built against `imbalanced/`, whose test set is 71% MBNL1.

    Because the density for class c is taken over every test row, that composition
    difference shifts the distributions on its own, independently of any training
    imbalance. Subsampling to equal class counts removes it. This is an approximation
    to re-running inference on the 574-row balanced test set, which needs the trained
    v_17 models; those are not in this checkout.
    """
    lab = DA.bundle_part(f"labels_{task}")
    y = lab.get((setting, seed)) if lab else None
    if y is None:
        return None
    y = np.asarray(y).argmax(1)
    counts = np.bincount(y, minlength=n_classes)
    if counts.min() == 0 or counts.max() == counts.min():
        return None                      # already balanced, nothing to do
    rng = np.random.default_rng(rng_seed)
    return np.concatenate([rng.choice(np.where(y == c)[0], counts.min(), replace=False)
                           for c in range(n_classes)])


def build(task, ref, alt, stem, out, log2, letter, aspect, seed=None, balance_test=False,
          sd_source=None, suffix=""):
    """
    sd_source : optional callable(model, seed) -> (sd_ref, sd_alt).
        Used to plot predictions that are not in the aggregated bundle -- in
        particular the RBP `train_imbalanced_test_balanced` runs, where both settings
        share one 574-sequence test set and no subsampling correction is needed.
    suffix : appended to the output filename, so a rebuild can sit beside the old file
        instead of overwriting it.
    """
    spec = CP.VERSIONS[task]
    seed = seed or spec["seeds"][0]
    comp = DA.load_comparison(task) if sd_source is None else None
    if sd_source is None and "sd" not in comp:
        print(f"  {stem}: no data")
        return None
    keep = (_balance_rows(task, alt, seed, spec["n_classes"])
            if (balance_test and sd_source is None) else None)
    if balance_test and sd_source is None:
        print(f"    {stem}: test set of '{alt}' subsampled to "
              f"{'n/a' if keep is None else len(keep)} rows with equal class counts")
    panels = {}
    for m in CP.MODELS:
        if sd_source is not None:
            sd_r, sd_a = sd_source(m, seed)
        else:
            sd_r, mn_r = PS._sd_for(comp, task, seed, ref, m)
            sd_a, mn_a = PS._sd_for(comp, task, seed, alt, m)
        if sd_r is None or sd_a is None:
            print(f"  {stem}: missing SDs for {m}")
            return None
        if keep is not None:
            sd_a = sd_a[keep]
        panels[m] = (sd_r, sd_a)

    nrow, ncol = spec["n_classes"], len(CP.MODELS)
    row_labels = [_wrap_label(c) for c in spec["class_names"]]
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * aspect), dpi=S.DPI)
    # room at the bottom for the legend + x label, at the top for the panel letter.
    # The left margin now has to hold a horizontal row label, so it scales with the
    # longest line rather than being fixed at the width of a rotated one.
    # One left margin for every sensitivity figure, sized for the longest row label of
    # any use case ("Erythroblast"), so that the panel grids of the RBP and single-cell
    # figures line up when they are stacked as a) and b).
    L = 0.20
    R, T, B, hgap, vgap = 0.988, 0.905, 0.205, 0.055, 0.030
    if letter is None:
        T = 0.945          # no letter to leave room for
    aw = (R - L - hgap * (ncol - 1)) / ncol
    ah = (T - B - vgap * (nrow - 1)) / nrow
    handles = None
    for ri in range(nrow):
        for mi, m in enumerate(CP.MODELS):
            ax = fig.add_axes([L + mi * (aw + hgap), T - ah - ri * (ah + vgap), aw, ah])
            sd_r, sd_a = panels[m]
            PS._density(ax, sd_r[:, ri], S.C_REF, DA.setting_label(task, ref), log2)
            PS._density(ax, sd_a[:, ri], S.C_ALT, DA.setting_label(task, alt), log2)
            if handles is None:
                handles = ax.get_legend_handles_labels()
            if ri == 0:
                ax.set_title(CP.MODEL_LABEL[m], fontsize=S.FS_LABEL - 0.5)
            # seaborn's kdeplot writes "Density" on EVERY axes; clearing it only on the
            # first column left it stamped down the middle of the figure twice over.
            ax.set_ylabel(row_labels[ri] if mi == 0 else "",
                          fontsize=S.FS_TICK - 0.5,
                          rotation=0, ha="right", va="center", labelpad=6)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=S.FS_TICK - 1.5)
            if ri == nrow - 1:
                ax.set_xlabel("log$_2$(SD)" if log2 else "SD",
                              fontsize=S.FS_LABEL - 0.5)
            else:
                # Only the bottom row is labelled, but every row keeps its ticks: the
                # x range differs per method by orders of magnitude, so a shared axis
                # would be wrong.
                ax.set_xticklabels([])
    if letter:
        fig.text(0.004, 0.982, f"{letter})", fontsize=S.FS_BASE, fontweight="bold",
                 va="top", ha="left")
    # The panels are kernel densities; the y scale carries no information beyond shape,
    # so ticks stay off, but the axis is named once so the reader knows what it is.
    fig.text(0.006, (T + B) / 2, "density (kernel estimate; y scale per panel)", rotation=90,
             fontsize=S.FS_TICK - 1.5, color="0.35", va="center", ha="left")
    if handles and handles[0]:
        fig.legend(*handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncols=2,
                   fontsize=S.FS_LEG, frameon=False)
    name = (f"SDSHIFT_{stem}" + ("_log2" if log2 else "_raw")
            + ("_baltest" if keep is not None else "") + suffix)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=S.DPI)
        paths.append(p)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  {name}: {w:.2f} x {h:.2f} in  (seed {seed})")
    return paths


def build_virus(out, seed=7, aspect=1.02, topn=4):
    """Distribution-shift figure: RBP model trained on human data, applied to coronavirus
    genomes. One row per genome, one column per UQ method.

    Differences from plots_virus.run():
      - vector PDF, no baked-in "(seed 7) [NEW]" suptitle
      - short genome labels. The full names ("SARS-CoV-2 (NC_045512)") are longer than
        the row is tall, so in the original they overlapped each other down the left edge
      - the legend sits below the x labels rather than on top of them
      - topn limits the figure to the first `topn` genomes; all eight make a 10.7 in
        figure, which needs its own page
    """
    import plots_virus as PV
    if seed == "all":
        # pool every seed in the bundle: concatenate human-side and virus-side arrays
        import numpy as _np
        parts = [PV._collect(s)[0] for s in CP.VERSIONS["rbp"]["seeds"]]
        parts = [p for p in parts if p]
        if not parts:
            print("  virus: no seeds in bundle"); return None
        got = {}
        for m in CP.MODELS:
            sd_in = _np.concatenate([p[m][0] for p in parts]); mean_in = _np.concatenate([p[m][1] for p in parts])
            pv = {v: _np.concatenate([p[m][2][v] for p in parts]) for v in parts[0][m][2]}
            got[m] = (sd_in, mean_in, pv)
        err = ""
    else:
        got, err = PV._collect(seed)
    if not got:
        print(f"  virus: {err}")
        return None
    names = [v for v in CP.VIRUSES if v in got[CP.MODELS[0]][2]][:topn]
    short = {"SARS_CoV_1": "SARS-CoV-1", "NC_045512": "SARS-CoV-2", "wuhCor1": "SARS-CoV-2*",
             "MERS": "MERS-CoV", "HKU1": "HCoV-HKU1", "OC43": "HCoV-OC43",
             "NL63": "HCoV-NL63", "CoV229E": "HCoV-229E"}
    nrow, ncol = len(names), len(CP.MODELS)
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * aspect), dpi=S.DPI)
    L, R, T, B, hgap, vgap = 0.115, 0.988, 0.925, 0.115, 0.055, 0.045
    aw = (R - L - hgap * (ncol - 1)) / ncol
    ah = (T - B - vgap * (nrow - 1)) / nrow
    # MC-dropout's SDs are ~1e-6 here (the saturation documented in
    # VERIFY_mcd_zero_variance.md), so matplotlib puts a "1e-6" offset label at the right
    # end of the axis, where it collided with "SD of the predicted class". Scale the
    # column instead and state the factor in the axis label.
    scale, sfx = {}, {}
    for m in CP.MODELS:
        sd_in, mean_in, per_v = got[m]
        hi = max(float(np.asarray(per_v[v]).max()) for v in names)
        e = 6 if hi < 1e-3 else (3 if hi < 1 else 0)
        scale[m] = 10.0 ** e
        sfx[m] = f" ($\\times 10^{{-{e}}}$)" if e else ""

    handles = None
    for ri, v in enumerate(names):
        for mi, m in enumerate(CP.MODELS):
            ax = fig.add_axes([L + mi * (aw + hgap), T - ah - ri * (ah + vgap), aw, ah])
            sd_in, mean_in, per_v = got[m]
            sd_in = np.asarray(sd_in); mean_in = np.asarray(mean_in)
            sd_v = np.asarray(per_v[v])
            # score = SD at the class the model predicts, matching the manuscript's
            # definition of the uncertainty score for a sample
            a = sd_in[np.arange(len(sd_in)), mean_in.argmax(1)] * scale[m]
            b = sd_v.max(axis=1) * scale[m]
            PS._density(ax, a, S.C_REF, "human test set", False)
            PS._density(ax, b, S.C_ALT, "virus genome", False)
            if handles is None:
                handles = ax.get_legend_handles_labels()
            if ri == 0:
                ax.set_title(CP.MODEL_LABEL[m], fontsize=S.FS_LABEL - 0.5)
            ax.set_ylabel(short.get(v, v) if mi == 0 else "", fontsize=S.FS_TICK - 0.5)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=S.FS_TICK - 1.5)
            if ri == nrow - 1:
                # short per-column label: the full "SD of the predicted class" repeated
                # three times with a multiplier appended overran the canvas
                ax.set_xlabel("SD" + sfx[m], fontsize=S.FS_LABEL - 0.5)
    if handles and handles[0]:
        fig.legend(*handles, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncols=2,
                   fontsize=S.FS_LEG, frameon=False)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"VIRUS_sdshift{'_pooled' if seed == 'all' else ''}.{ext}")
        fig.savefig(p, dpi=S.DPI)
        paths.append(p)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  VIRUS_sdshift: {w:.2f} x {h:.2f} in  ({nrow} genomes, seed {seed})")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sdshift")
    ap.add_argument("--aspect", type=float, default=0.60,
                    help="height/width. 0.60 -> 3.90 in, so two stack in 7.8 in")
    ap.add_argument("--only", nargs="*", help="stems to build, or 'virus'")
    ap.add_argument("--virus-seed", type=int, default=7)
    ap.add_argument("--balance-test", action="store_true",
                    help="subsample the perturbed test set to equal class counts "
                         "(RBP only; see _balance_rows for why)")
    # The seed is chosen per use case, not globally: it has to be one where the BNN
    # converged to an accuracy comparable to MCD and ENS, since a BNN that underfits
    # raises its uncertainty everywhere and would flatter the comparison. On RBP every
    # seed converges; on GEX seeds 7 and 5 do not (BNN 0.74 and 0.81 against 0.98).
    ap.add_argument("--seed", type=int, default=None,
                    help="override the seed for the stems being built")
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    S.apply_rc()
    print(f"writing to {out}")
    if args.only and "virus" in args.only:
        build_virus(out, seed=args.virus_seed)
        return 0
    for task, ref, alt, lab, stem, let_log2, let_raw in JOBS:
        if args.only and stem not in args.only:
            continue
        build(task, ref, alt, stem, out, True,  let_log2, args.aspect,
              seed=args.seed, balance_test=args.balance_test)
        build(task, ref, alt, stem, out, False, let_raw,  args.aspect,
              seed=args.seed, balance_test=args.balance_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
