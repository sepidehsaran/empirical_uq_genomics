#!/usr/bin/env python3
"""
cross_seed_tables.py -- five re-analyses of the stored predictions that need no
retraining (cross-seed tables of the supplement). Everything below reads data already on disk:

    figure_data_bundle/     aggregated predictive means and SDs, plus labels
    cluster_preds/rbp_v19/  raw (n_iter, n_samples, n_classes) predictions,
                            3 methods x 5 seeds x 2 settings

Run:
    cd <repo root>   # with figure_data_bundle/ and cluster_preds/ here, or UQFIG_BUNDLE / UQFIG_PREDS set
    python3 scripts/sensitivity/cross_seed_tables.py all

or one at a time:
    nll        multi-label NLL recomputed as binary cross-entropy   (Fig 5h)
    seeds      cross-seed test of the SD-shift direction            (Figs 9, 10)
    mcdzero    MC-dropout zero-variance fraction per setting        (Figs 6, 7)
    converge   BNN results with and without non-converged runs      (Fig 5)
    ensemble   ensemble-size / number-of-draws sensitivity          (new suppl.)

Needs only numpy, scipy, pandas.
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repository root

BUNDLE = os.environ.get(
    "UQFIG_BUNDLE",
    os.path.join(ROOT, "figure_data_bundle"))
PREDS = os.environ.get(
    "UQFIG_PREDS",
    os.path.join(ROOT, "cluster_preds"))

MODELS = ["mcd", "bnn", "ens"]

# Mirrors figures/config_paths.py VERSIONS so the keys match the bundle exactly.
VERSIONS = {
    "rbp": dict(version=17, n_classes=3, seeds=[7, 5, 11, 6, 9],
                settings=["bal_single", "imbal_single", "bal_multi", "imbal_multi"],
                class_names=["MBNL1", "PUM2", "QKI"]),
    "tf":  dict(version=18, n_classes=6, seeds=[7, 5, 11, 6, 9],
                settings=["none", "mislabeled", "no_motif"],
                class_names=["CTCF", "HNF4A", "JUN", "MEF2A", "MYC", "TAL1"]),
    "gex": dict(version=19, n_classes=4, seeds=[7, 5, 11, 6, 9],
                settings=["bal_single", "imbal_single"],
                class_names=["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]),
}

# Which (reference, perturbed) pairs the sensitivity figures compare.
SHIFT_PAIRS = [
    ("tf",  "none",       "mislabeled",   "TF: clean -> mislabelled (Fig 9)"),
    ("tf",  "none",       "no_motif",     "TF: clean -> missing motif (suppl.)"),
    ("rbp", "bal_single", "imbal_single", "RBP: balanced -> imbalanced, single (Fig 10a)"),
    ("rbp", "bal_multi",  "imbal_multi",  "RBP: balanced -> imbalanced, multi"),
    ("gex", "bal_single", "imbal_single", "GEX: balanced -> imbalanced (Fig 10b)"),
]

MULTI_LABEL_SETTINGS = {"bal_multi", "imbal_multi"}


# --------------------------------------------------------------------------- loading
_CACHE = {}


def bundle_part(name):
    if name in _CACHE:
        return _CACHE[name]
    path = os.path.join(BUNDLE, name + ".pkl.gz")
    if not os.path.exists(path):
        _CACHE[name] = None
        return None
    with gzip.open(path, "rb") as fh:
        _CACHE[name] = pickle.load(fh)
    return _CACHE[name]


def dict_key(task, seed, setting, model):
    return f"v_{VERSIONS[task]['version']}_s_{seed}_{setting}_{model}"


def labels_for(task, setting, seed):
    """y_test for one (setting, seed). The bundle keys these as a tuple; be lenient."""
    lab = bundle_part(f"labels_{task}")
    if not lab:
        return None
    for k, v in lab.items():
        if isinstance(k, tuple) and len(k) == 2:
            s, sd = k
            if str(s) == str(setting) and int(sd) == int(seed):
                return np.asarray(v)
        elif str(k) == f"{setting}_{seed}":
            return np.asarray(v)
    return None


def comp_arrays(task, seed, setting, model):
    """(mean, sd) for one run, or (None, None)."""
    comp = bundle_part(f"comparison_{task}")
    if not comp:
        return None, None
    k = dict_key(task, seed, setting, model)
    m = comp.get("mean", {}).get(k)
    s = comp.get("sd", {}).get(k)
    return (None if m is None else np.asarray(m),
            None if s is None else np.asarray(s))


def _write(df, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    df.to_csv(p, index=False)
    print(f"    -> {p}")


# ------------------------------------------------------------------------ 1. NLL fix
def nll_positives_only(p, y):
    """What the paper computes today (figures/metrics.py:36). Correct for one-hot
    labels, wrong for multi-label: the (1-y)log(1-p) term is missing, so a model that
    predicts 1 everywhere scores near-perfectly."""
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return float(-np.sum(np.asarray(y) * np.log(p)) / y.shape[0])


def nll_binary_ce(p, y, per_class=False):
    """Correct NLL for independent-sigmoid multi-label output."""
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    return float(-ll.mean()) if per_class else float(-ll.sum(1).mean())


def brier_sum(p, y):
    """Sum over classes, mean over samples -- what the code does."""
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.mean(np.sum(d ** 2, axis=1)))


def brier_mean(p, y):
    """K^-1 * sum over classes -- what supplement eq. BS says. Differs by a factor K."""
    return brier_sum(p, y) / np.asarray(y).shape[1]


def hamming_accuracy(p, y, th=0.5):
    return float(np.mean(np.where(np.asarray(p) < th, 0.0, 1.0) == y))


def run_nll(out):
    print("[1/5] multi-label NLL: positives-only vs binary cross-entropy")
    rows = []
    for task in ("rbp",):                       # only RBP has multi-label settings
        spec = VERSIONS[task]
        for setting in spec["settings"]:
            if setting not in MULTI_LABEL_SETTINGS:
                continue
            for seed in spec["seeds"]:
                y = labels_for(task, setting, seed)
                if y is None:
                    continue
                for model in MODELS:
                    m, _ = comp_arrays(task, seed, setting, model)
                    if m is None or m.shape[0] != y.shape[0]:
                        continue
                    rows.append(dict(
                        task=task, setting=setting, model=model.upper(), seed=seed,
                        hamming_acc=hamming_accuracy(m, y),
                        all_zero_baseline=float(np.mean(y == 0)),
                        NLL_published=nll_positives_only(m, y),
                        NLL_bce_sum=nll_binary_ce(m, y),
                        NLL_bce_mean=nll_binary_ce(m, y, per_class=True),
                        Brier_published=brier_sum(m, y),
                        Brier_eq_in_text=brier_mean(m, y)))
    if not rows:
        print("    no multi-label runs found in the bundle -- check UQFIG_BUNDLE")
        return
    df = pd.DataFrame(rows)
    _write(df, out, "nll_multilabel_recomputed.csv")

    piv = (df.groupby(["setting", "model"])[["NLL_published", "NLL_bce_sum"]]
             .mean().round(4))
    print("\n    mean over 5 seeds:")
    print(piv.to_string())
    print("\n    Read the ordering, not the values. If BNN ranks best on NLL_published")
    print("    and worst on NLL_bce_sum, the published panel is showing the artifact.\n")


# ------------------------------------------------------- 2. cross-seed shift direction
def median_sd_per_class(sd, mean, class_i):
    """Median SD for class_i over the samples this model predicts as class_i.
    Selecting with each run's own predictions is what lets reference and perturbed
    runs have different test-set sizes (metrics.ks_sd_shift_unpaired)."""
    sel = np.asarray(mean).argmax(axis=-1) == class_i
    if sel.sum() < 3:
        return np.nan
    return float(np.median(np.asarray(sd)[sel][:, class_i]))


def run_seeds(out):
    print("[2/5] cross-seed test of the SD-shift direction")
    rows = []
    for task, ref, alt, label in SHIFT_PAIRS:
        spec = VERSIONS[task]
        for model in MODELS:
            for c in range(spec["n_classes"]):
                per_seed = []
                for seed in spec["seeds"]:
                    m_r, s_r = comp_arrays(task, seed, ref, model)
                    m_a, s_a = comp_arrays(task, seed, alt, model)
                    if s_r is None or s_a is None:
                        continue
                    a = median_sd_per_class(s_r, m_r, c)
                    b = median_sd_per_class(s_a, m_a, c)
                    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
                        continue
                    per_seed.append((seed, a, b, np.log2(b / a)))
                if len(per_seed) < 3:
                    continue
                lr = np.array([p[3] for p in per_seed])
                n_up = int((lr > 0).sum())
                # Sign test. With 5 seeds the smallest attainable two-sided p is
                # 0.0625, so this can show perfect consistency but never p < 0.05 on
                # its own -- report it alongside the effect size, not instead of it.
                sign_p = stats.binomtest(n_up, len(lr), 0.5).pvalue
                t, t_p = stats.ttest_1samp(lr, 0.0)
                try:
                    w, w_p = stats.wilcoxon(lr)
                except ValueError:
                    w, w_p = np.nan, np.nan
                rows.append(dict(
                    comparison=label, task=task, ref=ref, alt=alt,
                    model=model.upper(),
                    cls=spec["class_names"][c] if c < len(spec["class_names"]) else c,
                    n_seeds=len(lr),
                    mean_log2_ratio=float(lr.mean()),
                    sd_log2_ratio=float(lr.std(ddof=1)),
                    fold_change=float(2 ** lr.mean()),
                    n_seeds_increasing=n_up,
                    sign_test_p=float(sign_p),
                    t_stat=float(t), t_p=float(t_p),
                    wilcoxon_p=float(w_p) if np.isfinite(w_p) else np.nan,
                    per_seed_log2=";".join(f"{s}:{v:+.3f}" for (s, _, _, v) in per_seed)))
    if not rows:
        print("    no comparable pairs found -- check UQFIG_BUNDLE")
        return
    df = pd.DataFrame(rows)
    _write(df, out, "seed_consistency_sdshift.csv")

    consistent = df[(df.n_seeds_increasing == df.n_seeds) |
                    (df.n_seeds_increasing == 0)]
    print(f"\n    {len(consistent)} of {len(df)} (comparison, model, class) cells shift "
          f"in the same direction in every seed.")
    print("    Cells where the direction is NOT unanimous:")
    bad = df[~df.index.isin(consistent.index)]
    if len(bad):
        print(bad[["comparison", "model", "cls", "n_seeds_increasing",
                   "mean_log2_ratio"]].to_string(index=False))
    else:
        print("    (none)")
    print("\n    Report mean_log2_ratio +/- sd_log2_ratio and t_p in the text. Say "
          "explicitly\n    that the sign test floors at p = 0.0625 with 5 seeds.\n")


# ------------------------------------------------------------- 3. MC-dropout zero SDs
def run_mcdzero(out):
    print("[3/5] MC-dropout zero-variance fraction, and its effect on Kendall's tau")
    rows = []
    for task, spec in VERSIONS.items():
        for setting in spec["settings"]:
            for seed in spec["seeds"]:
                y = labels_for(task, setting, seed)
                if y is None:
                    continue
                lab = np.asarray(y).argmax(1)
                idx = np.arange(len(lab))
                sds = {}
                for model in MODELS:
                    m, s = comp_arrays(task, seed, setting, model)
                    if s is None or s.shape[0] != y.shape[0]:
                        continue
                    sds[model] = (m, s)
                    s_true = np.asarray(s)[idx, lab]
                    rows.append(dict(
                        task=task, setting=setting, seed=seed, model=model.upper(),
                        n=len(lab),
                        frac_sd_zero_true_class=float(np.mean(s_true == 0)),
                        frac_sd_zero_all_classes=float(np.mean((s == 0).all(1))),
                        median_sd_true_class=float(np.median(s_true))))
                # tau with and without the tied block, for every pair
                for a in range(len(MODELS)):
                    for b in range(a + 1, len(MODELS)):
                        ma, mb = MODELS[a], MODELS[b]
                        if ma not in sds or mb not in sds:
                            continue
                        s1 = sds[ma][1][idx, lab]
                        s2 = sds[mb][1][idx, lab]
                        tau_all, p_all = stats.kendalltau(s1, s2)
                        keep = (s1 != 0) & (s2 != 0)
                        if keep.sum() > 2:
                            tau_nz, p_nz = stats.kendalltau(s1[keep], s2[keep])
                        else:
                            tau_nz, p_nz = np.nan, np.nan
                        rows.append(dict(
                            task=task, setting=setting, seed=seed,
                            model=f"{ma.upper()}-{mb.upper()}", n=len(lab),
                            tau_all=float(tau_all), tau_p=float(p_all),
                            tau_nonzero_only=float(tau_nz) if np.isfinite(tau_nz) else np.nan,
                            n_nonzero=int(keep.sum())))
    if not rows:
        print("    nothing loaded -- check UQFIG_BUNDLE")
        return
    df = pd.DataFrame(rows)
    _write(df, out, "mcd_zero_variance_and_tau.csv")
    mcd = df[df.model == "MCD"].dropna(subset=["frac_sd_zero_true_class"])
    if len(mcd):
        print("\n    MCD zero-SD fraction (true class), mean over seeds:")
        print((mcd.groupby(["task", "setting"])["frac_sd_zero_true_class"]
                  .mean().round(3)).to_string())
    print("\n    Put frac_sd_zero next to every MCD tau in Figs 6-7, and state that")
    print("    scipy's tau-b tie correction means the tau is valid but computed over")
    print("    a shrinking effective sample.\n")


# ------------------------------------------------------------ 4. BNN convergence filter
def run_converge(out):
    print("[4/5] BNN runs below the trivial baseline")
    rows = []
    for task, spec in VERSIONS.items():
        for setting in spec["settings"]:
            multi = setting in MULTI_LABEL_SETTINGS
            for seed in spec["seeds"]:
                y = labels_for(task, setting, seed)
                if y is None:
                    continue
                # A priori, method-agnostic rule, fixed before looking at which runs
                # fail it: multi-label -> the all-zeros predictor's Hamming accuracy;
                # single-label -> the majority-class rate.
                base = (float(np.mean(y == 0)) if multi
                        else float(np.asarray(y).sum(0).max() / len(y)))
                for model in MODELS:
                    m, _ = comp_arrays(task, seed, setting, model)
                    if m is None or m.shape[0] != y.shape[0]:
                        continue
                    acc = (hamming_accuracy(m, y) if multi
                           else float(np.mean(np.asarray(m).argmax(1) ==
                                              np.asarray(y).argmax(1))))
                    rows.append(dict(
                        task=task, setting=setting, seed=seed, model=model.upper(),
                        accuracy=acc, trivial_baseline=base,
                        below_baseline=bool(acc < base)))
    if not rows:
        print("    nothing loaded -- check UQFIG_BUNDLE")
        return
    df = pd.DataFrame(rows)
    _write(df, out, "convergence_flags.csv")
    bad = df[df.below_baseline]
    print(f"\n    {len(bad)} of {len(df)} runs fall below their trivial baseline:")
    if len(bad):
        print(bad[["task", "setting", "model", "seed", "accuracy",
                   "trivial_baseline"]].to_string(index=False))
    print("\n    Fig 5 should be shown with these excluded, with the exclusion rule")
    print("    stated in the caption and the full version kept in the supplement.\n")


# ---------------------------------------------------- 5. ensemble size / draw count
def _softmax(a, axis=-1):
    z = a - a.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _load_raw(setting_dir, seed, model):
    d = os.path.join(PREDS, "rbp_v19", setting_dir, f"3class_split{seed}")
    p = os.path.join(d, f"iter_preds_{model}_v_19_s_{seed}_100.pickle")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        raw = np.asarray(pickle.load(fh))
    # BNN's last layer has activation=None, so its stored output is logits
    # (models.py:247). MCD and ENS already carry softmax.
    return _softmax(raw) if model == "bnn" else raw


def run_ensemble(out):
    print("[5/5] ensemble-size and draw-count sensitivity (RBP, raw predictions)")
    if not os.path.isdir(os.path.join(PREDS, "rbp_v19")):
        print(f"    {PREDS}/rbp_v19 not found -- skipping")
        return
    rng = np.random.default_rng(0)
    rows = []
    for setting_dir in ("balanced", "train_imbalanced_test_balanced"):
        for seed in VERSIONS["rbp"]["seeds"]:
            for model in MODELS:
                raw = _load_raw(setting_dir, seed, model)
                if raw is None:
                    continue
                n_iter = raw.shape[0]

                if model == "ens":
                    # 100 rows are 10 members x 10 identical repeats
                    # (eval_functions.py:499). Deduplicate to the 10 real members
                    # before subsampling, or the "sensitivity" is over copies.
                    per = n_iter // 10
                    members = raw[::per][:10]
                    uniq = len(np.unique(members.round(10).reshape(10, -1), axis=0))
                    pool, sizes, unit = members, [2, 3, 5, 10], "members"
                    rows.append(dict(setting=setting_dir, seed=seed, model="ENS",
                                     check="distinct_members_of_10", value=float(uniq)))
                    # confirm the replication claim: SD over 100 == SD over 10
                    sd100 = raw.std(0)
                    sd10 = members.std(0)
                    rows.append(dict(setting=setting_dir, seed=seed, model="ENS",
                                     check="max_abs_diff_sd100_vs_sd10",
                                     value=float(np.abs(sd100 - sd10).max())))
                else:
                    pool, sizes, unit = raw, [5, 10, 20, 50, 100], "draws"

                full_mean = pool.mean(0)
                full_sd = pool.std(0)
                pred = full_mean.argmax(1)
                ref = full_sd[np.arange(len(pred)), pred]

                for k in sizes:
                    if k > len(pool):
                        continue
                    taus, meds = [], []
                    for _ in range(20 if k < len(pool) else 1):
                        sel = (rng.choice(len(pool), k, replace=False)
                               if k < len(pool) else np.arange(len(pool)))
                        sub = pool[sel]
                        sd = sub.std(0)[np.arange(len(pred)), pred]
                        t, _p = stats.kendalltau(ref, sd)
                        taus.append(t)
                        meds.append(np.median(sd))
                    rows.append(dict(
                        setting=setting_dir, seed=seed, model=model.upper(),
                        check=f"tau_vs_full_at_{k}_{unit}",
                        value=float(np.mean(taus)), spread=float(np.std(taus)),
                        median_sd=float(np.mean(meds))))
    if not rows:
        print("    no raw prediction files matched -- check UQFIG_PREDS")
        return
    df = pd.DataFrame(rows)
    _write(df, out, "ensemble_size_sensitivity.csv")
    chk = df[df.check.str.startswith("max_abs_diff")]
    if len(chk):
        print(f"\n    max |SD(100 rows) - SD(10 members)| = {chk.value.max():.2e}")
        print("    If this is at float precision, the 10x repetition changed no ENS")
        print("    number and the correction is purely to the text.")
    print("\n    Plot tau_vs_full against size for the supplement: it answers "
          "'would\n    more members / draws have changed the ranking?'\n")


# ------------------------------------------------------------------------------- main
COMMANDS = {"nll": run_nll, "seeds": run_seeds, "mcdzero": run_mcdzero,
            "converge": run_converge, "ensemble": run_ensemble}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all",
                    choices=["all"] + list(COMMANDS))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "tables"))
    a = ap.parse_args()

    print(f"bundle : {BUNDLE}  {'OK' if os.path.isdir(BUNDLE) else 'MISSING'}")
    print(f"preds  : {PREDS}  {'OK' if os.path.isdir(PREDS) else 'MISSING'}")
    print()
    if not os.path.isdir(BUNDLE):
        print("Set UQFIG_BUNDLE to the figure_data_bundle directory.")
        return 1

    for name in (COMMANDS if a.which == "all" else [a.which]):
        try:
            COMMANDS[name](a.out)
        except Exception as exc:
            print(f"    !! {name} failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
