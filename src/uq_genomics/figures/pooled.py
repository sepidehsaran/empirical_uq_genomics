"""
pooled.py - the sensitivity figures of the paper, pooled over the five seeds.

Two score definitions are used, and the figure names say which:

* ``*_pooled``           SD of each class's predicted probability over *all* test
                         samples (TF label noise, Fig. 9; TF missing motif, supplement).
* ``*_predclass_pooled`` SD at the *predicted* class only: for class c, the samples the
                         model predicts as c (class imbalance, Fig. 10 and supplement).

In both cases the test sets of the five seeds are concatenated before the densities are
drawn. ``log2=True`` gives the log-scale figures of the main text, ``log2=False`` the
raw-scale versions of the supplement.

The single-cell and TF panels come from the released bundle. The RBP panel of Fig. 10
needs the raw per-pass predictions (``UQFIG_PREDS``), because both RBP settings are
evaluated on the same balanced test set and that evaluation is not part of the bundle.
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as CP            # noqa: E402
import data_access as DA             # noqa: E402
import plots_sensitivity as PS       # noqa: E402
import plots_sdshift as SDS     # noqa: E402


def softmax(a):
    z = a - a.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def predclass_only(sd, mean):
    """keep sd[:, c] only where argmax(mean) == c, NaN elsewhere (densities drop NaN)"""
    out = np.full_like(sd, np.nan, dtype=float)
    pred = np.asarray(mean).argmax(1)
    for c in range(sd.shape[1]):
        m = pred == c
        out[m, c] = sd[m, c]
    return out


def pooled_source(task, ref, alt):
    """all-sample SD, concatenated over seeds, from the bundle"""
    comp = DA.load_comparison(task)

    def src(model, _seed):
        r, a = [], []
        for s in CP.VERSIONS[task]["seeds"]:
            sd_r, _ = PS._sd_for(comp, task, s, ref, model)
            sd_a, _ = PS._sd_for(comp, task, s, alt, model)
            if sd_r is None or sd_a is None:
                continue
            r.append(np.asarray(sd_r))
            a.append(np.asarray(sd_a))
        if not r:
            raise FileNotFoundError(f"no {task} {ref}/{alt} predictions in the bundle")
        return np.concatenate(r), np.concatenate(a)
    return src


def predclass_source_bundle(task, ref, alt):
    """predicted-class SD, concatenated over seeds, from the bundle (single-cell)"""
    comp = DA.load_comparison(task)

    def src(model, _seed):
        parts_r, parts_a = [], []
        for s in CP.VERSIONS[task]["seeds"]:
            sd_r, mn_r = PS._sd_for(comp, task, s, ref, model)
            sd_a, mn_a = PS._sd_for(comp, task, s, alt, model)
            if sd_r is None or sd_a is None:
                continue
            parts_r.append(predclass_only(np.asarray(sd_r), mn_r))
            parts_a.append(predclass_only(np.asarray(sd_a), mn_a))
        if not parts_r:
            raise FileNotFoundError(f"no {task} {ref}/{alt} predictions in the bundle")
        return np.concatenate(parts_r), np.concatenate(parts_a)
    return src


def predclass_source_rbp(model, _seed, preds=None):
    """predicted-class SD for the RBP balanced / imbalanced pair, both evaluated on the
    shared balanced test set, from the raw per-pass predictions under UQFIG_PREDS"""
    preds = preds or CP.PREDS
    parts_r, parts_a = [], []
    for s in CP.VERSIONS["rbp"]["seeds"]:
        pr = glob.glob(os.path.join(preds, "rbp_v19", "balanced", f"3class_split{s}", f"iter_preds_{model}_*.pickle"))
        pa = glob.glob(os.path.join(preds, "rbp_v19", "train_imbalanced_test_balanced", f"3class_split{s}",
                                    f"iter_preds_{model}_*.pickle"))
        if not pr or not pa:
            continue
        with open(pr[0], "rb") as fh:
            r = np.asarray(pickle.load(fh))
        with open(pa[0], "rb") as fh:
            a = np.asarray(pickle.load(fh))
        if model == "bnn":                     # BNN passes are stored as logits
            r, a = softmax(r), softmax(a)
        parts_r.append(predclass_only(r.std(0), r.mean(0)))
        parts_a.append(predclass_only(a.std(0), a.mean(0)))
    if not parts_r:
        raise FileNotFoundError(f"no rbp_v19 per-pass predictions under {preds} (set UQFIG_PREDS)")
    return np.concatenate(parts_r), np.concatenate(parts_a)


# ----------------------------------------------------------------------------------
# builders: one per paper figure (or figure pair)
# ----------------------------------------------------------------------------------
def build_tf(out, log2=True, aspect=0.95, letters=(None, None)):
    """TF label noise (Fig. 9) and TF missing motif (supplement), all-sample score.
    Writes SDSHIFT_tf_mislabeled_pooled_{log2|raw}.pdf and SDSHIFT_tf_no_motif_pooled_*."""
    os.makedirs(out, exist_ok=True)
    SDS.build("tf", "none", "mislabeled", "tf_mislabeled_pooled", out, log2, letters[0], aspect,
              sd_source=pooled_source("tf", "none", "mislabeled"))
    SDS.build("tf", "none", "no_motif", "tf_no_motif_pooled", out, log2, letters[1], aspect,
              sd_source=pooled_source("tf", "none", "no_motif"))


def build_imbalance(out, log2=True, aspect=0.56, preds=None, skip_missing=True):
    """Class imbalance, RBP (a) and single-cell (b), predicted-class score (Fig. 10).
    Writes SDSHIFT_rbp_imbalance_predclass_pooled_* and SDSHIFT_gex_imbalance_predclass_pooled_*.
    The RBP panel needs the raw per-pass predictions; with skip_missing it is skipped
    (with a message) when they are absent, so the single-cell panel still builds."""
    os.makedirs(out, exist_ok=True)
    kw = dict(aspect=aspect) if aspect else {}
    try:
        SDS.build("rbp", "bal_single", "imbal_single", "rbp_imbalance_predclass_pooled", out, log2, "a",
                  sd_source=lambda m, s: predclass_source_rbp(m, s, preds), suffix="", **kw)
    except FileNotFoundError as exc:
        if not skip_missing:
            raise
        print(f"  RBP panel skipped: {exc}")
    SDS.build("gex", "bal_single", "imbal_single", "gex_imbalance_predclass_pooled", out, log2, "b",
              sd_source=predclass_source_bundle("gex", "bal_single", "imbal_single"), suffix="", **kw)


def build_all(out, preds=None):
    """every pooled sensitivity figure of the paper, log and raw scale"""
    for log2 in (True, False):
        build_tf(out, log2=log2)
        build_imbalance(out, log2=log2, preds=preds)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=CP.OUT_DIR)
    ap.add_argument("--preds", default=None, help="raw per-pass predictions (default: UQFIG_PREDS)")
    a = ap.parse_args()
    build_all(a.out, preds=a.preds)
