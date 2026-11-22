"""
metrics.py - the metric functions, copied from the notebooks/src that produced the
published figures so numbers stay identical.

Provenance:
  accuracy_single_label / accuracy_multi_label -> src/eval_functions.py
  calculate_nll_one_hot / calculate_brier_score -> 2024_figures_section_3_3.ipynb cells 71/70
  kendall pair logic -> src/eval_functions.py get_kendall_stats()
  top-N uncertain per class -> src/eval_functions.py find_uncertain_samples_per_class()
  KS on SD distributions -> 2024_figures_section_3_3.ipynb cell 51
"""
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- predictive performance
def accuracy_single_label(preds, labels):
    return float(np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)))


def accuracy_multi_label(preds, labels, apply_sigmoid=False, th=0.5):
    """Element-wise (Hamming) accuracy — byte-for-byte the definition in
    src/eval_functions.py:24. Every cell of the label matrix counts equally.

    This deliberately does NOT use exact-match (subset) accuracy,
    mean(all(th_preds == labels, axis=1)). Subset accuracy is stricter and is the more
    common textbook choice, but the project's own definition is element-wise and all
    published numbers rest on it, so the pipeline follows src/ and does not silently
    substitute a different metric.
    """
    arr = 1.0 / (1.0 + np.exp(-np.asarray(preds))) if apply_sigmoid else np.asarray(preds)
    th_preds = np.where(arr < th, 0.0, 1.0)
    return float(np.mean(th_preds == labels))


def calculate_nll_one_hot(one_hot_pred, one_hot_labels):
    p = np.clip(np.asarray(one_hot_pred, dtype=float), 1e-15, 1 - 1e-15)
    nll = -np.sum(np.asarray(one_hot_labels) * np.log(p))
    return float(nll / one_hot_labels.shape[0])


def calculate_nll_multilabel(pred, labels):
    """Negative log-likelihood for independent-sigmoid (multi-label) output: mean over
    samples of the summed binary cross-entropy over classes. calculate_nll_one_hot()
    above is only correct for one-hot labels; applied to multi-label targets it drops the
    (1-y)log(1-p) term and rewards a model that predicts 1 everywhere."""
    p = np.clip(np.asarray(pred, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(labels, dtype=float)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    return float(-ll.sum(axis=1).mean())


def calculate_nll(pred, labels, multi_label=False):
    return (calculate_nll_multilabel(pred, labels) if multi_label
            else calculate_nll_one_hot(pred, labels))


def calculate_brier_score(one_hot_pred, one_hot_labels):
    d = np.asarray(one_hot_pred, dtype=float) - np.asarray(one_hot_labels, dtype=float)
    return float(np.mean(np.sum(d ** 2, axis=1)))


# ---------------------------------------------------------------------------- agreement
class ShapeMismatch(ValueError):
    """raised when arrays that must describe the same test set do not"""


def check_same_n(name, y_test, *arrays):
    """
    Every array must have the same number of rows as y_test. Real aggregated dicts do
    contain entries computed on different test sets (different seed/split written under
    the same setting), so this is checked rather than assumed.
    """
    n = np.asarray(y_test).shape[0]
    shapes = [np.asarray(a).shape for a in arrays]
    bad = [s for s in shapes if s[0] != n]
    if bad:
        raise ShapeMismatch(
            f"{name}: y_test has {n} rows but arrays have "
            f"{', '.join(str(s[0]) for s in shapes)} - these were computed on "
            f"different test sets")
    return True


def kendall_pair(y_test, mean1, sd1, mean2, sd2):
    """
    Kendall's tau between the SDs of two UQ methods, taken at the TRUE class of each
    sample; reported for all test samples and for the subset both models classify
    correctly. Mirrors get_kendall_stats().
    """
    check_same_n("kendall_pair", y_test, mean1, sd1, mean2, sd2)
    lab = np.asarray(y_test).argmax(axis=1)
    idx = np.arange(len(lab))
    s1 = np.asarray(sd1)[idx, lab]
    s2 = np.asarray(sd2)[idx, lab]
    ok = (np.asarray(mean1).argmax(axis=1) == lab) & (np.asarray(mean2).argmax(axis=1) == lab)
    tau_all, p_all = stats.kendalltau(s1, s2)
    if ok.sum() > 2:
        tau_ok, p_ok = stats.kendalltau(s1[ok], s2[ok])
    else:
        tau_ok, p_ok = np.nan, np.nan
    return dict(tau_all=tau_all, p_all=p_all, tau_correct=tau_ok, p_correct=p_ok,
                n_correct=int(ok.sum()), n_total=int(len(lab)))


def topn_uncertain_per_class(y_test, sd, mean, class_i, topn):
    """
    Indices (into the full test set) of the topn most uncertain samples of class_i,
    ranked by the SD of the PREDICTED class. Equivalent to
    find_uncertain_samples_per_class() but vectorised and taking mean/sd arrays
    instead of the full iteration stack.
    """
    y_test = np.asarray(y_test); sd = np.asarray(sd); mean = np.asarray(mean)
    cls_idx = np.where(y_test.argmax(axis=1) == class_i)[0]
    if len(cls_idx) == 0:
        return np.array([], dtype=int)
    pred_cls = mean[cls_idx].argmax(axis=1)
    scores = sd[cls_idx, pred_cls]
    order = (-scores).argsort()[:topn]
    return cls_idx[order]


def overlap_counts(idx_a, idx_b):
    return int(len(np.intersect1d(np.asarray(idx_a), np.asarray(idx_b))))


def ks_sd_shift(sd_ref, sd_alt, mean_ref, class_i):
    """
    Two-sided exact KS test between the reference and perturbed SD distributions for
    class_i, restricted to samples the reference model predicts as class_i (cell 51).
    """
    sel = np.asarray(mean_ref).argmax(axis=-1) == class_i
    a = np.asarray(sd_ref)[sel][:, class_i]
    b = np.asarray(sd_alt)[sel][:, class_i]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    st, p = stats.ks_2samp(a, b, method="exact", alternative="two-sided")
    return float(st), float(p)


def ks_sd_shift_unpaired(sd_ref, mean_ref, sd_alt, mean_alt, class_i, method="auto"):
    """KS between two SD distributions that need NOT have the same number of samples.

    ks_sd_shift() above selects samples with the REFERENCE model's predictions and
    applies that one mask to both arrays, so it silently requires the two runs to share a
    test set. That is true for TF (1800 vs 1800) and GEX (3600 vs 3600) but not for RBP,
    where balanced has 574 test samples and imbalanced 2411 - so the RBP KS figure was
    skipped entirely with "no comparable SD pairs".

    The restriction is not real. The paper's own methods section says the KS test
    "compares the underlying distributions of two independent samples that can be of
    different sizes". Here each sample is selected with its OWN model's predictions,
    which is the correct way to ask "is the SD distribution for class i shifted", and
    the two samples are then compared as independent samples.

    method: "exact" is what the manuscript specifies, but scipy's exact computation is
    O(n*m) and becomes very slow past a few thousand points. "auto" uses exact for small
    samples and the asymptotic approximation for large ones; at n>1000 the two agree to
    far more precision than the figure resolves.
    """
    sd_ref, mean_ref = np.asarray(sd_ref), np.asarray(mean_ref)
    sd_alt, mean_alt = np.asarray(sd_alt), np.asarray(mean_alt)
    a = sd_ref[mean_ref.argmax(axis=-1) == class_i][:, class_i]
    b = sd_alt[mean_alt.argmax(axis=-1) == class_i][:, class_i]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, len(a), len(b)
    st, p = stats.ks_2samp(a, b, method=method, alternative="two-sided")
    return float(st), float(p), len(a), len(b)


def pearson_spearman(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return dict(pearson=float(pr), pearson_p=float(pp),
                spearman=float(sr), spearman_p=float(sp))


# ------------------------------------------------------------------------------ utility
def average_precision_per_class(y_true, scores):
    from sklearn.metrics import average_precision_score
    out = []
    for i in range(y_true.shape[1]):
        try:
            out.append(float(average_precision_score(y_true[:, i], scores[:, i])))
        except Exception:
            out.append(np.nan)
    return np.array(out)
