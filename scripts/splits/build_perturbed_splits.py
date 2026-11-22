"""Rebuild the three composite dataset settings so that the test set is disjoint from
train and valid (see AUDIT_data_splits_and_MCD.md, section 2).

The original construction took train/valid from one random split of a pool and the test
set from a second, independent split of the same pool, so most test samples were also in
the training set (TF mislabeled 85%, RBP imbalanced 85%, GEX imbalanced 64%). Here the test
set is held out FIRST and train/valid are drawn from what remains:

  tf/.../train_mislabeled_test_clean   test  = clean split's test (clean labels)
                                        train/valid = clean split's train/valid sequences with the
                                        labels of half of the JUN and half of the MEF2A sequences
                                        swapped (the rule of the original noisy dataset, DS10
                                        "noisy2"; new realization, seeded)
  rbp/.../train_imbalanced_test_balanced (single and multi label)
                                        test  = balanced split's test
                                        train/valid = imbalanced pool minus the test sequences,
                                        deduplicated, split 80/20 with the seed
  gex/.../train_imbalanced_test_balanced test = balanced split's test
                                        train/valid = imbalanced pool minus the test cells,
                                        split 75/25 with the seed

Everything else is left as is: the new tree <out> gets symlinks (or copies with --copy) to the
unchanged setting directories, so it can be used as path_datasets directly.

    python3 scripts/splits/build_perturbed_splits.py --datasets <root> --out <root>_v2
    python3 scripts/splits/audit_splits.py --datasets <root>_v2      # must report 0 overlap
"""
import argparse, glob, hashlib, os, pickle, shutil
import numpy as np
from sklearn.model_selection import train_test_split

SEEDS = [5, 11, 6, 7, 9]


def rh(r):
    a = np.ascontiguousarray(r)
    a = a.astype(np.int8) if (a.dtype.kind in "biu" or np.all(np.mod(a, 1) == 0)) else a.astype(np.float32)
    return hashlib.md5(a.tobytes()).hexdigest()


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def dump(d, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(d, fh, protocol=4)


def pool(d):
    x = np.concatenate([np.asarray(d["x_train"]), np.asarray(d["x_valid"]), np.asarray(d["x_test"])])
    y = np.concatenate([np.asarray(d["y_train"]), np.asarray(d["y_valid"]), np.asarray(d["y_test"])])
    return x, y


def dedupe(x, y):
    seen, keep = set(), []
    for i, r in enumerate(x):
        h = rh(r)
        if h not in seen:
            seen.add(h); keep.append(i)
    return x[keep], y[keep], len(x) - len(keep)


def rebuild_holdout(pool_x, pool_y, test_x, test_y, valid_frac, seed):
    test_h = set(rh(r) for r in test_x)
    mask = np.array([rh(r) not in test_h for r in pool_x])
    x, y, ndup = dedupe(pool_x[mask], pool_y[mask])
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=valid_frac, random_state=seed)
    return dict(x_train=x_tr, y_train=y_tr, x_valid=x_va, y_valid=y_va, x_test=test_x, y_test=test_y), int((~mask).sum()), ndup


def report(name, d, removed, ndup):
    f = lambda k: np.asarray(d[k]).sum(axis=0).astype(int).tolist()
    print(f"  {name}: removed {removed} test samples from the pool, {ndup} duplicates dropped; "
          f"train {len(d['x_train'])} {f('y_train')} valid {len(d['x_valid'])} {f('y_valid')} test {len(d['x_test'])} {f('y_test')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true", help="copy unchanged settings instead of symlinking")
    ap.add_argument("--no-link", action="store_true", help="write only the perturbed settings")
    a = ap.parse_args()
    R, O = a.datasets.rstrip("/"), a.out.rstrip("/")
    os.makedirs(O, exist_ok=True)

    # ---- TF: mislabeled train/valid on the clean split's sequences. The original noisy
    # dataset (DS10 "noisy2") swaps the labels of HALF of the JUN and MEF2A sequences; the
    # same rule is applied here to the clean split's train and valid parts, seeded, and the
    # test part keeps its clean labels.
    print("== TF train_mislabeled_test_clean (swap half of JUN<->MEF2A in train/valid)")
    base = "tf/simulated_human/balanced/single_label"
    JUN, MEF2A = 2, 3
    for s in SEEDS:
        clean = load(f"{R}/{base}/no_noise/6class_split{s}.pickle")
        rng = np.random.default_rng(s)
        out = dict(x_test=np.asarray(clean["x_test"]), y_test=np.asarray(clean["y_test"]))
        flips = 0
        for part in ("train", "valid"):
            x = np.asarray(clean["x_" + part]); y = np.asarray(clean["y_" + part]).copy(); lab = y.argmax(1)
            for c, other in ((JUN, MEF2A), (MEF2A, JUN)):
                idx = np.where(lab == c)[0]; pick = rng.choice(idx, len(idx) // 2, replace=False)
                y[pick] = 0; y[pick, other] = 1; flips += len(pick)
            out["x_" + part], out["y_" + part] = x, y
        dump(out, f"{O}/{base}/train_mislabeled_test_clean/6class_split{s}.pickle")
        n = len(out["x_train"]) + len(out["x_valid"])
        print(f"  seed {s}: {flips}/{n} train+valid labels swapped ({100*flips/n:.1f}%); test {len(out['x_test'])} clean")

    # ---- RBP: imbalanced pool minus balanced test
    for lab in ("single_label", "multi_label"):
        print(f"== RBP train_imbalanced_test_balanced / {lab}")
        for s in SEEDS:
            imb = load(f"{R}/rbp/real_human/imbalanced/{lab}/3class_split{s}.pickle")
            bal = load(f"{R}/rbp/real_human/balanced/{lab}/3class_split{s}.pickle")
            px, py = pool(imb)
            d, removed, ndup = rebuild_holdout(px, py, np.asarray(bal["x_test"]), np.asarray(bal["y_test"]), 0.20, s)
            dump(d, f"{O}/rbp/real_human/train_imbalanced_test_balanced/{lab}/3class_split{s}.pickle")
            report(f"seed {s}", d, removed, ndup)

    # ---- GEX
    print("== GEX train_imbalanced_test_balanced")
    for s in SEEDS:
        imb = load(f"{R}/gex/real_human/imbalanced/single_label/4class_split{s}.pickle")
        bal = load(f"{R}/gex/real_human/balanced/single_label/4class_split{s}.pickle")
        px, py = pool(imb)
        d, removed, ndup = rebuild_holdout(px, py, np.asarray(bal["x_test"]), np.asarray(bal["y_test"]), 0.25, s)
        dump(d, f"{O}/gex/real_human/train_imbalanced_test_balanced/single_label/4class_split{s}.pickle")
        report(f"seed {s}", d, removed, ndup)

    # ---- everything else: link or copy
    print("== linking unchanged settings" if not a.no_link else "== perturbed settings only (--no-link)")
    for p in ([] if a.no_link else sorted(glob.glob(f"{R}/**/*.pickle", recursive=True))):
        rel = os.path.relpath(p, R); dst = f"{O}/{rel}"
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if a.copy:
            shutil.copy2(p, dst)
        else:
            os.symlink(os.path.abspath(p), dst)
    print(f"done -> {O}\nnow run: python3 scripts/splits/audit_splits.py --datasets {O}")


if __name__ == "__main__":
    main()
