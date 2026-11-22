"""Audit every dataset split pickle: shapes, label format, class counts, duplicates, and
leakage between train/valid/test -- within each setting and ACROSS the composite settings
that pair a train set from one split with a test set from another.

    python3 scripts/splits/audit_splits.py --datasets /path/to/data/datasets \
        [--out out/tables/split_audit.csv]

Composite settings checked against their sources:
    tf/.../train_mislabeled_test_clean   vs  tf/.../no_noise           (same sequences, labels flipped)
    rbp/.../train_imbalanced_test_balanced vs rbp/.../balanced         (single and multi label)
    gex/.../train_imbalanced_test_balanced vs gex/.../balanced
Runs on CPU in a minute or two; no TensorFlow needed. Fine inside an srun allocation.
"""
import argparse, glob, hashlib, os, pickle, re
import numpy as np
import pandas as pd


def rh(r):
    a = np.ascontiguousarray(r)
    a = a.astype(np.int8) if a.dtype.kind in "biu" or (a.dtype.kind == "f" and a.max() <= 1 and np.all(np.mod(a, 1) == 0)) else a.astype(np.float32)
    return hashlib.md5(a.tobytes()).hexdigest()


def hashes(x):
    return [rh(r) for r in np.asarray(x)]


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def describe(d):
    out = {}
    for k in ("x_train", "x_valid", "x_test"):
        out[k] = None if d.get(k) is None else tuple(np.asarray(d[k]).shape)
    for k in ("y_train", "y_valid", "y_test"):
        y = d.get(k)
        if y is None:
            out[k] = None; continue
        y = np.asarray(y)
        s = y.sum(axis=1) if y.ndim == 2 else None
        out[k] = dict(shape=tuple(y.shape), counts=y.sum(axis=0).astype(int).tolist() if y.ndim == 2 else None,
                      single=int((s == 1).sum()) if s is not None else None, multi=int((s > 1).sum()) if s is not None else None,
                      none=int((s == 0).sum()) if s is not None else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sources", default=None, help="root holding the source (clean/balanced) splits if not under --datasets")
    a = ap.parse_args()
    root = a.datasets.rstrip("/")
    files = sorted(glob.glob(os.path.join(root, "**", "*class_split*.pickle"), recursive=True))
    print(f"{len(files)} split files under {root}\n")
    rows = []
    cache = {}
    for p in files:
        rel = os.path.relpath(p, root)
        d = load(p); cache[rel] = d
        desc = describe(d)
        H = {k: hashes(d[k]) for k in ("x_train", "x_valid", "x_test") if d.get(k) is not None}
        S = {k: set(v) for k, v in H.items()}
        row = dict(setting=os.path.dirname(rel), seed=int(re.search(r"split(\d+)", rel).group(1)),
                   n_train=desc["x_train"][0] if desc["x_train"] else None,
                   n_valid=desc["x_valid"][0] if desc["x_valid"] else None,
                   n_test=desc["x_test"][0] if desc["x_test"] else None,
                   dup_train=len(H.get("x_train", [])) - len(S.get("x_train", [])),
                   dup_test=len(H.get("x_test", [])) - len(S.get("x_test", [])),
                   train_test=len(S.get("x_train", set()) & S.get("x_test", set())),
                   valid_test=len(S.get("x_valid", set()) & S.get("x_test", set())),
                   train_valid=len(S.get("x_train", set()) & S.get("x_valid", set())),
                   y_train_counts=desc["y_train"]["counts"] if desc["y_train"] else None,
                   y_test_counts=desc["y_test"]["counts"] if desc["y_test"] else None,
                   y_test_multi=desc["y_test"]["multi"] if desc["y_test"] else None,
                   y_test_nolabel=desc["y_test"]["none"] if desc["y_test"] else None)
        rows.append(row)
        flag = "  <-- LEAK" if row["train_test"] or row["valid_test"] else ""
        print(f"{rel:85s} train/valid/test {row['n_train']}/{row['n_valid']}/{row['n_test']}  "
              f"train∩test {row['train_test']}  valid∩test {row['valid_test']}  dup(train,test) {row['dup_train']},{row['dup_test']}{flag}")
    df = pd.DataFrame(rows)

    print("\n=== composite settings against their sources")
    pairs = [("tf/simulated_human/balanced/single_label/train_mislabeled_test_clean", "tf/simulated_human/balanced/single_label/no_noise"),
             ("rbp/real_human/train_imbalanced_test_balanced/single_label", "rbp/real_human/balanced/single_label"),
             ("rbp/real_human/train_imbalanced_test_balanced/multi_label", "rbp/real_human/balanced/multi_label"),
             ("gex/real_human/train_imbalanced_test_balanced/single_label", "gex/real_human/balanced/single_label")]
    comp_rows = []
    for comp, src in pairs:
        for rel, d in sorted(cache.items()):
            if not rel.startswith(comp + "/"):
                continue
            seed = re.search(r"split(\d+)", rel).group(1)
            s_rel = f"{src}/{os.path.basename(rel)}"
            if s_rel not in cache:
                alt = os.path.join(a.sources or root, s_rel)
                if not os.path.exists(alt):
                    print(f"  {rel}: source {s_rel} not found"); continue
                cache[s_rel] = load(alt)
            s = cache[s_rel]
            ct, ctr, cv = set(hashes(s["x_test"])), set(hashes(s["x_train"])), set(hashes(s["x_valid"]))
            mt = hashes(d["x_test"]); mtr, mv = set(hashes(d["x_train"])), set(hashes(d["x_valid"]))
            leak_idx = [i for i, h in enumerate(mt) if h in mtr or h in mv]
            y = np.asarray(d["y_test"]) if d.get("y_test") is not None else None
            per_class = (np.bincount(y[leak_idx].argmax(1), minlength=y.shape[1]).tolist() if y is not None and y.ndim == 2 and len(leak_idx) else None)
            tot = (y.sum(axis=0).astype(int).tolist() if y is not None and y.ndim == 2 else None)
            same_test = set(mt) == ct
            # label flips for shared training sequences (TF mislabeled)
            flips = None
            if comp.startswith("tf") and mtr == ctr:
                yc = dict(zip(hashes(s["x_train"]), np.asarray(s["y_train"]).argmax(1)))
                ym = dict(zip(hashes(d["x_train"]), np.asarray(d["y_train"]).argmax(1)))
                flips = sum(int(yc[k] != ym[k]) for k in yc)
            r = dict(setting=comp, seed=int(seed), n_test=len(mt), test_equals_source_test=same_test,
                     test_in_train_or_valid=len(leak_idx), pct=round(100 * len(leak_idx) / len(mt), 1),
                     leaked_per_class=per_class, test_per_class=tot, train_label_flips=flips)
            comp_rows.append(r)
            print(f"  {comp} seed {seed}: test={len(mt)} (same as source test: {same_test})  in train+valid: {len(leak_idx)} ({r['pct']}%)"
                  f"  per class {per_class} of {tot}" + (f"  label flips in train: {flips}" if flips is not None else "")
                  + ("   <-- LEAK" if leak_idx else ""))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)
        pd.DataFrame(comp_rows).to_csv(a.out.replace(".csv", "_composite.csv"), index=False)
        print(f"\nwritten: {a.out} and {a.out.replace('.csv', '_composite.csv')}")


if __name__ == "__main__":
    main()
