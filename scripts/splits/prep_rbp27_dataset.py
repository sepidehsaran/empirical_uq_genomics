"""Place the 27-class RBP application dataset where the training driver expects it, so MCD and
ENS can be trained on it with scripts/training/train.py --setting rbp_app27 (the paper's BNN was
trained on this file directly).

    python3 scripts/splits/prep_rbp27_dataset.py \
        --src /path/to/data/app27/uncertainty_datasets/real_imbalanced_singlelabel_27classlow_nonoise.pickle \
        --datasets /path/to/data/datasets_v2 --seed 9

Writes <datasets>/rbp/real_human/train_imbalanced_test_balanced/single_label/27class_split<seed>.pickle
with lower-case keys (x_train, y_train, x_valid, y_valid, x_test, y_test) and prints the shapes.
"""
import argparse, os, pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--datasets", required=True)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    d = pickle.load(open(a.src, "rb"))
    d = {k.lower(): v for k, v in d.items()}
    need = ["x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test"]
    missing = [k for k in need if k not in d]
    if missing:
        raise SystemExit(f"keys missing: {missing}; present: {sorted(d)}")
    out = {k: np.asarray(d[k]) for k in need}
    for k in need:
        print(f"  {k}: {out[k].shape}")
    print("  train per class:", out["y_train"].sum(0).astype(int).tolist())
    odir = os.path.join(a.datasets, "rbp/real_human/train_imbalanced_test_balanced/single_label")
    os.makedirs(odir, exist_ok=True)
    p = os.path.join(odir, f"27class_split{a.seed}.pickle")
    with open(p, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print("wrote", p)


if __name__ == "__main__":
    main()
