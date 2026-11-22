"""Label-noise sweep on the simulated TF data: swap the labels of a FRACTION of the JUN and
of the MEF2A training and validation sequences, test labels clean.

Same construction as the 50% setting of build_perturbed_splits.py (train/valid sequences of
the clean split, test set of the clean split held out), for other fractions:

  <out>/tf/simulated_human/balanced/single_label/train_mislabeled15_test_clean/6class_split<seed>.pickle
  <out>/tf/simulated_human/balanced/single_label/train_mislabeled30_test_clean/6class_split<seed>.pickle

The swapped sequences are nested across fractions: one seeded permutation of each class, the
first round(frac * n) sequences are swapped, so the 15% set is a subset of the 30% set. (The
existing 50% setting was drawn with rng.choice and is not nested with these.)

    python3 scripts/splits/build_label_noise_sweep.py --datasets <root> --out <root>_v2 --fracs 0.15 0.30
    python3 scripts/splits/audit_splits.py --datasets <root>_v2        # 0 overlap expected
"""
import argparse, os, pickle
import numpy as np

SEEDS = [7, 5, 11, 6, 9]
BASE = "tf/simulated_human/balanced/single_label"
JUN, MEF2A = 2, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="root holding the clean TF splits (.../no_noise)")
    ap.add_argument("--out", required=True, help="root to write into (datasets_v2)")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.15, 0.30])
    a = ap.parse_args()
    for frac in a.fracs:
        tag = f"mislabeled{int(round(frac * 100))}"
        odir = os.path.join(a.out, BASE, f"train_{tag}_test_clean")
        os.makedirs(odir, exist_ok=True)
        print(f"== {tag}: swap {frac:.0%} of JUN<->MEF2A in train/valid -> {odir}")
        for s in SEEDS:
            clean = pickle.load(open(os.path.join(a.datasets, BASE, "no_noise", f"6class_split{s}.pickle"), "rb"))
            rng = np.random.default_rng(s)
            out = dict(x_test=np.asarray(clean["x_test"]), y_test=np.asarray(clean["y_test"]))
            flips = 0
            for part in ("train", "valid"):
                x = np.asarray(clean["x_" + part]); y = np.asarray(clean["y_" + part]).copy(); lab = y.argmax(1)
                for c, other in ((JUN, MEF2A), (MEF2A, JUN)):
                    idx = np.where(lab == c)[0]
                    perm = rng.permutation(idx)                      # one permutation per class and part
                    pick = perm[:int(round(frac * len(idx)))]
                    y[pick] = 0; y[pick, other] = 1; flips += len(pick)
                out["x_" + part], out["y_" + part] = x, y
            with open(os.path.join(odir, f"6class_split{s}.pickle"), "wb") as fh:
                pickle.dump(out, fh, protocol=4)
            n = len(out["x_train"]) + len(out["x_valid"])
            print(f"  seed {s}: {flips}/{n} train+valid labels swapped ({100 * flips / n:.1f}%); "
                  f"train {np.asarray(out['y_train']).sum(0).astype(int).tolist()}; test {len(out['x_test'])} clean")


if __name__ == "__main__":
    main()
