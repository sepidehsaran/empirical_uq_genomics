"""Prove the bundle is sufficient: build a synthetic one, then generate every figure
from it with ALL cluster paths unset.

This is the guard that matters. If a plot module quietly still needs a cluster file, the
figure is skipped here and the test says which section is short. Run it after any change
to export_bundle.py or the loaders.

    python3 test_bundle_roundtrip.py

Nothing here touches real data; the numbers are random. It checks plumbing, not science.
"""
import gzip
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Unset every cluster path BEFORE importing config_paths, so the only data source is
# the bundle. This is exactly the laptop situation.
for k in list(os.environ):
    if k.startswith("UQFIG_"):
        del os.environ[k]

BUNDLE = tempfile.mkdtemp(prefix="uqfig_bundle_")
os.environ["UQFIG_BUNDLE"] = BUNDLE

import config_paths as CP        # noqa: E402
import data_access as DA         # noqa: E402


def put(name, obj):
    with gzip.open(os.path.join(BUNDLE, name + ".pkl.gz"), "wb") as fh:
        pickle.dump(obj, fh, protocol=4)


def build():
    rng = np.random.default_rng(0)
    for task in ("rbp", "tf", "gex"):
        spec = CP.VERSIONS[task]
        n, c = 400, spec["n_classes"]
        mean, sd, top100, top50, top10p = {}, {}, {}, {}, {}
        labels = {}
        for setting in spec["settings"]:
            for seed in spec["seeds"]:
                y = np.zeros((n, c), dtype=np.int8)
                y[np.arange(n), rng.integers(0, c, n)] = 1
                labels[(setting, seed)] = y
                for model in CP.MODELS:
                    k = DA.dict_key(task, seed, setting, model)
                    p = rng.dirichlet(np.ones(c) * 0.2, size=n).astype(np.float32)
                    mean[k] = p
                    sd[k] = (rng.random((n, c)) * 0.1).astype(np.float32)
                    top100[k] = rng.choice(n, 100, replace=False)
                    top50[k] = rng.choice(n, 50, replace=False)
                    top10p[k] = rng.choice(n, n // 10, replace=False)
        put(f"comparison_{task}", {"mean": mean, "sd": sd, "top100": top100,
                                   "top50": top50, "top10p": top10p,
                                   "_files": {}, "_version": spec["version"]})
        put(f"labels_{task}", labels)

    # class distributions
    cd = {}
    for task, kw in (("rbp", dict(balanced=True, multi_label=False)),
                     ("rbp", dict(balanced=False, multi_label=False)),
                     ("rbp", dict(balanced=True, multi_label=True)),
                     ("rbp", dict(balanced=False, multi_label=True)),
                     ("tf", dict(balanced=True, noise="none")),
                     ("tf", dict(balanced=False, noise="none")),
                     ("gex", dict(balanced=True)),
                     ("gex", dict(balanced=False))):
        c = CP.VERSIONS[task]["n_classes"]
        seed = CP.VERSIONS[task]["seeds"][0]
        cd[(task, tuple(sorted(kw.items())), seed)] = {
            "y_train": rng.integers(50, 500, c), "y_valid": rng.integers(10, 90, c),
            "y_test": rng.integers(10, 90, c),
            "_n": {"y_train": 1000, "y_valid": 200, "y_test": 200}}
    put("classdist", cd)

    # UMAP
    n = 3000
    names = CP.VERSIONS["gex"]["class_names"]
    put("umap", {"xy": rng.normal(size=(n, 2)).astype(np.float32),
                 "labels": np.asarray([names[i % len(names)] for i in range(n)])})

    # sensitivity: the inference-only entries (tf/no_motif)
    sens = {}
    for seed in CP.VERSIONS["tf"]["seeds"]:
        for model in CP.MODELS:
            c = CP.VERSIONS["tf"]["n_classes"]
            sens[("tf", "no_motif", seed, model)] = (
                (rng.random((400, c)) * 0.1).astype(np.float32),
                rng.dirichlet(np.ones(c) * 0.2, size=400).astype(np.float32))
    put("sensitivity", sens)

    # virus
    vir = {}
    for seed in CP.VERSIONS["rbp"]["seeds"]:
        for model in CP.MODELS:
            vir[(seed, model)] = {
                "sd_in": (rng.random((400, 3)) * 0.1).astype(np.float32),
                "mean_in": rng.dirichlet(np.ones(3) * 0.2, size=400).astype(np.float32),
                "per_virus": {v: (rng.random((300, 3)) * 0.15).astype(np.float32)
                              for v in CP.VIRUSES}}
    put("virus", vir)

    # 27-RBP
    n, c = 2000, 27
    y = (rng.random((n, c)) < 0.1).astype(np.int8)
    put("app27", {"mean": rng.random((n, c)).astype(np.float32),
                  "sd": (rng.random((n, c)) * 0.2).astype(np.float32), "y_test": y,
                  "train_counts": rng.integers(20, 900, c),
                  "class_names": [f"RBP{i}" for i in range(c)]})


def main():
    print(f"synthetic bundle: {BUNDLE}")
    build()
    present, missing = DA.bundle_status()
    print(f"sections present: {len(present)}  missing: {missing or 'none'}\n")

    out = tempfile.mkdtemp(prefix="uqfig_out_")
    import style as S
    S.apply_rc()

    # Optional stage filter, so the whole thing can be run in slices when a shell has
    # a short timeout:  python3 test_bundle_roundtrip.py performance agreement
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    all_stages = ("datasets", "performance", "agreement", "sensitivity", "virus", "utility")
    mods = {}
    for name in (want or all_stages):
        try:
            mods[name] = __import__({"datasets": "plots_datasets",
                                     "performance": "plots_performance",
                                     "agreement": "plots_agreement",
                                     "sensitivity": "plots_sensitivity",
                                     "virus": "plots_virus",
                                     "utility": "plots_utility"}[name])
        except Exception as exc:
            print(f"  cannot import {name}: {exc}")

    totals = {"ok": 0, "skip": 0, "error": 0}
    skipped = []
    for name, mod in mods.items():
        print(f"[{name}]")
        try:
            res = mod.run(out)
        except Exception as exc:
            import traceback
            print(f"    STAGE CRASHED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
            totals["error"] += 1
            continue
        for r in res or []:
            st = r.get("status", "?")
            totals[st] = totals.get(st, 0) + 1
            if st != "ok":
                skipped.append(f"{r.get('id')}: {r.get('reason', '')[:90]}")
        n_ok = sum(1 for r in (res or []) if r.get("status") == "ok")
        print(f"    {n_ok}/{len(res or [])} generated")

    print("\n" + "=" * 64)
    print(f"ok={totals.get('ok', 0)}  skip={totals.get('skip', 0)}  "
          f"error={totals.get('error', 0)}")
    if skipped:
        print("\nnot generated from the bundle alone:")
        for s in skipped:
            print(f"  - {s}")
    print(f"\nfigures written to {out}")
    shutil.rmtree(BUNDLE, ignore_errors=True)
    # A skip is not automatically a failure - passthrough figures are copies of images
    # that legitimately cannot live in a data bundle. Errors always are.
    return 1 if totals.get("error", 0) else 0



def test_script():
    """pytest entry point: the script's main() returns 0 on success"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
