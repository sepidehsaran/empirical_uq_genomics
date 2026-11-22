"""Export everything the figures need into one portable bundle.

Run once where the experiment trees are (datasets, per-run results, trained models; see
the UQFIG_* variables in docs/REPRODUCING.md). Then every figure regenerates anywhere in
seconds, because the bundle contains the numbers rather than the models and genomes.

    # where the experiment trees are
    export UQFIG_RESULTS_ROOT_RBP=... UQFIG_RESULTS_ROOT_TF=... UQFIG_RESULTS_ROOT_GEX=...
    export UQFIG_PATH_DATASETS_SEQ=... UQFIG_PATH_DATASETS_GEX=...
    python -m uq_genomics.figures.export_bundle --out figure_data_bundle

    # anywhere else
    export UQFIG_BUNDLE=/path/to/figure_data_bundle
    ./scripts/make_all_figures.sh

What goes in, and why it is small:

  labels + predictions   The aggregated dicts are per-sample predictive means and SDs,
                         a few thousand rows by a handful of classes. Tens of MB, not GB.
  inference outputs      Sensitivity, virus and TF-missing-motif normally re-run the
                         models on a GPU. Their *outputs* are the same shape as the
                         predictions, so exporting them makes those stages local too.
  class counts           Class-distribution figures need column sums, not the sequences.
  UMAP coordinates       Two float columns per cell, not the 6 GB .h5ad.

What is deliberately NOT included: model weights, x_train/x_test one-hot sequences, the
encoded virus genomes, adata.h5ad. Nothing downstream of this bundle needs them.

Re-running is cheap and idempotent; use --only to refresh one section.
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as CP          # noqa: E402
import data_access as DA           # noqa: E402

SECTIONS = ["comparison", "labels", "classdist", "umap", "sensitivity", "virus", "app27"]


# --------------------------------------------------------------------------- helpers
def _f32(a):
    """float64 -> float32 halves the bundle; predictions carry ~7 digits at most."""
    a = np.asarray(a)
    return a.astype(np.float32) if a.dtype == np.float64 else a


def _shrink(obj):
    """recursively downcast arrays inside the nested dicts"""
    if isinstance(obj, dict):
        return {k: _shrink(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (int, float, np.ndarray)):
        try:
            return _f32(np.asarray(obj))
        except Exception:
            return obj
    if isinstance(obj, np.ndarray):
        return _f32(obj)
    return obj


def write(bundle, name, obj):
    path = os.path.join(bundle, name + ".pkl.gz")
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        pickle.dump(obj, fh, protocol=4)
    os.replace(tmp, path)
    mb = os.path.getsize(path) / 1e6
    print(f"    wrote {os.path.basename(path):42s} {mb:8.1f} MB")
    return mb


def read(bundle, name):
    path = os.path.join(bundle, name + ".pkl.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- sections
def export_comparison(bundle):
    """aggregated predictive means / SDs / top-N index dicts, per use case"""
    total = 0.0
    for task in ("rbp", "tf", "gex"):
        comp = DA.load_comparison(task)
        if not comp:
            print(f"    {task}: no comparison dir - SKIPPED")
            continue
        keep = {k: _shrink(v) for k, v in comp.items() if k != "_files"}
        keep["_files"] = comp.get("_files", {})
        keep["_version"] = CP.VERSIONS[task]["version"]
        n = sum(len(v) for k, v in keep.items() if isinstance(v, dict) and not k.startswith("_"))
        print(f"    {task}: {n} entries across {len(keep)-2} dicts")
        total += write(bundle, f"comparison_{task}", keep)
    return total


def export_labels(bundle):
    """y_test for every (task, setting, seed) the figures ask for"""
    total = 0.0
    for task in ("rbp", "tf", "gex"):
        spec = CP.VERSIONS[task]
        comp = DA.load_comparison(task)
        dmean = comp.get("mean", {}) if comp else {}
        out = {}
        for setting in spec["settings"]:
            kw = DA.setting_to_split_args(task, setting)
            for seed in spec["seeds"]:
                # match the row count of the aggregated entry, which is what
                # disambiguates RBP imbalanced (2411 vs 574)
                expect = None
                for model in CP.MODELS:
                    key = DA.dict_key(task, seed, setting, model)
                    if key in dmean:
                        expect = np.asarray(dmean[key]).shape[0]
                        break
                y = DA.load_y_test(task, spec["species"], spec["n_classes"], seed,
                                   expect_n=expect, **kw)
                if y is not None:
                    out[(setting, seed)] = np.asarray(y).astype(np.int8)
        print(f"    {task}: {len(out)} label arrays")
        total += write(bundle, f"labels_{task}", out)
    return total


def export_classdist(bundle):
    """column sums only - the class-distribution bars need nothing else"""
    jobs = [("rbp", "real_human", 3, dict(balanced=True, multi_label=False)),
            ("rbp", "real_human", 3, dict(balanced=False, multi_label=False)),
            ("rbp", "real_human", 3, dict(balanced=True, multi_label=True)),
            ("rbp", "real_human", 3, dict(balanced=False, multi_label=True)),
            ("tf", "simulated_human", 6, dict(balanced=True, noise="none")),
            ("tf", "simulated_human", 6, dict(balanced=False, noise="none")),
            ("gex", "real_human", 4, dict(balanced=True)),
            ("gex", "real_human", 4, dict(balanced=False))]
    out = {}
    for task, species, ncl, kw in jobs:
        for seed in CP.VERSIONS[task]["seeds"][:1]:
            d = DA.load_split(task, species, ncl, seed,
                              keys=("y_train", "y_valid", "y_test"), **kw)
            if d is None:
                continue
            key = (task, tuple(sorted(kw.items())), seed)
            out[key] = {k: np.asarray(v).sum(axis=0).astype(np.int64)
                        for k, v in d.items()}
            out[key]["_n"] = {k: int(np.asarray(v).shape[0]) for k, v in d.items()}
    print(f"    {len(out)} class-count tables")
    return write(bundle, "classdist", out)


def export_umap(bundle):
    """UMAP xy + cell-type labels, instead of the 6 GB .h5ad"""
    try:
        import plots_datasets as PD
        xy, labels, err = PD._read_adata_umap()      # 3-tuple: (xy, labels, error)
    except Exception as exc:
        print(f"    UMAP unavailable ({type(exc).__name__}: {exc}) - SKIPPED")
        return 0.0
    if xy is None:
        print(f"    UMAP unavailable ({err}) - SKIPPED")
        return 0.0
    print(f"    {np.asarray(xy).shape[0]} cells")
    return write(bundle, "umap", {"xy": _f32(xy), "labels": np.asarray(labels)})


def export_sensitivity(bundle):
    """SD/mean per (task, setting, seed, model), including the inference-only conditions

    Most of these come straight from the aggregated dicts and are already covered by the
    comparison section. The ones that are NOT in any dict, and would otherwise force a
    cluster run forever, are the point of this section:
      - tf / no_motif : the aggregation loop only covered ls_noise=['mislabeled','none']
      - anything else _sd_for falls back to direct inference for
    """
    import plots_sensitivity as PS
    out, failed, from_dict, from_inference = {}, [], 0, 0
    for task in ("rbp", "tf", "gex"):
        spec = CP.VERSIONS[task]
        comp = DA.load_comparison(task)
        settings = list(spec["settings"]) + (["no_motif"] if task == "tf" else [])
        for setting in settings:
            for seed in spec["seeds"]:
                for model in CP.MODELS:
                    in_dict = DA.dict_key(task, seed, setting, model) in comp.get("sd", {})
                    try:
                        sd, mean = PS._sd_for(comp, task, seed, setting, model)
                    except Exception as exc:
                        failed.append(f"{task}/{setting}/s{seed}/{model}: "
                                      f"{type(exc).__name__}: {exc}")
                        continue
                    if sd is None:
                        failed.append(f"{task}/{setting}/s{seed}/{model}: unavailable")
                        continue
                    # Dict-backed entries are already in comparison_*; storing only the
                    # inference-derived ones keeps the bundle from doubling in size.
                    if in_dict:
                        from_dict += 1
                        continue
                    from_inference += 1
                    out[(task, setting, seed, model)] = (_f32(sd), _f32(mean))
    print(f"    {from_dict} already in the aggregated dicts (not duplicated)")
    print(f"    {from_inference} inference-only entries exported")
    print(f"    {len(failed)} unavailable")
    for f in failed[:6]:
        print(f"      - {f}")
    return write(bundle, "sensitivity", out)


def export_virus(bundle, seeds=None):
    """coronavirus OOD: in-distribution reference + per-virus SDs, all 8 genomes

    This is the expensive one - it runs MC inference over every encoded genome, which is
    why the genomes (~140 MB each) never need to leave the cluster. Output is one SD
    vector per (seed, model, virus), a few MB in total.
    """
    import plots_virus as PV
    spec = CP.VERSIONS["rbp"]
    use = seeds if seeds is not None else spec["seeds"]
    found, missing = PV._virus_inventory()
    print(f"    genomes present: {len(found)}/{len(CP.VIRUSES)}"
          + (f"  missing: {', '.join(missing)}" if missing else ""))
    if not found:
        print("    no genomes - SKIPPED (set UQFIG_VIRUS_DIR)")
        return 0.0
    out = {}
    for seed in use:
        t0 = time.time()
        try:
            data, err = PV._collect(seed)
        except Exception as exc:
            print(f"    seed {seed}: FAILED {type(exc).__name__}: {exc}")
            continue
        if data is None:
            print(f"    seed {seed}: skipped - {err}")
            continue
        for model, (sd_in, mean_in, per_virus) in data.items():
            out[(seed, model)] = {"sd_in": _f32(sd_in), "mean_in": _f32(mean_in),
                                  "per_virus": {v: _f32(a) for v, a in per_virus.items()}}
        print(f"    seed {seed}: {len(data)} models x {len(found)} genomes "
              f"({time.time() - t0:.0f}s)")
    if not out:
        print("    nothing collected - SKIPPED")
        return 0.0
    return write(bundle, "virus", out)


def export_app27(bundle):
    """27-RBP predictions + true labels for the uncertainty-filtering AP / PR figures"""
    import plots_utility as PU
    preds, err = PU._load_predictions()
    if preds is None:
        print(f"    {err} - SKIPPED")
        return 0.0
    n_classes = int(np.asarray(preds["mean"]).shape[1])
    y = PU._y_true(n_classes)
    if y is None:
        print("    APP27_DATASET missing y_test - exporting predictions only")
    # the class-distribution figure needs y_train column sums and the class names
    train_counts, class_names = None, None
    if CP.APP27_DATASET and os.path.exists(CP.APP27_DATASET):
        with open(CP.APP27_DATASET, "rb") as fh:
            d = pickle.load(fh)
        if d.get("y_train") is not None:
            train_counts = np.asarray(d["y_train"]).sum(axis=0).astype(np.int64)
        class_names = d.get("class_names")
    print(f"    predictions {np.asarray(preds['mean']).shape}, "
          f"labels {None if y is None else np.asarray(y).shape}, "
          f"train counts {None if train_counts is None else train_counts.shape}")
    return write(bundle, "app27", {"mean": _f32(preds["mean"]), "sd": _f32(preds["sd"]),
                                   "y_test": None if y is None else np.asarray(y).astype(np.int8),
                                   "train_counts": train_counts,
                                   "class_names": class_names})


EXPORTERS = {"comparison": export_comparison, "labels": export_labels,
             "classdist": export_classdist, "umap": export_umap,
             "sensitivity": export_sensitivity, "virus": export_virus,
             "app27": export_app27}


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figure_data_bundle")
    ap.add_argument("--only", nargs="*", choices=SECTIONS,
                    help="refresh only these sections")
    ap.add_argument("--virus-seeds", nargs="*", type=int, default=None,
                    help="limit virus inference to these seeds (default: all 5). "
                         "Virus is by far the slowest section.")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in SECTIONS:
            print(f"  {s:14s} {EXPORTERS[s].__doc__.strip().splitlines()[0]}")
        return 0

    bundle = os.path.abspath(args.out)
    os.makedirs(bundle, exist_ok=True)
    todo = args.only or SECTIONS
    print(f"bundle: {bundle}\nsections: {', '.join(todo)}\n")

    sizes, errors = {}, {}
    for s in todo:
        print(f"[{s}]")
        t0 = time.time()
        try:
            sizes[s] = (export_virus(bundle, args.virus_seeds) if s == "virus"
                        else EXPORTERS[s](bundle)) or 0.0
        except Exception as exc:
            import traceback
            errors[s] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {errors[s]}")
            traceback.print_exc(limit=3)
        print(f"    {time.time() - t0:.1f}s\n")

    # manifest, so the Mac side can tell what it has and when it came from
    man_path = os.path.join(bundle, "manifest.json")
    man = {}
    if os.path.exists(man_path):
        try:
            man = json.load(open(man_path))
        except Exception:
            man = {}
    man.setdefault("sections", {})
    for s in todo:
        man["sections"][s] = {"exported": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "mb": round(sizes.get(s, 0.0), 1),
                              "error": errors.get(s, "")}
    man["roots"] = {"rbp": CP.results_root_for("rbp"), "tf": CP.results_root_for("tf"),
                    "gex": CP.results_root_for("gex")}
    man["versions"] = {t: CP.VERSIONS[t]["version"] for t in ("rbp", "tf", "gex")}
    json.dump(man, open(man_path, "w"), indent=1)

    total = sum(sizes.values())
    print("=" * 62)
    print(f"total {total:.1f} MB in {bundle}")
    if errors:
        print(f"{len(errors)} section(s) failed: {', '.join(errors)}")
    print("\ncopy to your Mac:")
    print(f"  scp -r <user>@<cluster-host>:{bundle} .")
    print("then locally:")
    print("  export UQFIG_BUNDLE=$PWD/figure_data_bundle")
    print("  python3 run_all.py --out runs/local")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
