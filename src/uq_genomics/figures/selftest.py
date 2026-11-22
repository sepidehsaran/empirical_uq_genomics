#!/usr/bin/env python3
"""
selftest.py - validate every code path WITHOUT the cluster data.

Most figures depend on the aggregated comparison_results dicts (and the virus figures
on the trained models), which are not on this machine. This script fabricates dicts with
the correct key convention and the correct per-dataset shapes, points the pipeline at
them, and runs everything. It proves the plotting code works end to end, so that when
you copy the real pickles in, the only thing that changes is the numbers.

    python3 selftest.py

Writes to ./selftest_out (never to ./out) so real and synthetic figures never mix.
THE FIGURES IT PRODUCES ARE SYNTHETIC. Do not use them in the paper.
"""
import os
import pickle
import shutil
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gc
import matplotlib.pyplot as _plt
import config_paths as CP
import data_access as DA


def _between_tasks():
    _plt.close('all')
    gc.collect()


def _between_stages():
    import data_access as _DA
    _DA.clear_caches()
    _plt.close('all')
    gc.collect()


def build_fixture(root):
    """synthetic dict_iter_mean / _sd with real shapes and real key names"""
    rng = np.random.default_rng(0)
    made = 0
    for task, spec in CP.VERSIONS.items():
        d = os.path.join(root, spec["task"], spec["species"], "comparison_results")
        os.makedirs(d, exist_ok=True)
        dmean, dsd = {}, {}
        for seed in spec["seeds"]:
            for setting in spec["settings"]:
                kw = DA.setting_to_split_args(task, setting)
                y = DA.load_y_test(task, spec["species"], spec["n_classes"], seed, **kw)
                if y is None:
                    continue
                y = np.asarray(y)
                n, k = y.shape
                # a shared "difficulty" signal so the methods agree partially,
                # which is what the agreement figures are meant to detect
                shared = rng.random(n)
                for mi, model in enumerate(CP.MODELS):
                    logits = y * (2.5 + 0.4 * mi) + rng.normal(0, 0.8, size=(n, k))
                    e = np.exp(logits - logits.max(axis=1, keepdims=True))
                    mean = e / e.sum(axis=1, keepdims=True)
                    base = 0.02 + 0.05 * mi
                    sd = (base * (0.4 + shared[:, None])
                          * (1.0 + 0.5 * rng.random((n, k))))
                    if setting not in (spec["settings"][0],):      # perturbed setting
                        sd *= 1.8                                   # shift, as expected
                    dmean[DA.dict_key(task, seed, setting, model)] = mean
                    dsd[DA.dict_key(task, seed, setting, model)] = sd
                    made += 1
        with open(os.path.join(d, "dict_iter_mean.pickle"), "wb") as fh:
            pickle.dump(dmean, fh)
        with open(os.path.join(d, "dict_iter_sd.pickle"), "wb") as fh:
            pickle.dump(dsd, fh)
    return made


def fake_inference(task, species, n_classes, seed, setting, model_type, x,
                   n_iter=100, n_iter_ens=10):
    """stand-in for the trained models, so the virus code path can be exercised"""
    rng = np.random.default_rng(abs(hash((task, setting, model_type, seed))) % 2**32)
    n = min(len(x), 4000)
    it = 12
    logits = rng.normal(0, 1.0, size=(it, n, n_classes))
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    out = os.path.join(HERE, "selftest_out")
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="uqfig_fixture_")
    print(f"building synthetic comparison_results in {tmp} ...")
    n = build_fixture(tmp)
    print(f"  {n} synthetic (setting x model x seed) entries\n")

    CP.RESULTS_ROOT = tmp
    CP.MODELS_ROOT = tmp                      # non-empty so virus/TF paths are attempted
    DA.infer_with_models = fake_inference     # exercise inference-dependent figures

    import style as S
    S.apply_rc()

    import importlib
    ok = skip = err = 0
    for mod_name in ("plots_performance", "plots_agreement", "plots_sensitivity",
                     "plots_virus", "plots_utility"):
        mod = importlib.import_module(mod_name)
        print("=" * 78)
        print(mod_name)
        print("=" * 78)
        for tname, thunk in mod.tasks(out):
            try:
                res = thunk() or []
            except Exception as exc:
                import traceback
                traceback.print_exc()
                res = [dict(id=tname, status="error", reason=str(exc))]
            for r in res:
                s = r["status"]
                ok += s == "ok"; skip += s == "skip"; err += s == "error"
                tag = {"ok": "  OK  ", "skip": " SKIP ", "error": "ERROR "}[s]
                print(f"  [{tag}] {r['id']:46s} "
                      f"{(r.get('note') or r.get('reason') or '')[:56]}", flush=True)
            _between_tasks()
        _between_stages()
        print()

    print("=" * 78)
    print(f"SELFTEST  ok={ok}  skipped={skip}  errors={err}")
    print(f"synthetic figures in: {out}   (DO NOT use these in the paper)")
    print("=" * 78)
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
