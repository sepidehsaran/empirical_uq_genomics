#!/usr/bin/env python3
"""
run_all.py - generate EVERY figure for the paper from scratch, one by one, in one
consistent size/font, and report exactly which inputs are missing.

    python3 run_all.py                 # everything
    python3 run_all.py --only datasets performance
    python3 run_all.py --list          # show the stages and exit
    python3 run_all.py --check         # only resolve paths, generate nothing

Nothing existing is modified. Output goes to ./out, logs and reports to ./logs.
"""
import argparse
import datetime as _dt
import json
import subprocess
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import config_paths as CP

# matplotlib / style are imported lazily: `--check` and `--list` only resolve paths and
# must work in a bare interpreter (e.g. conda `base`) with nothing installed.
_plt = None
S = None


def _load_plotting():
    """import the plotting stack on demand; returns (ok, message)"""
    global _plt, S
    if S is not None:
        return True, ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import style as _S
        _plt, S = plt, _S
        return True, ""
    except ImportError as exc:
        return False, str(exc)

STAGES = [
    ("datasets",    "plots_datasets",    "class distributions, UMAP, MNIST, passthrough panels"),
    ("performance", "plots_performance", "accuracy / NLL / Brier  (RBP, GEX, TF[new])"),
    ("agreement",   "plots_agreement",   "Kendall, top-50/100 overlap, KS, Pearson/Spearman (+TF[new])"),
    ("sensitivity", "plots_sensitivity", "SD-shift grids + compact examples (+TF imbalance[new])"),
    ("virus",       "plots_virus",       "coronavirus OOD: 8 genomes, KS heatmap [new]"),
    ("utility",     "plots_utility",     "27-RBP uncertainty filtering: AP + PR curves"),
]


def _between_tasks():
    if _plt is not None:
        _plt.close("all")
    gc.collect()


def _between_stages():
    try:
        import data_access as _DA
        _DA.clear_caches()
    except Exception:
        pass
    if _plt is not None:
        _plt.close("all")
    gc.collect()


def _hr(ch="-", n=86):
    return ch * n


def _flush_manifest(results):
    """write the manifest after every figure so nothing is lost if the run dies"""
    os.makedirs(CP.LOG_DIR, exist_ok=True)
    with open(os.path.join(CP.LOG_DIR, "MANIFEST.csv"), "w") as fh:
        fh.write("stage,id,status,path,note_or_reason\n")
        for r in results:
            note = (r.get("note") or r.get("reason") or "").replace(",", ";").replace("\n", " ")
            fh.write(f"{r.get('stage','')},{r['id']},{r['status']},"
                     f"{os.path.basename(r.get('path',''))},{note}\n")


REQUIRED = ["numpy", "scipy", "matplotlib", "pandas", "seaborn"]
OPTIONAL = {"sklearn": "27-RBP PR curves",
            "tensorflow": "figures that re-run model inference",
            "anndata": "UMAP rebuilt from adata.h5ad (cached coords used otherwise)"}


def check_packages():
    import importlib.util as iu
    print("=" * 86)
    print("PYTHON ENVIRONMENT")
    print("=" * 86)
    print(f"  interpreter: {sys.executable}")
    missing = []
    for m in REQUIRED:
        ok = iu.find_spec(m) is not None
        print(f"  [{'ok' if ok else 'MISSING'}]      {m}")
        if not ok:
            missing.append(m)
    for m, why in OPTIONAL.items():
        ok = iu.find_spec(m) is not None
        print(f"  [{'ok' if ok else '  --  '}]      {m:12s} (optional: {why})")
    if missing:
        print()
        print(f"  !! missing required package(s): {', '.join(missing)}")
        print("     if you are in activate the project environment first:")
        print("         source .venv/bin/activate")
        print(f"     or install:  pip install {' '.join(missing)}")
    print()
    return missing


def check_paths():
    print(_hr("="))
    print("INPUT PATHS")
    print(_hr("="))
    missing = []
    for label, path, needed in CP.report():
        if path:
            print(f"  [ok]      {label:28s} {path}")
        else:
            print(f"  [MISSING] {label:28s} <empty>   -> needed for: {needed}")
            missing.append((label, needed))
    print()
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of stage names to run")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stage", default=None,
                    help=argparse.SUPPRESS)   # internal: run ONE stage in this process
    ap.add_argument("--no-isolate", action="store_true",
                    help="run all stages in one process (default: one subprocess per "
                         "stage, so an out-of-memory kill in one stage cannot lose the "
                         "figures already produced by the others)")
    ap.add_argument("--out", default=None,
                    help="output folder (default ./out, or $UQFIG_OUT_DIR). "
                         "Use a fresh name per run to compare side by side, "
                         "e.g. --out runs/2026-07-27_with_cluster_data")
    args = ap.parse_args()

    if args.list:
        for name, _, desc in STAGES:
            print(f"  {name:12s} {desc}")
        return 0

    if args.out:
        CP.set_out_dir(args.out)
    os.makedirs(CP.OUT_DIR, exist_ok=True)
    os.makedirs(CP.LOG_DIR, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------- worker: run exactly one stage and hand results back as JSON ----------
    if args.stage:
        ok, err = _load_plotting()
        if not ok:
            print(f"cannot import the plotting stack: {err}")
            return 1
        S.apply_rc()
        entry = next((x for x in STAGES if x[0] == args.stage), None)
        if entry is None:
            return 2
        name, mod_name, _ = entry
        out = []
        try:
            mod = __import__(mod_name)
            tl = mod.tasks(CP.OUT_DIR) if hasattr(mod, "tasks") \
                 else [(mod_name, lambda m=mod: m.run(CP.OUT_DIR))]
            for tname, thunk in tl:
                try:
                    res = thunk() or []
                except Exception as exc:
                    traceback.print_exc()
                    res = [dict(id=tname, status="error", reason=str(exc))]
                for r in res:
                    r["stage"] = name
                    tag = {"ok": "  OK  ", "skip": " SKIP ",
                           "error": "ERROR "}.get(r["status"], "  ?   ")
                    print(f"  [{tag}] {r['id']:44s} "
                          f"{(r.get('note') or r.get('reason') or '')[:70]}", flush=True)
                out.extend(res)
                _between_tasks()
        except Exception as exc:
            traceback.print_exc()
            out.append(dict(stage=name, id=mod_name, status="error", reason=str(exc)))
        with open(os.path.join(CP.LOG_DIR, f"_stage_{name}.json"), "w") as fh:
            json.dump(out, fh)
        return 0

    missing_pkgs = check_packages()
    missing_paths = check_paths()
    if args.check:
        return 1 if missing_pkgs else 0

    ok, err = _load_plotting()
    if not ok:
        print(f"cannot import the plotting stack: {err}")
        print("activate the project environment first:  source .venv/bin/activate")
        return 1
    S.apply_rc()
    print(f"style: every figure exported at the width it will occupy "
          f"(\\textwidth={S.TEXTWIDTH_IN}in), base font {S.FS_BASE}pt, {S.DPI} dpi")
    print(f"output: {CP.OUT_DIR}")
    print(f"        (nothing is written to the manuscript's figures/ folder - the figures the paper "
          f"currently uses are left untouched)\n")

    todo = [s for s in STAGES if not args.only or s[0] in args.only]
    results = []
    for name, mod_name, desc in todo:
        print(_hr("="))
        print(f"STAGE: {name}   ({desc})")
        print(_hr("="))
        try:
            mod = __import__(mod_name)
        except Exception as exc:
            print(f"  !! cannot import {mod_name}: {exc}")
            results.append(dict(stage=name, id=mod_name, status="error", reason=str(exc)))
            continue
        if not args.no_isolate:
            # Stage isolation: run the stage in a child process. Memory is fully
            # reclaimed between stages, and an OOM kill cannot take down the whole run.
            jf = os.path.join(CP.LOG_DIR, f"_stage_{name}.json")
            stale_mtime = os.path.getmtime(jf) if os.path.exists(jf) else None
            try:
                if stale_mtime is not None:
                    os.remove(jf)
                    stale_mtime = None
            except OSError:
                pass          # read-only / restricted fs: detect staleness by mtime
            cmd = [sys.executable, os.path.abspath(__file__), "--stage", name,
                   "--out", CP.OUT_DIR]
            rc = subprocess.call(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            fresh = (os.path.exists(jf) and
                     (stale_mtime is None or os.path.getmtime(jf) > stale_mtime))
            if fresh:
                with open(jf) as fh:
                    stage_res = json.load(fh)
                results.extend(stage_res)
            else:
                msg = (f"stage died (exit {rc}) with no results - most likely killed "
                       f"for memory; re-run just this stage with --only {name}")
                print(f"  !! {msg}")
                results.append(dict(stage=name, id=mod_name, status="error", reason=msg))
            _flush_manifest(results)
            print()
            continue

        try:
            task_list = mod.tasks(CP.OUT_DIR) if hasattr(mod, "tasks") \
                        else [(mod_name, lambda m=mod: m.run(CP.OUT_DIR))]
        except Exception as exc:
            print(f"  !! cannot build task list for {mod_name}: {exc}")
            results.append(dict(stage=name, id=mod_name, status="error", reason=str(exc)))
            continue

        for tname, thunk in task_list:
            try:
                out = thunk() or []
            except Exception as exc:
                traceback.print_exc()
                out = [dict(id=tname, status="error", reason=str(exc))]
            for r in out:
                r["stage"] = name
                results.append(r)
                tag = {"ok": "  OK  ", "skip": " SKIP ",
                       "error": "ERROR "}.get(r["status"], "  ?   ")
                extra = r.get("note") or r.get("reason") or ""
                print(f"  [{tag}] {r['id']:44s} {extra[:70]}", flush=True)
            _between_tasks()
            _flush_manifest(results)
        _between_stages()
        print()

    # ------------------------------------------------------------------ reports
    ok = [r for r in results if r["status"] == "ok"]
    sk = [r for r in results if r["status"] == "skip"]
    er = [r for r in results if r["status"] == "error"]

    man = os.path.join(CP.LOG_DIR, "MANIFEST.csv")
    _flush_manifest(results)

    rep = os.path.join(CP.LOG_DIR, "MISSING_INPUTS.md")
    with open(rep, "w") as fh:
        fh.write(f"# Missing inputs report\n\nGenerated {stamp}\n\n")
        fh.write(f"Generated **{len(ok)}** figures, skipped **{len(sk)}**, "
                 f"errors **{len(er)}**.\n\n")
        if missing_paths:
            fh.write("## 1. Paths to fill in `config_paths.py`\n\n")
            fh.write("| variable | unblocks |\n|---|---|\n")
            for label, needed in missing_paths:
                fh.write(f"| `{label}` | {needed} |\n")
            fh.write("\n")
        fh.write("## 2. Figures that could not be generated\n\n")
        if sk:
            fh.write("| stage | figure | why |\n|---|---|---|\n")
            for r in sk:
                fh.write(f"| {r['stage']} | `{r['id']}` | {r.get('reason','')} |\n")
        else:
            fh.write("None.\n")
        fh.write("\n## 3. Figures generated\n\n")
        fh.write("| stage | figure | note |\n|---|---|---|\n")
        for r in ok:
            fh.write(f"| {r['stage']} | `{r['id']}` | {r.get('note','')} |\n")
        if er:
            fh.write("\n## 4. Errors\n\n")
            for r in er:
                fh.write(f"- `{r['id']}`: {r.get('reason','')}\n")

    print(_hr("="))
    print(f"DONE  ok={len(ok)}  skipped={len(sk)}  errors={len(er)}")
    print(f"  figures : {CP.OUT_DIR}")
    print(f"  manifest: {man}")
    print(f"  gaps    : {rep}")
    print(_hr("="))
    if sk:
        print("\nTop reasons for skipping:")
        from collections import Counter
        for reason, n in Counter(r.get("reason", "") for r in sk).most_common(6):
            print(f"  {n:3d} x  {reason[:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
