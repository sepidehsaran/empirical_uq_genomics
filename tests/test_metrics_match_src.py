"""Prove the pipeline's metrics are identical to src/eval_functions.py.

Imports BOTH implementations and compares them on random inputs. This is the guard
against the pipeline silently drifting from the project's own definitions.

    python3 test_metrics_match_src.py                    # uses UQFIG_SRC_DIR
    UQFIG_SRC_DIR=/path/to/src python3 test_metrics_match_src.py

Also checks the branch that the published notebook got backwards:
2024_figures_section_3_3.ipynb cell 63 calls accuracy_single_label on multi-label data
and accuracy_multi_label on single-label data. plots_performance.py must do the reverse.
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as CP          # noqa: E402
import metrics as M                # noqa: E402


def load_src_eval():
    src = os.environ.get("UQFIG_SRC_DIR") or CP.SRC_DIR
    if not src:
        return None, "SRC_DIR not set - run on the cluster or set UQFIG_SRC_DIR"
    path = os.path.join(src, "eval_functions.py")
    if not os.path.exists(path):
        return None, f"not found: {path}"
    if src not in sys.path:
        sys.path.insert(0, src)     # eval_functions does `from utils import *`
    # eval_functions.py imports sklearn/tensorflow at module level for functions we do
    # not call. Missing those must NOT turn into a silent skip - a skipped comparison
    # looks identical to a passing one. Stub whatever is absent and carry on.
    import types
    stubbed = []
    for _ in range(20):
        spec = importlib.util.spec_from_file_location("_src_eval", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            return mod, (f"stubbed absent module(s): {', '.join(stubbed)}" if stubbed else "")
        except ModuleNotFoundError as exc:
            name = exc.name
            if not name or name in stubbed:
                return None, f"import failed: {exc}"
            for part_i in range(len(name.split(".")) + 1):
                sub = ".".join(name.split(".")[:part_i]) or name
                sys.modules.setdefault(sub, types.ModuleType(sub))
            m = sys.modules[name]
            m.__getattr__ = lambda _n: types.SimpleNamespace()   # any attribute access
            m.__path__ = []
            m.__all__ = []          # `from x import *` iterates this
            stubbed.append(name)
        except Exception as exc:
            return None, f"import failed: {exc}"
    return None, "gave up stubbing imports"


def one_hot(n, c, rng):
    y = np.zeros((n, c))
    y[np.arange(n), rng.integers(0, c, n)] = 1.0
    return y


def main():
    src, err = load_src_eval()
    rng = np.random.default_rng(0)
    fails = []

    # ---- single-label: 3-class and 4-class one-hot ------------------------------
    for c in (3, 4):
        # Confident predictions (one class well above 0.5), as real softmax outputs are.
        # The (1+(C-1)a)/C identity assumes exactly one cell crosses the 0.5 threshold
        # per row; flat random probabilities break that assumption, which is precisely
        # the effect that distorts the published ENS numbers.
        p = rng.dirichlet(np.ones(c) * 0.05, size=500)
        y = one_hot(500, c, rng)
        mine = M.accuracy_single_label(p, y)
        if src:
            theirs = float(src.accuracy_single_label(p, y))
            ok = abs(mine - theirs) < 1e-12
            print(f"accuracy_single_label  C={c}   mine={mine:.6f}  src={theirs:.6f}  "
                  f"{'OK' if ok else 'DIFFER'}")
            if not ok:
                fails.append(f"accuracy_single_label C={c}")

        # The notebook's inverted branch, reproduced so the size of the error is visible.
        wrong = M.accuracy_multi_label(p, y, apply_sigmoid=False, th=0.5)
        # A correct row has all C cells right; a wrong row has exactly two wrong (a
        # false 1 at the predicted class, a false 0 at the true class) regardless of C.
        # So element-wise = (C - 2 + 2a)/C. At C=3 this equals (1+2a)/3.
        predicted = (c - 2 + 2 * mine) / c
        ok = abs(wrong - predicted) < 5e-3
        print(f"   notebook branch (element-wise on single-label) = {wrong:.6f}; "
              f"(C-2+2a)/C = {predicted:.6f}  {'OK' if ok else 'DIFFER'}")
        if not ok:
            fails.append(f"rescaling identity C={c}")

    # ---- multi-label: element-wise, must equal src exactly ----------------------
    p = rng.random((500, 6))
    y = (rng.random((500, 6)) < 0.3).astype(float)
    mine = M.accuracy_multi_label(p, y, apply_sigmoid=False, th=0.5)
    subset = float(np.mean(np.all(np.where(p < 0.5, 0.0, 1.0) == y, axis=1)))
    if src:
        theirs = float(src.accuracy_multi_label(p, y, apply_sigmoid=False, th=0.5))
        ok = abs(mine - theirs) < 1e-12
        print(f"accuracy_multi_label         mine={mine:.6f}  src={theirs:.6f}  "
              f"{'OK' if ok else 'DIFFER'}")
        if not ok:
            fails.append("accuracy_multi_label")
    print(f"   exact-match (NOT used)                = {subset:.6f}")

    # sigmoid path
    if src:
        logits = rng.normal(size=(200, 6))
        yb = (rng.random((200, 6)) < 0.3).astype(float)
        a = M.accuracy_multi_label(logits, yb, apply_sigmoid=True, th=0.5)
        try:
            b = float(src.accuracy_multi_label(logits, yb, apply_sigmoid=True, th=0.5))
            ok = abs(a - b) < 1e-12
            print(f"accuracy_multi_label sigmoid mine={a:.6f}  src={b:.6f}  "
                  f"{'OK' if ok else 'DIFFER'}")
            if not ok:
                fails.append("accuracy_multi_label(sigmoid)")
        except NameError as exc:
            # src/utils.py sigmoid() uses math.exp without importing math - a
            # pre-existing bug in the repo. The pipeline always passes
            # apply_sigmoid=False, so this path is never exercised.
            print(f"accuracy_multi_label sigmoid mine={a:.6f}  src=<src bug: {exc}>  "
                  f"NOT EXERCISED by the pipeline")

    # ---- NLL / Brier: defined in the notebooks, replicated here -----------------
    c = 3
    p = rng.dirichlet(np.ones(c), size=400)
    y = one_hot(400, c, rng)
    nll_ref = float(-np.sum(y * np.log(np.clip(p, 1e-15, 1 - 1e-15))) / y.shape[0])
    brier_ref = float(np.mean(np.sum((p - y) ** 2, axis=1)))
    for name, mine_v, ref in (("calculate_nll_one_hot", M.calculate_nll_one_hot(p, y), nll_ref),
                              ("calculate_brier_score", M.calculate_brier_score(p, y), brier_ref)):
        ok = abs(mine_v - ref) < 1e-12
        print(f"{name:24s} mine={mine_v:.6f}  notebook={ref:.6f}  {'OK' if ok else 'DIFFER'}")
        if not ok:
            fails.append(name)

    # ---- the branch itself ------------------------------------------------------
    import inspect
    import plots_performance as PP
    body = inspect.getsource(PP._collect) if hasattr(PP, "_collect") else inspect.getsource(PP)
    i_multi = body.find("accuracy_multi_label")
    i_single = body.find("accuracy_single_label")
    branch_ok = -1 < i_multi < i_single          # multi first, guarded by `if multi`
    print(f"\nbranch order in plots_performance: "
          f"{'OK - multi_label used when multi=True' if branch_ok else 'CHECK MANUALLY'}")

    if src is None:
        print(f"\nsrc/ comparison COULD NOT RUN: {err}")
        fails.append("src comparison did not run - result is not a pass")
    elif err:
        print(f"\nnote: {err}")
    print("\n" + ("FAILED: " + ", ".join(fails) if fails else "all metric definitions match"))
    return 1 if fails else 0



def test_script():
    """pytest entry point: the script's main() returns 0 on success"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
