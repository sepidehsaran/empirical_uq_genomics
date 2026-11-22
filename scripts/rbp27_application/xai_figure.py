"""Figure and statistics from xai_uncertainty.py output.

    python scripts/rbp27_application/xai_figure.py --json out/xai_top3/entropy.json --out out/xai_top3
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

CELLS = ["correct_kept", "correct_filtered", "wrong_kept", "wrong_filtered"]
NICE = {"correct_kept": "correct\nconfident", "correct_filtered": "correct\nunsure",
        "wrong_kept": "wrong\nconfident", "wrong_filtered": "wrong\nunsure"}
RBP27 = ["MBNL1", "P53_NONO", "PUM2", "QKI", "AGO3", "FUS", "TAF15", "ZFP36",
         "DICER1", "EIF3A", "EIF3D", "EIF3G", "SSB", "PAPD5", "CPSF4", "CPSF3",
         "RTCB", "FXR1", "NOP58", "NOP56", "FBL", "LIN28A", "LIN28B", "UPF1",
         "G35", "G45", "XPO5"]


def load(path):
    with open(path) as fh:
        rows = json.load(fh)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["cls"]][r["group"]].append(r)
    return rows, by


def stats_table(by, metric):
    print(f"\n=== {metric} ===")
    for c in sorted(by):
        cells = by[c]
        nm = RBP27[c] if c < len(RBP27) else f"class {c}"
        line = f"  {nm:10s}"
        for g in CELLS:
            v = [r[metric] for r in cells.get(g, [])]
            if v:
                line += f"  {g.split('_')[0][0].upper()}{g.split('_')[1][0]}" \
                        f"={np.median(v):.4f}(n={len(v)})"
        print(line)
        ck = [r[metric] for r in cells.get("correct_kept", [])]
        for other in ("correct_filtered", "wrong_kept", "wrong_filtered"):
            o = [r[metric] for r in cells.get(other, [])]
            if len(ck) >= 8 and len(o) >= 8:
                _, p = mannwhitneyu(ck, o, alternative="greater")
                flag = "  *" if p < 0.05 else ""
                print(f"             correct_kept > {other:17s} p={p:.3g}{flag}")
        sh = [r[metric] for r in cells.get("dinuc_shuf", [])]
        if len(ck) >= 8 and len(sh) >= 8:
            _, p = mannwhitneyu(ck, sh, alternative="greater")
            print(f"             CONTROL vs dinuc_shuf            p={p:.3g}"
                  f"{'' if p < 0.05 else '   <-- no motif signal in this class'}")


def build(rows, by, metric, out, stem):
    try:
        from uq_genomics.figures import style as S
        S.apply_rc()
        pal = S.PALETTE if isinstance(S.PALETTE, (list, tuple)) else None
        W = S.TEXTWIDTH_IN
    except Exception:
        pal, W = None, 6.5
    colors = pal or ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"]

    classes = sorted(by)
    # taller than it looks necessary: the shared legend sits under panel a only, so the
    # bottom band has to clear panel b's axis label as well
    fig = plt.figure(figsize=(W, W * 0.50), dpi=300)
    axL = fig.add_axes([0.095, 0.255, 0.50, 0.665])
    axR = fig.add_axes([0.705, 0.255, 0.280, 0.665])

    # ---- a) 2x2 per class, shuffled control as a dashed reference
    pos, ticks, labels = 0, [], []
    for c in classes:
        cells = by[c]
        base = pos
        for gi, g in enumerate(CELLS):
            v = [r[metric] for r in cells.get(g, [])]
            if not v:
                pos += 1
                continue
            bp = axL.boxplot([v], positions=[pos], widths=0.72, patch_artist=True,
                             showfliers=False, medianprops=dict(color="0.2", lw=1.0))
            bp["boxes"][0].set(facecolor=colors[gi % len(colors)], lw=0, alpha=0.9)
            pos += 1
        sh = [r[metric] for r in cells.get("dinuc_shuf", [])]
        if sh:
            axL.plot([base - 0.5, pos - 0.5], [np.median(sh)] * 2,
                     ls="--", color="0.35", lw=1.0, zorder=5)
        ticks.append((base + pos - 1) / 2)
        labels.append(RBP27[c] if c < len(RBP27) else f"c{c}")
        pos += 1
    axL.set_xticks(ticks)
    axL.set_xticklabels(labels)
    axL.set_ylabel({"conc": "attribution in best 8nt window",
                    "gini": "Gini of attribution",
                    "entropy": "positional entropy (bits)"}.get(metric, metric))
    axL.text(0.0, 1.02, "a)", transform=axL.transAxes, fontweight="bold", va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i % len(colors)], lw=0)
               for i in range(4)]
    handles.append(plt.Line2D([0], [0], ls="--", color="0.35"))
    axL.legend(handles, [NICE[g].replace("\n", " ") for g in CELLS] + ["shuffled"],
               loc="upper center", bbox_to_anchor=(0.5, -0.115), ncols=3,
               frameon=False, fontsize=6.5, handletextpad=0.5, columnspacing=1.2,
               handlelength=1.4)

    # ---- b) continuous: uncertainty vs sharpness
    for i, c in enumerate(classes):
        v = [(r["sd"], r[metric]) for g in CELLS for r in by[c].get(g, [])]
        if len(v) < 10:
            continue
        s, m = map(np.array, zip(*v))
        axR.scatter(s, m, s=5, alpha=0.45, linewidths=0,
                    color=colors[i % len(colors)],
                    label=RBP27[c] if c < len(RBP27) else f"c{c}")
        rho, p = spearmanr(s, m)
        # rho printed as text rather than legend entries: three classes x two lines of
        # legend covered most of the scatter
        axR.text(0.03, 0.97 - 0.075 * i,
                 f"{RBP27[c] if c < len(RBP27) else c}  $\\rho$={rho:+.2f}"
                 f"{'*' if p < 0.05 else ''}",
                 transform=axR.transAxes, fontsize=6.5, va="top",
                 color=colors[i % len(colors)])
    axR.set_xlabel("uncertainty score (SD)", fontsize=8)
    axR.set_ylabel("sharpness", fontsize=8)
    axR.tick_params(labelsize=7)
    axR.text(0.0, 1.02, "b)", transform=axR.transAxes, fontweight="bold", va="bottom")

    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"{stem}.{ext}")
        fig.savefig(p, dpi=300)
        paths.append(p)
    plt.close(fig)
    print(f"\nwrote {paths[0]}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="conc", choices=["conc", "gini", "entropy"])
    args = ap.parse_args()
    out = args.out or os.path.dirname(args.json) or "."
    os.makedirs(out, exist_ok=True)

    rows, by = load(args.json)
    print(f"{len(rows)} attributions over {len(by)} classes")
    for m in ("conc", "gini", "entropy"):
        if any(m in r for r in rows[:1]):
            stats_table(by, m)
    build(rows, by, args.metric, out, f"XAI_sharpness_{args.metric}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
