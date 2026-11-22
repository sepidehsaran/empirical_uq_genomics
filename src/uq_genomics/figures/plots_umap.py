"""
plots_umap.py - Fig. 2: UMAP of the single-cell data, all cells (a) and the four selected
types in the Liam embedding (b). Reads the cached coordinates that ship with the bundle.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_paths as CP            # noqa: E402
import style as S                    # noqa: E402

SELECTED = ["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]
SEL_COLORS = {"CD14+ Mono": "#8c2d5a", "CD4+ T activated": "#4c72b0", "Erythroblast": "#2ca25f", "NK": "#9467bd"}


def build(out):
    """Fig. 2: writes UMAP_liam_embedding_v2.pdf/.png into `out`"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    os.makedirs(out, exist_ok=True)
    # Panel a: the dataset's own UMAP of all 69,249 cells (X_umap, as in the
    # figure). Panel b: UMAP of the 20-D Liam embedding of the 28,214 selected cells.
    if not (CP.UMAP_COORDS_FULL and CP.UMAP_COORDS_4CT):
        raise FileNotFoundError("cached UMAP coordinates not found: put _full_xumap_labels.pkl and "
                                "umap_liam_4_cell_types_coords.pkl into the bundle directory, or set "
                                "UQFIG_UMAP_COORDS_FULL / UQFIG_UMAP_COORDS_4CT")
    with open(CP.UMAP_COORDS_FULL, "rb") as fh:
        full = pickle.load(fh)
    xy, lab = np.asarray(full["X_umap"]), np.asarray(full["labels"]).astype(str)
    with open(CP.UMAP_COORDS_4CT, "rb") as fh:
        xy4, lab4 = pickle.load(fh)
    xy4, lab4 = np.asarray(xy4), np.asarray(lab4).astype(str)
    types = sorted(set(lab))
    cmap = plt.get_cmap("tab20")
    colors = {t: cmap(i % 20) for i, t in enumerate(types)}
    rng = np.random.default_rng(0)
    order = rng.permutation(len(xy))           # draw in random order so no type sits on top

    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * 0.46), dpi=S.DPI)
    ax = axes[0]
    ax.scatter(xy[order, 0], xy[order, 1], c=[colors[t] for t in lab[order]], s=0.4, linewidths=0, rasterized=True)
    ax.set_title("a) all cell types", fontsize=S.FS_LABEL, loc="left")
    h = [Line2D([0], [0], marker="o", color=colors[t], lw=0, ms=4) for t in types]
    ax.legend(h, types, loc="upper left", bbox_to_anchor=(1.0, 1.0), ncols=1, fontsize=S.FS_LEG - 2.5,
              frameon=False, handletextpad=0.3, labelspacing=0.25, borderaxespad=0.0)

    ax = axes[1]
    o2 = rng.permutation(len(xy4))
    ax.scatter(xy4[o2, 0], xy4[o2, 1], c=[SEL_COLORS.get(t, "0.7") for t in lab4[o2]], s=0.4, linewidths=0, rasterized=True)
    ax.set_title("b) four selected types (Liam)", fontsize=S.FS_LABEL, loc="left")
    h = [Line2D([0], [0], marker="o", color=SEL_COLORS[t], lw=0, ms=4) for t in SELECTED]
    ax.legend(h, SELECTED, loc="lower center", bbox_to_anchor=(0.5, -0.26), ncols=2, fontsize=S.FS_LEG - 1,
              frameon=False, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=S.FS_TICK); ax.set_ylabel("UMAP 2", fontsize=S.FS_TICK)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.20, wspace=0.90)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"UMAP_liam_embedding_v2.{ext}"), dpi=S.DPI)
    print("->", os.path.join(out, "UMAP_liam_embedding_v2.pdf"), len(types), "types")



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=CP.OUT_DIR)
    build(ap.parse_args().out)
