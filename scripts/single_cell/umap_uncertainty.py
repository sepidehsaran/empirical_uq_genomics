"""UMAP of the single-cell test cells colored by uncertainty score.

Panel a: the 28,214 selected cells in the Liam-embedding UMAP, test cells of one seed colored
by cell type, the rest in gray. Panels b-d: the same test cells colored by log10 of each
method's SD at the predicted class (balanced setting, one seed); misclassified cells are
drawn with a black edge. Cells are matched to UMAP coordinates by exact row match of the
20-D embedding (same mapping as donor_breakdown.py).

    UQFIG_BUNDLE=... UQFIG_PALETTE=set2_terracotta python3 scripts/single_cell/umap_uncertainty.py [--seed 7] [--setting bal_single]
"""
import argparse, os, pickle, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "uq_genomics", "figures"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
import config_paths as CP, data_access as DA, style as S   # noqa: E402

FIG = os.environ.get("UQFIG_FIGOUT", os.path.join(ROOT, "out"))
OUT = os.environ.get("UQFIG_OUT", os.path.join(ROOT, "out", "tables"))
TYPES = ["CD14+ Mono", "CD4+ T activated", "Erythroblast", "NK"]
SEL_COLORS = {"CD14+ Mono": "#8c2d5a", "CD4+ T activated": "#4c72b0", "Erythroblast": "#2ca25f", "NK": "#9467bd"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("UQFIG_PATH_DATASETS_GEX", ""), help="dataset root (UQFIG_PATH_DATASETS_GEX)")
    ap.add_argument("--umap", default=os.environ.get("UQFIG_UMAP_COORDS_4CT", ""), help="cached UMAP coordinates (UQFIG_UMAP_COORDS_4CT)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--setting", default="bal_single")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    S.apply_rc() if hasattr(S, "apply_rc") else None

    emb = pickle.load(open(os.path.join(a.data, "gex", "liam_embeddings", "liam_seed_0_allsd_df_gex_embedding_subset.pkl"), "rb"))
    X = emb.drop(columns=["cell_type"]).to_numpy(dtype=np.float32).round(5)
    key = {tuple(r): i for i, r in enumerate(X)}
    xy, lab = pickle.load(open(a.umap, "rb")); xy = np.asarray(xy); lab = np.asarray(lab).astype(str)
    assert len(xy) == len(X) and (lab == emb.cell_type.values.astype(str)).all(), "UMAP rows do not line up with the embedding"

    # the balanced split's test set is the test set of both settings
    d = pickle.load(open(os.path.join(a.data, "gex", "real_human", "balanced", "single_label", f"4class_split{a.seed}.pickle"), "rb"))
    xt = np.asarray(d["x_test"], dtype=np.float32).round(5); y = np.asarray(d["y_test"]).argmax(1)
    idx = np.array([key[tuple(r)] for r in xt]); txy = xy[idx]
    comp = DA.load_comparison("gex")

    fig, axes = plt.subplots(1, 4, figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * 0.34), dpi=S.DPI, layout=None)
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.24, wspace=0.14)
    rng = np.random.default_rng(0); order = rng.permutation(len(idx))
    for ax in axes:
        ax.scatter(xy[:, 0], xy[:, 1], c="0.88", s=0.3, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[0]
    ax.scatter(txy[order, 0], txy[order, 1], c=[SEL_COLORS[TYPES[c]] for c in y[order]], s=1.2, linewidths=0, rasterized=True)
    ax.set_title("a) cell type", fontsize=S.FS_LABEL, loc="left", fontweight="normal")
    h = [Line2D([0], [0], marker="o", color=SEL_COLORS[t], lw=0, ms=3.5) for t in TYPES]
    ax.legend(h, TYPES, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncols=2, fontsize=S.FS_LEG - 1.5, frameon=False,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.0)
    rows = []
    for ax, model, letter in zip(axes[1:], CP.MODELS, "bcd"):
        k = DA.dict_key("gex", a.seed, a.setting, model)
        mean, sd = np.asarray(comp["mean"][k]), np.asarray(comp["sd"][k]); pred = mean.argmax(1)
        score = sd[np.arange(len(pred)), pred]; wrong = pred != y
        lo = np.log10(np.clip(score, 1e-8, None))
        vmin, vmax = np.percentile(lo, [2, 98])
        sc = ax.scatter(txy[order, 0], txy[order, 1], c=lo[order], cmap="viridis", vmin=vmin, vmax=vmax, s=1.2, linewidths=0, rasterized=True)
        ax.scatter(txy[wrong, 0], txy[wrong, 1], facecolors="none", edgecolors="k", s=6, linewidths=0.4, rasterized=True)
        ax.set_title(f"{letter}) {model.upper()}", fontsize=S.FS_LABEL, loc="left", fontweight="normal")
        bb = ax.get_position()
        cax = fig.add_axes([bb.x0 + 0.08 * bb.width, bb.y0 - 0.10, 0.84 * bb.width, 0.035])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label(r"$\log_{10}$ SD, predicted class", fontsize=S.FS_TICK - 2); cb.ax.tick_params(labelsize=S.FS_TICK - 1.5)
        # where are the uncertain cells? fraction of top-10% uncertain cells that are misclassified, and per type
        top = score >= np.quantile(score, 0.9)
        rows.append(dict(model=model.upper(), seed=a.seed, setting=a.setting, n_wrong=int(wrong.sum()),
                         frac_wrong_in_top10=float(wrong[top].mean()), frac_wrong_overall=float(wrong.mean()),
                         **{f"top10_{t}": float((y[top] == c).mean()) for c, t in enumerate(TYPES)}))
    stem = f"UMAP_test_uncertainty_{a.setting}_seed{a.seed}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), dpi=S.DPI, bbox_inches="tight", pad_inches=0.02)
    # the same enrichment over all five seeds
    for seed in CP.VERSIONS["gex"]["seeds"]:
        if seed == a.seed:
            continue
        dd = pickle.load(open(os.path.join(a.data, "gex", "real_human", "balanced", "single_label", f"4class_split{seed}.pickle"), "rb"))
        yy = np.asarray(dd["y_test"]).argmax(1)
        for model in CP.MODELS:
            k = DA.dict_key("gex", seed, a.setting, model)
            mean, sd = np.asarray(comp["mean"][k]), np.asarray(comp["sd"][k]); pred = mean.argmax(1)
            score = sd[np.arange(len(pred)), pred]; wrong = pred != yy; top = score >= np.quantile(score, 0.9)
            rows.append(dict(model=model.upper(), seed=seed, setting=a.setting, n_wrong=int(wrong.sum()),
                             frac_wrong_in_top10=float(wrong[top].mean()), frac_wrong_overall=float(wrong.mean()),
                             **{f"top10_{t}": float((yy[top] == c).mean()) for c, t in enumerate(TYPES)}))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, f"umap_uncertainty_{a.setting}.csv"), index=False)
    g = df.groupby("model")[["frac_wrong_in_top10", "frac_wrong_overall"] + [f"top10_{t}" for t in TYPES]].mean()
    g["enrichment"] = g.frac_wrong_in_top10 / g.frac_wrong_overall
    print(g.round(3).to_string()); print(f"figure -> {FIG}/{stem}.pdf")


if __name__ == "__main__":
    main()
