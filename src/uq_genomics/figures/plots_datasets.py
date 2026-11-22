"""
plots_datasets.py - dataset-description figures (Section 2 of the manuscript).

These only need the dataset pickles, which are all present locally, so this module
runs end to end with no cluster access.

Covers:  class distribution (TF / RBP / GEX / 27-RBP / MNIST), train-vs-test split
distribution, UMAP of the Liam embedding, and pass-through of the motif / framework
panels that cannot be re-plotted from data.
"""
import os
import shutil
import numpy as np
import config_paths as CP
import data_access as DA
import style as S


def _bar_class_dist(ax, counts, class_names, ylabel="number of samples", labels=True):
    x = np.arange(len(counts))
    bars = ax.bar(x, counts, color=S.C_REF, width=0.66)
    if labels:
        ax.bar_label(bars, fmt="%d", padding=2, fontsize=S.FS_LEG - 1.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right",
                       fontsize=S.FS_TICK - (1.0 if max(map(len, class_names)) > 8 else 0))
    ax.set_ylabel(ylabel)
    ax.margins(y=0.26)


def class_distributions(out, seed=7):
    """one figure per use case x balance setting"""
    res = []
    jobs = [
        ("tf",  "simulated_human", 6, dict(balanced=True,  noise="none"),      "TF (simulated), balanced"),
        ("tf",  "simulated_human", 6, dict(balanced=False, noise="none"),      "TF (simulated), imbalanced"),
        ("rbp", "real_human",      3, dict(balanced=True,  multi_label=False), "RBP (real), balanced single-label"),
        ("rbp", "real_human",      3, dict(balanced=False, multi_label=False), "RBP (real), imbalanced single-label"),
        ("rbp", "real_human",      3, dict(balanced=True,  multi_label=True),  "RBP (real), balanced multi-label"),
        ("rbp", "real_human",      3, dict(balanced=False, multi_label=True),  "RBP (real), imbalanced multi-label"),
        ("gex", "real_human",      4, dict(balanced=True),                     "single-cell GEX, balanced"),
        ("gex", "real_human",      4, dict(balanced=False),                    "single-cell GEX, imbalanced"),
    ]
    for task, species, ncl, kw, title in jobs:
        fid = f"dataset_classdist_{task}_{'bal' if kw.get('balanced', True) else 'imbal'}" \
              f"{'_multi' if kw.get('multi_label') else ''}"
        d = DA.load_split(task, species, ncl, seed, **kw)
        if d is None:
            # laptop path: class-distribution bars need column sums only, and the
            # bundle stores exactly those under (task, sorted(kw), seed).
            cd = DA.bundle_part("classdist")
            key = (task, tuple(sorted(kw.items())), seed)
            if cd and key in cd:
                d = {k: v for k, v in cd[key].items() if not k.startswith("_")}
            else:
                res.append(dict(id=fid, status="skip",
                                reason=f"dataset pickle not found ({task}, seed {seed})"))
                continue
        names = CP.VERSIONS[task]["class_names"]
        # full text width: two panels of 6 rotated class names do not fit at 0.48
        fig, axes = S.new_fig(1.0, aspect=0.40, ncols=2)
        for ax, key, sub in ((axes[0], "y_train", "train"), (axes[1], "y_test", "test")):
            arr = np.asarray(d[key])
            counts = arr if arr.ndim == 1 else arr.sum(axis=0)   # bundle stores column sums
            _bar_class_dist(ax, counts, names)
            ax.set_title(sub, fontsize=S.FS_LABEL)
        S.fig_suptitle(fig, None, f"Class distribution - {title} (seed {seed})")
        p = os.path.join(out, f"{fid}.png")
        S.save(fig, p)
        res.append(dict(id=fid, status="ok", path=p))
    return res


def _read_adata_umap():
    """
    Pull (xy, labels, error) out of an .h5ad.

    LAPTOP PATH FIRST: if the bundle carries exported UMAP coordinates, they are used
    and the .h5ad is never opened.

    The files on the cluster do NOT have obsm['X_umap'] - they carry 'GEX_X_umap'
    (and 'ATAC_umap'). Cell-type labels likewise live under different column names in
    different exports. Both are therefore searched by preference order rather than
    assumed, and the file is read BACKED so the ~6 GB matrix never enters memory.
    """
    import numpy as _np
    _b = DA.bundle_part("umap")
    if _b is not None:
        return _np.asarray(_b["xy"]), _np.asarray(_b["labels"]), ""
    if not CP.ADATA_H5AD or not os.path.exists(CP.ADATA_H5AD):
        return None, None, "no adata.h5ad (ADATA_H5AD)"
    try:
        import anndata as ad
    except ImportError:
        return None, None, "anndata not installed"
    try:
        a = ad.read_h5ad(CP.ADATA_H5AD, backed="r")
    except Exception as exc:
        return None, None, f"cannot open adata: {exc}"

    umap_keys = ["X_umap", "GEX_X_umap", "umap", "X_UMAP"]
    key = next((k for k in umap_keys if k in a.obsm), None)
    if key is None:
        key = next((k for k in a.obsm if "umap" in k.lower()
                    and "atac" not in k.lower()), None)
    if key is None:
        return None, None, (f"no UMAP in obsm; available: "
                            f"{[k for k in list(a.obsm)[:8]]}")
    xy = _np.asarray(a.obsm[key])

    ct_cols = ["cell_type", "celltype", "cell_types", "CellType",
               "cell_type_l1", "cell_type_l2", "GEX_cell_type", "label"]
    col = next((c for c in ct_cols if c in a.obs.columns), None)
    if col is None:
        col = next((c for c in a.obs.columns
                    if "cell" in c.lower() and "type" in c.lower()), None)
    if col is None:
        return None, None, (f"no cell-type column in obs; available: "
                            f"{list(a.obs.columns)[:10]}")
    labels = a.obs[col].astype(str).values
    print(f"    [adata] umap='{key}'  labels='{col}'  n={len(xy)}")
    return xy, _np.asarray(labels), ""


def gex_umap(out):
    """
    UMAP of the Liam embedding: full dataset (4 selected cell types highlighted) and
    the 4 selected alone.

    Prefers the cached coordinate pickles (fast, no 6 GB read); falls back to the
    .h5ad, read backed.
    """
    import pickle
    res = []
    sel = CP.VERSIONS["gex"]["class_names"]

    xy_full = lab_full = xy_4 = lab_4 = None
    src_note = ""

    if CP.UMAP_COORDS_FULL and os.path.exists(CP.UMAP_COORDS_FULL):
        with open(CP.UMAP_COORDS_FULL, "rb") as fh:
            d = pickle.load(fh)
        for k in ("X_umap", "GEX_X_umap"):
            if k in d:
                xy_full = np.asarray(d[k]); break
        if "labels" in d:
            lab_full = np.asarray(d["labels"]).astype(str)
        src_note = "cached coords"
    if CP.UMAP_COORDS_4CT and os.path.exists(CP.UMAP_COORDS_4CT):
        with open(CP.UMAP_COORDS_4CT, "rb") as fh:
            t = pickle.load(fh)
        xy_4, lab_4 = np.asarray(t[0]), np.asarray(t[1]).astype(str)

    if xy_full is None:
        xy_full, lab_full, err = _read_adata_umap()
        src_note = "adata.h5ad"
        if xy_full is None and xy_4 is None:
            return [dict(id="dataset_gex_umap", status="skip",
                         reason=f"no cached coords and {err}")]

    if xy_full is not None and lab_full is not None:
        fid = "dataset_gex_umap_full"
        fig, ax = S.new_fig(0.55, aspect=0.92)
        other = ~np.isin(lab_full, sel)
        ax.scatter(xy_full[other, 0], xy_full[other, 1], s=0.5, c="#dcdcdc",
                   linewidths=0, rasterized=True, label="other cell types")
        n_hit = 0
        for name in sel:
            m = lab_full == name
            n_hit += int(m.sum())
            if m.sum():
                ax.scatter(xy_full[m, 0], xy_full[m, 1], s=0.8, linewidths=0,
                           rasterized=True, label=name)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=8, loc="best", fontsize=S.FS_LEG - 1)
        S.fig_suptitle(fig, None, "Single-cell GEX: full dataset in the Liam embedding")
        note = f"{len(xy_full)} cells, {n_hit} in the 4 selected ({src_note})"
        if n_hit == 0:
            note += f" !! no label matched {sel}; found e.g. {sorted(set(lab_full))[:6]}"
        res.append(dict(id=fid, status="ok", note=note,
                        path=S.save(fig, os.path.join(out, f"{fid}.png"))))

    if xy_4 is None and xy_full is not None and lab_full is not None:
        keep = np.isin(lab_full, sel)
        if keep.sum():
            xy_4, lab_4 = xy_full[keep], lab_full[keep]

    if xy_4 is not None:
        fid = "dataset_gex_umap_4celltypes"
        fig, ax = S.new_fig(0.55, aspect=0.92)
        for name in sorted(set(lab_4)):
            m = lab_4 == name
            ax.scatter(xy_4[m, 0], xy_4[m, 1], s=1.0, linewidths=0, rasterized=True,
                       label=name)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=8, loc="best", fontsize=S.FS_LEG - 1)
        S.fig_suptitle(fig, None, "Single-cell GEX: the four selected cell types")
        res.append(dict(id=fid, status="ok", note=f"{len(xy_4)} cells",
                        path=S.save(fig, os.path.join(out, f"{fid}.png"))))
    return res


def mnist_class_distribution(out):
    """MNIST imbalanced class distribution (supplement); regenerated from the
    sub-sampling fractions stated in the supplement text."""
    fid = "dataset_mnist_imbalance"
    FRAC = [0.5, 0.7, 0.8, 0.3, 0.6, 0.8, 0.4, 0.2, 0.9, 0.8]
    # MNIST train-set class counts are fixed constants of the dataset, so no download
    # is needed; if keras happens to be installed we read them from it instead.
    MNIST_TRAIN = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
    try:
        from tensorflow.keras.datasets import mnist
        (_, y_tr), _ = mnist.load_data()
        base = [int((y_tr == i).sum()) for i in range(10)]
    except Exception:
        base = MNIST_TRAIN
    counts = np.array([int(base[i] * FRAC[i]) for i in range(10)])
    fig, ax = S.new_fig(0.62, aspect=0.60)
    _bar_class_dist(ax, counts, [str(i) for i in range(10)])
    ax.set_xlabel("digit class")
    S.fig_suptitle(fig, None, "MNIST: class sizes after sub-sampling (imbalanced variant)")
    p = os.path.join(out, f"{fid}.png")
    S.save(fig, p)
    return [dict(id=fid, status="ok", path=p)]


def app27_class_distribution(out):
    fid = "dataset_27rbp_classdist"
    counts, names = None, None
    b = DA.bundle_part("app27")                       # laptop path: counts, not the 96 MB
    if b is not None and b.get("train_counts") is not None:
        counts = np.asarray(b["train_counts"])
        names = b.get("class_names")
    elif CP.APP27_DATASET and os.path.exists(CP.APP27_DATASET):
        import pickle
        with open(CP.APP27_DATASET, "rb") as fh:
            d = pickle.load(fh)
        y = np.asarray(d.get("y_train"))
        if y is None:
            return [dict(id=fid, status="skip", reason="no y_train in 27-class pickle")]
        counts = y.sum(axis=0)
        names = d.get("class_names")
    else:
        return [dict(id=fid, status="skip",
                     reason="27-class dataset pickle not found (APP27_DATASET)")]
    names = names or [f"c{i}" for i in range(len(counts))]
    fig, ax = S.new_fig(1.0, aspect=0.42)
    _bar_class_dist(ax, counts, names)
    ax.set_xlabel("RBP class")
    S.fig_suptitle(fig, None, "27-RBP application: class sizes")
    p = os.path.join(out, f"{fid}.png")
    S.save(fig, p)
    return [dict(id=fid, status="ok", path=p)]


def passthrough(out):
    """
    Copy figures that are images rather than data (motif panels, framework overview,
    XAI panel). They cannot be re-plotted here: the motif panels need the JASPAR PFMs
    and the DeepRiPe attribution maps, the overview is a hand-drawn diagram.
    """
    res = []
    if not CP.FIGDIR_EXISTING:
        return [dict(id="passthrough", status="skip",
                     reason="existing figures dir not found (FIGDIR_EXISTING)")]
    for fid, rel in CP.PASSTHROUGH.items():
        src = os.path.join(CP.FIGDIR_EXISTING, rel)
        if not os.path.exists(src):
            res.append(dict(id=f"passthrough_{fid}", status="skip",
                            reason=f"source image missing: {rel}"))
            continue
        dst = os.path.join(out, f"passthrough_{fid}.png")
        shutil.copy2(src, dst)
        res.append(dict(id=f"passthrough_{fid}", status="ok", path=dst,
                        note="copied as-is (not re-plotted)"))
    return res


def run(out):
    r = []
    r += class_distributions(out)
    r += gex_umap(out)
    r += mnist_class_distribution(out)
    r += app27_class_distribution(out)
    r += passthrough(out)
    return r


def tasks(out):
    return [
        ("class distributions",      lambda: class_distributions(out)),
        ("GEX UMAP",                 lambda: gex_umap(out)),
        ("MNIST class distribution", lambda: mnist_class_distribution(out)),
        ("27-RBP class sizes",       lambda: app27_class_distribution(out)),
        ("passthrough panels",       lambda: passthrough(out)),
    ]
