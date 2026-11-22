"""Sequence-logo panel for the RBP and TF motifs used in the simulated data.

Draws the logos from the position weight matrices themselves. Letter heights are
information content in bits. See docs/REPRODUCING.md for the --trim caveat.

    python scripts/simulated_tf/motif_logos_figure.py \
        --rbp-pwms /path/to/rbp_motif_matrix_nonbinary_orig \
        --tf-pwms  /path/to/new_jasper_based \
        --out      out/motifs
"""
import argparse
import os
import sys

import numpy as np

ALPHABET = "ACGU"          # default; TF panels use ACGT
# The colouring used throughout the paper's sequence figures.
BASE_COLOR = {"A": "#3DA02D", "C": "#2B6CB0", "G": "#E8A33D",
              "T": "#D6473E", "U": "#D6473E"}


def read_pwm(path):
    """Read a position weight matrix in either of the two formats used here."""
    lines = [l.strip() for l in open(path).read().strip().split("\n") if l.strip()]
    rows = []
    for l in lines:
        if l.lower().startswith(("pos", "letter-probability", "meme", "alphabet")):
            continue
        parts = [f for f in l.replace(",", " ").split() if f]
        try:
            vals = [float(f) for f in parts]
        except ValueError:
            continue
        if len(vals) == 5:          # leading position index
            vals = vals[1:]
        if len(vals) == 4:
            rows.append(vals)
    if not rows:
        raise ValueError(f"no numeric rows parsed from {path}")
    m = np.clip(np.array(rows), 1e-9, None)
    return m / m.sum(1, keepdims=True)


def information_content(pwm):
    """Bits per position: 2 - H(p) for a four-letter alphabet."""
    h = -(pwm * np.log2(pwm)).sum(1)
    return np.clip(2.0 - h, 0.0, 2.0)


def letter(ax, base, x, y, height, width=0.92):
    """One nucleotide glyph scaled to `height` in data units."""
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    from matplotlib.font_manager import FontProperties
    if height <= 0:
        return
    tp = TextPath((0, 0), base, size=1,
                  prop=FontProperties(family="DejaVu Sans", weight="bold"))
    bb = tp.get_extents()
    if bb.width == 0 or bb.height == 0:
        return
    t = (Affine2D().translate(-bb.x0, -bb.y0)
         .scale(1.0 / bb.width, 1.0 / bb.height)
         .scale(width, height)
         .translate(x + (1 - width) / 2, y))
    ax.add_patch(PathPatch(t.transform_path(tp),
                           facecolor=BASE_COLOR.get(base, "0.4"), edgecolor="none"))


def draw_logo(ax, pwm, title=None, fs_title=8, fs_tick=7, alphabet=ALPHABET):
    ic = information_content(pwm)
    for i in range(len(pwm)):
        order = np.argsort(pwm[i])          # smallest at the bottom, as is conventional
        y = 0.0
        for j in order:
            h = float(pwm[i, j] * ic[i])
            letter(ax, alphabet[j], i, y, h)
            y += h
    ax.set_xlim(0, len(pwm))
    ax.set_ylim(0, 2.02)
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=fs_tick)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=fs_tick - 0.5)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=fs_title, loc="left", pad=2)


def load(root, names, alphabet, label):
    import glob
    out = {}
    print(f"{label}:")
    for r in names:
        files = sorted(glob.glob(os.path.join(root, r, "*.txt")))
        if not files:
            print(f"  warning: no PWM for {r}")
            continue
        m = read_pwm(files[0])
        out[r] = m
        # A position with no information is degenerate, and argmax on a tie silently
        # returns the first base. PUM2 position 5 is exactly uniform -- the N of the
        # Pumilio element UGUANAUA -- so it must print as N, not as an arbitrary A.
        ic_pos = information_content(m)
        cons = "".join(alphabet[i] if ic_pos[k] > 0.05 else "N"
                       for k, i in enumerate(m.argmax(1)))
        print(f"  {r.upper():8s} {os.path.basename(files[0]):16s} len {m.shape[0]:2d}  "
              f"consensus {cons:20s} {information_content(m).sum():5.2f} bits")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rbp-pwms", required=True,
                    help="rbp_motif_matrix_nonbinary_orig: one subdirectory per RBP")
    ap.add_argument("--tf-pwms", required=True,
                    help="new_jasper_based: one subdirectory per TF")
    ap.add_argument("--rbps", default="mbnl1,pum2,qki")
    ap.add_argument("--tfs", default="ctcf,hnf4a,jun,mef2a,myc,tal1")
    ap.add_argument("--out", default="out/motifs")
    ap.add_argument("--row-height", type=float, default=0.60,
                    help="inches per logo row")
    # Trimming a motif to fewer positions than the source entry. Only use this for
    # flanks that carry little information; cutting informative positions makes the
    # panel disagree with the database it cites and with what was planted in the data.
    # Format: name:start:end (1-based, inclusive), comma separated.
    ap.add_argument("--trim", default="",
                    help="e.g. pum2:1:4 to show only the first four positions")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    try:
        from uq_genomics.figures import style as S
        S.apply_rc(); W = S.TEXTWIDTH_IN
    except Exception:
        W = 6.5
        matplotlib.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                                    "pdf.fonttype": 42, "font.family": "sans-serif"})

    rbps = [r.strip() for r in args.rbps.split(",")]
    tfs = [r.strip() for r in args.tfs.split(",")]
    A = load(args.rbp_pwms, rbps, "ACGU", "a) RBP motifs (RNA)")
    for spec in [s for s in args.trim.split(",") if s.strip()]:
        nm, a, b = spec.split(":")
        a, b = int(a), int(b)
        for store, alpha in ((A, "ACGU"), (None, None)):
            if store is not None and nm in store:
                full = store[nm]
                lost = information_content(full).sum() - information_content(full[a-1:b]).sum()
                store[nm] = full[a - 1:b]
                print(f"  TRIM {nm.upper()} to positions {a}-{b} of {len(full)}: "
                      f"{lost:.2f} of {information_content(full).sum():.2f} bits removed")
    print()
    B = load(args.tf_pwms, tfs, "ACGT", "b) TF motifs (DNA)")

    ra = [r for r in rbps if r in A]
    rb = [r for r in tfs if r in B]

    # Three columns: RBPs on the left, the TFs split across two columns on the right.
    # With three RBPs and six TFs, a two-column layout leaves the left column half
    # empty; splitting b) balances the block and halves the figure height.
    half = (len(rb) + 1) // 2
    groups = [(ra, A, "ACGU", "a)"), (rb[:half], B, "ACGT", "b)"),
              (rb[half:], B, "ACGT", None)]
    nrow = max(len(g[0]) for g in groups)

    fig = plt.figure(figsize=(W, args.row_height * nrow + 0.35), dpi=300)
    TOP, BOT, VG = 0.905, 0.055, 0.115
    h = (TOP - BOT - VG * (nrow - 1)) / nrow
    LEFTS = [0.055, 0.385, 0.715]
    WID = 0.245
    for (names, store, alpha, tag), L in zip(groups, LEFTS):
        for k, r in enumerate(names):
            ax = fig.add_axes([L, TOP - (k + 1) * h - k * VG, WID, h])
            draw_logo(ax, store[r], title=r.upper(), alphabet=alpha,
                      fs_title=7, fs_tick=6)
            if k == 0 and tag:
                ax.text(-0.21, 1.34, tag, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="top", ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"MOTIFS_rbp_tf.{ext}"), dpi=300)
    plt.close(fig)
    w_in, h_in = fig.get_size_inches()
    print(f"\n  MOTIFS_rbp_tf: {w_in:.2f} x {h_in:.2f} in")
    print(f"  wrote {os.path.join(args.out, 'MOTIFS_rbp_tf.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
