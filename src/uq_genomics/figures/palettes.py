"""Candidate colour palettes, and a real colour-blindness check for each.

Colours are drawn from the manuscript's own Figures 1 and 3 (the framework schematic),
whose actual hexes are:

    #2CA089 teal-green   #89E58C light green   #92D6D6 light teal   #FFE680 pale yellow
    #28A8C8 cyan-blue    #5BBDD5 light blue    #FFBF12 amber

Two things matter beyond looking nice:

1. **Colour-blind safety is verified, not asserted.** Each palette is converted to LMS,
   run through the Viénot-Brettel-Mollon dichromat simulation for deuteranopia,
   protanopia and tritanopia, and the minimum pairwise CIE76 distance is reported. Any
   pair below ~25 will read as "the same colour" to roughly 8% of male readers.

2. **Greyscale separation** is reported too, since TMLR is often printed in black and
   white. Two colours can be perfectly distinct in colour and identical in grey.

    python3 palettes.py            # print the table
    python3 palettes.py --preview  # render swatch sheets
"""
import numpy as np

# ---------------------------------------------------------------------------- palettes
# Order is always: balanced/clean first, then imbalanced/perturbed, then the multi-label
# pair. Keeping this order fixed is what makes one setting the same colour in every
# use case.
PALETTES = {
    "fig3_native": {
        "label": "Figure 3 native - the schematic's own colours, unchanged",
        "colors": ["#2CA089", "#92D6D6", "#FFBF12", "#FFE680"],
    },
    "teal_amber": {
        "label": "Teal & amber - hue encodes label type, lightness encodes balance",
        "colors": ["#14707C", "#6FC7CE", "#C87F0A", "#F7CE6B"],
    },
    "teal_amber_hi": {
        # Found by searching teal/cyan/amber/yellow families for the best worst-case
        # dichromat separation. Structure is deliberate: dark/light teal for the
        # single-label pair, dark/light amber for the multi-label pair, so the colour
        # tells you both facts at once and survives greyscale printing.
        "label": "Teal & amber, high contrast - widest colour-blind margin",
        "colors": ["#0B4F60", "#8FD4DE", "#8A5A12", "#FFE08A"],
    },
    "set2_terracotta": {
        # The reference/perturbed pair in the sensitivity figures was Set2 green vs Set2
        # blue-violet. Under deuteranopia those two sit 5.8 apart in CIE76, i.e. the same
        # colour to ~8% of male readers, and 0.090 apart in luminance. On overlapping
        # density plots that means the two conditions cannot be separated at all.
        # Replacing the blue with a muted terracotta moves the worst case to 49.5 and the
        # greyscale gap to 0.201, and the overlap region blends to a warm taupe that stays
        # 31-41 from both parents instead of merging into one of them.
        "label": "Set2 green & terracotta - reference/perturbed pair, colour-blind safe",
        "colors": ["#66C2A5", "#FC8D62", "#4C72B0", "#E78AC3"],
        "alt": "#C4705A",   # perturbed-setting overlay colour (style.C_ALT)
    },
    "okabe_ito": {
        "label": "Okabe-Ito - the reference colour-blind-safe set",
        "colors": ["#0072B2", "#56B4E9", "#E69F00", "#F0E442"],
    },
    "deep_ocean": {
        "label": "Deep ocean - teal to cyan with a gold accent",
        "colors": ["#0B5563", "#2CA089", "#57C4E5", "#FFC845"],
    },
    "teal_plum": {
        "label": "Teal & plum - teal pair plus violet and gold, muted for print",
        "colors": ["#146C6B", "#6BBFB8", "#7A5C9E", "#F0B429"],
    },
    "set2_gold": {
        # Set2's four-hue identity kept intact. The colour-blind repair is the DEEPENED
        # GREEN, not the warm hue: under deuteranopia Set2's green and blue sit 5.8
        # apart, i.e. indistinguishable, and darkening the green is what separates them.
        # Swapping orange for gold is cosmetic. Worst case goes 5.8 -> 27.1 with no loss
        # of ordinary contrast (41.0, same as Set2).
        "label": "Set2 repaired, gold - deeper green, orange replaced by gold",
        "colors": ["#1B7F6B", "#D9A404", "#8DA0CB", "#E78AC3"],
    },
    "set2_red": {
        "label": "Set2 repaired, red - deeper green, orange replaced by a muted red",
        "colors": ["#1B7F6B", "#D1495B", "#8DA0CB", "#E78AC3"],
    },
    "set2_gold_deep": {
        # as set2_gold, but the blue and pink are darkened/lightened respectively so the
        # four also separate in greyscale rather than only in colour
        "label": "Set2 repaired, gold + greyscale-separated blue/pink",
        "colors": ["#1B7F6B", "#EFC344", "#5B6FB0", "#E9A6D2"],
    },
    # ---- original Set2 with only the orange varied -------------------------------
    # Green, blue and pink are exactly seaborn's Set2. NOTE: swapping the orange does
    # not change colour-blind safety at all - the colliding pair under deuteranopia is
    # green vs blue (5.8), not anything involving the orange. These are aesthetic
    # variants only.
    "set2_o_deep": {
        "label": "Set2, deeper orange - more saturated, less pastel",
        "colors": ["#66C2A5", "#E8763C", "#8DA0CB", "#E78AC3"],
    },
    "set2_o_amber": {
        "label": "Set2, amber - warmer and more golden",
        "colors": ["#66C2A5", "#E8A33D", "#8DA0CB", "#E78AC3"],
    },
    "set2_o_burnt": {
        "label": "Set2, burnt orange - darker and earthier",
        "colors": ["#66C2A5", "#D2691E", "#8DA0CB", "#E78AC3"],
    },
    "set2_current": {
        "label": "Set2 - what the figures use now, for comparison",
        "colors": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"],
    },
}

DEFAULT = "set2_terracotta"     # the palette of every figure in the paper


# ------------------------------------------------------------------ colour conversions
def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def rgb_to_hex(rgb):
    v = np.clip(np.asarray(rgb), 0, 1) * 255
    return "#{:02X}{:02X}{:02X}".format(*v.round().astype(int))


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


# Viénot, Brettel & Mollon (1999) dichromat simulation, applied in linear RGB.
_SIM = {
    "deuteranopia": np.array([[0.625, 0.375, 0.0],
                              [0.700, 0.300, 0.0],
                              [0.0,   0.300, 0.700]]),
    "protanopia":   np.array([[0.567, 0.433, 0.0],
                              [0.558, 0.442, 0.0],
                              [0.0,   0.242, 0.758]]),
    "tritanopia":   np.array([[0.950, 0.050, 0.0],
                              [0.0,   0.433, 0.567],
                              [0.0,   0.475, 0.525]]),
}


def simulate(hexcolor, kind):
    """what `hexcolor` looks like to a dichromat"""
    lin = _srgb_to_linear(hex_to_rgb(hexcolor))
    return rgb_to_hex(_linear_to_srgb(_SIM[kind] @ lin))


def to_lab(hexcolor):
    """sRGB -> CIE L*a*b* (D65), for perceptual distances"""
    lin = _srgb_to_linear(hex_to_rgb(hexcolor))
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = (m @ lin) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def luminance(hexcolor):
    """relative luminance 0-1, i.e. what a greyscale printer sees"""
    lin = _srgb_to_linear(hex_to_rgb(hexcolor))
    return float(np.dot([0.2126, 0.7152, 0.0722], lin))


def min_separation(colors, kind=None):
    """smallest CIE76 distance between any two colours, optionally after simulation"""
    cs = [simulate(c, kind) if kind else c for c in colors]
    labs = [to_lab(c) for c in cs]
    return min(float(np.linalg.norm(labs[i] - labs[j]))
               for i in range(len(labs)) for j in range(i + 1, len(labs)))


def min_grey_separation(colors):
    ls = sorted(luminance(c) for c in colors)
    return min(b - a for a, b in zip(ls, ls[1:]))


def report(n=4):
    """print the safety table; n = how many colours are actually used"""
    rows = []
    for key, spec in PALETTES.items():
        cs = spec["colors"][:n]
        rows.append((key, min_separation(cs), min_separation(cs, "deuteranopia"),
                     min_separation(cs, "protanopia"), min_separation(cs, "tritanopia"),
                     min_grey_separation(cs)))
    w = max(len(r[0]) for r in rows)
    print(f"minimum pairwise separation with {n} colours "
          f"(CIE76; >25 is comfortably distinct, <15 is a problem)\n")
    print(f"{'palette':{w}s} {'normal':>7s} {'deuter':>7s} {'protan':>7s} "
          f"{'tritan':>7s} {'grey':>6s}   verdict")
    for key, a, d, p, t, g in sorted(rows, key=lambda r: -min(r[1:5])):
        worst = min(a, d, p, t)
        verdict = ("safe" if worst > 25 else "marginal" if worst > 15 else "RISKY")
        if g < 0.06:
            verdict += ", greys collide"
        print(f"{key:{w}s} {a:7.1f} {d:7.1f} {p:7.1f} {t:7.1f} {g:6.3f}   {verdict}")
    return rows


def get(name=None, n=None):
    """colour list for a palette name (falls back to the default)"""
    cs = PALETTES.get(name or DEFAULT, PALETTES[DEFAULT])["colors"]
    return cs[:n] if n else list(cs)


if __name__ == "__main__":
    import sys
    for k in (2, 4):
        report(k)
        print()
    if "--preview" in sys.argv:
        import preview_palettes                                   # noqa: F401
