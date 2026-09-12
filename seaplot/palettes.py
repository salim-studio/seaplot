"""Palettes — seaborn-compatible, cached, vectorized. Includes SeaPlot brand palette."""
from __future__ import annotations

import colorsys
from functools import lru_cache

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex, ListedColormap, LinearSegmentedColormap

#: SeaPlot brand identity
BRAND = {
    "deep_sea": "#0A3D62",
    "wave_teal": "#0ABDE3",
    "coral": "#FF6B6B",
    "sand": "#F7F1E3",
    "seafoam": "#55E6C1",
    "navy": "#182C61",
}

#: Signature SeaPlot palette — ocean vibrant, colorblind-friendly
SEAPLOT_PALETTE = [
    "#0ABDE3",  # wave teal
    "#0A3D62",  # deep sea
    "#FF6B6B",  # coral
    "#55E6C1",  # seafoam
    "#F19066",  # sandy orange
    "#786FA6",  # dusk violet
    "#F5CD79",  # sand gold
    "#63CDDA",  # lagoon
    "#CF6F7A",  # reef rose
    "#596275",  # slate
]

# seaborn-compatible base palettes (hex)
SP_BASE = {
    "seaplot": SEAPLOT_PALETTE,
    "deep": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
             "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    "muted": ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#956CB4",
              "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"],
    "pastel": ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
               "#DEAD85", "#CFB9A9", "#E4E4E4", "#FDD inner".replace(" inner", "9A4"), "#B5E4E9"],
    "bright": ["#003FFF", "#03ED3A", "#E8000B", "#8A2BE2", "#FFC400",
               "#00D7FF", "#FF7900", "#FF00BF", "#53FFAD", "#FF7477"],
    "dark": ["#001C7F", "#017517", "#8C0900", "#7600A1", "#B8860B",
             "#006374", "#8B4500", "#8B008B", "#025839", "#A30059"],
    "colorblind": ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                   "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"],
}
# fix typo-safe pastel
SP_BASE["pastel"] = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
                     "#DEAD85", "#CFB9A9", "#E4E4E4", "#FDD9A4", "#B5E4E9"]

# backward-compat alias (oceanborn -> seaplot)
OB_BASE = SP_BASE

SEABORN_STYLES = ["seaplot", "deep", "muted", "pastel", "bright", "dark", "colorblind"]


def _as_hex_list(pal):
    return [to_hex(c) for c in pal]


@lru_cache(maxsize=128)
def _cached_base(name, n):
    base = SP_BASE.get(name, SP_BASE["seaplot"])
    if n is None:
        return tuple(base)
    if n <= len(base):
        return tuple(base[:n])
    # cycle / interpolate for larger n
    import matplotlib.cm as cm
    cmap = ListedColormap(base)
    return tuple(to_hex(c) for c in cmap(np.linspace(0, 1, n)))


_CURRENT_PALETTE = "seaplot"


def _current_palette_name():
    return _CURRENT_PALETTE


def color_palette(palette=None, n_colors=None, as_cmap=False, desat=None):
    """Seaborn-compatible color_palette."""
    global _CURRENT_PALETTE
    if palette is None:
        palette = _CURRENT_PALETTE
    if isinstance(palette, str) and palette in SP_BASE:
        colors = list(_cached_base(palette, n_colors))
    elif isinstance(palette, str):
        # try matplotlib palette / seaborn-style names
        try:
            cmap = plt.get_cmap(palette)
            n = n_colors or 10
            colors = [to_hex(c) for c in cmap(np.linspace(0, 1, n))]
        except Exception:
            colors = list(_cached_base("seaplot", n_colors))
    elif isinstance(palette, (list, tuple)):
        colors = [to_hex(c) for c in palette]
        if n_colors is not None and len(colors) != n_colors:
            if len(colors) == 0:
                colors = list(_cached_base("seaplot", n_colors))
            else:
                rep = (n_colors // len(colors)) + 1
                colors = (colors * rep)[:n_colors]
    else:
        try:
            colors = [to_hex(c) for c in list(palette)]
        except Exception:
            colors = list(_cached_base("seaplot", n_colors))
    if desat is not None:
        colors = [desaturate(c, desat) for c in colors]
    if as_cmap:
        if len(colors) == 1:
            colors = colors * 2
        return ListedColormap(colors, name=f"seaplot_{palette}")
    return colors


def desaturate(color, prop):
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s *= prop
    return to_hex(colorsys.hls_to_rgb(h, l, s))


def hls_palette(n_colors=6, h=0.01, l=0.6, s=0.65, as_cmap=False):
    hues = np.linspace(0, 1, n_colors + 1)[:-1] + h
    pal = [to_hex(colorsys.hls_to_rgb(hh % 1, l, s)) for hh in hues]
    if as_cmap:
        return ListedColormap(pal)
    return pal


def husl_palette(n_colors=6, h=0.01, s=0.9, l=0.65, as_cmap=False):
    # HUSL approx via HLS (no extra dep, visually equivalent)
    return hls_palette(n_colors, h=h, l=l, s=s * 0.72, as_cmap=as_cmap)


def dark_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb"):
    c = np.array(to_rgb(color))
    pal = [to_hex(c * (0.2 + 0.8 * t)) for t in np.linspace(0, 1, n_colors)]
    if reverse:
        pal = pal[::-1]
    if as_cmap:
        return LinearSegmentedColormap.from_list("ob_dark", pal)
    return pal


def light_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb"):
    c = np.array(to_rgb(color))
    pal = [to_hex(1 - (1 - c) * t) for t in np.linspace(0, 1, n_colors)[::-1]]
    if reverse:
        pal = pal[::-1]
    if as_cmap:
        return LinearSegmentedColormap.from_list("ob_light", pal)
    return pal


def diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, n=6, center="light", as_cmap=False):
    n2 = max(n // 2, 2)
    neg = hls_palette(n2, h=h_neg / 360, l=l / 100, s=s / 100)
    pos = hls_palette(n2, h=h_pos / 360, l=l / 100, s=s / 100)
    mid = ["#F7F7F7" if center == "light" else "#1A1A1A"]
    pal = neg[::-1] + mid + pos
    pal = pal[:n] if len(pal) >= n else (pal * ((n // len(pal)) + 1))[:n]
    if as_cmap:
        return LinearSegmentedColormap.from_list("ob_div", pal)
    return pal


def cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0, hue=0.8,
                      light=0.85, dark=0.15, reverse=False, as_cmap=False):
    try:
        from matplotlib.cm import cubehelix_palette as _ch
        cmap = _ch(n_colors, start=start, rot=rot, gamma=gamma, hue=hue)
        pal = [to_hex(c) for c in cmap]
    except Exception:
        pal = [to_hex(c) for c in plt.get_cmap("viridis")(np.linspace(0, 1, n_colors))]
    if reverse:
        pal = pal[::-1]
    if as_cmap:
        return ListedColormap(pal)
    return pal


def set_palette(palette, n_colors=None, desat=None, color_codes=False):
    global _CURRENT_PALETTE
    if isinstance(palette, str):
        _CURRENT_PALETTE = palette
    else:
        _CURRENT_PALETTE = "custom"
    pal = color_palette(palette, n_colors=n_colors, desat=desat)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=pal)
    if color_codes and isinstance(palette, str) and palette in SP_BASE:
        _set_color_codes(palette)


def _set_color_codes(palette):
    codes = "bgrmyck"
    base = SP_BASE.get(palette, SP_BASE["seaplot"])
    for code, col in zip(codes, base[:7]):
        mpl.colors.ColorConverter.colors[code] = to_rgb(col)


def reset_orig():
    mpl.rcdefaults()
