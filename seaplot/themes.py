"""Themes — set_theme / set_style / set_context / despine (seaborn-compatible)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

_STYLE_DEFAULTS = {
    "darkgrid": {"axes.facecolor": "#EAEAF2", "axes.edgecolor": "white",
                 "axes.grid": True, "grid.color": "white", "grid.linestyle": "-",
                 "axes.axisbelow": True},
    "whitegrid": {"axes.facecolor": "white", "axes.edgecolor": "#CCCCCC",
                  "axes.grid": True, "grid.color": "#EAEAEA", "grid.linestyle": "-",
                  "axes.axisbelow": True},
    "dark": {"axes.facecolor": "#212121", "axes.edgecolor": "white",
             "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white",
             "text.color": "white", "axes.grid": False},
    "white": {"axes.facecolor": "white", "axes.grid": False},
    "ticks": {"axes.facecolor": "white", "axes.grid": False},
}
_CONTEXT_DEFAULTS = {
    "paper": {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
              "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
              "lines.linewidth": 1.2, "lines.markersize": 5},
    "notebook": {"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
                 "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                 "lines.linewidth": 1.5, "lines.markersize": 6},
    "talk": {"font.size": 14, "axes.labelsize": 14, "axes.titlesize": 16,
             "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
             "lines.linewidth": 2.0, "lines.markersize": 8},
    "poster": {"font.size": 17, "axes.labelsize": 17, "axes.titlesize": 20,
               "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
               "lines.linewidth": 2.5, "lines.markersize": 10},
}


def axes_style(style="darkgrid", rc=None):
    base = dict(_STYLE_DEFAULTS.get(style, _STYLE_DEFAULTS["darkgrid"]))
    if rc:
        base.update(rc)
    return base


def plotting_context(context="notebook", font_scale=1.0, rc=None):
    base = dict(_CONTEXT_DEFAULTS.get(context, _CONTEXT_DEFAULTS["notebook"]))
    for k in [kk for kk in base if "size" in kk]:
        base[k] = base[k] * font_scale
    if rc:
        base.update(rc)
    return base


def set_style(style="darkgrid", rc=None):
    mpl.rcParams.update(axes_style(style, rc))


def set_context(context="notebook", font_scale=1.0, rc=None):
    mpl.rcParams.update(plotting_context(context, font_scale, rc))


def set_theme(context="notebook", style="darkgrid", palette="seaplot", font="sans-serif",
              font_scale=1.0, color_codes=True, rc=None):
    from .palettes import set_palette
    set_style(style, rc=None)
    set_context(context, font_scale, rc=None)
    set_palette(palette, color_codes=color_codes)
    mpl.rcParams["font.family"] = font
    if rc:
        mpl.rcParams.update(rc)


# seaborn aliases
set = set_theme


def reset_defaults():
    mpl.rcdefaults()
    from .palettes import set_palette
    try:
        set_palette("seaplot")
    except Exception:
        pass


def despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False,
            offset=None, trim=False):
    fig = fig or plt.gcf()
    axes = [ax] if ax is not None else fig.axes
    for a in axes:
        if top:
            a.spines["top"].set_visible(False)
        if right:
            a.spines["right"].set_visible(False)
        if left:
            a.spines["left"].set_visible(False)
        if bottom:
            a.spines["bottom"].set_visible(False)
        if offset is not None:
            for side in (["top", "right", "left", "bottom"]):
                try:
                    a.spines[side].set_position(("outward", offset))
                except Exception:
                    pass
        if trim:
            try:
                a.margins(x=0.02)
            except Exception:
                pass
