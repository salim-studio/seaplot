"""Distributions: histplot / kdeplot / ecdfplot / rugplot / displot."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._data import get_vector, factorize
from ._fast import gaussian_kde_1d, fast_hist, ecdf_vals, kde_2d
from .palettes import color_palette


def _get_ax(ax):
    return ax if ax is not None else plt.gca()


def histplot(data=None, *, x=None, y=None, hue=None, weights=None, stat="count",
             bins="auto", binwidth=None, binrange=None, discrete=False,
             cumulative=False, common_bins=True, common_norm=True,
             multiple="layer", element="bars", fill=True, shrink=1.0,
             kde=False, kde_kws=None, line_kws=None, color=None, palette="seaplot",
             legend="auto", ax=None, **kwargs):
    ax = _get_ax(ax)
    xv = get_vector(data, x) if x is not None else get_vector(data, y)
    orient = "v" if x is not None else "h"
    if xv is None:
        raise ValueError("histplot requires x or y")
    xv = np.asarray(xv)
    hue_v = get_vector(data, hue) if hue is not None else None
    w = np.asarray(get_vector(data, weights)) if weights is not None and not isinstance(weights, (str, type(None))) else None
    if isinstance(weights, str):
        w = np.asarray(get_vector(data, weights))
    pal = color_palette(palette)

    # bins resolution
    if binwidth is not None:
        try:
            lo = float(np.nanmin(xv.astype(float))); hi = float(np.nanmax(xv.astype(float)))
            bins = max(1, int(np.ceil((hi - lo) / binwidth)))
        except Exception:
            pass
    if discrete:
        try:
            ux = np.unique(xv[np.isfinite(xv.astype(float))] if xv.dtype.kind in "fiu" else xv)
            bins = len(ux)
        except Exception:
            pass

    def _one(v, wv, col, label=None, alpha=0.7):
        v = np.asarray(v)
        try:
            vf = v.astype(float)
            vf = vf[np.isfinite(vf)]
        except Exception:
            cats, codes = factorize(v.astype(object))
            counts = np.bincount(codes)
            pos = np.arange(len(cats))
            ax.bar(pos, counts, color=col, alpha=alpha, label=label, width=0.8 * shrink, **kwargs)
            ax.set_xticks(pos, [str(c) for c in cats])
            return
        counts, edges = fast_hist(vf, bins=bins, range=binrange,
                                  weights=(wv[:len(v)] if wv is not None else None))
        if stat == "density":
            counts = counts / (counts.sum() * np.diff(edges) + 1e-12)
        elif stat == "probability":
            counts = counts / (counts.sum() + 1e-12)
        elif stat == "percent":
            counts = 100 * counts / (counts.sum() + 1e-12)
        if cumulative:
            counts = np.cumsum(counts)
        centers = (edges[:-1] + edges[1:]) / 2
        width = np.diff(edges) * (0.95 * shrink)
        if element == "step":
            ax.stairs(counts, edges, color=col, label=label, fill=fill, **{k: v for k, v in kwargs.items() if k not in ("width",)})
        elif element == "poly":
            ax.plot(centers, counts, color=col, label=label)
            if fill:
                ax.fill_between(centers, counts, alpha=0.25, color=col)
        else:
            ax.bar(centers, counts, width=width, color=col, alpha=alpha if multiple == "layer" else 0.9,
                   label=label)
        if kde:
            g, d = gaussian_kde_1d(vf)
            scale = counts.max() / (d.max() + 1e-12) if stat == "count" else 1.0
            ax.plot(g, d * scale, color=col, **(line_kws or {}))

    if hue_v is None:
        _one(xv, w, color or pal[0], label=None)
    else:
        hue_v = np.asarray(hue_v)
        cats, _ = factorize(hue_v.astype(object))
        subpal = color_palette(palette, n_colors=len(cats))
        # common bins: compute once
        for i, c in enumerate(cats):
            mm = hue_v.astype(object) == c
            _one(np.asarray(xv)[mm], w[mm] if w is not None and len(w) == len(xv) else None,
                 subpal[i % len(subpal)], label=str(c))
        if legend == "auto":
            ax.legend(title=str(hue) if isinstance(hue, str) else None)
    if isinstance(x, str):
        ax.set_xlabel(x)
    if isinstance(y, str):
        ax.set_ylabel(y)
    ax.set_ylabel(stat if stat != "count" else "Count")
    return ax


def kdeplot(data=None, *, x=None, y=None, hue=None, weights=None, palette="seaplot",
            hue_order=None, hue_norm=None, color=None, fill=None, multiple="layer",
            common_norm=True, common_grid=False, cumulative=False, bw_method="silverman",
            bw_adjust=1.0, gridsize=256, cut=3.0, clip=None, legend="auto",
            cbar=False, ax=None, levels=10, thresh=0.05, **kwargs):
    ax = _get_ax(ax)
    pal = color_palette(palette)
    bw = None
    if isinstance(bw_method, (int, float)):
        bw = float(bw_method) * bw_adjust
    xv = get_vector(data, x) if x is not None else None
    yv = get_vector(data, y) if y is not None else None

    def _scale_bw(v):
        if bw is not None:
            return bw
        from ._fast import _silverman_bw
        try:
            return _silverman_bw(np.asarray(v, dtype=float)[np.isfinite(np.asarray(v, dtype=float))]) * bw_adjust
        except Exception:
            return 1.0

    if xv is not None and yv is not None:
        # bivariate: fast smoothed 2d hist -> contour
        xv, yv = np.asarray(xv), np.asarray(yv)
        gx, gy, H = kde_2d(xv, yv, gridsize=min(gridsize, 128))
        col = color or pal[0]
        if fill:
            ax.contourf(gx, gy, H, levels=levels, cmap=kwargs.pop("cmap", None) or "Blues")
        else:
            ax.contour(gx, gy, H, levels=levels, colors=[col], **kwargs)
        if cbar:
            try:
                plt.colorbar(ax.collections[-1], ax=ax)
            except Exception:
                pass
    else:
        v0 = np.asarray(xv if xv is not None else yv)
        orient = "x" if xv is not None else "y"
        hue_v = get_vector(data, hue) if hue is not None else None
        if hue_v is None:
            try:
                vf = v0.astype(float)
            except Exception:
                return ax
            g, d = gaussian_kde_1d(vf, bw=_scale_bw(vf), gridsize=gridsize, cut=cut, clip=clip)
            if cumulative:
                d = np.cumsum(d) / (d.sum() + 1e-12)
            col = color or pal[0]
            if orient == "x":
                ax.plot(g, d, color=col, **kwargs)
                if fill:
                    ax.fill_between(g, d, alpha=0.25, color=col)
            else:
                ax.plot(d, g, color=col, **kwargs)
                if fill:
                    ax.fill_betweenx(g, d, alpha=0.25, color=col)
        else:
            hue_v = np.asarray(hue_v)
            cats, _ = factorize(hue_v.astype(object))
            subpal = color_palette(palette, n_colors=len(cats))
            for i, c in enumerate(cats):
                mm = hue_v.astype(object) == c
                vf = np.asarray(v0)[mm].astype(float)
                vf = vf[np.isfinite(vf)]
                if vf.size == 0:
                    continue
                g, d = gaussian_kde_1d(vf, bw=_scale_bw(vf), gridsize=gridsize, cut=cut, clip=clip)
                if cumulative:
                    d = np.cumsum(d) / (d.sum() + 1e-12)
                if orient == "x":
                    ax.plot(g, d, color=subpal[i % len(subpal)], label=str(c), **kwargs)
                    if fill:
                        ax.fill_between(g, d, alpha=0.2, color=subpal[i % len(subpal)])
                else:
                    ax.plot(d, g, color=subpal[i % len(subpal)], label=str(c), **kwargs)
                    if fill:
                        ax.fill_betweenx(g, d, alpha=0.2, color=subpal[i % len(subpal)])
            if legend == "auto":
                ax.legend(title=str(hue) if isinstance(hue, str) else None)
    if isinstance(x, str):
        ax.set_xlabel(x)
    if isinstance(y, str):
        ax.set_ylabel(y)
    return ax


def ecdfplot(data=None, *, x=None, y=None, hue=None, weights=None, stat="proportion",
             complementary=False, palette="seaplot", legend="auto", color=None, ax=None, **kwargs):
    ax = _get_ax(ax)
    pal = color_palette(palette)
    v0 = get_vector(data, x if x is not None else y)
    if v0 is None:
        raise ValueError("ecdfplot requires x or y")
    v0 = np.asarray(v0)
    hue_v = get_vector(data, hue) if hue is not None else None
    orient = "x" if x is not None else "h"

    def _one(v, col, label=None):
        v = np.asarray(v, dtype=float)
        xs, ys = ecdf_vals(v)
        if stat == "count":
            ys = ys * len(xs)
        if complementary:
            ys = 1 - ys
        if orient == "x":
            ax.plot(xs, ys, color=col, label=label, drawstyle="steps-post", **kwargs)
        else:
            ax.plot(ys, xs, color=col, label=label, drawstyle="steps-post", **kwargs)

    if hue_v is None:
        _one(v0, color or pal[0])
    else:
        hue_v = np.asarray(hue_v)
        cats, _ = factorize(hue_v.astype(object))
        subpal = color_palette(palette, n_colors=len(cats))
        for i, c in enumerate(cats):
            mm = hue_v.astype(object) == c
            _one(np.asarray(v0)[mm], subpal[i % len(subpal)], label=str(c))
        if legend == "auto":
            ax.legend(title=str(hue) if isinstance(hue, str) else None)
    return ax


def rugplot(data=None, *, x=None, y=None, hue=None, height=0.025, expand_margins=True,
            palette="seaplot", color=None, legend="auto", ax=None, **kwargs):
    ax = _get_ax(ax)
    pal = color_palette(palette)
    xv = get_vector(data, x) if x is not None else None
    yv = get_vector(data, y) if y is not None else None
    col = color or pal[0]
    if xv is not None:
        xv = np.asarray(xv, dtype=float)
        xv = xv[np.isfinite(xv)]
        if xv.size > 5000:  # rug of 100k ticks is noise; thin it
            xv = np.random.default_rng(0).choice(xv, 5000, replace=False)
        ymin, ymax = ax.get_ylim() if ax.lines or ax.collections else (0, 1)
        h = height * (ymax - ymin if ymax > ymin else 1)
        y0 = ymin
        ax.vlines(xv, y0, y0 + h, colors=col, linewidths=0.8, **kwargs)
    if yv is not None:
        yv = np.asarray(yv, dtype=float)
        yv = yv[np.isfinite(yv)]
        if yv.size > 5000:
            yv = np.random.default_rng(0).choice(yv, 5000, replace=False)
        xmin, xmax = ax.get_xlim() if ax.lines or ax.collections else (0, 1)
        w = height * (xmax - xmin if xmax > xmin else 1)
        ax.hlines(yv, xmin, xmin + w, colors=col, linewidths=0.8)
    return ax


def displot(data=None, *, x=None, y=None, hue=None, row=None, col=None, kind="hist",
            palette="seaplot", height=5, aspect=1, facet_kws=None, legend="auto", **kwargs):
    from .grids import FacetGrid
    from ._data import get_vector as _gv
    g = FacetGrid(data=data, row=row, col=col, hue=None, palette=palette,
                  height=height, aspect=aspect, **(facet_kws or {}))
    fn = {"hist": histplot, "kde": kdeplot, "ecdf": ecdfplot}[kind]
    row_v = _gv(data, row) if row is not None else None
    col_v = _gv(data, col) if col is not None else None
    for (ri, ci, axm) in g._iter_axes(row_v, col_v):
        mask = np.ones(g._n, dtype=bool)
        if row_v is not None:
            mask &= np.asarray(row_v, dtype=object) == g._row_cats[ri]
        if col_v is not None:
            mask &= np.asarray(col_v, dtype=object) == g._col_cats[ci]
        sub = g._subset(data, mask)
        fn(data=sub, x=x, y=y, hue=hue, palette=palette, ax=axm, legend=(ri == 0 and ci == 0), **kwargs)
    return g
