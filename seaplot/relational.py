"""Relational plots: scatterplot / lineplot / relplot — fast paths."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._data import get_vector, factorize
from ._fast import decimate, minmax_decimate_line
from .palettes import color_palette


def _get_ax(ax):
    return ax if ax is not None else plt.gca()


def _hue_colors(hue_v, palette, ax):
    if hue_v is None:
        return None, None
    hue_v = np.asarray(hue_v)
    try:
        hv = hue_v.astype(float)
        isnumeric = np.all(np.isfinite(hv[np.isfinite(hv)])) if hv.size else False
        # numeric with many uniques -> continuous
        if isnumeric and np.unique(hv[np.isfinite(hv)]).size > 12:
            return hv, None
    except Exception:
        pass
    cats, codes = factorize(hue_v.astype(object))
    pal = color_palette(palette, n_colors=len(cats))
    colmap = {c: pal[i % len(pal)] for i, c in enumerate(cats)}
    colors = [colmap[v] for v in hue_v.astype(object)]
    return colors, cats


def scatterplot(data=None, *, x=None, y=None, hue=None, size=None, style=None,
                palette="seaplot", markers=True, alpha=0.8, s=36, linewidths=None,
                edgecolor=None, legend="auto", ax=None, rasterized=None,
                max_points=200_000, **kwargs):
    """Seaborn-compatible scatterplot with decimation + single-Collection draw."""
    ax = _get_ax(ax)
    xv = np.asarray(get_vector(data, x)) if get_vector(data, x) is not None else None
    yv = np.asarray(get_vector(data, y)) if get_vector(data, y) is not None else None
    if xv is None or yv is None:
        raise ValueError("scatterplot requires x and y")
    n = min(len(xv), len(yv))
    xv, yv = np.asarray(xv)[:n], np.asarray(yv)[:n]
    m = np.ones(n, dtype=bool)
    try:
        m &= np.isfinite(xv.astype(float))
    except Exception:
        pass
    try:
        m &= np.isfinite(yv.astype(float))
    except Exception:
        pass
    xv, yv = xv[m], yv[m]
    hue_v = get_vector(data, hue)
    hue_v = np.asarray(hue_v)[m] if hue_v is not None and len(hue_v) >= m.sum() or (hue_v is not None and len(hue_v) == n) else hue_v
    # fix mask alignment
    if hue_v is not None and len(hue_v) == n:
        hue_v = np.asarray(hue_v)[m]

    sv = get_vector(data, size)
    if sv is not None and len(np.asarray(sv)) == n:
        sv = np.asarray(sv, dtype=float)[m]
        try:
            smin, smax = np.nanmin(sv), np.nanmax(sv)
            sizes = 20 + 180 * (sv - smin) / (smax - smin + 1e-12)
        except Exception:
            sizes = s
    else:
        sizes = s

    # decimate huge data (keeps hue proportions via stratified sample)
    idx_map = None
    if len(xv) > max_points:
        if hue_v is not None:
            # stratified: sample per category
            cats, codes = factorize(np.asarray(hue_v, dtype=object))
            keep = []
            rng = np.random.default_rng(0)
            per = max(1, max_points // max(len(cats), 1))
            for g in range(len(cats)):
                gi = np.where(codes == g)[0]
                if len(gi) > per:
                    gi = rng.choice(gi, per, replace=False)
                keep.append(gi)
            keep = np.sort(np.concatenate(keep))
            xv, yv = xv[keep], yv[keep]
            hue_v = np.asarray(hue_v)[keep]
            if isinstance(sizes, np.ndarray):
                sizes = sizes[keep]
        else:
            xv, yv, keep = decimate(np.asarray(xv), np.asarray(yv), max_points)
            if isinstance(sizes, np.ndarray) and keep is not None:
                sizes = sizes[keep]
            if hue_v is not None and keep is not None:
                hue_v = np.asarray(hue_v)[keep]

    colors, cats = _hue_colors(hue_v, palette, ax) if hue_v is not None else (None, None)
    # single scatter call (fast) — split by hue only if categorical for legend
    if cats is not None and not isinstance(colors, np.ndarray):
        for i, c in enumerate(cats):
            mm = np.asarray(hue_v, dtype=object) == c
            ax.scatter(xv[mm], yv[mm], s=sizes[mm] if isinstance(sizes, np.ndarray) else sizes,
                       c=[color_palette(palette, len(cats))[i]], alpha=alpha,
                       linewidths=linewidths, edgecolors=edgecolor, label=str(c),
                       rasterized=rasterized if rasterized is not None else (len(xv) > 50000),
                       **kwargs)
        if legend == "auto":
            ax.legend(title=str(hue) if isinstance(hue, str) else None, markerscale=1.2,
                      scatterpoints=3, frameon=True)
    else:
        c_arg = colors if colors is not None else kwargs.pop("color", kwargs.pop("c", None))
        ax.scatter(xv, yv, s=sizes, c=c_arg, alpha=alpha, linewidths=linewidths,
                   edgecolors=edgecolor,
                   rasterized=rasterized if rasterized is not None else (len(xv) > 50000),
                   **kwargs)
        if hue_v is not None and legend == "auto":
            try:
                sm = ax.collections[-1]
                plt.colorbar(sm, ax=ax, label=str(hue))
            except Exception:
                pass
    try:
        if isinstance(x, str):
            ax.set_xlabel(x)
        if isinstance(y, str):
            ax.set_ylabel(y)
    except Exception:
        pass
    return ax


def lineplot(data=None, *, x=None, y=None, hue=None, size=None, style=None,
             units=None, estimator="mean", errorbar=("ci", 95), n_boot=500,
             seed=0, orient="x", color=None, palette="seaplot", linewidth=1.8,
             linestyle="-", marker=None, markers=None, dashes=True, legend="auto",
             ax=None, sort=True, max_points=50_000, **kwargs):
    """Fast lineplot: numpy sort + bincount aggregation (no pandas groupby)."""
    from ._fast import group_stats, bootstrap_ci
    ax = _get_ax(ax)
    xv = get_vector(data, x); yv = get_vector(data, y)
    if xv is None or yv is None:
        raise ValueError("lineplot requires x and y")
    xv, yv = np.asarray(xv), np.asarray(yv)
    n = min(len(xv), len(yv))
    xv, yv = xv[:n], yv[:n]
    hue_v = get_vector(data, hue)
    if hue_v is not None:
        hue_v = np.asarray(hue_v)[:n]
    pal = color_palette(palette)

    def _draw_single(xs, ys, col, label=None):
        try:
            xs_f = xs.astype(float); ys_f = ys.astype(float)
            ok = np.isfinite(xs_f) & np.isfinite(ys_f)
            xs, ys = xs[ok], ys[ok]
            order = np.argsort(xs_f[ok], kind="mergesort") if sort else np.arange(ok.sum())
            xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
            # aggregate duplicates: mean per unique x (vectorized)
            ux, inv = np.unique(xs, return_inverse=True)
            if len(ux) < len(xs) and estimator is not None:
                if estimator == "mean" or estimator == "median":
                    ys = group_stats(ys.astype(float), inv, len(ux),
                                     stat="mean" if estimator == "mean" else "median")
                    xs = ux
                else:
                    xs, ys = ux, group_stats(ys.astype(float), inv, len(ux), stat="mean")
            else:
                if estimator is None:
                    pass
            xs_d, ys_d = minmax_decimate_line(np.asarray(xs), np.asarray(ys), max_points)
            ln, = ax.plot(xs_d, ys_d, color=col, lw=linewidth, ls=linestyle,
                          marker=marker, label=label, **kwargs)
            # error band
            if errorbar is not None and estimator is not None and len(ux) < len(order):
                kind = errorbar[0] if isinstance(errorbar, tuple) else errorbar
                if kind == "ci":
                    ci = errorbar[1] if isinstance(errorbar, tuple) and len(errorbar) > 1 else 95
                    lo = np.empty(len(ux)); hi = np.empty(len(ux))
                    yf = ys if len(ys) == len(ux) else ys
                    # per-x bootstrap would be slow; use global approx only if few uniques
                    if len(ux) <= 200:
                        orig_xs = np.asarray(xs)
                        for gi in range(len(ux)):
                            g = ys_f[ok][order][inv == gi] if len(inv) == ok.sum() else yf[gi:gi+1]
                            try:
                                lo[gi], hi[gi] = bootstrap_ci(g, ci=ci, n_boot=min(n_boot, 200), seed=seed)
                            except Exception:
                                lo[gi], hi[gi] = np.nan, np.nan
                        ax.fill_between(ux, lo, hi, color=col, alpha=0.18, linewidth=0)
                elif kind == "sd":
                    sds = np.empty(len(ux))
                    for gi in range(len(ux)):
                        g = ys_f[ok][order][inv == gi] if len(inv) == ok.sum() else np.array([np.nan])
                        sds[gi] = float(np.nanstd(g)) if len(g) else np.nan
                    ax.fill_between(ux, yf - sds, yf + sds, color=col, alpha=0.18, linewidth=0)
            return ln
        except Exception:
            xs_d, ys_d = minmax_decimate_line(np.asarray(xs), np.asarray(ys), max_points)
            ln, = ax.plot(xs_d, ys_d, color=col, lw=linewidth, ls=linestyle,
                          marker=marker, label=label, **kwargs)
            return ln

    if hue_v is None:
        _draw_single(xv, yv, color or (pal[0] if pal else None))
    else:
        try:
            hv_f = hue_v.astype(float)
            if np.unique(hv_f[np.isfinite(hv_f)]).size > 12:
                _draw_single(xv, yv, color or (pal[0] if pal else None))
                try:
                    plt.colorbar(ax.lines[-1], ax=ax)
                except Exception:
                    pass
            else:
                raise ValueError("categorical")
        except Exception:
            cats, codes = factorize(np.asarray(hue_v, dtype=object))
            subpal = color_palette(palette, n_colors=len(cats))
            for i, c in enumerate(cats):
                mm = np.asarray(hue_v, dtype=object) == c
                _draw_single(xv[mm], yv[mm], subpal[i % len(subpal)], label=str(c))
            if legend == "auto":
                ax.legend(title=str(hue) if isinstance(hue, str) else None)
    try:
        if isinstance(x, str):
            ax.set_xlabel(x)
        if isinstance(y, str):
            ax.set_ylabel(y)
    except Exception:
        pass
    return ax


def relplot(data=None, *, x=None, y=None, hue=None, size=None, style=None,
            row=None, col=None, kind="scatter", palette="seaplot", height=5,
            aspect=1, facet_kws=None, **kwargs):
    """Figure-level relational plot backed by fast scatterplot/lineplot."""
    from .grids import FacetGrid
    g = FacetGrid(data=data, row=row, col=col, hue=hue, palette=palette,
                  height=height, aspect=aspect, **(facet_kws or {}))
    fn = scatterplot if kind == "scatter" else lineplot
    # map across facets
    row_v = get_vector(data, row) if row is not None else None
    col_v = get_vector(data, col) if col is not None else None
    for (ri, ci, axm) in g._iter_axes(row_v, col_v):
        mask = np.ones(g._n, dtype=bool)
        if row_v is not None:
            mask &= np.asarray(row_v, dtype=object) == g._row_cats[ri]
        if col_v is not None:
            mask &= np.asarray(col_v, dtype=object) == g._col_cats[ci]
        sub = g._subset(data, mask)
        fn(data=sub, x=x, y=y, hue=hue, size=size, style=style,
           palette=palette, ax=axm, legend="brief" if (ri == 0 and ci == 0) else False,
           **kwargs)
    return g
