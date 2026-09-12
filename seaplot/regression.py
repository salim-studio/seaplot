"""Regression plots: regplot / lmplot / residplot (numpy polyfit, no statsmodels)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._data import get_vector
from .palettes import color_palette
from .relational import scatterplot


def _get_ax(ax):
    return ax if ax is not None else plt.gca()


def _polyfit_ci(x, y, order=1, ci=95, n_boot=500, seed=0, grid=None):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < order + 1:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    coef = np.polyfit(x, y, order)
    if grid is None:
        grid = np.linspace(x.min(), x.max(), 100)
    pred = np.polyval(coef, grid)
    if ci is None:
        return grid, pred, pred, pred
    rng = np.random.default_rng(seed)
    boots = np.empty((min(n_boot, 300), len(grid)))
    for b in range(boots.shape[0]):
        idx = rng.integers(0, x.size, x.size)
        try:
            c = np.polyfit(x[idx], y[idx], order)
            boots[b] = np.polyval(c, grid)
        except Exception:
            boots[b] = pred
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    lo = np.percentile(boots, lo_q, axis=0)
    hi = np.percentile(boots, hi_q, axis=0)
    return grid, pred, lo, hi


def regplot(data=None, *, x=None, y=None, x_estimator=None, x_bins=None,
            x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=500, seed=0,
            units=None, order=1, logistic=False, lowess=False, robust=False,
            logx=False, x_partial=None, y_partial=None, truncate=True,
            dropna=True, x_jitter=None, y_jitter=None, label=None,
            color=None, palette="seaplot", scatter_kws=None, line_kws=None,
            ax=None, **kwargs):
    ax = _get_ax(ax)
    xv = get_vector(data, x); yv = get_vector(data, y)
    if xv is None or yv is None:
        raise ValueError("regplot requires x and y")
    xv, yv = np.asarray(xv, dtype=float), np.asarray(yv, dtype=float)
    n = min(len(xv), len(yv))
    xv, yv = xv[:n], yv[:n]
    if dropna:
        m = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[m], yv[m]
    pal = color_palette(palette)
    col = color or pal[0]
    if scatter:
        sk = dict(s=30, alpha=0.6, color=col)
        if scatter_kws:
            sk.update(scatter_kws)
        ax.scatter(xv, yv, rasterized=(len(xv) > 50000), **sk)
    if fit_reg and len(xv) > 1:
        grid = np.linspace(xv.min(), xv.max(), 100)
        _, pred, lo, hi = _polyfit_ci(xv, yv, order=order, ci=ci, n_boot=n_boot, seed=seed, grid=grid)
        lk = dict(color=col, lw=2.0)
        if line_kws:
            lk.update(line_kws)
        if label:
            lk["label"] = label
        ax.plot(grid, pred, **lk)
        if ci is not None:
            ax.fill_between(grid, lo, hi, color=col, alpha=0.18, linewidth=0)
    if logx:
        ax.set_xscale("log")
    if isinstance(x, str):
        ax.set_xlabel(x)
    if isinstance(y, str):
        ax.set_ylabel(y)
    if label:
        ax.legend()
    return ax


def lmplot(data=None, *, x=None, y=None, hue=None, row=None, col=None,
           palette="seaplot", height=5, aspect=1, ci=95, order=1, scatter=True,
           fit_reg=True, legend="auto", facet_kws=None, **kwargs):
    from .grids import FacetGrid
    from ._data import get_vector as _gv
    g = FacetGrid(data=data, row=row, col=col, hue=hue, palette=palette,
                  height=height, aspect=aspect, **(facet_kws or {}))
    row_v = _gv(data, row) if row is not None else None
    col_v = _gv(data, col) if col is not None else None
    hue_v = _gv(data, hue) if hue is not None else None
    pal = color_palette(palette)
    for (ri, ci, axm) in g._iter_axes(row_v, col_v):
        mask = np.ones(g._n, dtype=bool)
        if row_v is not None:
            mask &= np.asarray(row_v, dtype=object) == g._row_cats[ri]
        if col_v is not None:
            mask &= np.asarray(col_v, dtype=object) == g._col_cats[ci]
        sub = g._subset(data, mask)
        if hue is None:
            regplot(data=sub, x=x, y=y, ci=ci, order=order, scatter=scatter,
                    fit_reg=fit_reg, ax=axm, **kwargs)
        else:
            import numpy as _np
            from ._data import factorize
            hv = _np.asarray(_gv(sub, hue) if isinstance(hue, str) else hue)
            cats, _ = factorize(hv.astype(object))
            subpal = color_palette(palette, n_colors=len(cats))
            for i, c in enumerate(cats):
                mm = hv.astype(object) == c
                d = {k: _np.asarray(v)[mm] if hasattr(v, "__len__") and len(v) == len(hv) else v
                     for k, v in (sub.items() if isinstance(sub, dict) else {}.items())} if isinstance(sub, dict) else sub
                # slice rows for pandas-like via mask on full data
                regplot(data=sub, x=_np.asarray(_gv(sub, x))[mm] if isinstance(hue, str) else x,
                        y=_np.asarray(_gv(sub, y))[mm] if isinstance(hue, str) else y,
                        ci=ci, order=order, scatter=scatter, fit_reg=fit_reg,
                        color=subpal[i % len(subpal)], label=str(c), ax=axm, **kwargs)
            if legend == "auto":
                axm.legend(title=str(hue) if isinstance(hue, str) else None)
    return g


def residplot(data=None, *, x=None, y=None, x_partial=None, y_partial=None,
              lowess=False, order=1, robust=False, ci=None, scatter_kws=None,
              line_kws=None, color=None, palette="seaplot", ax=None, **kwargs):
    ax = _get_ax(ax)
    xv = np.asarray(get_vector(data, x), dtype=float)
    yv = np.asarray(get_vector(data, y), dtype=float)
    m = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[m], yv[m]
    coef = np.polyfit(xv, yv, order)
    resid = yv - np.polyval(coef, xv)
    pal = color_palette(palette)
    col = color or pal[0]
    sk = dict(s=30, alpha=0.6, color=col)
    if scatter_kws:
        sk.update(scatter_kws)
    ax.scatter(xv, resid, rasterized=(len(xv) > 50000), **sk)
    lk = dict(color=col, lw=2.0)
    if line_kws:
        lk.update(line_kws)
    gx = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(gx, np.zeros_like(gx), **lk)
    ax.axhline(0, color=".4", lw=1, ls="--", alpha=0.7)
    ax.set_ylabel("Residuals")
    if isinstance(x, str):
        ax.set_xlabel(x)
    return ax
