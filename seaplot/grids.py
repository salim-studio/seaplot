"""Grids: FacetGrid / PairGrid / JointGrid + pairplot / jointplot."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._data import get_vector, factorize
from .palettes import color_palette


class FacetGrid:
    def __init__(self, data=None, *, row=None, col=None, hue=None, col_wrap=None,
                 sharex=True, sharey=True, height=5, aspect=1, palette="seaplot",
                 hue_order=None, hue_norm=None, legend_out=True, despine=True,
                 margin_titles=False, **kwargs):
        self.data = data
        self._row_name, self._col_name, self._hue_name = row, col, hue
        row_v = get_vector(data, row) if row is not None else None
        col_v = get_vector(data, col) if col is not None else None
        self._row_cats = list(dict.fromkeys([str(v) for v in np.asarray(row_v, dtype=object)])) if row_v is not None else [None]
        self._col_cats = list(dict.fromkeys([str(v) for v in np.asarray(col_v, dtype=object)])) if col_v is not None else [None]
        # map back to original values for masking
        if row_v is not None:
            uniq = []
            seen = set()
            for v, s in zip(np.asarray(row_v, dtype=object), [str(v) for v in np.asarray(row_v, dtype=object)]):
                if s not in seen:
                    seen.add(s); uniq.append(v)
            self._row_cats = uniq
        if col_v is not None:
            uniq = []
            seen = set()
            for v, s in zip(np.asarray(col_v, dtype=object), [str(v) for v in np.asarray(col_v, dtype=object)]):
                if s not in seen:
                    seen.add(s); uniq.append(v)
            self._col_cats = uniq
        nr, nc = len(self._row_cats), len(self._col_cats)
        if col_wrap is not None and row is None:
            nc = min(col_wrap, nc)
            nr = int(np.ceil(len(self._col_cats) / nc))
        self._n = len(np.asarray(row_v)) if row_v is not None else (len(np.asarray(col_v)) if col_v is not None else (len(data) if hasattr(data, "__len__") else 0))
        fig, axes = plt.subplots(nr, nc, figsize=(height * aspect * nc, height * nr),
                                 sharex=sharex, sharey=sharey, squeeze=False)
        self.fig = fig
        self.axes = axes
        self._nr, self._nc = nr, nc
        self.palette = color_palette(palette)
        if despine:
            try:
                from .themes import despine as _d
                _d(fig=fig)
            except Exception:
                pass

    def _iter_axes(self, row_v, col_v):
        for ri in range(self._nr):
            for ci in range(self._nc):
                if ri < len(self._row_cats) or self._row_cats == [None]:
                    yield ri, ci, self.axes[ri, ci]

    def _subset(self, data, mask):
        if data is None:
            return None
        try:
            import pandas as pd  # type: ignore
            if hasattr(data, "iloc"):
                return data.iloc[np.where(mask)[0]]
        except Exception:
            pass
        if isinstance(data, dict):
            return {k: np.asarray(v)[mask] if hasattr(v, "__len__") and len(v) == len(mask) else v for k, v in data.items()}
        try:
            import polars as pl  # type: ignore
            if hasattr(data, "filter"):
                import polars as _pl
                return data.filter(_pl.Series(values=mask))
        except Exception:
            pass
        return data

    def map(self, func, *args, **kwargs):
        row_v = get_vector(self.data, self._row_name) if self._row_name else None
        col_v = get_vector(self.data, self._col_name) if self._col_name else None
        for (ri, ci, axm) in self._iter_axes(row_v, col_v):
            mask = np.ones(self._n, dtype=bool)
            if row_v is not None and ri < len(self._row_cats):
                mask &= np.asarray(row_v, dtype=object) == self._row_cats[ri]
            if col_v is not None and ci < len(self._col_cats):
                mask &= np.asarray(col_v, dtype=object) == self._col_cats[ci]
            sub = self._subset(self.data, mask)
            # resolve args against subset
            rargs = [get_vector(sub, a) if isinstance(a, str) else a for a in args]
            func(*rargs, ax=axm, **kwargs)
        return self

    def map_dataframe(self, func, *args, **kwargs):
        row_v = get_vector(self.data, self._row_name) if self._row_name else None
        col_v = get_vector(self.data, self._col_name) if self._col_name else None
        for (ri, ci, axm) in self._iter_axes(row_v, col_v):
            mask = np.ones(self._n, dtype=bool)
            if row_v is not None and ri < len(self._row_cats):
                mask &= np.asarray(row_v, dtype=object) == self._row_cats[ri]
            if col_v is not None and ci < len(self._col_cats):
                mask &= np.asarray(col_v, dtype=object) == self._col_cats[ci]
            sub = self._subset(self.data, mask)
            func(data=sub, ax=axm, *args, **kwargs)
        return self

    def set_titles(self, template="{col_name} = {col_var}", **kwargs):
        for ci, c in enumerate(self._col_cats):
            for ri in range(self._nr):
                t = template.replace("{col_var}", str(c)).replace("{col_name}", str(self._col_name))
                t = t.replace("{row_var}", str(self._row_cats[ri] if ri < len(self._row_cats) else ""))
                t = t.replace("{row_name}", str(self._row_name))
                self.axes[ri, ci].set_title(t, **kwargs)
        return self

    def set_axis_labels(self, x_var=None, y_var=None, **kwargs):
        for ax in self.axes.flat:
            if x_var:
                ax.set_xlabel(x_var, **kwargs)
            if y_var:
                ax.set_ylabel(y_var, **kwargs)
        return self

    def add_legend(self, **kwargs):
        try:
            self.axes[0, 0].legend(**kwargs)
        except Exception:
            pass
        return self

    def tight_layout(self, *a, **k):
        self.fig.tight_layout(*a, **k)
        return self


class PairGrid:
    def __init__(self, data, *, hue=None, hue_order=None, palette="seaplot",
                 vars=None, x_vars=None, y_vars=None, corner=False,
                 diag_sharey=True, height=2.5, aspect=1, despine=True, **kwargs):
        self.data = data
        self._hue_name = hue
        if vars is not None:
            self.x_vars = list(vars); self.y_vars = list(vars)
        else:
            try:
                cols = list(data.columns) if hasattr(data, "columns") else list(data.keys())
            except Exception:
                cols = []
            if hue in cols:
                cols = [c for c in cols if c != hue]
            # numeric only
            num = []
            for c in cols:
                try:
                    np.asarray(get_vector(data, c), dtype=float)
                    num.append(c)
                except Exception:
                    pass
            self.x_vars = list(x_vars) if x_vars else num
            self.y_vars = list(y_vars) if y_vars else num
        nx, ny = len(self.x_vars), len(self.y_vars)
        fig, axes = plt.subplots(ny, nx, figsize=(height * aspect * nx, height * ny), squeeze=False)
        self.fig, self.axes = fig, axes
        self.palette = color_palette(palette)
        if despine:
            try:
                from .themes import despine as _d
                _d(fig=fig)
            except Exception:
                pass

    def map(self, func, **kwargs):
        return self.map_offdiag(func, **kwargs).map_diag(func, **kwargs)

    def map_offdiag(self, func, **kwargs):
        for i, yv in enumerate(self.y_vars):
            for j, xv in enumerate(self.x_vars):
                if xv == yv:
                    continue
                func(data=self.data, x=xv, y=yv, hue=self._hue_name, ax=self.axes[i, j], **kwargs)
        return self

    def map_diag(self, func, **kwargs):
        for i, yv in enumerate(self.y_vars):
            for j, xv in enumerate(self.x_vars):
                if xv == yv:
                    func(data=self.data, x=xv, ax=self.axes[i, j], **kwargs)
        return self


class JointGrid:
    def __init__(self, data=None, *, x=None, y=None, hue=None, height=6, ratio=5,
                 space=0.2, palette="seaplot", **kwargs):
        self.fig = plt.figure(figsize=(height, height))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[ratio, 1], height_ratios=[1, ratio],
                                   hspace=space, wspace=space)
        self.ax_joint = self.fig.add_subplot(gs[1, 0])
        self.ax_marg_x = self.fig.add_subplot(gs[0, 0], sharex=self.ax_joint)
        self.ax_marg_y = self.fig.add_subplot(gs[1, 1], sharey=self.ax_joint)
        self.x, self.y, self.data = x, y, data

    def plot(self, joint_func, marginal_func, **kwargs):
        joint_func(data=self.data, x=self.x, y=self.y, ax=self.ax_joint, **kwargs.get("joint_kws", {}))
        try:
            marginal_func(data=self.data, x=self.x, ax=self.ax_marg_x, **kwargs.get("marginal_kws", {}))
            marginal_func(data=self.data, y=self.y, ax=self.ax_marg_y, **kwargs.get("marginal_kws", {}))
        except Exception:
            pass
        plt.setp(self.ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(self.ax_marg_y.get_yticklabels(), visible=False)
        return self


def pairplot(data, *, hue=None, hue_order=None, palette="seaplot", vars=None,
             x_vars=None, y_vars=None, kind="scatter", diag_kind="auto",
             markers=None, height=2.5, aspect=1, corner=False, plot_kws=None,
             diag_kws=None, grid_kws=None, **kwargs):
    from . import relational as _r, distributions as _d
    g = PairGrid(data, hue=hue, palette=palette, vars=vars, x_vars=x_vars,
                 y_vars=y_vars, corner=corner, height=height, aspect=aspect,
                 **(grid_kws or {}))
    pk, dk = dict(plot_kws or {}), dict(diag_kws or {})
    off = _r.scatterplot if kind == "scatter" else _r.lineplot if kind == "line" else _r.scatterplot
    if kind == "reg":
        from .regression import regplot as _reg
        off = _reg
    diag = _d.histplot if (diag_kind in ("auto", "hist")) else _d.kdeplot
    if diag_kind == "kde":
        diag = _d.kdeplot
    g.map_offdiag(off, palette=palette, **{**kwargs, **pk})
    g.map_diag(diag, palette=palette, **dk)
    # axis labels
    for i, yv in enumerate(g.y_vars):
        g.axes[i, 0].set_ylabel(str(yv))
    for j, xv in enumerate(g.x_vars):
        g.axes[-1, j].set_xlabel(str(xv))
    return g


def jointplot(data=None, *, x=None, y=None, hue=None, kind="scatter", height=6,
              ratio=5, space=0.2, palette="seaplot", marginal_kws=None,
              joint_kws=None, **kwargs):
    from . import relational as _r, distributions as _d
    from .regression import regplot as _reg
    g = JointGrid(data=data, x=x, y=y, hue=hue, height=height, ratio=ratio, space=space)
    jf = {"scatter": _r.scatterplot, "line": _r.lineplot, "reg": _reg,
          "kde": _d.kdeplot, "hist": _d.histplot}[kind]
    mf = _d.histplot
    g.plot(jf, mf, joint_kws={**(joint_kws or {}), **kwargs}, marginal_kws=marginal_kws or {})
    return g
