"""Matrix plots: heatmap / clustermap — single-pcolormesh fast path."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


def _get_ax(ax):
    return ax if ax is not None else plt.gca()


def heatmap(data, *, vmin=None, vmax=None, cmap=None, center=None,
            robust=False, annot=None, fmt=".2g", annot_kws=None,
            linewidths=0, linecolor="white", cbar=True, cbar_kws=None,
            square=False, xticklabels="auto", yticklabels="auto",
            mask=None, ax=None, **kwargs):
    ax = _get_ax(ax)
    D = np.asarray(data, dtype=float)
    if mask is not None:
        D = np.where(np.asarray(mask, dtype=bool), np.nan, D)
    if robust:
        vmin = np.nanpercentile(D, 2) if vmin is None else vmin
        vmax = np.nanpercentile(D, 98) if vmax is None else vmax
    if center is not None:
        vmax = max(abs(np.nanmin(D) - center), abs(np.nanmax(D) - center)) if vmax is None else vmax
        vmin = center - (vmax - center) if vmin is None else vmin
    cmap = cmap or ("coolwarm" if center is not None else "viridis")
    mesh = ax.pcolormesh(D, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
                         edgecolors=linecolor if linewidths else "face",
                         linewidths=linewidths)
    ax.set_xlim(0, D.shape[1])
    ax.set_ylim(D.shape[0], 0)
    # ticks
    try:
        import pandas as pd  # type: ignore
        if hasattr(data, "columns"):
            cols = list(data.columns)
            idx = list(data.index)
        else:
            cols, idx = None, None
    except Exception:
        cols, idx = None, None
    if cols is None:
        cols = [str(i) for i in range(D.shape[1])]
        idx = [str(i) for i in range(D.shape[0])]
    if xticklabels == "auto":
        step = max(1, len(cols) // 20)
        ax.set_xticks(np.arange(len(cols)) + 0.5, cols[::step] if False else cols, rotation=45, ha="right")
        ax.set_xticks(np.arange(0, len(cols), step) + 0.5, [cols[i] for i in range(0, len(cols), step)])
    elif xticklabels is not False:
        ax.set_xticks(np.arange(len(cols)) + 0.5, list(xticklabels))
    if yticklabels == "auto":
        step = max(1, len(idx) // 20)
        ax.set_yticks(np.arange(0, len(idx), step) + 0.5, [idx[i] for i in range(0, len(idx), step)])
    elif yticklabels is not False:
        ax.set_yticks(np.arange(len(idx)) + 0.5, list(yticklabels))
    if square:
        ax.set_aspect("equal")
    if annot:
        # vectorized text: only annotate if small (avoid 1M Text objects)
        if D.size <= 2000:
            ak = dict(ha="center", va="center", fontsize=8)
            if annot_kws:
                ak.update(annot_kws)
            if isinstance(annot, np.ndarray) and annot.shape == D.shape:
                labels = annot
            else:
                labels = np.vectorize(lambda v: format(v, fmt))(D)
            xs, ys = np.meshgrid(np.arange(D.shape[1]) + 0.5, np.arange(D.shape[0]) + 0.5)
            for xx, yy, lab, val in zip(xs.ravel(), ys.ravel(), np.asarray(labels).ravel(), D.ravel()):
                if np.isfinite(val):
                    ax.text(xx, yy, str(lab), **ak)
        else:
            import warnings
            warnings.warn("seaplot.heatmap: annot skipped for large matrix (>2000 cells) for speed")
    if cbar:
        try:
            plt.colorbar(mesh, ax=ax, **(cbar_kws or {}))
        except Exception:
            pass
    return ax


def clustermap(data, *, pivot_kws=None, method="average", metric="euclidean",
               z_score=None, standard_scale=None, figsize=(10, 10), cbar_pos=(0.02, 0.8, 0.05, 0.18),
               row_cluster=True, col_cluster=True, dendrogram_ratio=0.2,
               colors_ratio=0.03, cbar=True, cmap=None, **kwargs):
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import pdist
    D = np.asarray(data, dtype=float)
    # scale
    if z_score == 0 or standard_scale == 0:
        D = (D - D.mean(axis=0)) / (D.std(axis=0) + 1e-12)
    elif z_score == 1 or standard_scale == 1:
        D = (D - D.mean(axis=1, keepdims=True)) / (D.std(axis=1, keepdims=True) + 1e-12)
    row_ind = np.arange(D.shape[0])
    col_ind = np.arange(D.shape[1])
    try:
        if row_cluster and D.shape[0] > 1:
            row_ind = leaves_list(linkage(pdist(np.nan_to_num(D), metric=metric), method=method))
        if col_cluster and D.shape[1] > 1:
            col_ind = leaves_list(linkage(pdist(np.nan_to_num(D.T), metric=metric), method=method))
    except Exception:
        pass
    Dr = D[np.ix_(row_ind, col_ind)]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[dendrogram_ratio, 1],
                          height_ratios=[dendrogram_ratio, 1], hspace=0.02, wspace=0.02)
    ax_hm = fig.add_subplot(gs[1, 1])
    heatmap(Dr, ax=ax_hm, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False, **kwargs)
    if row_cluster:
        ax_r = fig.add_subplot(gs[1, 0], sharey=ax_hm)
        try:
            with plt.rc_context({"lines.linewidth": 0.8}):
                dendrogram(linkage(pdist(np.nan_to_num(D), metric=metric), method=method),
                           orientation="left", ax=ax_r, no_labels=True)
            ax_r.axis("off")
        except Exception:
            ax_r.axis("off")
    if col_cluster:
        ax_c = fig.add_subplot(gs[0, 1], sharex=ax_hm)
        try:
            with plt.rc_context({"lines.linewidth": 0.8}):
                dendrogram(linkage(pdist(np.nan_to_num(D.T), metric=metric), method=method),
                           ax=ax_c, no_labels=True)
            ax_c.axis("off")
        except Exception:
            ax_c.axis("off")
    if cbar:
        try:
            ax_cb = fig.add_axes(cbar_pos)
            import matplotlib as _mpl
            norm = _mpl.colors.Normalize(vmin=np.nanmin(Dr), vmax=np.nanmax(Dr))
            _mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap or "viridis", norm=norm)
        except Exception:
            pass
    try:
        labels = list(data.index[row_ind]) if hasattr(data, "index") else row_ind
        cols = list(data.columns[col_ind]) if hasattr(data, "columns") else col_ind
    except Exception:
        labels, cols = row_ind, col_ind
    fig._ob_row_order = row_ind
    fig._ob_col_order = col_ind
    return fig
