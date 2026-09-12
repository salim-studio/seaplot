"""One-line EDA: profile() + missing/corr/dist plots.

Built on top of the fast plotting core::

    import seaplot as sp
    report = sp.profile(df)          # dict + overview figure
    sp.plot_missing(df)
    sp.plot_corr(df)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def profile(df, plots=True, max_cats=10, figsize=(12, 8)):
    """Return dict report (shape, dtypes, missing, describe, corr) + optional fig."""
    from .stats import describe as _describe
    report = {
        "shape": tuple(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": df.isna().sum().to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "n_duplicates": int(df.duplicated().sum()),
        "n_unique": df.nunique(dropna=True).to_dict(),
    }
    try:
        report["describe"] = _describe(df)
    except Exception:
        report["describe"] = None
    try:
        num = df.select_dtypes(include=[np.number])
        report["corr"] = num.corr() if num.shape[1] >= 2 else None
    except Exception:
        report["corr"] = None
    fig = None
    if plots:
        try:
            from .matrix import heatmap as _hm
            from .distributions import histplot as _hist
            nums = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
            n = max(len(nums), 1)
            fig, axes = plt.subplots(2, max(n, 2), figsize=figsize)
            axes = np.asarray(axes).ravel()
            # missing bar
            miss = df.isna().mean().sort_values(ascending=False).head(10)
            axes[0].barh(list(map(str, miss.index))[::-1], miss.values[::-1])
            axes[0].set_title("Missing % (top 10)")
            # corr heatmap thumb
            if report["corr"] is not None:
                try:
                    plt.sca(axes[1])
                    _hm(report["corr"], ax=axes[1], annot=False, cbar=False)
                    axes[1].set_title("Correlation")
                except Exception:
                    axes[1].axis("off")
            else:
                axes[1].axis("off")
            for i, c in enumerate(nums[: max(len(axes) - 2, 0)]):
                try:
                    _hist(data=df, x=c, ax=axes[2 + i])
                    axes[2 + i].set_title(str(c))
                except Exception:
                    axes[2 + i].axis("off")
            for j in range(2 + len(nums), len(axes)):
                axes[j].axis("off")
            fig.tight_layout()
            report["fig"] = fig
        except Exception:
            report["fig"] = None
    return report


def plot_missing(df, ax=None):
    """Horizontal bar of % missing per column."""
    ax = ax or plt.gca()
    miss = df.isna().mean().sort_values() * 100
    miss = miss[miss > 0]
    if len(miss) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
        return ax
    ax.barh(list(map(str, miss.index)), miss.values)
    ax.set_xlabel("Missing %")
    ax.set_title("Missing values")
    return ax


def plot_corr(df, method="pearson", annot=True, ax=None, **kwargs):
    """Correlation heatmap shortcut."""
    from .matrix import heatmap as _hm
    from .stats import corr as _corr
    ax = ax or plt.gca()
    C = _corr(df, method=method)
    return _hm(C, annot=annot if C.size <= 2000 else False, ax=ax, **kwargs)


def plot_dist_grid(df, columns=None, max_cols=4, kind="hist", figsize=None):
    """Grid of hist/kde per numeric column."""
    from .distributions import histplot as _hist, kdeplot as _kde
    nums = list(columns) if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    nums = nums[: max_cols * 2]
    n = len(nums)
    if n == 0:
        raise ValueError("no numeric columns to plot")
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4 * cols, 3 * rows), squeeze=False)
    fn = _hist if kind == "hist" else _kde
    for i, c in enumerate(nums):
        axm = axes[i // cols, i % cols]
        try:
            fn(data=df, x=c, ax=axm)
            axm.set_title(str(c))
        except Exception:
            axm.axis("off")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout()
    return fig
