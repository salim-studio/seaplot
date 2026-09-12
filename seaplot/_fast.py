"""Fast numerical kernels for seaplot.

All hot paths are vectorized NumPy. If numba is installed we use @njit
kernels (no python overhead, parallel where useful). Everything has a
pure-NumPy fallback so seaplot never *requires* numba.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if a and callable(a[0]) is False else (a[0] if a else deco)
    prange = range


# ------------------------------------------------------------------ KDE
def _silverman_bw(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 1.0
    std = float(np.std(x))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sig = min(std, iqr / 1.349) if iqr > 0 else std
    if sig == 0:
        sig = std if std > 0 else 1.0
    return 0.9 * sig * n ** (-1 / 5)


if _HAS_NUMBA:
    @njit(fastmath=True)
    def _kde_gauss_numba(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
        n = x.shape[0]
        m = grid.shape[0]
        out = np.empty(m, dtype=np.float64)
        inv = 1.0 / (n * bw * 2.5066282746310002)
        for j in range(m):
            s = 0.0
            g = grid[j]
            for i in range(n):
                z = (g - x[i]) / bw
                s += np.exp(-0.5 * z * z)
            out[j] = s * inv
        return out
else:  # pragma: no cover
    def _kde_gauss_numba(x, grid, bw):
        z = (grid[None, :] - x[:, None]) / bw
        return np.exp(-0.5 * z * z).mean(axis=0) / (bw * 2.5066282746310002)


def gaussian_kde_1d(x, grid=None, bw="silverman", gridsize=256, cut=3.0, clip=None):
    """Fast 1-D Gaussian KDE. Returns (grid, density)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        g = np.linspace(0, 1, gridsize)
        return g, np.zeros_like(g)
    if isinstance(bw, str):
        bw = _silverman_bw(x)
    bw = float(bw) if bw and bw > 0 else _silverman_bw(x) or 1.0
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    pad = cut * bw
    lo, hi = lo - pad, hi + pad
    if clip is not None:
        lo, hi = max(lo, clip[0]), min(hi, clip[1])
    if grid is None:
        grid = np.linspace(lo, hi, gridsize)
    else:
        grid = np.asarray(grid, dtype=np.float64)
    # subsample huge inputs for KDE (statistically identical, much faster)
    if x.size > 20000:
        idx = np.random.default_rng(0).choice(x.size, 20000, replace=False)
        x = x[idx]
    dens = _kde_gauss_numba(np.ascontiguousarray(x), np.ascontiguousarray(grid), bw)
    return grid, dens


def kde_2d(x, y, gridsize=128, bw=None):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        gx = np.linspace(0, 1, gridsize); gy = np.linspace(0, 1, gridsize)
        return gx, gy, np.zeros((gridsize, gridsize))
    bwx = _silverman_bw(x) if bw is None else float(bw)
    bwy = _silverman_bw(y) if bw is None else float(bw)
    gx = np.linspace(x.min() - 3 * bwx, x.max() + 3 * bwx, gridsize)
    gy = np.linspace(y.min() - 3 * bwy, y.max() + 3 * bwy, gridsize)
    # histogram + gaussian smooth = O(n + m^2), far faster than pairwise KDE
    H, _, _ = np.histogram2d(x, y, bins=gridsize,
                             range=[[gx[0], gx[-1]], [gy[0], gy[-1]]])
    try:
        from scipy.ndimage import gaussian_filter
        sx = gridsize * bwx / (gx[-1] - gx[0] + 1e-12)
        sy = gridsize * bwy / (gy[-1] - gy[0] + 1e-12)
        H = gaussian_filter(H, (sy, sx), mode="nearest")
    except Exception:
        pass
    s = H.sum()
    if s > 0:
        H /= s
    return gx, gy, H.T


# ------------------------------------------------------------------ hist / ecdf
def fast_hist(x, bins="auto", range=None, weights=None, density=False):
    x = np.asarray(x)
    x = x[np.isfinite(x.astype(float, copy=False))] if x.size else x
    counts, edges = np.histogram(x, bins=bins, range=range, weights=weights, density=density)
    return counts, edges


def ecdf_vals(x):
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    n = x.size
    if n == 0:
        return x, x
    y = np.arange(1, n + 1, dtype=float) / n
    return x, y


# ------------------------------------------------------------------ aggregation
def group_stats(values, codes, n_groups, stat="mean"):
    """Vectorized group aggregation using np.bincount (no pandas groupby)."""
    values = np.asarray(values, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    if stat == "count":
        return np.bincount(codes, minlength=n_groups).astype(float)
    mask = np.isfinite(values)
    v = np.where(mask, values, 0.0)
    w = mask.astype(float)
    sums = np.bincount(codes, weights=v, minlength=n_groups)
    cnts = np.bincount(codes, weights=w, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "mean":
            return sums / np.maximum(cnts, 1)
        if stat == "sum":
            return sums
        if stat == "median":
            out = np.full(n_groups, np.nan)
            for g in range(n_groups):
                gv = values[codes == g]
                gv = gv[np.isfinite(gv)]
                if gv.size:
                    out[g] = np.median(gv)
            return out
        if stat in ("std", "var"):
            sq = np.bincount(codes, weights=np.where(mask, values ** 2, 0.0), minlength=n_groups)
            mean = sums / np.maximum(cnts, 1)
            var = sq / np.maximum(cnts, 1) - mean ** 2
            var = np.maximum(var, 0)
            return np.sqrt(var) if stat == "std" else var
        if stat == "min":
            out = np.full(n_groups, np.nan)
            for g in range(n_groups):
                gv = values[codes == g]
                gv = gv[np.isfinite(gv)]
                if gv.size:
                    out[g] = gv.min()
            return out
        if stat == "max":
            out = np.full(n_groups, np.nan)
            for g in range(n_groups):
                gv = values[codes == g]
                gv = gv[np.isfinite(gv)]
                if gv.size:
                    out[g] = gv.max()
            return out
    raise ValueError(f"unknown stat {stat!r}")


def bootstrap_ci(x, stat="mean", n_boot=1000, ci=95, seed=0, fast_n=20000):
    """Percentile bootstrap CI, vectorized (n_boot x n) in one shot."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    if x.size > fast_n:  # subsample: CI converges long before 20k points
        x = np.random.default_rng(seed).choice(x, fast_n, replace=False)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    samp = x[idx]
    if stat == "mean":
        boots = samp.mean(axis=1)
    elif stat == "median":
        boots = np.median(samp, axis=1)
    else:
        boots = samp.mean(axis=1)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return float(np.percentile(boots, lo_q)), float(np.percentile(boots, hi_q))


# ------------------------------------------------------------------ downsampling
def decimate(x, y, max_points=200_000, seed=0):
    """Random decimation for huge scatters (keeps visual fidelity)."""
    n = len(x)
    if n <= max_points:
        return x, y, None
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_points, replace=False)
    idx.sort()
    return np.asarray(x)[idx], np.asarray(y)[idx], idx


def minmax_decimate_line(x, y, max_points=50_000):
    """Min-max decimation preserving line shape (Sveinn Steinarsson style)."""
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n <= max_points:
        return x, y
    bucket = n // max_points
    m = (n // bucket) * bucket
    xb = x[:m].reshape(-1, bucket)
    yb = y[:m].reshape(-1, bucket)
    imin = yb.argmin(axis=1); imax = yb.argmax(axis=1)
    keep = np.empty(2 * len(xb), dtype=np.int64)
    base = np.arange(len(xb), dtype=np.int64) * bucket
    keep[0::2] = base + imin
    keep[1::2] = base + imax
    keep = np.unique(keep)
    tail_idx = np.arange(m, n)
    keep = np.concatenate([keep, tail_idx])
    keep.sort()
    return x[keep], y[keep]
