"""Zero-overhead data handling: pandas / polars / dict / numpy agnostic."""
from __future__ import annotations

import numpy as np


def _is_pandas(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "pandas" or hasattr(obj, "iloc") and hasattr(obj, "columns") and not hasattr(obj, "to_pandas")


def _to_vector(values) -> np.ndarray:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values
    try:
        import pandas as pd  # type: ignore
        if isinstance(values, (pd.Series, pd.Index)):
            return values.to_numpy()
    except Exception:
        pass
    try:
        import polars as pl  # type: ignore
        if isinstance(values, (pl.Series,)):
            return values.to_numpy()
    except Exception:
        pass
    if isinstance(values, (list, tuple)):
        return np.asarray(values)
    try:
        return np.asarray(values)
    except Exception:
        return np.asarray(list(values))


def get_vector(data, key):
    """Extract a column/vector from flexible `data` + `key` like seaborn."""
    if key is None:
        return None
    if isinstance(key, (list, tuple, np.ndarray)) and not isinstance(key, str):
        arr = _to_vector(key)
        if arr is not None and not (isinstance(key, str)):
            # could still be a list of column names? treat as values
            return arr
    if isinstance(key, str):
        if data is None:
            return None
        if isinstance(data, dict):
            return _to_vector(data.get(key))
        # pandas
        try:
            if hasattr(data, "__getitem__") and hasattr(data, "columns"):
                try:
                    return _to_vector(data[key])
                except Exception:
                    return None
        except Exception:
            pass
        # polars
        try:
            if hasattr(data, "get_column"):
                try:
                    return _to_vector(data.get_column(key))
                except Exception:
                    return None
        except Exception:
            pass
        # record array / structured
        try:
            if isinstance(data, np.ndarray) and data.dtype.names and key in data.dtype.names:
                return np.asarray(data[key])
        except Exception:
            pass
        return None
    return _to_vector(key)


def resolve_xy(data, x, y):
    return get_vector(data, x), get_vector(data, y)


def factorize(arr):
    """Fast factorize preserving order of appearance (like seaborn categorical)."""
    arr = np.asarray(arr, dtype=object)
    _, idx = np.unique(arr, return_index=True)
    order = np.unique(arr[np.sort(idx)]).tolist() if False else None
    # order of appearance:
    seen = {}
    cats = []
    codes = np.empty(len(arr), dtype=np.int64)
    for i, v in enumerate(arr):
        k = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
        if k not in seen:
            seen[k] = len(cats)
            cats.append(v)
        codes[i] = seen[k]
    # sort cats naturally? keep appearance order (fast, stable)
    return np.asarray(cats, dtype=object), codes


def n_colors(n, palette):
    from .palettes import color_palette
    pal = color_palette(palette, n_colors=n if isinstance(n, int) else None)
    return pal
