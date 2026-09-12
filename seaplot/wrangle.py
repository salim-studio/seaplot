"""Data wrangling: clean / encode / scale / split — one-liners for analysts.

All functions accept pandas DataFrames and return new frames (no inplace
surprises)::

    import seaplot as sp
    df = sp.clean(df)                              # dedupe + fix dtypes + missing
    X, y = sp.encode(df, target="churn")           # one-hot + label target
    Xs = sp.scale(X)                               # standardize numerics
    tr, te = sp.train_test_split(df, test_size=0.2)
"""
from __future__ import annotations

import numpy as np


def _pd():
    import pandas as pd
    return pd


def missing_table(df):
    """Per-column missing report sorted by % missing."""
    pd = _pd()
    m = df.isna().sum()
    out = pd.DataFrame({"missing": m, "pct": (m / max(len(df), 1) * 100).round(2)})
    return out[out["missing"] > 0].sort_values("pct", ascending=False)


def fill_missing(df, num="median", cat="most_frequent", columns=None, value=None):
    """Fill NaNs: numerics median/mean, categoricals mode/constant."""
    out = df.copy()
    cols = list(columns) if columns else list(out.columns)
    for c in cols:
        if c not in out.columns or not out[c].isna().any():
            continue
        if value is not None:
            out[c] = out[c].fillna(value)
            continue
        try:
            out[c].astype(float)
            is_num = True
        except Exception:
            is_num = False
        if is_num:
            fill = out[c].median() if num == "median" else out[c].mean()
            if num == "constant":
                fill = 0
            out[c] = out[c].fillna(fill)
        else:
            if cat == "most_frequent":
                try:
                    fill = out[c].mode(dropna=True).iloc[0]
                except Exception:
                    fill = "missing"
            else:
                fill = "missing"
            out[c] = out[c].fillna(fill)
    return out


def coerce_dtypes(df, parse_dates=True):
    """Best-effort dtype fixer: numerics + datetimes + low-card strings→category."""
    pd = _pd()
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if parse_dates and s.dtype == object:
            # only try date parse when it looks like a date
            sample = s.dropna().astype(str).head(20)
            if len(sample) and all(any(ch in v for ch in "-/:") for v in sample):
                try:
                    conv = pd.to_datetime(s, errors="raise")
                    out[c] = conv
                    continue
                except Exception:
                    pass
        if s.dtype == object:
            try:
                conv = pd.to_numeric(s, errors="raise")
                out[c] = conv
                continue
            except Exception:
                pass
            try:
                if s.nunique(dropna=True) <= max(20, len(s) // 20):
                    out[c] = s.astype("category")
            except Exception:
                pass
    return out


def drop_duplicates(df, **kwargs):
    return df.drop_duplicates(**kwargs).reset_index(drop=True)


def add_date_features(df, column):
    """Expand datetime col into year/month/day/dow/hour/weekend flags."""
    pd = _pd()
    out = df.copy()
    s = pd.to_datetime(out[column], errors="coerce")
    out[f"{column}_year"] = s.dt.year
    out[f"{column}_month"] = s.dt.month
    out[f"{column}_day"] = s.dt.day
    out[f"{column}_dow"] = s.dt.dayofweek
    try:
        out[f"{column}_hour"] = s.dt.hour
    except Exception:
        pass
    out[f"{column}_is_weekend"] = (s.dt.dayofweek >= 5).astype("int8")
    return out


def clean(df, drop_dupes=True, parse_dates=True, fill=True, lowercase_columns=False):
    """The famous one-liner: dedupe → fix dtypes → fill missing → tidy index."""
    out = df.copy()
    if lowercase_columns:
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    if drop_dupes:
        out = out.drop_duplicates()
    out = coerce_dtypes(out, parse_dates=parse_dates)
    if fill:
        out = fill_missing(out)
    return out.reset_index(drop=True)


def encode(df, target=None, max_onehot=20, drop_first=False):
    """Split into numeric-ready (X, y): one-hot low-card categoricals, label target.

    Returns (X: DataFrame, y: Series or None).
    """
    pd = _pd()
    data = df.copy()
    y = None
    if target is not None and target in data.columns:
        y = data.pop(target)
        try:
            if y.dtype == object or str(y.dtype) == "category":
                y = pd.Series(pd.factorize(y)[0], index=data.index, name=target)
            else:
                y = pd.to_numeric(y, errors="coerce")
        except Exception:
            pass
    cat_cols = []
    for c in data.columns:
        dt = data[c].dtype
        try:
            pd.to_numeric(data[c].dropna().head(20), errors="raise")
            # numeric-like -> not categorical (unless bool/low-int strings?)
            if str(dt) in ("object", "category", "bool", "string") or str(dt).startswith("string"):
                # check if truly categorical strings
                try:
                    pd.to_numeric(data[c], errors="raise")
                    continue
                except Exception:
                    pass
            else:
                continue
        except Exception:
            pass
        cat_cols.append(c)
    cat_cols = [c for c in cat_cols if data[c].nunique(dropna=True) <= max_onehot]
    high_card = [c for c in list(data.columns)
                 if c not in cat_cols and _is_non_numeric(data[c])]
    if cat_cols:
        data = pd.get_dummies(data, columns=cat_cols, drop_first=drop_first, dtype=float)
    # ensure numeric: factorize leftovers (high-card strings), coerce numerics
    for c in list(data.columns):
        if _is_non_numeric(data[c]):
            try:
                v = pd.to_numeric(data[c], errors="coerce")
                if v.isna().all():
                    data[c] = pd.factorize(data[c])[0].astype(float)
                else:
                    data[c] = v.fillna(0).astype(float)
            except Exception:
                data[c] = pd.factorize(data[c])[0].astype(float)
    return data.astype(float), y


def _is_non_numeric(s):
    import numpy as _np
    try:
        if str(s.dtype) in ("object", "category", "bool", "string"):
            return True
        if str(s.dtype).startswith("string"):
            return True
        _np.asarray(s.to_numpy() if hasattr(s, "to_numpy") else s, dtype=float)
        return False
    except Exception:
        return True


def scale(X, method="standard", columns=None):
    """Standardize (default) or minmax/normalize numeric columns."""
    out = X.copy()
    cols = list(columns) if columns else [c for c in out.columns]
    nums = []
    for c in cols:
        try:
            out[c].astype(float)
            nums.append(c)
        except Exception:
            continue
    V = out[nums].astype(float).to_numpy()
    if method in ("standard", "zscore", "std"):
        mu = np.nanmean(V, axis=0)
        sd = np.nanstd(V, axis=0)
        sd[sd == 0] = 1.0
        V = (V - mu) / sd
    elif method in ("minmax", "min-max"):
        lo = np.nanmin(V, axis=0)
        hi = np.nanmax(V, axis=0)
        V = (V - lo) / np.maximum(hi - lo, 1e-12)
    elif method == "robust":
        med = np.nanmedian(V, axis=0)
        iqr = np.nanpercentile(V, 75, axis=0) - np.nanpercentile(V, 25, axis=0)
        iqr[iqr == 0] = 1.0
        V = (V - med) / iqr
    out[nums] = V
    return out


def train_test_split(df, test_size=0.2, shuffle=True, seed=42, stratify=None):
    """Split DataFrame (or arrays) — pandas-friendly wrapper."""
    try:
        from sklearn.model_selection import train_test_split as _tts
        if isinstance(df, (tuple, list)) and len(df) == 2:
            X, y = df
            return _tts(X, y, test_size=test_size, random_state=seed,
                        shuffle=shuffle, stratify=stratify)
        return _tts(df, test_size=test_size, random_state=seed, shuffle=shuffle,
                    stratify=(df[stratify] if isinstance(stratify, str) else stratify))
    except ImportError:
        rng = np.random.default_rng(seed)
        n = len(df)
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        k = int(n * (1 - test_size))
        if hasattr(df, "iloc"):
            return df.iloc[idx[:k]].reset_index(drop=True), df.iloc[idx[k:]].reset_index(drop=True)
        arr = np.asarray(df)
        return arr[idx[:k]], arr[idx[k:]]
