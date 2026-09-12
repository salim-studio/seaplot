"""Statistics for analysts: describe / corr / tests / outliers.

Vectorized NumPy + SciPy when available, pure fallbacks otherwise::

    import seaplot as sp
    sp.describe(df)
    sp.corr(df, method="pearson")
    sp.ttest(df, "tip", group="sex")
    sp.outliers(df, "total_bill")
"""
from __future__ import annotations

import numpy as np


def _pd():
    import pandas as pd
    return pd


def describe(df, include="all", percentiles=(0.25, 0.5, 0.75)):
    """Extended describe: adds skew, kurt, missing%, n_unique."""
    pd = _pd()
    base = df.describe(include=include, percentiles=list(percentiles)).T
    try:
        base["skew"] = df.skew(numeric_only=True)
    except Exception:
        pass
    try:
        base["kurt"] = df.kurt(numeric_only=True)
    except Exception:
        pass
    try:
        base["missing"] = df.isna().sum()
        base["missing_pct"] = (df.isna().mean() * 100).round(2)
        base["n_unique"] = df.nunique(dropna=True)
    except Exception:
        pass
    return base


def corr(df, method="pearson", min_periods=1):
    """Correlation matrix over numeric columns only."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError("corr requires at least one numeric column")
    return num.corr(method=method, min_periods=min_periods)


def _groups(df, value, group):
    g = df[[value, group]].dropna()
    cats = list(dict.fromkeys(g[group].astype(str)))
    out = {}
    for c in cats:
        v = pd_vals(g[g[group].astype(str) == c][value])
        out[c] = v
    return out


def pd_vals(s):
    return np.asarray(s, dtype=float)


def ttest(df, value, group, equal_var=False):
    """Welch t-test between the first two groups of ``group`` col."""
    from scipy import stats as _st
    cats = list(dict.fromkeys(df[group].dropna().astype(str)))
    if len(cats) < 2:
        raise ValueError("ttest needs >= 2 groups")
    a = pd_vals(df[df[group].astype(str) == cats[0]][value].dropna())
    b = pd_vals(df[df[group].astype(str) == cats[1]][value].dropna())
    r = _st.ttest_ind(a, b, equal_var=equal_var, nan_policy="omit")
    return {"groups": (cats[0], cats[1]), "statistic": float(r.statistic),
            "pvalue": float(r.pvalue)}


def chi2_test(df, col1, col2):
    """Chi-square test of independence for two categorical cols."""
    pd = _pd()
    from scipy import stats as _st
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, exp = _st.chi2_contingency(ct)
    return {"chi2": float(chi2), "pvalue": float(p), "dof": int(dof),
            "crosstab": ct}


def anova(df, value, group):
    """One-way ANOVA across groups."""
    from scipy import stats as _st
    cats = list(dict.fromkeys(df[group].dropna().astype(str)))
    samples = [pd_vals(df[df[group].astype(str) == c][value].dropna()) for c in cats]
    samples = [s[np.isfinite(s)] for s in samples if len(s)]
    r = _st.f_oneway(*samples)
    return {"groups": cats, "statistic": float(r.statistic), "pvalue": float(r.pvalue)}


def outliers(df, column=None, method="iqr", factor=1.5, z=3.0):
    """Boolean mask of outliers (iqr or zscore). Series if column given else DataFrame."""
    if column is not None:
        v = pd_vals(df[column])
        m = np.isfinite(v)
        if method == "zscore":
            mu, sd = np.nanmean(v), np.nanstd(v)
            mask = np.abs(v - mu) > z * (sd if sd else 1.0)
        else:
            q1, q3 = np.nanpercentile(v[m], [25, 75]) if m.any() else (0, 0)
            iqr = q3 - q1
            mask = (v < q1 - factor * iqr) | (v > q3 + factor * iqr)
        mask[~m] = False
        return mask
    out = {}
    for c in df.columns:
        try:
            out[c] = outliers(df, c, method=method, factor=factor, z=z)
        except Exception:
            continue
    pd = _pd()
    return pd.DataFrame(out, index=df.index)


def normality(df, column):
    """Shapiro/D'Agostino normality quick check."""
    from scipy import stats as _st
    v = pd_vals(df[column].dropna())
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {"statistic": float("nan"), "pvalue": float("nan"), "normal": False}
    try:
        if len(v) <= 5000:
            s, p = _st.shapiro(v)
        else:
            s, p = _st.normaltest(v)
    except Exception:
        s, p = float("nan"), float("nan")
    return {"statistic": float(s), "pvalue": float(p), "normal": bool(p > 0.05)}
