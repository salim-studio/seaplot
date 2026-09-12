"""Machine learning shortcuts on top of scikit-learn (optional).

Graceful fallback: clear error if sklearn missing, numpy fallback for
train_test_split already in wrangle::

    import seaplot as sp
    model, report = sp.train(df, target="survived", task="classification")
    sp.evaluate(model, X_test, y_test)
    sp.cross_validate(model, X, y)
    sp.clusters(df, n_clusters=3)
    sp.feature_importance(model, feature_names=[...])
"""
from __future__ import annotations

import numpy as np


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except Exception as e:
        raise ImportError("seaplot.ml needs scikit-learn: pip install scikit-learn") from e


def make_pipeline(task="classification", seed=42, scale_numeric=True):
    """Sensible default pipeline (impute→scale→model)."""
    _require_sklearn()
    from sklearn.pipeline import make_pipeline as _mp
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    steps = [SimpleImputer(strategy="median")]
    if scale_numeric:
        steps.append(StandardScaler())
    if task == "classification":
        steps.append(RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))
    else:
        steps.append(RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1))
    return _mp(*steps)


def train(df=None, target=None, task="auto", test_size=0.2, seed=42, model=None, X=None, y=None):
    """End-to-end: encode → split → fit → evaluate. Returns (model, report dict).

    Accepts either (df, target) or (X, y) arrays/frames.
    """
    _require_sklearn()
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 mean_squared_error, mean_absolute_error, r2_score)
    from .wrangle import encode, train_test_split as _tts
    if X is None or y is None:
        if df is None or target is None:
            raise ValueError("pass df+target or X+y")
        from .wrangle import clean
        X, y = encode(clean(df), target=target)
    y = np.asarray(y)
    if task == "auto":
        task = "classification" if len(np.unique(y[np.isfinite(y)])) <= 20 else "regression"
    Xtr, Xte, ytr, yte = _tts_local(X, y, test_size, seed)
    mdl = model or make_pipeline(task=task, seed=seed)
    mdl.fit(Xtr, ytr)
    pred = mdl.predict(Xte)
    report = {"task": task, "n_train": len(ytr), "n_test": len(yte)}
    if task == "classification":
        report["accuracy"] = float(accuracy_score(yte, pred))
        try:
            report["f1_macro"] = float(f1_score(yte, pred, average="macro"))
        except Exception:
            pass
        try:
            if len(np.unique(y)) == 2 and hasattr(mdl, "predict_proba"):
                report["roc_auc"] = float(roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1]))
        except Exception:
            pass
    else:
        report["rmse"] = float(mean_squared_error(yte, pred) ** 0.5)
        report["mae"] = float(mean_absolute_error(yte, pred))
        try:
            report["r2"] = float(r2_score(yte, pred))
        except Exception:
            pass
    report["model"] = mdl
    return mdl, report


def _tts_local(X, y, test_size, seed):
    from sklearn.model_selection import train_test_split as _tts
    try:
        return _tts(X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y)) <= 20 else None)
    except Exception:
        return _tts(X, y, test_size=test_size, random_state=seed)


def evaluate(model, X_test, y_test, task="auto"):
    """Score a fitted model. Returns dict."""
    _require_sklearn()
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 mean_squared_error, mean_absolute_error, r2_score)
    y_test = np.asarray(y_test)
    pred = model.predict(X_test)
    if task == "auto":
        task = "classification" if len(np.unique(y_test)) <= 20 else "regression"
    if task == "classification":
        return {"accuracy": float(accuracy_score(y_test, pred)),
                "confusion_matrix": confusion_matrix(y_test, pred),
                "report": classification_report(y_test, pred, output_dict=True)}
    return {"rmse": float(mean_squared_error(y_test, pred) ** 0.5),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred))}


def cross_validate(model, X, y, cv=5, scoring=None):
    _require_sklearn()
    from sklearn.model_selection import cross_validate as _cv
    return _cv(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)


def clusters(df_or_X, n_clusters=3, seed=42, columns=None):
    """KMeans labels + centers (DataFrame-aware). Returns (labels, centers)."""
    _require_sklearn()
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    import pandas as pd
    if hasattr(df_or_X, "columns"):
        X = df_or_X[columns] if columns else df_or_X.select_dtypes(include=[np.number])
        Xv = SimpleImputer(strategy="median").fit_transform(X.astype(float))
    else:
        Xv = np.asarray(df_or_X, dtype=float)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    lab = km.fit_predict(Xv)
    return lab, km.cluster_centers_


def feature_importance(model, feature_names=None, top_n=15):
    """Sorted importances from tree/linear models. Returns DataFrame."""
    import pandas as pd
    names = list(feature_names) if feature_names is not None else None
    imp = None
    try:
        # pipeline: last step
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(est, "feature_importances_"):
            imp = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            imp = np.abs(np.asarray(est.coef_).ravel())
    except Exception:
        imp = None
    if imp is None:
        raise ValueError("model has no feature_importances_ / coef_")
    if names is None:
        names = [f"f{i}" for i in range(len(imp))]
    out = pd.DataFrame({"feature": names[: len(imp)], "importance": imp})
    return out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def save_model(model, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return str(path)


def load_model(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
