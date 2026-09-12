"""Tests for seaplot data toolkit (io/db/wrangle/stats/eda/ml/dl)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaplot as sp


def _df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(5, 2, size=n),
        "g": rng.choice(list("AB"), n), "y": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 10, replace=False), "a"] = np.nan
    return df


def test_io_roundtrip(tmp_path):
    df = _df()
    p = str(tmp_path / "t.csv")
    sp.write(df, p)
    d2 = sp.read(p)
    assert d2.shape == df.shape


def test_db_memory():
    df = _df()
    db = sp.connect(":memory:")
    db.write_df(df, "t")
    assert "t" in db.tables()
    out = db.query("SELECT COUNT(*) AS n FROM t")
    assert int(out["n"].iloc[0]) == len(df)
    db.close()


def test_wrangle():
    df = _df()
    dc = sp.clean(df)
    assert dc.isna().sum().sum() == 0
    X, y = sp.encode(df, target="y")
    assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]
    assert sp.scale(X).shape == X.shape
    tr, te = sp.train_test_split(df, test_size=0.2)
    assert len(tr) + len(te) == len(df)


def test_stats():
    df = _df()
    assert sp.describe(df).shape[0] >= 3
    assert sp.corr(df).shape[0] >= 2
    assert "pvalue" in sp.ttest(sp.clean(df), "a", group="g")
    assert sp.outliers(df, "b").sum() >= 0


def test_eda():
    df = _df()
    r = sp.profile(df, plots=False)
    assert r["shape"] == df.shape
    plt.figure(); sp.plot_missing(df); plt.close("all")
    plt.figure(); sp.plot_corr(sp.clean(df)); plt.close("all")


def test_ml_dl():
    df = sp.clean(_df())
    model, rep = sp.train(df, target="y", task="classification")
    assert "accuracy" in rep
    X, y = sp.encode(df, target="y")
    assert "accuracy" in sp.evaluate(model, X, y)
    lab, centers = sp.clusters(df, n_clusters=2)
    assert len(lab) == len(df)
    mlp = sp.MLP(hidden=(16,), epochs=3, verbose=False)
    mlp.fit(X.to_numpy(), np.asarray(y))
    assert len(mlp.predict(X.to_numpy()[:5])) == 5
