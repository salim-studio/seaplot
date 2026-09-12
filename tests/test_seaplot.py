"""Smoke + correctness tests for seaplot (seaborn-compatible API)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaplot as sp


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(size=n), "y": rng.normal(size=n),
        "h": rng.choice(list("AB"), n), "c": rng.choice(list("XYZ"), n),
        "v": rng.normal(5, 2, n),
    })


def test_api_surface():
    for fn in ["scatterplot", "lineplot", "relplot", "histplot", "kdeplot",
               "ecdfplot", "rugplot", "displot", "barplot", "countplot",
               "boxplot", "violinplot", "stripplot", "swarmplot", "pointplot",
               "boxenplot", "catplot", "regplot", "lmplot", "residplot",
               "heatmap", "clustermap", "pairplot", "jointplot",
               "color_palette", "set_theme", "set_style", "set_context",
               "despine", "load_dataset"]:
        assert hasattr(sp, fn), fn


def test_relational():
    df = _df()
    for fn in [lambda: sp.scatterplot(data=df, x="x", y="y", hue="h"),
               lambda: sp.lineplot(data=df, x="x", y="y", hue="h"),
               lambda: sp.regplot(data=df, x="x", y="y"),
               lambda: sp.residplot(data=df, x="x", y="y")]:
        plt.figure(); fn(); plt.close("all")


def test_distributions():
    df = _df()
    plt.figure(); sp.histplot(data=df, x="x", hue="h"); plt.close("all")
    plt.figure(); sp.kdeplot(data=df, x="x", hue="h"); plt.close("all")
    plt.figure(); sp.ecdfplot(data=df, x="x", hue="h"); plt.close("all")
    plt.figure(); sp.rugplot(data=df, x="x"); plt.close("all")


def test_categorical():
    df = _df()
    plt.figure(); sp.barplot(data=df, x="c", y="v"); plt.close("all")
    plt.figure(); sp.countplot(data=df, x="c"); plt.close("all")
    plt.figure(); sp.boxplot(data=df, x="c", y="v"); plt.close("all")
    plt.figure(); sp.violinplot(data=df, x="c", y="v"); plt.close("all")
    plt.figure(); sp.stripplot(data=df, x="c", y="v"); plt.close("all")
    plt.figure(); sp.pointplot(data=df, x="c", y="v"); plt.close("all")
    plt.figure(); sp.boxenplot(data=df, x="c", y="v"); plt.close("all")


def test_matrix_grids():
    rng = np.random.default_rng(0)
    plt.figure(); sp.heatmap(np.corrcoef(rng.normal(size=(5, 50)))); plt.close("all")
    df = _df(200)
    sp.pairplot(df[["x", "y", "v"]]); plt.close("all")
    sp.jointplot(data=df, x="x", y="y"); plt.close("all")
    sp.catplot(data=df, x="c", y="v", kind="box"); plt.close("all")


def test_dict_and_numpy_inputs():
    # no pandas required
    x = np.arange(50.0); y = x * 2 + 1
    plt.figure(); sp.scatterplot(x=x, y=y); plt.close("all")
    plt.figure(); sp.lineplot(x=x, y=y); plt.close("all")
    plt.figure(); sp.histplot(x=x); plt.close("all")
    plt.figure(); sp.kdeplot(x=x); plt.close("all")
    plt.figure(); sp.barplot(x=["a", "b", "a"], y=[1, 2, 3]); plt.close("all")


def test_large_scatter_decimation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300_000); y = rng.normal(size=300_000)
    plt.figure(); sp.scatterplot(x=x, y=y); plt.close("all")


def test_palettes_themes():
    sp.set_theme(style="whitegrid", context="notebook", palette="muted")
    assert len(sp.color_palette("deep")) == 10
    assert len(sp.hls_palette(4)) == 4
    assert len(sp.color_palette("deep", 20)) == 20
    sp.despine()
    plt.close("all")
