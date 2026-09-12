"""seaplot — seaborn-compatible plotting + full data-science toolkit.

Plotting (drop-in for seaborn)::

    import seaplot as sp
    sp.scatterplot(...); sp.histplot(...); sp.heatmap(...)

Data + DB + ML/DL::

    df = sp.read("data.csv")
    db = sp.connect("sqlite:///app.db"); db.write_df(df, "t"); db.query("SELECT ...")
    df = sp.clean(df)
    sp.profile(df)
    model, report = sp.train(df, target="y")
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "__version__",
    # relational
    "scatterplot", "lineplot", "relplot",
    # distributions
    "histplot", "kdeplot", "ecdfplot", "rugplot", "displot",
    # categorical
    "barplot", "countplot", "boxplot", "violinplot", "stripplot",
    "swarmplot", "pointplot", "boxenplot", "catplot",
    # regression
    "regplot", "lmplot", "residplot",
    # matrix
    "heatmap", "clustermap",
    # grids
    "FacetGrid", "PairGrid", "JointGrid", "pairplot", "jointplot",
    # palettes / themes
    "color_palette", "hls_palette", "husl_palette", "dark_palette",
    "light_palette", "diverging_palette", "cubehelix_palette",
    "set_palette", "reset_orig",
    "set_theme", "set", "set_style", "set_context",
    "axes_style", "plotting_context", "despine", "reset_defaults",
    # datasets / utils (legacy)
    "load_dataset", "get_dataset_names", "move_legend",
    # IO
    "read", "write", "smart_read", "smart_write", "read_sql",
    # DB
    "connect", "query", "OceanDB",
    # wrangle
    "clean", "fill_missing", "coerce_dtypes", "encode", "scale",
    "train_test_split", "missing_table", "add_date_features",
    # stats
    "describe", "corr", "ttest", "chi2_test", "anova", "outliers", "normality",
    # eda
    "profile", "plot_missing", "plot_corr", "plot_dist_grid",
    # ml
    "make_pipeline", "train", "evaluate", "cross_validate", "clusters",
    "feature_importance", "save_model", "load_model",
    # dl
    "MLP", "torch_mlp",
    # utils
    "seed_everything", "Timer", "memory_usage",
    "numeric_columns", "categorical_columns",
]

from .relational import scatterplot, lineplot, relplot
from .distributions import histplot, kdeplot, ecdfplot, rugplot, displot
from .categorical import (barplot, countplot, boxplot, violinplot, stripplot,
                          swarmplot, pointplot, boxenplot, catplot)
from .regression import regplot, lmplot, residplot
from .matrix import heatmap, clustermap
from .grids import FacetGrid, PairGrid, JointGrid, pairplot, jointplot
from .palettes import (color_palette, hls_palette, husl_palette, dark_palette,
                       light_palette, diverging_palette, cubehelix_palette,
                       set_palette, reset_orig)
from .themes import (set_theme, set, set_style, set_context, axes_style,
                     plotting_context, despine, reset_defaults)
from .datasets import load_dataset, get_dataset_names, move_legend

# new toolkit surface
from .io import smart_read, smart_write, read_sql, read, write
from .db import OceanDB, connect, query
from .wrangle import (clean, fill_missing, coerce_dtypes, encode, scale,
                      train_test_split, missing_table, add_date_features)
from .stats import describe, corr, ttest, chi2_test, anova, outliers, normality
from .eda import profile, plot_missing, plot_corr, plot_dist_grid
from .ml import (make_pipeline, train, evaluate, cross_validate, clusters,
                 feature_importance, save_model, load_model)
from .dl import MLP, torch_mlp
from .utils import (seed_everything, Timer, memory_usage,
                    numeric_columns, categorical_columns)
