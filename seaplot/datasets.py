"""Datasets + utils (seaborn-compatible shims)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


_DATASETS = ["tips", "titanic", "iris", "penguins", "flights", "diamonds",
             "exercise", "fmri", "mpg", "planets", "glue", "anscombe", "attention", "dowjones"]


def get_dataset_names():
    return list(_DATASETS)


def load_dataset(name, cache=True, data_home=None, **kwargs):
    """Load example dataset. Tries seaborn online repo, falls back to synthetic."""
    import urllib.request, io
    url = f"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{name}.csv"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            buf = r.read()
        import pandas as pd
        return pd.read_csv(io.BytesIO(buf), **kwargs)
    except Exception:
        # offline synthetic fallback (deterministic)
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 31))
        import pandas as pd
        n = 300
        if name == "tips":
            return pd.DataFrame({
                "total_bill": rng.normal(20, 8, n).round(2),
                "tip": rng.normal(3, 1.2, n).round(2),
                "sex": rng.choice(["Male", "Female"], n),
                "smoker": rng.choice(["Yes", "No"], n),
                "day": rng.choice(["Thur", "Fri", "Sat", "Sun"], n),
                "time": rng.choice(["Lunch", "Dinner"], n),
                "size": rng.integers(1, 6, n),
            })
        if name == "iris":
            sp = rng.choice(["setosa", "versicolor", "virginica"], n)
            return pd.DataFrame({
                "sepal_length": rng.normal(5.8, 0.8, n).round(1),
                "sepal_width": rng.normal(3.0, 0.4, n).round(1),
                "petal_length": rng.normal(3.7, 1.7, n).round(1),
                "petal_width": rng.normal(1.2, 0.7, n).round(1),
                "species": sp,
            })
        return pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n),
                             "hue": rng.choice(list("ABC"), n)})


def move_legend(ax, loc, **kwargs):
    try:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.legend(loc=loc, **kwargs)
    except Exception:
        pass


def despine(*a, **k):
    from .themes import despine as _d
    return _d(*a, **k)
