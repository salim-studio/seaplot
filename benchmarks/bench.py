"""Benchmark seaplot internals (no seaborn dependency needed).

Run:  python benchmarks/bench.py
Compares seaplot fast paths against naive pandas-loop baselines
that mirror what seaborn does internally.
"""
import sys, time
sys.path.insert(0, "seaplot")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaplot as sp
from seaplot._fast import group_stats, gaussian_kde_1d, bootstrap_ci


def timed(fn, n=3):
    best = 1e18
    for _ in range(n):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def bench_groupby(n=200_000, k=50):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, k, n)
    vals = rng.normal(size=n)
    cats = np.array([f"c{i}" for i in codes])
    df = pd.DataFrame({"c": cats, "v": vals})
    t_pandas = timed(lambda: df.groupby("c")["v"].mean(), 3)
    t_fast = timed(lambda: group_stats(vals, codes, k, "mean"), 5)
    print(f"groupby-mean n={n} k={k}: pandas {t_pandas*1000:.1f}ms  seaplot {t_fast*1000:.1f}ms  speedup x{t_pandas/max(t_fast,1e-9):.1f}")


def bench_kde(n=20_000):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    grid = np.linspace(-4, 4, 256)
    # naive pairwise KDE (seaborn/scipy style)
    def naive():
        z = (grid[None, :] - x[:, None]) / 0.5
        np.exp(-0.5 * z * z).mean(axis=0)
    t_naive = timed(naive, 3)
    t_fast = timed(lambda: gaussian_kde_1d(x, grid=grid, bw=0.5), 5)
    print(f"kde n={n}: naive {t_naive*1000:.1f}ms  seaplot {t_fast*1000:.1f}ms  speedup x{t_naive/max(t_fast,1e-9):.1f}")


def bench_plots(n=100_000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n),
                       "h": rng.choice(list("AB"), n), "c": rng.choice(list("XYZ"), n)})
    for name, fn in [
        ("scatter 100k", lambda: sp.scatterplot(data=df, x="x", y="y", hue="h")),
        ("hist 100k", lambda: sp.histplot(data=df, x="x", hue="h")),
        ("kde 100k", lambda: sp.kdeplot(data=df, x="x", hue="h")),
        ("bar", lambda: sp.barplot(data=df, x="c", y="x")),
        ("box", lambda: sp.boxplot(data=df, x="c", y="x")),
    ]:
        plt.figure()
        t = timed(fn, 3)
        plt.close("all")
        print(f"{name}: {t*1000:.1f}ms")


if __name__ == "__main__":
    print("== seaplot benchmark ==")
    bench_groupby()
    bench_kde()
    bench_plots()
