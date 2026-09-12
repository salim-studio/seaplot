"""Utils: seeding, timing, memory, column helpers."""
from __future__ import annotations

import random
import time

import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


class Timer:
    """Context manager + manual timer: ``with sp.Timer('query'): ...``."""

    def __init__(self, name="block", verbose=True):
        self.name = name
        self.verbose = verbose
        self.elapsed = None

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self._t
        if self.verbose:
            print(f"[seaplot] {self.name}: {self.elapsed:.3f}s")
        return False


def memory_usage(df):
    """Per-column + total memory (MB)."""
    mem = df.memory_usage(deep=True, index=True) / 1e6
    out = mem.to_frame("MB").sort_values("MB", ascending=False)
    out.loc["TOTAL"] = mem.sum()
    return out


def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df):
    return [c for c in df.columns if c not in numeric_columns(df)]
