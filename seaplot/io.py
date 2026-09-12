"""Universal IO: smart_read / smart_write for analysts.

Supports csv / parquet / json / excel / pickle / sql with zero config::

    import seaplot as sp
    df = sp.read("data.csv")
    df = sp.read("data.parquet")
    sp.write(df, "out.parquet")
"""
from __future__ import annotations

import os


def _pd():
    import pandas as pd
    return pd


def smart_read(path_or_buf, engine="auto", **kwargs):
    """Read almost anything into a pandas DataFrame.

    - path.csv / .csv.gz -> read_csv
    - path.parquet -> read_parquet
    - path.json -> read_json
    - path.xlsx/xls -> read_excel
    - path.pkl/pickle -> read_pickle
    - sql:// or (query, conn) via ``read_sql`` kwarg
    - dict / list -> DataFrame constructor
    """
    pd = _pd()
    if isinstance(path_or_buf, (dict, list)):
        return pd.DataFrame(path_or_buf)
    if hasattr(path_or_buf, "to_pandas"):  # polars frame
        try:
            return path_or_buf.to_pandas()
        except Exception:
            pass
    if hasattr(path_or_buf, "iloc") and hasattr(path_or_buf, "columns"):
        return path_or_buf  # already pandas
    if not isinstance(path_or_buf, str):
        try:
            return pd.read_csv(path_or_buf, **kwargs)
        except Exception:
            return pd.DataFrame(path_or_buf)

    p = path_or_buf
    low = p.lower().split("?")[0].split("#")[0]
    if low.startswith(("sqlite://", "postgresql://", "mysql://", "duckdb://", "sql://")):
        return read_sql(kwargs.pop("query", "SELECT 1"), p, **kwargs)
    if low.endswith((".csv", ".csv.gz", ".gz", ".tsv", ".txt")):
        sep = kwargs.pop("sep", "," if not low.endswith(".tsv") else "\t")
        return pd.read_csv(p, sep=sep, **kwargs)
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(p, **kwargs)
    if low.endswith((".json", ".jsonl", ".ndjson")):
        try:
            return pd.read_json(p, **kwargs)
        except Exception:
            return pd.read_json(p, lines=True, **kwargs)
    if low.endswith((".xlsx", ".xls", ".xlsm", ".ods")):
        return pd.read_excel(p, **kwargs)
    if low.endswith((".pkl", ".pickle")):
        return pd.read_pickle(p, **kwargs)
    if low.endswith((".feather",)):
        try:
            return pd.read_feather(p, **kwargs)
        except Exception:
            pass
    # fallback: try csv then excel
    try:
        return pd.read_csv(p, **kwargs)
    except Exception:
        return pd.read_excel(p, **kwargs)


def smart_write(df, path, index=False, **kwargs):
    """Write DataFrame by extension. Returns path."""
    pd = _pd()
    low = str(path).lower()
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    if low.endswith((".csv", ".csv.gz", ".gz", ".tsv", ".txt")):
        sep = "," if not low.endswith(".tsv") else "\t"
        df.to_csv(path, index=index, sep=kwargs.pop("sep", sep), **kwargs)
    elif low.endswith((".parquet", ".pq")):
        df.to_parquet(path, index=index, **kwargs)
    elif low.endswith(".json"):
        df.to_json(path, index=index, **kwargs)
    elif low.endswith((".xlsx", ".xls")):
        df.to_excel(path, index=index, **kwargs)
    elif low.endswith((".pkl", ".pickle")):
        df.to_pickle(path, **kwargs)
    else:
        df.to_csv(path, index=index, **kwargs)
    return str(path)


def read_sql(query, conn_or_uri, **kwargs):
    """Read SQL query from sqlite URI, sqlalchemy engine, sqlite3/duckdb conn."""
    pd = _pd()
    if isinstance(conn_or_uri, str):
        uri = conn_or_uri
        if uri.startswith("sql://"):
            uri = uri[6:]
        if uri.startswith("duckdb:"):
            import duckdb
            f = uri.split("://", 1)[-1] or ":memory:"
            con = duckdb.connect(f)
            try:
                return con.execute(query).fetchdf()
            finally:
                try:
                    con.close()
                except Exception:
                    pass
        # sqlalchemy for everything else (sqlite included)
        try:
            from sqlalchemy import create_engine
            eng = create_engine(uri)
            with eng.connect() as con:
                return pd.read_sql(query, con, **kwargs)
        except Exception:
            import sqlite3
            path = uri.split("://")[-1] or ":memory:"
            con = sqlite3.connect(path)
            try:
                return pd.read_sql(query, con, **kwargs)
            finally:
                con.close()
    return pd.read_sql(query, conn_or_uri, **kwargs)


# seaborn-style short aliases
read = smart_read
write = smart_write
