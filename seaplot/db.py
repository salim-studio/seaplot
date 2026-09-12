"""Databases for everyone: OceanDB + query() one-liners.

Works with stdlib sqlite3 out of the box; upgrades automatically to
SQLAlchemy / DuckDB when installed::

    import seaplot as sp

    db = sp.connect("sqlite:///shop.db")     # or ":memory:" / "duckdb://..."
    db.write_df(df, "sales", if_exists="replace")
    print(db.tables())
    df2 = db.query("SELECT day, AVG(total_bill) FROM sales GROUP BY day")
    df2 = sp.query("SELECT * FROM sales LIMIT 5", "sqlite:///shop.db")
"""
from __future__ import annotations

import sqlite3


def _pd():
    import pandas as pd
    return pd


def _is_duckdb_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.lower().startswith("duckdb:")


class OceanDB:
    """Tiny universal DB wrapper (sqlite / duckdb / sqlalchemy)."""

    def __init__(self, uri=":memory:"):
        self.uri = uri
        self._duck = _is_duckdb_uri(uri)
        self._mem_sqlite = None
        if uri == ":memory:" and not self._duck:
            self._mem_sqlite = sqlite3.connect(":memory:")

    # -- internals --------------------------------------------------
    def _sqlalchemy_engine(self):
        from sqlalchemy import create_engine
        return create_engine(self.uri if "://" in self.uri else f"sqlite:///{self.uri}")

    def _sqlite_path(self):
        u = self.uri
        if u in (":memory:", "sqlite:///:memory:"):
            return ":memory:"
        if u.startswith("sqlite:///"):
            return u[10:]
        if u.startswith("sqlite://"):
            return u[9:]
        return u

    # -- public API -------------------------------------------------
    def query(self, sql, params=None, **kwargs):
        """Run SELECT and return DataFrame."""
        pd = _pd()
        if self._duck:
            import duckdb
            f = self.uri.split("://", 1)[-1] or ":memory:"
            con = duckdb.connect(f)
            try:
                return con.execute(sql, params or []).fetchdf()
            finally:
                try:
                    con.close()
                except Exception:
                    pass
        if self._mem_sqlite is not None:
            return pd.read_sql(sql, self._mem_sqlite, params=params, **kwargs)
        try:
            eng = self._sqlalchemy_engine()
            with eng.connect() as con:
                return pd.read_sql(sql, con, params=params, **kwargs)
        except Exception:
            path = self._sqlite_path()
            con = sqlite3.connect(path if path != ":memory:" else ":memory:")
            try:
                return pd.read_sql(sql, con, params=params, **kwargs)
            finally:
                if self._mem_sqlite is None:
                    con.close()

    def execute(self, sql, params=None):
        """Run INSERT/DDL/DML. Returns affected rowcount when known."""
        if self._duck:
            import duckdb
            f = self.uri.split("://", 1)[-1] or ":memory:"
            con = duckdb.connect(f)
            try:
                con.execute(sql, params or [])
                return 0
            finally:
                try:
                    con.close()
                except Exception:
                    pass
        if self._mem_sqlite is not None:
            cur = self._mem_sqlite.execute(sql, params or [])
            self._mem_sqlite.commit()
            return cur.rowcount
        try:
            from sqlalchemy import text
            eng = self._sqlalchemy_engine()
            with eng.begin() as con:
                r = con.execute(text(sql), params or {})
                return r.rowcount
        except Exception:
            con = sqlite3.connect(self._sqlite_path())
            try:
                cur = con.execute(sql, params or [])
                con.commit()
                return cur.rowcount
            finally:
                con.close()

    def tables(self):
        """List table names."""
        try:
            if self._duck:
                import duckdb
                f = self.uri.split("://", 1)[-1] or ":memory:"
                con = duckdb.connect(f)
                try:
                    return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                finally:
                    con.close()
            df = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            return df["name"].tolist() if "name" in df.columns else []
        except Exception:
            return []

    def read_table(self, table, limit=None, **kwargs):
        q = f'SELECT * FROM "{table}"' + (f" LIMIT {int(limit)}" if limit else "")
        return self.query(q, **kwargs)

    def write_df(self, df, table, if_exists="replace", index=False, **kwargs):
        """Write DataFrame to table (fast path per backend)."""
        if self._mem_sqlite is not None:
            df.to_sql(table, self._mem_sqlite, if_exists=if_exists, index=index, **kwargs)
            return table
        if self._duck:
            import duckdb
            f = self.uri.split("://", 1)[-1] or ":memory:"
            con = duckdb.connect(f)
            try:
                mode = "OR REPLACE " if if_exists == "replace" else ""
                con.register("_ob_df", df)
                con.execute(f'CREATE {mode}TABLE "{table}" AS SELECT * FROM _ob_df')
                con.unregister("_ob_df")
                return table
            finally:
                try:
                    con.close()
                except Exception:
                    pass
        try:
            eng = self._sqlalchemy_engine()
            df.to_sql(table, eng, if_exists=if_exists, index=index, **kwargs)
            return table
        except Exception:
            con = sqlite3.connect(self._sqlite_path())
            try:
                df.to_sql(table, con, if_exists=if_exists, index=index, **kwargs)
                return table
            finally:
                con.close()

    def close(self):
        try:
            if self._mem_sqlite is not None:
                self._mem_sqlite.close()
                self._mem_sqlite = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def __repr__(self):
        return f"OceanDB({self.uri!r})"


def connect(uri=":memory:"):
    """Connect to a database. ``sp.connect('sqlite:///app.db')``."""
    return OceanDB(uri)


def query(sql, conn_or_uri, params=None, **kwargs):
    """One-liner: ``sp.query('SELECT 1', 'sqlite:///:memory:')``."""
    if isinstance(conn_or_uri, OceanDB):
        return conn_or_uri.query(sql, params=params, **kwargs)
    with OceanDB(conn_or_uri) as db:
        # keep :memory: sqlite alive only inside call (documented); prefer OceanDB for multi-step
        return db.query(sql, params=params, **kwargs)
