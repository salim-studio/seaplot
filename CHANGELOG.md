# Changelog

## 1.0.0 — SeaPlot launch (renamed from oceanborn)

- Project renamed `oceanborn` → `seaplot`; new home `https://github.com/salim-studio/seaplot`.
- New visual identity: wave logo (`assets/logo.svg`), banner (`assets/banner.svg`), signature `seaplot` palette (default), `docs/BRAND.md`.
- English-first README with badges, migration table, and brand section.
- Packaging: `pyproject.toml` name `seaplot`, URLs pointed at seaplot, GitHub CI workflow.
- Code: all imports now `import seaplot as sp`; seaborn-compatible API unchanged.

## 0.2.0 (as oceanborn)

- Full data toolkit: `io`, `db` (OceanDB), `wrangle`, `stats`, `eda`, `ml`, `dl`, `utils`.
- Examples `fullstack.py`, toolkit tests.

## 0.1.0 (as oceanborn)

- Initial release: seaborn-compatible plotting, faster (NumPy-vectorized + optional Numba).
