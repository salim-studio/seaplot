# Contributing to SeaPlot

Thanks for helping make fast, friendly data visualization + analysis for everyone.

## Setup

```bash
git clone https://github.com/salim-studio/seaplot.git
cd seaplot
pip install -e ".[all]"
python -m pytest tests -q
```

## Guidelines

- Keep the seaborn-compatible surface stable: same function names and `data, x, y, hue` conventions.
- Hot paths must stay vectorized (NumPy) with pure-Python/NumPy fallbacks; Numba is optional only.
- New IO/DB/ML dependencies must be optional extras, never hard requirements (except the core: numpy, matplotlib, pandas, scipy, scikit-learn).
- Add a test in `tests/` for every new feature; keep `python -m pytest tests -q` green.
- Follow the brand: default palette `seaplot`, wave logo unmodified, English docs.

## Release checklist

1. Bump `__version__` in `seaplot/__init__.py` and `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. `python -m pytest tests -q` + `python examples/fullstack.py`.
4. Tag `vX.Y.Z` and push; publish with `python -m build && twine upload dist/*`.
