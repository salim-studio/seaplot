# SeaPlot brand guide

SeaPlot = calm seas, fast ships. Friendly for beginners, credible for researchers.

## Logo

- `assets/logo.svg` — rounded square gradient (`#0ABDE3` → `#0A3D62`), white trend line, scatter dots, double wave.
- `assets/banner.svg` — 1200×400 hero for GitHub social preview / README header.
- Clear space: keep padding ≥ 15% of logo width. Minimum display size: 32px.
- Do not: rotate, recolor, add shadows, or place on low-contrast backgrounds.

## Colors

| Token | Hex | Use |
|---|---|---|
| `wave_teal` | `#0ABDE3` | primary, links, first series |
| `deep_sea` | `#0A3D62` | headers, text on light, second series |
| `coral` | `#FF6B6B` | accent, highlights |
| `seafoam` | `#55E6C1` | waves, success accents |
| `sand` | `#F7F1E3` | light backgrounds |
| `navy` | `#182C61` | dark mode surfaces |

Signature chart palette `seaplot`:

```
#0ABDE3, #0A3D62, #FF6B6B, #55E6C1, #F19066,
#786FA6, #F5CD79, #63CDDA, #CF6F7A, #596275
```

```python
import seaplot as sp
sp.set_theme(style="whitegrid", palette="seaplot")
sp.color_palette("seaplot")
```

## Voice

- English-first, plain, concrete. Show code before theory.
- Tagline: "Seaborn-compatible plotting, 20x faster — plus DB, EDA, ML & DL."
- README headers, issues, and releases stay in English.

## GitHub repo settings (recommended)

- Description: "Seaborn-compatible statistical plotting — 20x faster — plus DB, EDA, ML & DL toolkit."
- Website: PyPI page once published.
- Topics: `visualization`, `seaborn`, `matplotlib`, `data-science`, `eda`, `machine-learning`, `deep-learning`, `database`, `statistics`, `plotting`.
- Social preview: upload `assets/banner.svg` (or PNG export) at 1280×640.
