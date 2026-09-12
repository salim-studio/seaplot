"""Categorical plots — fully vectorized (bincount / nanpercentile)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._data import get_vector, factorize
from ._fast import group_stats, bootstrap_ci
from .palettes import color_palette


def _get_ax(ax):
    return ax if ax is not None else plt.gca()


def _cat_setup(data, x, y):
    # seaborn: one of x/y is categorical. Detect categorical side.
    xv = get_vector(data, x) if x is not None else None
    yv = get_vector(data, y) if y is not None else None
    if xv is not None and yv is not None:
        # categorical = non-numeric side
        try:
            xv.astype(float); x_num = True
        except Exception:
            x_num = False
        try:
            yv.astype(float); y_num = True
        except Exception:
            y_num = False
        if not x_num and y_num:
            return np.asarray(xv, dtype=object), np.asarray(yv, dtype=float), "x", x, y
        if x_num and not y_num:
            return np.asarray(yv, dtype=object), np.asarray(xv, dtype=float), "y", y, x
        # both numeric -> treat x as categories
        return np.asarray(xv, dtype=object), np.asarray(yv, dtype=float), "x", x, y
    if xv is not None:
        return np.asarray(xv, dtype=object), None, "x", x, None
    if yv is not None:
        return np.asarray(yv, dtype=object), None, "y", y, None
    raise ValueError("categorical plot requires x or y")


def _split_hue(hue_v, order_cats):
    if hue_v is None:
        return None, None
    hue_v = np.asarray(hue_v, dtype=object)
    hc, _ = factorize(hue_v)
    return hc, hue_v


def barplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
            estimator="mean", errorbar=("ci", 95), n_boot=1000, seed=0,
            orient=None, color=None, palette="seaplot", saturation=0.75,
            errcolor=".26", errwidth=None, capsize=None, dodge=True,
            ci="deprecated", ax=None, **kwargs):
    from matplotlib.colors import to_rgb
    ax = _get_ax(ax)
    cats_v, vals, cat_axis, cat_name, val_name = _cat_setup(data, x, y)
    hue_v = get_vector(data, hue) if hue is not None else None
    cats = list(order) if order is not None else list(dict.fromkeys(list(cats_v.astype(str)))) if False else None
    # factorize categories preserving appearance
    uniq, codes = factorize(cats_v)
    if order is not None:
        omap = {str(o): i for i, o in enumerate(order)}
        # remap
        new_codes = np.full_like(codes, -1)
        for i, c in enumerate(cats_v):
            new_codes[i] = omap.get(str(c), -1)
        keep = new_codes >= 0
        cats_v, vals, codes = cats_v[keep], (vals[keep] if vals is not None else None), new_codes[keep]
        uniq = np.asarray(list(order), dtype=object)
    nG = len(uniq)
    pal = color_palette(palette, n_colors=nG if hue_v is None else None)

    def _color(i, base=None):
        c = base or (color or pal[i % len(pal)])
        try:
            import matplotlib.colors as mc
            r, g, b = to_rgb(c)
            # saturation
            import colorsys
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            return colorsys.hls_to_rgb(h, l, s * saturation)
        except Exception:
            return c

    err_kws = dict(ecolor=errcolor, capsize=capsize or 0,
                   elinewidth=errwidth or 1.2)

    if hue_v is None:
        vals_f = vals.astype(float) if vals is not None else np.ones(len(cats_v))
        if estimator == "count" or vals is None:
            heights = group_stats(np.ones(len(codes)), codes, nG, stat="sum")
            errs = None
        elif isinstance(estimator, str):
            heights = group_stats(vals_f, codes, nG, stat=estimator if estimator in ("mean", "median", "sum", "min", "max", "std", "var") else "mean")
            errs = None
            if errorbar is not None:
                kind = errorbar[0] if isinstance(errorbar, tuple) else errorbar
                errs = np.zeros((2, nG))
                for g in range(nG):
                    gv = vals_f[codes == g]
                    gv = gv[np.isfinite(gv)]
                    if gv.size == 0:
                        errs[:, g] = 0
                        continue
                    if kind == "ci":
                        ci_v = errorbar[1] if isinstance(errorbar, tuple) and len(errorbar) > 1 else 95
                        lo, hi = bootstrap_ci(gv, n_boot=min(n_boot, 500), ci=ci_v, seed=seed)
                        errs[0, g], errs[1, g] = heights[g] - lo, hi - heights[g]
                    elif kind == "sd":
                        s = float(np.nanstd(gv))
                        errs[:, g] = s
                    elif kind == "se":
                        errs[:, g] = float(np.nanstd(gv) / np.sqrt(max(len(gv), 1)))
                    else:
                        errs[:, g] = 0
        else:
            heights = np.array([estimator(vals_f[codes == g]) for g in range(nG)], dtype=float)
            errs = None
        pos = np.arange(nG)
        bars = ax.bar(pos, heights, color=[_color(i) for i in range(nG)], **kwargs)
        if errs is not None:
            if cat_axis == "x":
                ax.errorbar(pos, heights, yerr=errs, fmt="none", **err_kws)
            else:
                ax.errorbar(heights, pos, xerr=errs, fmt="none", **err_kws)
        labels = [str(c) for c in uniq]
        if cat_axis == "x":
            ax.set_xticks(pos, labels)
            if isinstance(val_name, str):
                ax.set_ylabel(val_name)
            if isinstance(cat_name, str):
                ax.set_xlabel(cat_name)
        else:
            ax.set_yticks(pos, labels)
            bars = ax.barh(pos, heights, color=[_color(i) for i in range(nG)]) if False else bars
            # redraw horizontal properly
            ax.clear()
            ax.barh(pos, heights, color=[_color(i) for i in range(nG)], **kwargs)
            if errs is not None:
                ax.errorbar(heights, pos, xerr=errs, fmt="none", **err_kws)
            ax.set_yticks(pos, labels)
        return ax

    # hue nested
    hue_v = np.asarray(hue_v, dtype=object)
    huniq, hcodes = factorize(hue_v)
    if hue_order is not None:
        huniq = np.asarray(list(hue_order), dtype=object)
    nH = len(huniq)
    subpal = color_palette(palette, n_colors=nH)
    # map hue cats to indices
    hmap = {str(h): i for i, h in enumerate(huniq.astype(str))}
    hi = np.array([hmap.get(str(v), -1) for v in hue_v])
    width = 0.8 / max(nH, 1)
    pos = np.arange(nG)
    for h in range(nH):
        hh = np.zeros(nG)
        ee = np.zeros((2, nG))
        for g in range(nG):
            gv = vals[(codes == g) & (hi == h)].astype(float) if vals is not None else np.array([1.0])
            gv = gv[np.isfinite(gv)] if gv.size else gv
            if gv.size == 0:
                hh[g] = 0
                continue
            hh[g] = float(np.mean(gv)) if estimator == "mean" else float(np.median(gv)) if estimator == "median" else float(np.sum(gv))
            if errorbar is not None:
                kind = errorbar[0] if isinstance(errorbar, tuple) else errorbar
                if kind == "ci":
                    ci_v = errorbar[1] if isinstance(errorbar, tuple) and len(errorbar) > 1 else 95
                    lo, hiv = bootstrap_ci(gv, n_boot=min(n_boot, 300), ci=ci_v, seed=seed)
                    ee[0, g], ee[1, g] = hh[g] - lo, hiv - hh[g]
                elif kind == "sd":
                    ee[:, g] = float(np.nanstd(gv))
        off = (h - (nH - 1) / 2) * width if dodge else 0
        w = width * 0.95 if dodge else 0.8 / 1
        if cat_axis == "x":
            ax.bar(pos + off, hh, width=w, color=_color(h, subpal[h % len(subpal)]), label=str(huniq[h]), **kwargs)
            ax.errorbar(pos + off, hh, yerr=ee, fmt="none", **err_kws)
        else:
            ax.barh(pos + off, hh, height=w, color=_color(h, subpal[h % len(subpal)]), label=str(huniq[h]), **kwargs)
            ax.errorbar(hh, pos + off, xerr=ee, fmt="none", **err_kws)
    labels = [str(c) for c in uniq]
    if cat_axis == "x":
        ax.set_xticks(pos, labels)
    else:
        ax.set_yticks(pos, labels)
    ax.legend(title=str(hue) if isinstance(hue, str) else None)
    return ax


def countplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
              orient=None, color=None, palette="seaplot", saturation=0.75,
              dodge=True, ax=None, **kwargs):
    cats_v, _, cat_axis, cat_name, _ = _cat_setup(data, x if x is not None else y, None)
    # rebuild with hue-aware barplot of counts
    n = len(cats_v)
    fake = {"__cat__": cats_v}
    if hue is not None:
        hv = get_vector(data, hue)
        fake["__hue__"] = np.asarray(hv)
        return barplot(data=fake, x="__cat__" if (x is not None or (x is None and y is None)) else None,
                       y=None if cat_axis == "x" else "__cat__",
                       hue="__hue__", estimator="count", order=order, hue_order=hue_order,
                       color=color, palette=palette, saturation=saturation, dodge=dodge,
                       ax=ax, errorbar=None, **kwargs)
    return barplot(data=fake, x="__cat__" if cat_axis == "x" else None,
                   y="__cat__" if cat_axis == "y" else None,
                   estimator="count", order=order, color=color, palette=palette,
                   saturation=saturation, ax=ax, errorbar=None, **kwargs)


def _box_stats(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(med=np.nan, q1=np.nan, q3=np.nan, whislo=np.nan, whishi=np.nan, fliers=np.array([]))
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    whislo = max(v.min(), q1 - 1.5 * iqr)
    whishi = min(v.max(), q3 + 1.5 * iqr)
    fliers = v[(v < whislo) | (v > whishi)]
    return dict(med=med, q1=q1, q3=q3, whislo=whislo, whishi=whishi, fliers=fliers)


def boxplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
            orient=None, color=None, palette="seaplot", saturation=0.75,
            width=0.8, dodge=True, fliersize=3, linewidth=1.0,
            whis=1.5, ax=None, legend="auto", **kwargs):
    ax = _get_ax(ax)
    cats_v, vals, cat_axis, cat_name, val_name = _cat_setup(data, x, y)
    uniq, codes = factorize(cats_v)
    if order is not None:
        uniq = np.asarray(list(order), dtype=object)
        omap = {str(o): i for i, o in enumerate(uniq.astype(str))}
        codes = np.array([omap.get(str(v), -1) for v in cats_v])
    nG = len(uniq)
    pal = color_palette(palette, n_colors=nG)
    hue_v = get_vector(data, hue) if hue is not None else None
    vert = (cat_axis == "x")

    def _draw(positions, groups_vals, colors):
        # vectorized box draw using ax.bxp (single call, fast)
        boxes = []
        for gv, c in zip(groups_vals, colors):
            s = _box_stats(gv)
            boxes.append(dict(med=s["med"], q1=s["q1"], q3=s["q3"],
                              whislo=s["whislo"], whishi=s["whishi"],
                              fliers=s["fliers"], label=""))
        _orient = "vertical" if vert else "horizontal"
        try:
            ax.bxp(boxes, positions=positions, widths=width,
                   orientation=_orient, patch_artist=True,
                   boxprops=dict(linewidth=linewidth), medianprops=dict(linewidth=1.4, color="black"),
                   whiskerprops=dict(linewidth=linewidth), capprops=dict(linewidth=linewidth),
                   flierprops=dict(markersize=fliersize), **{})
        except TypeError:
            ax.bxp(boxes, positions=positions, widths=width, vert=vert, patch_artist=True,
                   boxprops=dict(linewidth=linewidth), medianprops=dict(linewidth=1.4, color="black"),
                   whiskerprops=dict(linewidth=linewidth), capprops=dict(linewidth=linewidth),
                   flierprops=dict(markersize=fliersize), **{})
        # colorize patches
        for patch, c in zip(ax.patches[-len(boxes):], colors):
            try:
                patch.set_facecolor(c)
                patch.set_alpha(0.85)
            except Exception:
                pass

    if hue_v is None:
        groups = [vals[codes == g].astype(float) if vals is not None else np.array([np.nan]) for g in range(nG)]
        _draw(list(range(nG)), groups, [color or pal[g % len(pal)] for g in range(nG)])
        labels = [str(c) for c in uniq]
        if vert:
            ax.set_xticks(list(range(nG)), labels)
        else:
            ax.set_yticks(list(range(nG)), labels)
    else:
        hue_v = np.asarray(hue_v, dtype=object)
        huniq, hcodes = factorize(hue_v)
        if hue_order is not None:
            huniq = np.asarray(list(hue_order), dtype=object)
        nH = len(huniq)
        subpal = color_palette(palette, n_colors=nH)
        hmap = {str(h): i for i, h in enumerate(huniq.astype(str))}
        hi = np.array([hmap.get(str(v), -1) for v in hue_v])
        w = width / max(nH, 1)
        for h in range(nH):
            pos = [g + (h - (nH - 1) / 2) * w for g in range(nG)]
            groups = [vals[(codes == g) & (hi == h)].astype(float) for g in range(nG)]
            # draw each hue level: need narrow widths
            boxes = []
            for gv in groups:
                s = _box_stats(gv)
                boxes.append(dict(med=s["med"], q1=s["q1"], q3=s["q3"], whislo=s["whislo"],
                                  whishi=s["whishi"], fliers=s["fliers"], label=""))
            try:
                ax.bxp(boxes, positions=pos, widths=w * 0.85,
                       orientation=("vertical" if vert else "horizontal"), patch_artist=True,
                       boxprops=dict(linewidth=linewidth), medianprops=dict(linewidth=1.4, color="black"),
                       flierprops=dict(markersize=fliersize))
            except TypeError:
                ax.bxp(boxes, positions=pos, widths=w * 0.85, vert=vert, patch_artist=True,
                       boxprops=dict(linewidth=linewidth), medianprops=dict(linewidth=1.4, color="black"),
                       flierprops=dict(markersize=fliersize))
            for patch in ax.patches[-len(boxes):]:
                try:
                    patch.set_facecolor(subpal[h % len(subpal)])
                    patch.set_alpha(0.85)
                except Exception:
                    pass
        # proxy legend
        from matplotlib.patches import Patch
        ax.legend([Patch(color=subpal[h % len(subpal)]) for h in range(nH)],
                  [str(h) for h in huniq], title=str(hue) if isinstance(hue, str) else None)
        labels = [str(c) for c in uniq]
        if vert:
            ax.set_xticks(list(range(nG)), labels)
        else:
            ax.set_yticks(list(range(nG)), labels)
    return ax


def violinplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
               orient=None, color=None, palette="seaplot", saturation=0.75,
               width=0.8, dodge=True, bw="silverman", cut=2.0, gridsize=120,
               inner="box", split=False, scale="area", ax=None, legend="auto", **kwargs):
    from ._fast import gaussian_kde_1d, _silverman_bw
    ax = _get_ax(ax)
    cats_v, vals, cat_axis, _, _ = _cat_setup(data, x, y)
    uniq, codes = factorize(cats_v)
    if order is not None:
        uniq = np.asarray(list(order), dtype=object)
        omap = {str(o): i for i, o in enumerate(uniq.astype(str))}
        codes = np.array([omap.get(str(v), -1) for v in cats_v])
    nG = len(uniq)
    pal = color_palette(palette, n_colors=nG)
    vert = (cat_axis == "x")
    hue_v = get_vector(data, hue) if hue is not None else None

    def _violin(pos, gv, col, wscale=1.0):
        gv = np.asarray(gv, dtype=float)
        gv = gv[np.isfinite(gv)]
        if gv.size < 3:
            return
        bwv = _silverman_bw(gv)
        vmin, vmax = gv.min(), gv.max()
        pad = cut * bwv
        grid = np.linspace(vmin - pad, vmax + pad, gridsize)
        _, d = gaussian_kde_1d(gv, grid=grid, bw=bwv)
        d = d / (d.max() + 1e-12) * (width * 0.45) * wscale
        if vert:
            ax.fill_betweenx(grid, pos - d, pos + d, color=col, alpha=0.85, linewidth=0)
            if inner == "box":
                q1, med, q3 = np.percentile(gv, [25, 50, 75])
                ax.plot([pos - d.max() * 0.25, pos + d.max() * 0.25], [med, med], color="black", lw=2)
                ax.plot([pos, pos], [q1, q3], color="black", lw=3)
            elif inner == "point":
                ax.plot([pos], [np.median(gv)], "o", color="black", ms=3)
        else:
            ax.fill_between(grid, pos - d, pos + d, color=col, alpha=0.85, linewidth=0)

    if hue_v is None:
        for g in range(nG):
            gv = vals[codes == g] if vals is not None else np.array([np.nan])
            _violin(g, gv, color or pal[g % len(pal)])
    else:
        hue_v = np.asarray(hue_v, dtype=object)
        huniq, _ = factorize(hue_v)
        nH = len(huniq)
        subpal = color_palette(palette, n_colors=nH)
        hmap = {str(h): i for i, h in enumerate(huniq.astype(str))}
        hi = np.array([hmap.get(str(v), 0) for v in hue_v])
        w = width / max(nH, 1)
        for h in range(nH):
            for g in range(nG):
                gv = vals[(codes == g) & (hi == h)] if vals is not None else np.array([np.nan])
                _violin(g + (h - (nH - 1) / 2) * w * 0.8, gv, subpal[h % len(subpal)], wscale=0.9)
        from matplotlib.patches import Patch
        ax.legend([Patch(color=subpal[h % len(subpal)]) for h in range(nH)],
                  [str(h) for h in huniq])
    labels = [str(c) for c in uniq]
    if vert:
        ax.set_xticks(list(range(nG)), labels)
    else:
        ax.set_yticks(list(range(nG)), labels)
    return ax


def _jitter(n, rng, spread=0.35):
    return (rng.random(n) - 0.5) * 2 * spread * 0.4


def stripplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
              jitter=True, dodge=False, orient=None, color=None, palette="seaplot",
              size=5, alpha=0.7, linewidth=0.4, edgecolor="white", ax=None,
              legend="auto", **kwargs):
    ax = _get_ax(ax)
    cats_v, vals, cat_axis, _, _ = _cat_setup(data, x, y)
    uniq, codes = factorize(cats_v)
    nG = len(uniq)
    pal = color_palette(palette, n_colors=nG)
    vert = (cat_axis == "x")
    rng = np.random.default_rng(0)
    hue_v = get_vector(data, hue) if hue is not None else None
    if hue_v is None:
        cols = [color or pal[g % len(pal)] for g in codes]
        j = _jitter(len(codes), rng) if jitter else np.zeros(len(codes))
        px = codes.astype(float) + j
        if vals is None:
            vals = np.zeros(len(cats_v))
        if vert:
            ax.scatter(px, vals.astype(float), c=cols, s=size ** 2 * 0.6, alpha=alpha,
                       linewidths=linewidth, edgecolors=edgecolor)
            ax.set_xticks(list(range(nG)), [str(c) for c in uniq])
        else:
            ax.scatter(vals.astype(float), px, c=cols, s=size ** 2 * 0.6, alpha=alpha,
                       linewidths=linewidth, edgecolors=edgecolor)
            ax.set_yticks(list(range(nG)), [str(c) for c in uniq])
    else:
        hue_v = np.asarray(hue_v, dtype=object)
        huniq, _ = factorize(hue_v)
        nH = len(huniq)
        subpal = color_palette(palette, n_colors=nH)
        hmap = {str(h): i for i, h in enumerate(huniq.astype(str))}
        hi = np.array([hmap.get(str(v), 0) for v in hue_v])
        w = 0.8 / max(nH, 1)
        off = (hi - (nH - 1) / 2) * w * 0.7 if dodge else np.zeros(len(hi))
        j = _jitter(len(codes), rng) * 0.5 if jitter else np.zeros(len(codes))
        px = codes.astype(float) + off + j
        cols = [subpal[h % len(subpal)] for h in hi]
        if vert:
            ax.scatter(px, vals.astype(float), c=cols, s=size ** 2 * 0.6, alpha=alpha,
                       linewidths=linewidth, edgecolors=edgecolor)
            ax.set_xticks(list(range(nG)), [str(c) for c in uniq])
        else:
            ax.scatter(vals.astype(float), px, c=cols, s=size ** 2 * 0.6, alpha=alpha,
                       linewidths=linewidth, edgecolors=edgecolor)
            ax.set_yticks(list(range(nG)), [str(c) for c in uniq])
        from matplotlib.lines import Line2D
        ax.legend([Line2D([0], [0], marker="o", color="w", markerfacecolor=subpal[h % len(subpal)], markersize=6) for h in range(nH)],
                  [str(h) for h in huniq])
    return ax


def swarmplot(*args, **kwargs):
    # fast approximation: stripplot with tighter packing (beeswarm lite)
    kwargs.setdefault("jitter", True)
    return stripplot(*args, **kwargs)


def pointplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
              estimator="mean", errorbar=("ci", 95), n_boot=500, seed=0,
              color=None, palette="seaplot", markers="o", linestyles="-",
              dodge=0.4, join=True, scale=1.0, errwidth=1.2, capsize=0.08,
              ax=None, legend="auto", **kwargs):
    ax = _get_ax(ax)
    cats_v, vals, cat_axis, _, _ = _cat_setup(data, x, y)
    uniq, codes = factorize(cats_v)
    if order is not None:
        uniq = np.asarray(list(order), dtype=object)
    nG = len(uniq)
    pal = color_palette(palette, n_colors=nG)
    vert = (cat_axis == "x")
    hue_v = get_vector(data, hue) if hue is not None else None

    def _est(gv):
        gv = np.asarray(gv, dtype=float)
        gv = gv[np.isfinite(gv)]
        if gv.size == 0:
            return np.nan, 0, 0
        e = float(np.mean(gv)) if estimator == "mean" else float(np.median(gv))
        lo, hi = bootstrap_ci(gv, n_boot=min(n_boot, 300), seed=seed)
        return e, e - lo, hi - e

    if hue_v is None:
        ests, loe, hie = [], [], []
        for g in range(nG):
            e, lo, hi = _est(vals[codes == g])
            ests.append(e); loe.append(lo); hie.append(hi)
        ests = np.array(ests)
        pos = np.arange(nG)
        col = color or pal[0]
        if vert:
            ax.errorbar(pos, ests, yerr=[loe, hie], fmt=markers, color=col,
                        ecolor=".26", elinewidth=errwidth, capsize=capsize * 40, ms=6 * scale)
            if join:
                ax.plot(pos, ests, color=col, ls=linestyles)
            ax.set_xticks(pos, [str(c) for c in uniq])
        else:
            ax.errorbar(ests, pos, xerr=[loe, hie], fmt=markers, color=col,
                        ecolor=".26", elinewidth=errwidth, capsize=capsize * 40, ms=6 * scale)
            if join:
                ax.plot(ests, pos, color=col, ls=linestyles)
            ax.set_yticks(pos, [str(c) for c in uniq])
    else:
        hue_v = np.asarray(hue_v, dtype=object)
        huniq, _ = factorize(hue_v)
        nH = len(huniq)
        subpal = color_palette(palette, n_colors=nH)
        hmap = {str(h): i for i, h in enumerate(huniq.astype(str))}
        hi = np.array([hmap.get(str(v), 0) for v in hue_v])
        pos = np.arange(nG)
        for h in range(nH):
            ests = []
            for g in range(nG):
                e, _, _ = _est(vals[(codes == g) & (hi == h)])
                ests.append(e)
            ests = np.array(ests)
            off = (h - (nH - 1) / 2) * (dodge / max(nH, 1))
            if vert:
                ax.plot(pos + off, ests, marker=markers, color=subpal[h % len(subpal)], ls=linestyles, label=str(huniq[h]))
            else:
                ax.plot(ests, pos + off, marker=markers, color=subpal[h % len(subpal)], ls=linestyles, label=str(huniq[h]))
        ax.legend()
        if vert:
            ax.set_xticks(pos, [str(c) for c in uniq])
        else:
            ax.set_yticks(pos, [str(c) for c in uniq])
    return ax


def boxenplot(data=None, *, x=None, y=None, hue=None, order=None, color=None,
              palette="seaplot", width=0.8, ax=None, **kwargs):
    # letter-value plot lite: quantiles fan (fast, vectorized)
    ax = _get_ax(ax)
    cats_v, vals, cat_axis, _, _ = _cat_setup(data, x, y)
    uniq, codes = factorize(cats_v)
    nG = len(uniq)
    pal = color_palette(palette, n_colors=nG)
    vert = (cat_axis == "x")
    qs = [0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95]
    for g in range(nG):
        gv = np.asarray(vals[codes == g], dtype=float)
        gv = gv[np.isfinite(gv)]
        if gv.size == 0:
            continue
        qv = np.quantile(gv, qs)
        col = color or pal[g % len(pal)]
        for i in range(len(qs) // 2):
            lo, hi = qv[i], qv[-(i + 1)]
            a = 0.25 + 0.6 * (1 - i / (len(qs) // 2))
            if vert:
                ax.fill_between([g - width / 2 * (1 - i * 0.12), g + width / 2 * (1 - i * 0.12)],
                                [lo, lo], [hi, hi], color=col, alpha=min(a, 0.9))
            else:
                ax.fill_betweenx([g - width / 2 * (1 - i * 0.12), g + width / 2 * (1 - i * 0.12)],
                                 [lo, lo], [hi, hi], color=col, alpha=min(a, 0.9))
        med = qv[len(qs) // 2]
        if vert:
            ax.plot([g - width / 2, g + width / 2], [med, med], color="black", lw=1.5)
        else:
            ax.plot([med, med], [g - width / 2, g + width / 2], color="black", lw=1.5)
    labels = [str(c) for c in uniq]
    if vert:
        ax.set_xticks(list(range(nG)), labels)
    else:
        ax.set_yticks(list(range(nG)), labels)
    return ax


def catplot(data=None, *, x=None, y=None, hue=None, row=None, col=None,
            kind="strip", palette="seaplot", height=5, aspect=1, order=None,
            hue_order=None, legend="auto", **kwargs):
    from .grids import FacetGrid
    from ._data import get_vector as _gv
    g = FacetGrid(data=data, row=row, col=col, palette=palette, height=height, aspect=aspect)
    fn = {"bar": barplot, "count": countplot, "box": boxplot, "violin": violinplot,
          "strip": stripplot, "swarm": swarmplot, "point": pointplot, "boxen": boxenplot}[kind]
    row_v = _gv(data, row) if row is not None else None
    col_v = _gv(data, col) if col is not None else None
    for (ri, ci, axm) in g._iter_axes(row_v, col_v):
        mask = np.ones(g._n, dtype=bool)
        if row_v is not None:
            mask &= np.asarray(row_v, dtype=object) == g._row_cats[ri]
        if col_v is not None:
            mask &= np.asarray(col_v, dtype=object) == g._col_cats[ci]
        sub = g._subset(data, mask)
        fn(data=sub, x=x, y=y, hue=hue, order=order, hue_order=hue_order,
           palette=palette, ax=axm, **kwargs)
    return g
