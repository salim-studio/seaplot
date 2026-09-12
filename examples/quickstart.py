"""seaplot quickstart — mirrors common seaborn workflows."""
import numpy as np
import matplotlib.pyplot as plt
import seaplot as sp

sp.set_theme(style="whitegrid", palette="seaplot")

rng = np.random.default_rng(0)
n = 500
import pandas as pd
df = pd.DataFrame({
    "x": rng.normal(size=n), "y": rng.normal(size=n),
    "group": rng.choice(list("ABC"), n),
    "cat": rng.choice(["Mon", "Tue", "Wed"], n),
    "val": rng.normal(10, 3, n),
})

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
sp.scatterplot(data=df, x="x", y="y", hue="group", ax=axes[0, 0])
sp.lineplot(data=df.sort_values("x"), x="x", y="y", hue="group", ax=axes[0, 1])
sp.histplot(data=df, x="val", hue="group", kde=True, ax=axes[0, 2])
sp.boxplot(data=df, x="cat", y="val", ax=axes[1, 0])
sp.violinplot(data=df, x="cat", y="val", ax=axes[1, 1])
sp.heatmap(np.corrcoef(rng.normal(size=(6, 100))), annot=True, ax=axes[1, 2])
fig.tight_layout()
fig.savefig("examples_gallery.png", dpi=100)
print("saved examples_gallery.png")
