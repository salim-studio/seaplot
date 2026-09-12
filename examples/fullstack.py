"""seaplot full-stack demo: viz + DB + wrangle + EDA + ML + DL."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaplot as sp

sp.seed_everything(0)
sp.set_theme(style="whitegrid", palette="seaplot")

# 1) data
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "total_bill": rng.normal(20, 8, 500).round(2),
    "tip": rng.normal(3, 1.2, 500).round(2),
    "sex": rng.choice(["Male", "Female"], 500),
    "day": rng.choice(["Thur", "Fri", "Sat", "Sun"], 500),
    "size": rng.integers(1, 6, 500),
})
df.loc[rng.choice(500, 20, replace=False), "tip"] = np.nan

# 2) DB round-trip
db = sp.connect(":memory:")
db.write_df(df, "tips")
print("tables:", db.tables())
print(db.query("SELECT day, AVG(total_bill) AS m FROM tips GROUP BY day"))

# 3) wrangle + EDA + stats
df = sp.clean(df)
print(sp.describe(df).head())
print(sp.missing_table(df).head())
rep = sp.profile(df, plots=False)
print("profile shape:", rep["shape"], "dupes:", rep["n_duplicates"])
print(sp.ttest(df, "tip", group="sex"))

# 4) viz gallery
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sp.scatterplot(data=df, x="total_bill", y="tip", hue="day", ax=axes[0])
sp.histplot(data=df, x="total_bill", hue="sex", kde=True, ax=axes[1])
sp.boxplot(data=df, x="day", y="total_bill", ax=axes[2])
fig.tight_layout()
fig.savefig("examples_gallery.png", dpi=100)
print("saved examples_gallery.png")

# 5) ML quick win
df["big_tip"] = (df["tip"] > df["tip"].median()).astype(int)
model, report = sp.train(df, target="big_tip", task="classification")
print({k: v for k, v in report.items() if k != "model"})

# 6) DL lite
X, y = sp.encode(df, target="big_tip")
mlp = sp.MLP(hidden=(32,), epochs=5, verbose=False)
mlp.fit(X.to_numpy()[:400], np.asarray(y)[:400])
print("mlp preds:", mlp.predict(X.to_numpy()[400:405]))
