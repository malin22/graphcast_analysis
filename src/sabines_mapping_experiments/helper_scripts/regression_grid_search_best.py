import json
import pandas as pd

def load_grid_as_dataframe(results_path):
    with open(results_path) as f:
        results = json.load(f)

    rows = []
    for pc_key, pc_result in results.items():
        for g in pc_result["grid_search"]:
            rows.append({"pc": pc_key, **g})

    return pd.DataFrame(rows)


df = load_grid_as_dataframe("regression_results.json")

def best_row(group, tol=0.02):
    best_r2 = group["val_r2"].max()
    candidates = group[group["val_r2"] >= best_r2 - tol]
    return candidates.loc[candidates["n_nonzero"].idxmin()]

summary = df.groupby("pc").apply(best_row).reset_index(drop=True)
print(summary[["pc", "l1_ratio", "alpha", "val_r2", "n_nonzero"]])