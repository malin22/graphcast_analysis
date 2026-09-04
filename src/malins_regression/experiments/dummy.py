
import numpy as np

x = np.load(
    "results/logistic_regression/TC/"
    "Node_Hierarchy_Level_M6/l1_pc_selection/"
    "c_sweep_results/c_01.npz"
)

coef = np.abs(x["coef_l1"])

print("C:", x["C"])
print("min:", coef.min())
print("max:", coef.max())

for threshold in [
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
]:
    print(
        f"{threshold:.0e}: "
        f"{np.sum(coef > threshold)} PCs"
    )

print()
print("Coefficient percentiles:")
for p in [0, 25, 50, 75, 90, 95, 99, 100]:
    print(p, np.percentile(coef, p))
