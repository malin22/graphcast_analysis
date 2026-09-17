import os
import numpy as np
import pandas as pd

acts_dir = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"

expected_times = np.arange(
    np.datetime64("2021-01-01T00"),
    np.datetime64("2022-01-01T00"),
    np.timedelta64(6, "h"),
)

missing = []
existing = []

for t in expected_times:
    t_str = np.datetime_as_string(t, unit="h")
    fname = f"layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t{t_str}.npy"
    path = os.path.join(acts_dir, fname)

    if os.path.exists(path):
        existing.append(path)
    else:
        missing.append(t_str)

print(f"Expected activations: {len(expected_times)}")
print(f"Existing activations: {len(existing)}")
print(f"Missing activations:  {len(missing)}")

if missing:
    print("\nMissing times:")
    for t in missing[:50]:
        print(t)
    if len(missing) > 50:
        print(f"... and {len(missing) - 50} more")
else:
    print("\nAll activations are present")