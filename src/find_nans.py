# save as find_nans.py and run: python find_nans.py /path/to/acts_dir
import sys, os
import numpy as np
from glob import glob

acts_dir = '/share/prj-4d/graphcast_shared/data/graphcast_activation_2021'

for f in sorted(glob(os.path.join(acts_dir, "*.npy"))):
    x = np.load(f, mmap_mode="r")
    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2:
        print(f"{os.path.basename(f)}: unexpected shape {x.shape}; skipping")
        continue
    total_nans = int(np.isnan(x).sum())
    if total_nans:
        print(f"{os.path.basename(f)}: NaNs={total_nans}, shape={x.shape}")