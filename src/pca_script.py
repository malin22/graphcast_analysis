import os
import numpy as np
from glob import glob
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

# Directory containing .npy activation files
ACTS_DIR = '/share/prj-4d/graphcast_shared/data/graphcast_activation'
BATCH_SIZE = 10  # Number of files to process per batch (tune as needed)
N_COMPONENTS = 20  # Number of principal components

# Find all .npy files (sorted by time if filenames allow)
npy_files = sorted(glob(os.path.join(ACTS_DIR, '*.npy')))

# Initialize IncrementalPCA
ipca = IncrementalPCA(n_components=N_COMPONENTS)

# First pass: partial_fit in batches
for i in range(0, len(npy_files), BATCH_SIZE):
    batch_files = npy_files[i:i+BATCH_SIZE]
    batch_data = []
    for f in batch_files:
        data = np.load(f)  # shape: (num_nodes, 512)
        if data.dtype == np.dtype("|V2"):
            data = data.view(np.float16)
        data = np.asarray(data)
        data = np.squeeze(data)  # Remove singleton dimensions if present
        batch_data.append(data)
    batch_data = np.vstack(batch_data)  # shape: (batch_nodes, 512)
    print(f"Fitting batch {i//BATCH_SIZE+1}: {batch_data.shape}")
    ipca.partial_fit(batch_data)

# Explained variance
print("Explained variance ratio (first 20 components):")
print(ipca.explained_variance_ratio_)
print("Cumulative explained variance (first 20 components):")
print(np.cumsum(ipca.explained_variance_ratio_))

# Plot cumulative explained variance
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, N_COMPONENTS+1), np.cumsum(ipca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Incremental PCA on Activation Nodes')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally, project data in batches and save results
# for i in range(0, len(npy_files), BATCH_SIZE):
#     batch_files = npy_files[i:i+BATCH_SIZE]
#     batch_data = []
#     for f in batch_files:
#         data = np.load(f)
#         batch_data.append(data)
#     batch_data = np.vstack(batch_data)
#     projected = ipca.transform(batch_data)
#     np.save(f'projected_batch_{i//BATCH_SIZE+1}.npy', projected)