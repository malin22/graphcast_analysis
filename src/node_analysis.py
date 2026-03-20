import numpy as np
import matplotlib.pyplot as plt
from graphcast import icosahedral_mesh

ACTS_PATH = "/share/prj-4d/graphcast_shared/data/graphcast_activation/layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t2021-08-29T06.npy"

# helper functions for analyzing node activations across the icosahedral mesh hierarchy
def load_activations(path: str) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    # Saved activations can come back as raw 2-byte blobs.
    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    # Sometimes shape is [nodes, batch, features].
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got shape {x.shape}")

    # Cast up for stable reductions.
    return x.astype(np.float32)

def mesh_hierarchy_indices(splits: int = 6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    verts = [m.vertices for m in meshes]
    v6 = verts[6]

    def to_key(v):
        return tuple(np.round(v, 12))

    m6_map = {to_key(v): i for i, v in enumerate(v6)}

    def level_idx_in_m6(level):
        return np.array([m6_map[to_key(v)] for v in verts[level]], dtype=np.int64)

    idx_by_level_cumulative = {}
    idx_by_level_only = {}

    prev = np.array([], dtype=np.int64)
    for level in range(splits + 1):
        idx = level_idx_in_m6(level)
        idx_by_level_cumulative[level] = idx
        idx_by_level_only[level] = np.setdiff1d(idx, prev)
        prev = idx

    return verts[6], idx_by_level_cumulative, idx_by_level_only

def vertices_to_latlon(vertices: np.ndarray):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat, lon

def summarize_groups(A: np.ndarray, idx_m0: np.ndarray, idx_m6_only: np.ndarray):
    A_m0 = A[idx_m0]
    A_m6_only = A[idx_m6_only]

    mean_m0 = A_m0.mean(axis=0)
    mean_m6_only = A_m6_only.mean(axis=0)
    delta = mean_m0 - mean_m6_only

    abs_delta = np.abs(delta)
    top_idx = np.argsort(abs_delta)[-10:][::-1]

    print(f"A shape: {A.shape}, dtype: {A.dtype}")
    print(f"M0 nodes: {len(idx_m0)}")
    print(f"M6-only nodes: {len(idx_m6_only)}")
    print("Top 10 separating features:")
    for i in top_idx:
        print(
            f"feature {i:4d} | "
            f"M0 mean = {mean_m0[i]: .5f} | "
            f"M6-only mean = {mean_m6_only[i]: .5f} | "
            f"delta = {delta[i]: .5f}"
        )

    return delta, top_idx


# plotting functions to visualize feature differences across the mesh levels
def plot_top_feature_deltas(delta: np.ndarray, top_idx: np.ndarray):
    plt.figure(figsize=(10, 4))
    plt.bar([str(i) for i in top_idx], delta[top_idx], color="dimgray")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("M0 mean - M6-only mean")
    plt.xlabel("Feature index")
    plt.title("Top feature differences between M0 and M6-only nodes")
    plt.tight_layout()
    plt.savefig("top_feature_deltas.png", dpi=300)

def plot_feature_across_levels(
    A: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    idx_by_level: dict,
    feature_idx: int,
    mode_name: str,
    out_name: str,
):
    # Shared color scale across levels for fair comparison.
    all_vals = np.concatenate([A[idx, feature_idx] for idx in idx_by_level.values()])
    vmax = np.percentile(np.abs(all_vals), 99)
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.ravel()

    for level in range(7):
        ax = axes[level]
        idx = idx_by_level[level]
        vals = A[idx, feature_idx]

        # faint global mesh background
        ax.scatter(lon, lat, s=1, c="lightgray", alpha=0.12, linewidths=0)

        sc = ax.scatter(
            lon[idx], lat[idx],
            c=vals, s=10 if len(idx) > 200 else 50,
            cmap="coolwarm", vmin=vmin, vmax=vmax,
            edgecolors="black" if len(idx) < 100 else "none",
            linewidths=0.3 if len(idx) < 100 else 0.0
        )

        ax.set_title(f"Level {level} (n={len(idx)})")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.grid(alpha=0.2)

    # 8th panel for colorbar/metadata
    axes[7].axis("off")
    cbar = fig.colorbar(sc, ax=axes[:7], shrink=0.85)
    cbar.set_label("Activation")

    fig.suptitle(f"Feature {feature_idx} across mesh levels (mode: {mode_name})", fontsize=14)
    fig.savefig(out_name, dpi=300, bbox_inches="tight")


def main():

    #plot_top_feature_deltas(delta, top_idx)


    A = load_activations(ACTS_PATH)
    vertices, idx_by_level_cum, idx_by_level_only = mesh_hierarchy_indices(splits=6)
    lat, lon = vertices_to_latlon(vertices)

    idx_m0 = idx_by_level_cum[0]
    idx_m6_only = idx_by_level_only[6]
    delta, top_idx = summarize_groups(A, idx_m0, idx_m6_only)

    for feature_idx in top_idx[:3]:
        plot_feature_across_levels(
            A, lat, lon, idx_by_level_cum, int(feature_idx),
            mode_name="cumulative", out_name=f"feature_{feature_idx}_levels_cumulative.png"
        )
        plot_feature_across_levels(
            A, lat, lon, idx_by_level_only, int(feature_idx),
            mode_name="only-new-nodes", out_name=f"feature_{feature_idx}_levels_only.png"
        )
        

if __name__ == "__main__":
    main()