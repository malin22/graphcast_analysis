import numpy as np


def to_float32(array: np.ndarray) -> np.ndarray:
    """Convert normal and bfloat16-like arrays to float32."""
    array = np.asarray(array)

    if array.dtype == np.dtype("|V2"):
        array = array.view(np.float16)

    return array.astype(np.float32, copy=False)


def get_node_selector(
    selected_indices: np.ndarray,
    total_nodes: int,
):
    """
    Return slice(None) when all nodes are selected in their original order.

    A slice is cheaper than NumPy advanced indexing.
    """
    expected = np.arange(
        total_nodes,
        dtype=selected_indices.dtype,
    )

    if (
        len(selected_indices) == total_nodes
        and np.array_equal(selected_indices, expected)
    ):
        return slice(None)

    return selected_indices


def correlations_streaming(
    pca_sources,
    feature_grid: np.ndarray,
    selected_indices: np.ndarray,
    n_pcs: int,
    time_chunk_size: int,
):
    """
    Calculate correlations between one ERA5 feature and all PCs without
    materializing the complete sample-by-PC matrix.

    feature_grid must have shape:
        (all_times, selected_nodes)

    PCA data are read from their memory-mapped files in small time chunks.
    """
    feature_grid = np.asarray(
        feature_grid,
        dtype=np.float32,
    )

    expected_times = sum(
        len(source["timestamps"])
        for source in pca_sources
    )

    if feature_grid.ndim != 2:
        raise ValueError(
            f"feature_grid must be 2D, got {feature_grid.shape}"
        )

    if feature_grid.shape[0] != expected_times:
        raise ValueError(
            f"Feature has {feature_grid.shape[0]} timesteps, "
            f"but PCA sources have {expected_times}"
        )

    if feature_grid.shape[1] != len(selected_indices):
        raise ValueError(
            f"Feature has {feature_grid.shape[1]} nodes, "
            f"but {len(selected_indices)} nodes were selected"
        )

    sum_x = np.zeros(n_pcs, dtype=np.float64)
    sum_x2 = np.zeros(n_pcs, dtype=np.float64)
    sum_xy = np.zeros(n_pcs, dtype=np.float64)

    sum_y = 0.0
    sum_y2 = 0.0
    n_valid = 0

    global_time_offset = 0

    for source in pca_sources:
        scores = source["scores"]
        source_times = scores.shape[0]

        node_selector = get_node_selector(
            selected_indices,
            scores.shape[1],
        )

        for start in range(
            0,
            source_times,
            time_chunk_size,
        ):
            stop = min(
                start + time_chunk_size,
                source_times,
            )

            global_start = global_time_offset + start
            global_stop = global_time_offset + stop

            x_chunk = to_float32(
                scores[
                    start:stop,
                    node_selector,
                    :n_pcs,
                ]
            )

            y_chunk = feature_grid[
                global_start:global_stop
            ]

            x_chunk = x_chunk.reshape(
                -1,
                n_pcs,
            )

            y_chunk = y_chunk.reshape(-1)

            valid_y = np.isfinite(y_chunk)

            if not valid_y.any():
                del x_chunk
                continue

            valid_x = np.all(
                np.isfinite(x_chunk),
                axis=1,
            )

            valid = valid_y & valid_x

            if not valid.any():
                del x_chunk
                continue

            xv = x_chunk[valid]
            yv = y_chunk[valid]

            chunk_n = int(yv.size)
            n_valid += chunk_n

            sum_y += np.sum(
                yv,
                dtype=np.float64,
            )

            sum_y2 += np.einsum(
                "i,i->",
                yv,
                yv,
                dtype=np.float64,
            )

            sum_x += np.sum(
                xv,
                axis=0,
                dtype=np.float64,
            )

            sum_x2 += np.einsum(
                "ij,ij->j",
                xv,
                xv,
                dtype=np.float64,
            )

            sum_xy += np.einsum(
                "ij,i->j",
                xv,
                yv,
                dtype=np.float64,
            )

            del x_chunk
            del xv
            del yv

        global_time_offset += source_times

    correlations = np.full(
        n_pcs,
        np.nan,
        dtype=np.float64,
    )

    if n_valid < 3:
        return correlations, n_valid

    numerator = (
        sum_xy
        - sum_x * sum_y / n_valid
    )

    x_ss = (
        sum_x2
        - sum_x * sum_x / n_valid
    )

    y_ss = (
        sum_y2
        - sum_y * sum_y / n_valid
    )

    x_ss = np.maximum(
        x_ss,
        0.0,
    )

    y_ss = max(
        y_ss,
        0.0,
    )

    denominator = np.sqrt(
        x_ss * y_ss
    )

    usable = denominator > 0.0

    correlations[usable] = (
        numerator[usable]
        / denominator[usable]
    )

    correlations = np.clip(
        correlations,
        -1.0,
        1.0,
    )

    return correlations, n_valid