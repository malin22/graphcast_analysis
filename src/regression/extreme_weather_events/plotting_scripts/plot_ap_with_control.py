from pathlib import Path

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

RESULTS_ROOT = Path(
    "results/regression/extreme_weather_events"
)

PLOTS_DIR = Path(
    f"plots/regression/extreme_weather_events/"
    f"decodability/")

os.makedirs(PLOTS_DIR, exist_ok=True)

NODE_LEVEL = 6
LABEL_MODE = "intersection"
MAX_HOURS = 3

EVENTS = ["AR", "TC"]

colors = [
    "gray",  # positive fraction (random ranking)
    "lightseagreen",  # shuffled days
    "firebrick",  # decoding result
]


# ============================================================
# LOAD CORRECTLY ALIGNED 512-D RESULT
# ============================================================

def load_real_result(event):
    """
    Load the correctly aligned raw-activation result.
    """

    path = (
        RESULTS_ROOT
        / event
        / f"Node_Hierarchy_Level_M{NODE_LEVEL}"
        / "raw_activations"
        / (
            f"logistic_probe_{event}_raw_activations_"
            f"{LABEL_MODE}_M{NODE_LEVEL}_max_{MAX_HOURS}hour.csv"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Real result not found:\n{path}"
        )

    df = pd.read_csv(path)

    # Select complete 512-dimensional representation
    row = df[df["n_features"] == 512]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one 512-d row in {path}, "
            f"found {len(row)}."
        )

    return row.iloc[0]


# ============================================================
# LOAD SHUFFLED RESULTS
# ============================================================

def load_shuffled_results(event):
    """
    Load all shuffled-timestep 512-d runs across seeds.
    """

    root = (
        RESULTS_ROOT
        / event
        / f"Node_Hierarchy_Level_M{NODE_LEVEL}"
        / "permuted_masks_raw_activations"
    )

    seed_dirs = sorted(
        root.glob("seed_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )

    if not seed_dirs:
        raise FileNotFoundError(
            f"No shuffled seed directories found:\n{root}"
        )

    rows = []

    for seed_dir in seed_dirs:

        seed = int(seed_dir.name.split("_")[-1])

        # We don't need to know the exact filename because there
        # should only be one logistic-probe CSV for this experiment.
        csv_files = list(seed_dir.glob("*.csv"))

        # Ignore mask-permutation metadata CSV if present
        csv_files = [
            p for p in csv_files
            if "mask_permutation" not in p.name
        ]

        result_found = False

        for path in csv_files:

            df = pd.read_csv(path)

            if "n_features" not in df.columns:
                continue

            row = df[df["n_features"] == 512]

            if len(row) != 1:
                continue

            row = row.iloc[0]

            rows.append({
                "seed": seed,
                "test_average_precision":
                    float(row["test_average_precision"]),
                "test_positive_rate":
                    float(row["test_positive_rate"]),
                "test_roc_auc":
                    float(row["test_roc_auc"]),
                "test_f1":
                    float(row["test_f1"]),
            })

            result_found = True
            break

        if not result_found:
            print(
                f"WARNING: no valid result CSV found in "
                f"{seed_dir}"
            )

    if not rows:
        raise RuntimeError(
            f"No shuffled results loaded for {event}."
        )

    return pd.DataFrame(rows).sort_values("seed")


# ============================================================
# COLLECT RESULTS FOR ONE EVENT
# ============================================================

def collect_event_results(event):

    real = load_real_result(event)
    shuffled = load_shuffled_results(event)

    prevalence = float(real["test_positive_rate"])
    real_ap = float(real["test_average_precision"])

    shuffled_ap = shuffled["test_average_precision"]

    # --------------------------------------------------------
    # Sanity check:
    # permutation should preserve class prevalence
    # --------------------------------------------------------

    if not np.allclose(
        shuffled["test_positive_rate"],
        prevalence,
        atol=1e-10,
    ):
        raise ValueError(
            f"{event}: shuffled test prevalence differs "
            f"from correctly aligned test prevalence."
        )

    # --------------------------------------------------------
    # Print diagnostics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(event)
    print("=" * 60)

    print(f"Number of shuffled runs : {len(shuffled)}")
    print(f"Positive fraction       : {prevalence:.4f}")
    print(f"Correctly aligned AP    : {real_ap:.4f}")
    print(
        f"Shuffled AP             : "
        f"{shuffled_ap.mean():.4f} "
        f"± {shuffled_ap.std():.4f}"
    )

    print("\nIndividual shuffled runs:")
    print(
        shuffled[
            ["seed", "test_average_precision"]
        ].to_string(index=False)
    )

    return {
        "prevalence": prevalence,
        "shuffled": shuffled_ap.mean(),
        "real": real_ap,
    }


# ============================================================
# LOAD BOTH EVENT TYPES
# ============================================================

results = {
    event: collect_event_results(event)
    for event in EVENTS
}


# ============================================================
# PLOT
# ============================================================

conditions = [
    "Random ranking",
    "Shuffled timesteps",
    "Correctly aligned",
]

keys = [
    "prevalence",
    "shuffled",
    "real",
]

x = np.arange(len(EVENTS))
width = 0.24

fig, ax = plt.subplots(figsize=(7.0, 4.6))


for i, (condition, key, color) in enumerate(
    zip(conditions, keys, colors)):


    values = [
        results[event][key]
        for event in EVENTS
    ]

    offset = (i - 1) * width

    bars = ax.bar(
        x + offset,
        values,
        width=width,
        label=condition,
        edgecolor="black",
        linewidth=0.7,
        color=color,
    )

    # Values above bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ============================================================
# FORMATTING
# ============================================================

ax.set_ylabel("Average precision (AP)", fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(
    [
        "Atmospheric river (AR)",
        "Tropical cyclone (TC)",
    ],
    fontsize=10,
)

ax.set_ylim(bottom=0)

ax.yaxis.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.35,
)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    fontsize=9,
)

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    PLOTS_DIR / "extreme_weather_decodability_AP.png",
    bbox_inches="tight",
)


plt.show()