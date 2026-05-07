"""Plot a 3x3 heatmap of mean weighted F1 across the CatBoost CV grid."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


CV_RESULTS_PATH = Path("processed_output/models/catboost/cv_tuning_results.json")
OUTPUT_PATH = Path("processed_output/models/catboost/cv_heatmap.png")


def main():
    payload = json.loads(CV_RESULTS_PATH.read_text())
    learning_rates = sorted(payload["grid"]["learning_rate"])
    depths = sorted(payload["grid"]["depth"])

    matrix = np.zeros((len(depths), len(learning_rates)))
    for entry in payload["trace"]:
        i = depths.index(entry["params"]["depth"])
        j = learning_rates.index(entry["params"]["learning_rate"])
        matrix[i, j] = entry["mean_weighted_f1"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        xticklabels=learning_rates,
        yticklabels=depths,
        cbar_kws={"label": "Mean weighted F1 (5-fold CV)"},
        ax=ax,
    )
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("depth")
    ax.set_title("CatBoost CV grid search — mean weighted F1")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
