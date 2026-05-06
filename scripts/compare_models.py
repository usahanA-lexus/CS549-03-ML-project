"""Head-to-head comparison: CatBoost vs LinearSVC vs LogisticRegression.

CatBoost results are read from existing artifacts under
`processed_output/models/catboost/` (multi-seed mean across seeds 0-4).
SVM and LogReg are trained inline against the same shared baseline so the
schema lines up. SVM and LogReg are deterministic with fixed C/random_state
and run with a single seed.

SVM mirrors `scripts/train_svm.py` (Deryn's script) with the addition of an
`unbalanced` no-description variant for symmetry with CatBoost's 4 cells.
LogReg mirrors a vanilla sklearn LogisticRegression pipeline with one-hot +
StandardScaler.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


TARGET = "Category"
SHARED_BASELINE = Path("processed_output/shared_baseline")
NO_DESC_BASELINE = Path("processed_output/baseline_no_description")
CATBOOST_RESULTS = Path("processed_output/models/catboost")
COMPARISON_OUTPUT = Path("processed_output/comparison")

# (cell_name, baseline_dir, include_description, class_weight)
ABLATION_CELLS = [
    ("baseline", SHARED_BASELINE, True, None),
    ("class_weighted", SHARED_BASELINE, True, "balanced"),
    ("no_description", NO_DESC_BASELINE, False, None),
    ("no_description_weighted", NO_DESC_BASELINE, False, "balanced"),
]

SVM_C_GRID = [0.01, 0.1, 1.0, 10.0]
LOGREG_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def load_split(baseline_dir):
    train = pd.read_csv(baseline_dir / "train_raw_split.csv")
    valid = pd.read_csv(baseline_dir / "valid_raw_split.csv")
    feature_columns = [c for c in train.columns if c != TARGET]
    return (
        train[feature_columns],
        train[TARGET],
        valid[feature_columns],
        valid[TARGET],
        feature_columns,
    )


def build_preprocessor(feature_columns):
    cat_features = [
        c for c in ["Description", "Transaction Type", "Account Name"] if c in feature_columns
    ]
    num_features = [c for c in ["Amount"] if c in feature_columns]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )


def evaluate(y_true, y_pred, training_time, classes):
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    per_class = {
        key: {m: float(v) if isinstance(v, (int, float)) else v for m, v in val.items()}
        for key, val in report.items()
        if isinstance(val, dict) and key != "accuracy"
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "training_time_seconds": float(training_time),
        "per_class_metrics": per_class,
    }


def train_svm(X_train, y_train, X_valid, y_valid, feature_columns, class_weight):
    pipe = Pipeline(
        [
            ("pre", build_preprocessor(feature_columns)),
            ("clf", LinearSVC(max_iter=5000, class_weight=class_weight)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        {"clf__C": SVM_C_GRID},
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    classes = sorted(y_train.unique())
    return grid.best_estimator_.predict(X_valid), elapsed, classes, grid.best_params_


def train_logreg(X_train, y_train, X_valid, y_valid, feature_columns, class_weight):
    pipe = Pipeline(
        [
            ("pre", build_preprocessor(feature_columns)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    grid = GridSearchCV(
        pipe,
        {"clf__C": LOGREG_C_GRID},
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    classes = sorted(y_train.unique())
    return grid.best_estimator_.predict(X_valid), elapsed, classes, grid.best_params_


def load_catboost_cell(cell_name):
    """Read pre-computed CatBoost metrics; return mean across seeds + per-class from seed 0."""
    metrics = json.loads((CATBOOST_RESULTS / cell_name / "metrics.json").read_text())
    return {
        "accuracy": metrics["accuracy"]["mean"],
        "weighted_f1": metrics["weighted_f1"]["mean"],
        "macro_f1": metrics["macro_f1"]["mean"],
        "training_time_seconds": metrics["training_time_seconds"]["mean"],
        "per_class_metrics": metrics["per_class_metrics_representative_seed"],
    }


def save_grouped_bar(records, metric, output_path, title):
    """Grouped bar chart: 4 cells × 3 models for the chosen metric."""
    df = pd.DataFrame(records)
    pivot = df.pivot(index="cell", columns="model", values=metric).reindex(
        [c[0] for c in ABLATION_CELLS]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("ablation cell")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(title="model")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_per_class_f1_baseline(records_with_per_class, output_path):
    """Per-class F1 for the baseline cell across all three models."""
    baseline_records = [r for r in records_with_per_class if r["cell"] == "baseline"]
    classes = sorted(
        {label for r in baseline_records for label in r["per_class_metrics"] if label not in {"macro avg", "weighted avg"}}
    )
    rows = []
    for r in baseline_records:
        for label in classes:
            rows.append(
                {
                    "model": r["model"],
                    "class": label,
                    "f1": r["per_class_metrics"].get(label, {}).get("f1-score", 0.0),
                }
            )
    df = pd.DataFrame(rows).pivot(index="class", columns="model", values="f1").reindex(classes)
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("class")
    ax.set_ylabel("F1 (baseline cell)")
    ax.set_title("Per-class F1 — baseline cell, all three models")
    ax.legend(title="model")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    COMPARISON_OUTPUT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    full_records = []

    for cell_name, baseline_dir, include_desc, class_weight in ABLATION_CELLS:
        X_train, y_train, X_valid, y_valid, feature_columns = load_split(baseline_dir)

        # CatBoost — read pre-computed
        catboost_result = load_catboost_cell(cell_name)
        full_records.append(
            {
                "model": "CatBoost",
                "cell": cell_name,
                "seeds_run": 5,
                **catboost_result,
            }
        )

        # SVM — train inline
        print(f"[{cell_name}] training SVM...", flush=True)
        svm_pred, svm_time, classes, svm_best = train_svm(
            X_train, y_train, X_valid, y_valid, feature_columns, class_weight
        )
        svm_result = evaluate(y_valid, svm_pred, svm_time, classes)
        full_records.append(
            {"model": "SVM", "cell": cell_name, "seeds_run": 1, **svm_result}
        )
        print(
            f"  SVM best={svm_best} weighted_f1={svm_result['weighted_f1']:.4f} "
            f"acc={svm_result['accuracy']:.4f} time={svm_time:.2f}s",
            flush=True,
        )

        # LogReg — train inline
        print(f"[{cell_name}] training LogReg...", flush=True)
        lr_pred, lr_time, classes, lr_best = train_logreg(
            X_train, y_train, X_valid, y_valid, feature_columns, class_weight
        )
        lr_result = evaluate(y_valid, lr_pred, lr_time, classes)
        full_records.append(
            {"model": "LogReg", "cell": cell_name, "seeds_run": 1, **lr_result}
        )
        print(
            f"  LogReg best={lr_best} weighted_f1={lr_result['weighted_f1']:.4f} "
            f"acc={lr_result['accuracy']:.4f} time={lr_time:.2f}s",
            flush=True,
        )

    # Summary CSV (drops per-class for readability)
    summary_rows = [
        {k: v for k, v in r.items() if k != "per_class_metrics"} for r in full_records
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        ["model", "cell", "weighted_f1", "macro_f1", "accuracy", "training_time_seconds", "seeds_run"]
    ]
    summary_df.to_csv(COMPARISON_OUTPUT / "comparison_summary.csv", index=False)

    # Full results JSON (with per-class)
    (COMPARISON_OUTPUT / "comparison_full.json").write_text(
        json.dumps(full_records, indent=4)
    )

    # Figures
    save_grouped_bar(
        summary_rows,
        "weighted_f1",
        COMPARISON_OUTPUT / "weighted_f1_bar.png",
        "Weighted F1 by model and ablation cell",
    )
    save_grouped_bar(
        summary_rows,
        "macro_f1",
        COMPARISON_OUTPUT / "macro_f1_bar.png",
        "Macro F1 by model and ablation cell",
    )
    save_grouped_bar(
        summary_rows,
        "accuracy",
        COMPARISON_OUTPUT / "accuracy_bar.png",
        "Accuracy by model and ablation cell",
    )
    save_per_class_f1_baseline(full_records, COMPARISON_OUTPUT / "per_class_f1_baseline.png")

    # Print final summary
    print("\n=== Comparison summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
