"""CatBoost training for the financial transaction classification project.

Tunes hyperparameters via 5-fold stratified CV on the training split only, then
runs the 2x2 ablation (with/without Description x with/without class weights)
with multi-seed reporting on the held-out validation split.
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold


TARGET = "Category"
CATEGORICAL_COLUMNS = ["Description", "Transaction Type", "Account Name"]
GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "depth": [4, 6, 8],
}
TUNE_ITERATIONS = 1000
EARLY_STOPPING_ROUNDS = 50
ABLATION_SEEDS = [0, 1, 2, 3, 4]
CV_FOLDS = 5
CV_RANDOM_STATE = 42
CV_MODEL_SEED = 0


def load_baseline(baseline_dir):
    train = pd.read_csv(baseline_dir / "train_raw_split.csv")
    valid = pd.read_csv(baseline_dir / "valid_raw_split.csv")

    feature_columns = [c for c in train.columns if c != TARGET]
    cat_features = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    for column in cat_features:
        train[column] = train[column].astype(str)
        valid[column] = valid[column].astype(str)

    summary = json.loads((baseline_dir / "preprocessing_summary.json").read_text())
    return train, valid, feature_columns, cat_features, summary


def label_encode(y_train, y_valid):
    classes = sorted(pd.concat([y_train, y_valid]).unique())
    mapping = {label: i for i, label in enumerate(classes)}
    return y_train.map(mapping).to_numpy(), y_valid.map(mapping).to_numpy(), classes


def make_classifier(params, cat_indices, iterations, class_weights, seed):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Weighted",
        cat_features=cat_indices,
        class_weights=class_weights,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def cross_validate_grid(X_train, y_train_enc, cat_indices):
    """5-fold stratified CV grid search. Returns sorted trace; best is trace[0]."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    folds = list(skf.split(X_train, y_train_enc))
    combos = [dict(zip(GRID.keys(), c)) for c in product(*GRID.values())]

    trace = []
    for params in combos:
        fold_scores = []
        fold_best_iters = []
        for tr_idx, va_idx in folds:
            model = make_classifier(
                params=params,
                cat_indices=cat_indices,
                iterations=TUNE_ITERATIONS,
                class_weights=None,
                seed=CV_MODEL_SEED,
            )
            model.fit(
                X_train.iloc[tr_idx],
                y_train_enc[tr_idx],
                eval_set=(X_train.iloc[va_idx], y_train_enc[va_idx]),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
            preds = model.predict(X_train.iloc[va_idx]).flatten().astype(int)
            fold_scores.append(f1_score(y_train_enc[va_idx], preds, average="weighted"))
            fold_best_iters.append(int(model.get_best_iteration() or TUNE_ITERATIONS))

        trace.append(
            {
                "params": params,
                "mean_weighted_f1": float(np.mean(fold_scores)),
                "std_weighted_f1": float(np.std(fold_scores)),
                "fold_scores": [float(s) for s in fold_scores],
                "median_best_iteration": int(np.median(fold_best_iters)),
                "fold_best_iterations": fold_best_iters,
            }
        )

    return sorted(trace, key=lambda r: r["mean_weighted_f1"], reverse=True)


def per_class_report(y_true_labels, y_pred_labels, classes):
    report = classification_report(
        y_true_labels, y_pred_labels, labels=classes, output_dict=True, zero_division=0
    )
    cleaned = {}
    for key, value in report.items():
        if key == "accuracy" or not isinstance(value, dict):
            continue
        cleaned[key] = {
            metric: float(score) if isinstance(score, (int, float)) else score
            for metric, score in value.items()
        }
    return cleaned


def save_confusion_matrix(y_true_labels, y_pred_labels, classes, output_path, run_name):
    matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"CatBoost confusion matrix — {run_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_importance(model, feature_columns, output_dir, run_name):
    importances = model.get_feature_importance(type="PredictionValuesChange")
    paired = sorted(zip(feature_columns, importances), key=lambda r: r[1], reverse=True)

    csv_path = output_dir / "feature_importance.csv"
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("feature,importance\n")
        for feature, importance in paired:
            fp.write(f"{feature},{float(importance):.6f}\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([p[0] for p in paired], [float(p[1]) for p in paired], color="#4c72b0")
    ax.invert_yaxis()
    ax.set_xlabel("PredictionValuesChange importance")
    ax.set_title(f"CatBoost feature importance — {run_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def aggregate(records, key):
    values = [r[key] for r in records]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(v) for v in values],
    }


def run_ablation(
    run_name,
    baseline_dir,
    output_root,
    use_class_weights,
    best_params,
    iterations,
):
    train, valid, feature_columns, cat_features, summary = load_baseline(baseline_dir)
    cat_indices = [feature_columns.index(c) for c in cat_features]

    X_train, y_train = train[feature_columns], train[TARGET]
    X_valid, y_valid = valid[feature_columns], valid[TARGET]
    y_train_enc, y_valid_enc, classes = label_encode(y_train, y_valid)

    class_weights = None
    class_weights_used = None
    if use_class_weights:
        suggested = summary["suggested_class_weights_from_training_split"]
        class_weights = [float(suggested[label]) for label in classes]
        class_weights_used = dict(zip(classes, class_weights))

    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_records = []
    representative_model = None
    representative_preds = None

    for seed in ABLATION_SEEDS:
        model = make_classifier(
            params=best_params,
            cat_indices=cat_indices,
            iterations=iterations,
            class_weights=class_weights,
            seed=seed,
        )
        start = time.perf_counter()
        model.fit(X_train, y_train_enc, verbose=False)
        elapsed = time.perf_counter() - start

        preds = model.predict(X_valid).flatten().astype(int)
        seed_records.append(
            {
                "seed": int(seed),
                "accuracy": float(accuracy_score(y_valid_enc, preds)),
                "weighted_f1": float(f1_score(y_valid_enc, preds, average="weighted")),
                "macro_f1": float(f1_score(y_valid_enc, preds, average="macro")),
                "training_time_seconds": float(elapsed),
            }
        )
        if seed == ABLATION_SEEDS[0]:
            representative_model = model
            representative_preds = preds

    rep_pred_labels = [classes[i] for i in representative_preds]
    actual_labels = y_valid.tolist()

    metrics = {
        "run_name": run_name,
        "baseline_dir": str(baseline_dir),
        "feature_columns": feature_columns,
        "cat_features": cat_features,
        "class_weights_used": class_weights_used,
        "best_params": best_params,
        "iterations": iterations,
        "seeds": ABLATION_SEEDS,
        "per_seed": seed_records,
        "accuracy": aggregate(seed_records, "accuracy"),
        "weighted_f1": aggregate(seed_records, "weighted_f1"),
        "macro_f1": aggregate(seed_records, "macro_f1"),
        "training_time_seconds": aggregate(seed_records, "training_time_seconds"),
        "representative_seed": ABLATION_SEEDS[0],
        "per_class_metrics_representative_seed": per_class_report(
            actual_labels, rep_pred_labels, classes
        ),
        "classes": classes,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))
    save_confusion_matrix(
        actual_labels, rep_pred_labels, classes, output_dir / "confusion_matrix.png", run_name
    )
    save_feature_importance(representative_model, feature_columns, output_dir, run_name)
    (output_dir / "best_params.json").write_text(
        json.dumps(
            {
                "params": best_params,
                "iterations": iterations,
                "tuning_metric": "weighted_f1",
                "tuned_on_baseline_dir": str(baseline_dir),
                "use_class_weights": use_class_weights,
            },
            indent=4,
        )
    )
    return metrics


def save_per_class_f1_delta(output_root):
    """Plot F1(class_weighted) - F1(baseline) per class — class-weight ablation headline."""
    baseline_metrics = json.loads((output_root / "baseline" / "metrics.json").read_text())
    weighted_metrics = json.loads(
        (output_root / "class_weighted" / "metrics.json").read_text()
    )

    classes = baseline_metrics["classes"]
    baseline_f1 = baseline_metrics["per_class_metrics_representative_seed"]
    weighted_f1 = weighted_metrics["per_class_metrics_representative_seed"]

    deltas = []
    for label in classes:
        b = baseline_f1.get(label, {}).get("f1-score", 0.0)
        w = weighted_f1.get(label, {}).get("f1-score", 0.0)
        deltas.append((label, w - b, baseline_f1.get(label, {}).get("support", 0)))

    deltas.sort(key=lambda r: r[2])  # rare classes first

    labels = [d[0] for d in deltas]
    values = [d[1] for d in deltas]
    colors = ["#c44e52" if v < 0 else "#55a868" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("F1(class_weighted) − F1(baseline)")
    ax.set_title("Per-class F1 delta from class weighting (sorted by support)")
    fig.tight_layout()
    fig.savefig(output_root / "per_class_f1_delta.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        default="processed_output/shared_baseline",
        help="Directory with the with-Description train/valid splits",
    )
    parser.add_argument(
        "--no-description-baseline-dir",
        default="processed_output/baseline_no_description",
        help="Directory with the no-Description train/valid splits",
    )
    parser.add_argument(
        "--output-root",
        default="processed_output/models/catboost",
        help="Where to write per-run output directories",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--all-ablations", action="store_true")
    args = parser.parse_args()

    np.random.seed(0)

    baseline_dir = Path(args.baseline_dir)
    no_desc_dir = Path(args.no_description_baseline_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Tune once on the with-Description train split.
    train, _, feature_columns, cat_features, _ = load_baseline(baseline_dir)
    cat_indices = [feature_columns.index(c) for c in cat_features]
    X_train = train[feature_columns]
    y_train_enc, _, _ = label_encode(train[TARGET], train[TARGET])

    print(f"Tuning on {baseline_dir} ({CV_FOLDS}-fold CV, {len(GRID['learning_rate']) * len(GRID['depth'])} configs)...", flush=True)
    trace = cross_validate_grid(X_train, y_train_enc, cat_indices)
    best = trace[0]

    cv_payload = {
        "tuned_on_baseline_dir": str(baseline_dir),
        "grid": GRID,
        "cv_folds": CV_FOLDS,
        "cv_random_state": CV_RANDOM_STATE,
        "iterations_cap": TUNE_ITERATIONS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "cv_model_seed": CV_MODEL_SEED,
        "metric": "weighted_f1",
        "best": best,
        "trace": trace,
    }
    (output_root / "cv_tuning_results.json").write_text(json.dumps(cv_payload, indent=4))

    best_params = best["params"]
    final_iterations = max(int(best["median_best_iteration"]), 100)
    print(
        f"  best params: {best_params}  median_best_iteration={final_iterations}  "
        f"weighted_f1={best['mean_weighted_f1']:.4f}±{best['std_weighted_f1']:.4f}",
        flush=True,
    )

    if args.all_ablations:
        ablations = [
            ("baseline", baseline_dir, False),
            ("class_weighted", baseline_dir, True),
            ("no_description", no_desc_dir, False),
            ("no_description_weighted", no_desc_dir, True),
        ]
    else:
        if args.run_name is None:
            parser.error("Provide --run-name or use --all-ablations")
        target_dir = no_desc_dir if "no_description" in args.run_name else baseline_dir
        ablations = [(args.run_name, target_dir, args.use_class_weights)]

    summary = []
    for run_name, dir_path, use_weights in ablations:
        result = run_ablation(
            run_name=run_name,
            baseline_dir=dir_path,
            output_root=output_root,
            use_class_weights=use_weights,
            best_params=best_params,
            iterations=final_iterations,
        )
        summary.append(
            {
                "run": run_name,
                "weighted_f1_mean": result["weighted_f1"]["mean"],
                "weighted_f1_std": result["weighted_f1"]["std"],
                "macro_f1_mean": result["macro_f1"]["mean"],
                "accuracy_mean": result["accuracy"]["mean"],
                "training_time_mean": result["training_time_seconds"]["mean"],
            }
        )
        print(
            f"[{run_name}] weighted_f1={result['weighted_f1']['mean']:.4f}±{result['weighted_f1']['std']:.4f} "
            f"macro_f1={result['macro_f1']['mean']:.4f} "
            f"acc={result['accuracy']['mean']:.4f} "
            f"train_s={result['training_time_seconds']['mean']:.2f}",
            flush=True,
        )

    if args.all_ablations:
        (output_root / "ablation_summary.json").write_text(json.dumps(summary, indent=4))
        save_per_class_f1_delta(output_root)


if __name__ == "__main__":
    main()
