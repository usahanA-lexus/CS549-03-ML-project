import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


TARGET_COLUMN = "Category"
NUMERIC_FEATURES = ["Amount"]
BASE_CATEGORICAL_FEATURES = ["Transaction Type", "Account Name"]
OPTIONAL_DESCRIPTION_COLUMN = "Description"
DROP_COLUMNS = ["User ID", "Date"]


def normalize_columns(df):
    df = df.copy()
    df.columns = [column.strip() for column in df.columns]

    rename_map = {}
    for column in df.columns:
        normalized = column.lower().replace("_", " ").strip()
        if normalized == "user id":
            rename_map[column] = "User ID"
        elif normalized == "date":
            rename_map[column] = "Date"
        elif normalized == "description":
            rename_map[column] = "Description"
        elif normalized == "amount":
            rename_map[column] = "Amount"
        elif normalized == "transaction type":
            rename_map[column] = "Transaction Type"
        elif normalized == "category":
            rename_map[column] = "Category"
        elif normalized == "account name":
            rename_map[column] = "Account Name"

    return df.rename(columns=rename_map)


def determine_feature_columns(columns, include_description):
    feature_columns = NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES

    if include_description and OPTIONAL_DESCRIPTION_COLUMN in columns:
        feature_columns.insert(0, OPTIONAL_DESCRIPTION_COLUMN)

    return feature_columns


def fill_missing_values(X_train, X_valid, numeric_columns, categorical_columns):
    X_train = X_train.copy()
    X_valid = X_valid.copy()

    imputation_values = {}
    missing_value_handling = {"numeric": {}, "categorical": {}}

    for column in numeric_columns:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_valid[column] = pd.to_numeric(X_valid[column], errors="coerce")

        fill_value = X_train[column].median()
        if pd.isna(fill_value):
            fill_value = 0.0

        imputation_values[column] = float(fill_value)
        missing_value_handling["numeric"][column] = "Filled missing values with the training-split median."

        X_train[column] = X_train[column].fillna(fill_value)
        X_valid[column] = X_valid[column].fillna(fill_value)

    for column in categorical_columns:
        mode = X_train[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Missing"
        fill_value = str(fill_value)

        imputation_values[column] = fill_value
        missing_value_handling["categorical"][column] = "Filled missing values with the training-split mode."

        X_train[column] = X_train[column].fillna(fill_value)
        X_valid[column] = X_valid[column].fillna(fill_value)

    return X_train, X_valid, imputation_values, missing_value_handling


def value_counts_by_label(series):
    return {str(label): int(count) for label, count in series.value_counts().items()}


def main():
    parser = argparse.ArgumentParser(
        description="Create one shared baseline preprocessing package for team models."
    )
    parser.add_argument("--input", required=True, help="Path to the raw transaction CSV")
    parser.add_argument(
        "--output-dir",
        default="processed_output",
        help="Parent directory for baseline and experiment runs",
    )
    parser.add_argument(
        "--run-name",
        default="shared_baseline",
        help="Subdirectory name for this preprocessing run",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split size",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/validation split",
    )
    parser.add_argument(
        "--drop-description",
        action="store_true",
        help="Exclude Description for a documented ablation run",
    )
    args = parser.parse_args()

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    raw_rows = len(df)

    df = normalize_columns(df)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.drop_duplicates().copy()
    rows_after_deduplication = len(df)

    dropped_columns = [column for column in DROP_COLUMNS if column in df.columns]
    if dropped_columns:
        df = df.drop(columns=dropped_columns)

    required_columns = ["Amount", "Transaction Type", "Account Name", TARGET_COLUMN]
    missing_required = [column for column in required_columns if column not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    requested_description = not args.drop_description
    feature_columns = determine_feature_columns(df.columns, include_description=requested_description)
    description_included = OPTIONAL_DESCRIPTION_COLUMN in feature_columns

    selected_columns = feature_columns + [TARGET_COLUMN]
    df = df[selected_columns].copy()

    rows_before_target_drop = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    rows_dropped_missing_target = rows_before_target_drop - len(df)

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    stratify = y if y.nunique() > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    missing_values_before_imputation = {
        "train_raw_split": {
            column: int(X_train[column].isna().sum()) for column in feature_columns
        },
        "valid_raw_split": {
            column: int(X_valid[column].isna().sum()) for column in feature_columns
        },
    }

    numeric_columns = [column for column in NUMERIC_FEATURES if column in feature_columns]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    X_train, X_valid, imputation_values, missing_value_handling = fill_missing_values(
        X_train,
        X_valid,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)],
        axis=1,
    )
    valid_df = pd.concat(
        [X_valid.reset_index(drop=True), y_valid.reset_index(drop=True)],
        axis=1,
    )

    train_df.to_csv(run_dir / "train_raw_split.csv", index=False)
    valid_df.to_csv(run_dir / "valid_raw_split.csv", index=False)

    classes = np.array(sorted(y_train.unique()))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    suggested_class_weights = {
        str(label): float(weight) for label, weight in zip(classes, class_weights)
    }

    summary = {
        "run_name": args.run_name,
        "source_dataset": args.input,
        "raw_rows": int(raw_rows),
        "rows_after_deduplication": int(rows_after_deduplication),
        "duplicates_removed": int(raw_rows - rows_after_deduplication),
        "rows_dropped_missing_target": int(rows_dropped_missing_target),
        "final_rows_before_split": int(len(df)),
        "duplicate_handling": "Removed exact duplicate rows before the shared split was created.",
        "dropped_columns": dropped_columns,
        "selected_feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "description_requested": requested_description,
        "description_included": description_included,
        "missing_value_handling": {
            "target": "Dropped rows with missing Category before splitting.",
            **missing_value_handling,
        },
        "imputation_values_from_training_split": imputation_values,
        "missing_values_before_imputation": missing_values_before_imputation,
        "missing_values_after_imputation": {
            "train_raw_split": {
                column: int(train_df[column].isna().sum()) for column in feature_columns
            },
            "valid_raw_split": {
                column: int(valid_df[column].isna().sum()) for column in feature_columns
            },
        },
        "split": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "validation_fraction": args.test_size,
            "random_state": args.random_state,
            "stratified": bool(stratify is not None),
        },
        "class_distribution": {
            "full_clean_dataset": value_counts_by_label(y),
            "train_raw_split": value_counts_by_label(y_train),
            "valid_raw_split": value_counts_by_label(y_valid),
        },
        "suggested_class_weights_from_training_split": suggested_class_weights,
        "evaluation_setup": {
            "shared_baseline_contract": (
                "All models must start from the same cleaned train_raw_split.csv and "
                "valid_raw_split.csv files produced by this script."
            ),
            "holdout_validation": (
                "Use valid_raw_split.csv as the common held-out validation set so model "
                "comparisons differ by model behavior rather than baseline data handling."
            ),
            "model_specific_step_boundary": (
                "Encoding, scaling, text featurization, and other model-specific transforms "
                "must happen after this baseline handoff."
            ),
        },
    }

    with open(run_dir / "preprocessing_summary.json", "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=4)


if __name__ == "__main__":
    main()
