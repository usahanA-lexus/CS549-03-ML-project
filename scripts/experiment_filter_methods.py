import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAIN_PATH = os.getenv("TRAIN_PATH", "processed_output/shared_baseline/train_raw_split_cleaned.csv")
VAL_PATH = os.getenv("VAL_PATH", "processed_output/shared_baseline/valid_raw_split.csv")
NOISE_SCORES_PATH = os.getenv(
    "NOISE_SCORES_PATH", "processed_output/final_capped_filter/training_noise_scores.csv"
)
OUT_DIR = Path(os.getenv("OUT_DIR", "processed_output/filter_method_experiments"))

TARGET = "Category"
CAPS = [0.05, 0.10, 0.15]


def clean_description(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"#\d+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def amount_bucket(v):
    try:
        v = float(v)
    except Exception:
        return "unknown"
    a = abs(v)
    if a < 10:
        return "0_10"
    if a < 25:
        return "10_25"
    if a < 50:
        return "25_50"
    if a < 100:
        return "50_100"
    if a < 250:
        return "100_250"
    return "250_plus"


def add_features(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "Description" not in df.columns:
        df["Description"] = ""
    if "Transaction Type" not in df.columns:
        df["Transaction Type"] = "unknown"
    if "Amount" not in df.columns:
        df["Amount"] = np.nan

    df["Description"] = df["Description"].fillna("").astype(str).apply(clean_description)
    df["Transaction Type"] = df["Transaction Type"].fillna("unknown").astype(str)
    df["Amount_bucket"] = df["Amount"].apply(amount_bucket)
    return df


def build_model():
    text_col = "Description"
    cat_cols = ["Amount_bucket", "Transaction Type"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95, stop_words="english"), text_col),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline(
        [("prep", preprocessor), ("clf", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))]
    )


def kalman_1d(z, q=1e-4, r=1e-2):
    xhat = np.zeros_like(z)
    p = 1.0
    xhat[0] = z[0]
    for k in range(1, len(z)):
        x_pred = xhat[k - 1]
        p_pred = p + q
        k_gain = p_pred / (p_pred + r)
        xhat[k] = x_pred + k_gain * (z[k] - x_pred)
        p = (1 - k_gain) * p_pred
    return xhat


def whittaker_smooth(y, lam=50.0, d=2):
    m = len(y)
    e = np.ones(m)
    dmat = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(m - 2, m), format="csc")
    if d != 2:
        raise ValueError("This implementation currently supports d=2 only.")
    w = sparse.eye(m, format="csc")
    a = w + lam * (dmat.T @ dmat)
    return spsolve(a, y)


def eval_with_scores(train_df, val_df, scores, method_name, params_label):
    feature_cols = ["Description", "Amount_bucket", "Transaction Type"]
    out_rows = []
    for cap in CAPS:
        n_remove = max(1, int(len(train_df) * cap))
        worst_idx = np.argsort(scores)[::-1][:n_remove]
        keep_mask = np.ones(len(train_df), dtype=bool)
        keep_mask[worst_idx] = False

        filtered = train_df.loc[keep_mask].copy()
        model = build_model()
        model.fit(filtered[feature_cols], filtered[TARGET])
        pred = model.predict(val_df[feature_cols])

        out_rows.append(
            {
                "method": method_name,
                "params": params_label,
                "cap_fraction": cap,
                "rows_removed": int(n_remove),
                "accuracy": float(accuracy_score(val_df[TARGET], pred)),
                "weighted_f1": float(f1_score(val_df[TARGET], pred, average="weighted")),
                "macro_f1": float(f1_score(val_df[TARGET], pred, average="macro")),
            }
        )
    return out_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = add_features(pd.read_csv(TRAIN_PATH))
    val_df = add_features(pd.read_csv(VAL_PATH))
    noise_df = pd.read_csv(NOISE_SCORES_PATH)

    required = ["true_label_prob"]
    missing = [c for c in required if c not in noise_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in noise scores: {missing}")

    wrong_col = "wrong_flag" if "wrong_flag" in noise_df.columns else "flagged_suspicious"
    wrong = noise_df[wrong_col].astype(float).to_numpy() if wrong_col in noise_df.columns else np.zeros(len(noise_df))
    base_score = (1.0 - noise_df["true_label_prob"].to_numpy()) + wrong

    results = []

    # Baseline (no smoothing)
    results.extend(eval_with_scores(train_df, val_df, base_score, "raw", "none"))

    # Median filter
    for k in [3, 5, 7]:
        s = medfilt(base_score, kernel_size=k)
        results.extend(eval_with_scores(train_df, val_df, s, "median", f"kernel={k}"))

    # Savitzky-Golay
    for window, poly in [(7, 2), (11, 2), (11, 3), (15, 3)]:
        if window < len(base_score):
            s = savgol_filter(base_score, window_length=window, polyorder=poly, mode="nearest")
            results.extend(eval_with_scores(train_df, val_df, s, "savgol", f"window={window},poly={poly}"))

    # Kalman
    for q, r in [(1e-4, 1e-2), (1e-5, 1e-2), (1e-4, 1e-1), (1e-3, 1e-2)]:
        s = kalman_1d(base_score, q=q, r=r)
        results.extend(eval_with_scores(train_df, val_df, s, "kalman", f"q={q},r={r}"))

    # Whittaker
    for lam in [10.0, 50.0, 200.0, 1000.0]:
        s = whittaker_smooth(base_score, lam=lam, d=2)
        results.extend(eval_with_scores(train_df, val_df, s, "whittaker", f"lambda={lam}"))

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(["weighted_f1", "macro_f1", "accuracy"], ascending=False)
    res_df.to_csv(OUT_DIR / "filter_method_results.csv", index=False)

    best_per_method = (
        res_df.sort_values(["weighted_f1", "macro_f1"], ascending=False)
        .groupby("method", as_index=False)
        .first()
        .sort_values("weighted_f1", ascending=False)
    )
    best_per_method.to_csv(OUT_DIR / "best_per_method.csv", index=False)

    print("Top 10 overall:")
    print(res_df.head(10).to_string(index=False))
    print("\nBest per method:")
    print(best_per_method.to_string(index=False))


if __name__ == "__main__":
    main()
