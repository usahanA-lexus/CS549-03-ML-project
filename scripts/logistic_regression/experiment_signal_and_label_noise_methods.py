import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path("processed_output/iterative_exp2_selective_cleaning/train_best_iterative_cleaned.csv")
VAL_PATH = Path("processed_output/shared_baseline/valid_raw_split.csv")
NOISE_PATH = Path("processed_output/exp2_best_tuned/training_noise_scores.csv")
OUT_DIR = Path("processed_output/signal_labelnoise_experiments")
TARGET = "Category"


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
    if "Account Name" not in df.columns:
        df["Account Name"] = "unknown"

    df["Description"] = df["Description"].fillna("").astype(str).apply(clean_description)
    df["Transaction Type"] = df["Transaction Type"].fillna("unknown").astype(str)
    df["Account Name"] = df["Account Name"].fillna("unknown").astype(str)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Amount_bucket"] = df["Amount"].apply(amount_bucket)
    return df


def build_main_model(include_signal_cols=False):
    text_col = "Description"
    cat_cols = ["Amount_bucket", "Transaction Type", "Account Name"]
    num_cols = ["Amount"]
    if include_signal_cols:
        num_cols += ["noise_raw", "noise_smooth", "noise_resid"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95, stop_words="english"), text_col),
            (
                "cat",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
                ),
                cat_cols,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ]
    )
    return Pipeline(
        [("prep", preprocessor), ("clf", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))]
    )


def evaluate_model(model, train_df, val_df, sample_weight=None):
    feature_cols = ["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]
    if "noise_raw" in train_df.columns:
        feature_cols += ["noise_raw", "noise_smooth", "noise_resid"]

    model.fit(train_df[feature_cols], train_df[TARGET], clf__sample_weight=sample_weight)
    pred = model.predict(val_df[feature_cols])
    return {
        "accuracy": float(accuracy_score(val_df[TARGET], pred)),
        "weighted_f1": float(f1_score(val_df[TARGET], pred, average="weighted")),
        "macro_f1": float(f1_score(val_df[TARGET], pred, average="macro")),
    }


def add_noise_signal_features(df, noise_df):
    df = df.copy().reset_index(drop=True)
    noise = noise_df.copy().reset_index(drop=True)
    base = (1.0 - noise["true_label_prob"].to_numpy()) + noise["flagged_suspicious"].astype(float).to_numpy()
    order = np.argsort(base)[::-1]
    z = base[order]
    z_smooth = savgol_filter(z, window_length=21, polyorder=6, mode="nearest")
    smooth = np.zeros_like(base)
    smooth[order] = z_smooth
    resid = base - smooth
    df["noise_raw"] = base
    df["noise_smooth"] = smooth
    df["noise_resid"] = resid
    return df


def run_label_smoothing(train_df, val_df, noise_df, smooth_main=0.9):
    # Approximate soft targets by duplicating ambiguous rows with model-pred label and lower weight.
    train = train_df.copy().reset_index(drop=True)
    noise = noise_df.copy().reset_index(drop=True)
    alt = train.copy()
    alt[TARGET] = noise["cv_pred"].astype(str).values

    mismatch = (noise["cv_pred"].astype(str).values != train[TARGET].astype(str).values)
    alt = alt.loc[mismatch].copy()

    merged = pd.concat([train, alt], axis=0, ignore_index=True)
    w_main = np.full(len(train), smooth_main, dtype=float)
    w_alt = np.full(len(alt), 1.0 - smooth_main, dtype=float)
    weights = np.concatenate([w_main, w_alt])

    model = build_main_model(include_signal_cols=False)
    metrics = evaluate_model(model, merged, val_df, sample_weight=weights)
    return metrics


def run_classaware_iforest(train_df, val_df, contamination=0.05):
    train = train_df.copy().reset_index(drop=True)
    # Embedding for class-aware density: text tfidf -> svd + tabular one-hot/numeric.
    text_vec = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, stop_words="english")
    x_text = text_vec.fit_transform(train["Description"])
    svd = TruncatedSVD(n_components=40, random_state=42)
    x_text_svd = svd.fit_transform(x_text)

    tab_prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Amount_bucket", "Transaction Type", "Account Name"]),
            ("num", StandardScaler(), ["Amount"]),
        ]
    )
    x_tab = tab_prep.fit_transform(train[["Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    x_tab = x_tab.toarray() if hasattr(x_tab, "toarray") else x_tab
    x_all = np.hstack([x_text_svd, x_tab])

    keep_mask = np.ones(len(train), dtype=bool)
    y = train[TARGET].astype(str).values
    for cls in sorted(np.unique(y)):
        idx = np.where(y == cls)[0]
        if len(idx) < 30:
            continue
        iso = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=1,
        )
        iso.fit(x_all[idx])
        pred = iso.predict(x_all[idx])  # -1 anomaly
        keep_mask[idx[pred == -1]] = False

    filtered = train.loc[keep_mask].copy()
    model = build_main_model(include_signal_cols=False)
    metrics = evaluate_model(model, filtered, val_df)
    metrics["rows_removed"] = int((~keep_mask).sum())
    return metrics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = add_features(pd.read_csv(TRAIN_PATH))
    val_df = add_features(pd.read_csv(VAL_PATH))
    noise_df = pd.read_csv(NOISE_PATH)

    results = []

    # Baseline current best-tuned setup (no extra tricks here, direct model on cleaned train)
    baseline_model = build_main_model(include_signal_cols=False)
    baseline = evaluate_model(baseline_model, train_df, val_df)
    results.append({"method": "baseline_retrain", "rows_removed": 0, **baseline})

    # 1) Residual features
    train_sig = add_noise_signal_features(train_df, noise_df)
    # For validation rows, signal features are unknown; use training distribution medians.
    val_sig = val_df.copy()
    val_sig["noise_raw"] = float(train_sig["noise_raw"].median())
    val_sig["noise_smooth"] = float(train_sig["noise_smooth"].median())
    val_sig["noise_resid"] = float(train_sig["noise_resid"].median())
    sig_model = build_main_model(include_signal_cols=True)
    sig_metrics = evaluate_model(sig_model, train_sig, val_sig)
    results.append({"method": "residual_features_raw_smooth_resid", "rows_removed": 0, **sig_metrics})

    # 2) Label smoothing / soft targets approximation
    for sm in [0.85, 0.9, 0.95]:
        ls_metrics = run_label_smoothing(train_df, val_df, noise_df, smooth_main=sm)
        results.append({"method": f"label_smoothing_softdup_{sm}", "rows_removed": 0, **ls_metrics})

    # 3) Class-aware Isolation Forest
    for c in [0.03, 0.05, 0.08]:
        if_metrics = run_classaware_iforest(train_df, val_df, contamination=c)
        results.append({"method": f"classaware_iforest_{c}", **if_metrics})

    out = pd.DataFrame(results).sort_values(["weighted_f1", "macro_f1", "accuracy"], ascending=False)
    out.to_csv(OUT_DIR / "results.csv", index=False)
    out.head(1).to_csv(OUT_DIR / "best_overall.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
