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


TRAIN_PATH = Path("processed_output/iterative_exp2_selective_cleaning/train_best_iterative_cleaned.csv")
VAL_PATH = Path("processed_output/shared_baseline/valid_raw_split.csv")
NOISE_PATH = Path("processed_output/exp2_best_tuned/training_noise_scores.csv")
OUT_DIR = Path("processed_output/hybrid_classaware_experiments")

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


def whittaker_smooth(y, lam=50.0):
    m = len(y)
    e = np.ones(m)
    dmat = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(m - 2, m), format="csc")
    w = sparse.eye(m, format="csc")
    a = w + lam * (dmat.T @ dmat)
    return spsolve(a, y)


def smooth_rank_signal(base_score, method):
    # Smooth over sorted-by-score order, then map back.
    order = np.argsort(base_score)[::-1]
    z = base_score[order]
    if method == "raw":
        zs = z
    elif method == "median":
        zs = medfilt(z, kernel_size=7)
    elif method == "savgol":
        zs = savgol_filter(z, window_length=15, polyorder=3, mode="nearest")
    elif method == "kalman":
        zs = kalman_1d(z, q=1e-4, r=1e-2)
    elif method == "whittaker":
        zs = whittaker_smooth(z, lam=50.0)
    else:
        raise ValueError(method)
    out = np.zeros_like(base_score, dtype=float)
    out[order] = zs
    return out


def evaluate(train_df, val_df, remove_mask):
    feature_cols = ["Description", "Amount_bucket", "Transaction Type"]
    filtered = train_df.loc[~remove_mask].copy()
    model = build_model()
    model.fit(filtered[feature_cols], filtered[TARGET])
    pred = model.predict(val_df[feature_cols])
    return {
        "accuracy": float(accuracy_score(val_df[TARGET], pred)),
        "weighted_f1": float(f1_score(val_df[TARGET], pred, average="weighted")),
        "macro_f1": float(f1_score(val_df[TARGET], pred, average="macro")),
        "rows_removed": int(remove_mask.sum()),
        "rows_kept": int((~remove_mask).sum()),
    }


def build_classaware_remove_mask(train_df, candidate_idx, score, frac, min_keep):
    remove_mask = np.zeros(len(train_df), dtype=bool)
    n_target = int(len(candidate_idx) * frac)
    class_counts = train_df[TARGET].value_counts().to_dict()
    class_removed = {k: 0 for k in class_counts}

    ranked = candidate_idx[np.argsort(score[candidate_idx])[::-1]]
    removed = 0
    for idx in ranked:
        cls = train_df.at[idx, TARGET]
        if class_counts[cls] - (class_removed[cls] + 1) < min_keep:
            continue
        remove_mask[idx] = True
        class_removed[cls] += 1
        removed += 1
        if removed >= n_target:
            break
    return remove_mask


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = add_features(pd.read_csv(TRAIN_PATH))
    val_df = add_features(pd.read_csv(VAL_PATH))
    noise_df = pd.read_csv(NOISE_PATH)

    base = (1.0 - noise_df["true_label_prob"].to_numpy()) + noise_df["flagged_suspicious"].astype(float).to_numpy()
    cond = (noise_df["flagged_suspicious"].astype(bool).to_numpy()) & (noise_df["true_label_prob"].to_numpy() < 0.15)
    candidate_idx = np.where(cond)[0]

    methods = ["raw", "median", "savgol", "kalman", "whittaker"]
    alphas = [0.0, 0.25, 0.5, 0.75]
    fracs = [0.6, 0.75, 0.9, 1.0]
    min_keeps = [20, 40, 60]

    results = []

    for method in methods:
        sm = smooth_rank_signal(base, method)
        sm_norm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-12)
        base_norm = (base - base.min()) / (base.max() - base.min() + 1e-12)
        for alpha in alphas:
            score = (1 - alpha) * base_norm + alpha * sm_norm
            for frac in fracs:
                # Hybrid: rank-based subset removal from tuned candidates.
                remove_mask = np.zeros(len(train_df), dtype=bool)
                ranked = candidate_idx[np.argsort(score[candidate_idx])[::-1]]
                n_remove = int(len(candidate_idx) * frac)
                remove_mask[ranked[:n_remove]] = True
                m = evaluate(train_df, val_df, remove_mask)
                results.append(
                    {
                        "family": "hybrid",
                        "method": method,
                        "alpha": alpha,
                        "frac_candidates_removed": frac,
                        "min_keep_per_class": "",
                        **m,
                    }
                )

                # Macro-focused: class-aware guardrail.
                for min_keep in min_keeps:
                    ca_mask = build_classaware_remove_mask(train_df, candidate_idx, score, frac, min_keep)
                    m2 = evaluate(train_df, val_df, ca_mask)
                    results.append(
                        {
                            "family": "classaware",
                            "method": method,
                            "alpha": alpha,
                            "frac_candidates_removed": frac,
                            "min_keep_per_class": min_keep,
                            **m2,
                        }
                    )

    res = pd.DataFrame(results)
    res = res.sort_values(["weighted_f1", "macro_f1", "accuracy"], ascending=False)
    res.to_csv(OUT_DIR / "all_results.csv", index=False)

    best_weighted = res.iloc[0].to_dict()
    best_macro = res.sort_values(["macro_f1", "weighted_f1", "accuracy"], ascending=False).iloc[0].to_dict()
    pd.DataFrame([best_weighted]).to_csv(OUT_DIR / "best_weighted.csv", index=False)
    pd.DataFrame([best_macro]).to_csv(OUT_DIR / "best_macro.csv", index=False)

    print("Top 10 by weighted_f1")
    print(res.head(10).to_string(index=False))
    print("\nBest by macro_f1")
    print(pd.DataFrame([best_macro]).to_string(index=False))


if __name__ == "__main__":
    main()
