import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path("processed_output/iterative_exp2_selective_cleaning/train_best_iterative_cleaned.csv")
VAL_PATH = Path("processed_output/shared_baseline/valid_raw_split.csv")
NOISE_PATH = Path("processed_output/exp2_best_tuned/training_noise_scores.csv")
OUT_DIR = Path("processed_output/hybrid_second_stage_experiments")
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
    for c, d in [("Description", ""), ("Transaction Type", "unknown"), ("Account Name", "unknown"), ("Amount", 0.0)]:
        if c not in df.columns:
            df[c] = d
    df["Description"] = df["Description"].fillna("").astype(str).apply(clean_description)
    df["Transaction Type"] = df["Transaction Type"].fillna("unknown").astype(str)
    df["Account Name"] = df["Account Name"].fillna("unknown").astype(str)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Amount_bucket"] = df["Amount"].apply(amount_bucket)
    return df


def build_model(extra_num_cols=None):
    if extra_num_cols is None:
        extra_num_cols = []
    cat_cols = ["Amount_bucket", "Transaction Type", "Account Name"]
    num_cols = ["Amount"] + list(extra_num_cols)
    prep = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95, stop_words="english"), "Description"),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ]
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))])


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def current_winner_mask(train_df, noise_df):
    base = (1.0 - noise_df["true_label_prob"].to_numpy()) + noise_df["flagged_suspicious"].astype(float).to_numpy()
    candidate_idx = np.where((noise_df["flagged_suspicious"].astype(bool).to_numpy()) & (noise_df["true_label_prob"].to_numpy() < 0.15))[0]
    score = base
    frac = 0.75
    min_keep = 40

    remove_mask = np.zeros(len(train_df), dtype=bool)
    class_counts = train_df[TARGET].value_counts().to_dict()
    removed_by_class = {k: 0 for k in class_counts}
    ranked = candidate_idx[np.argsort(score[candidate_idx])[::-1]]
    target_remove = int(len(candidate_idx) * frac)
    removed = 0
    for idx in ranked:
        cls = train_df.at[idx, TARGET]
        if class_counts[cls] - (removed_by_class[cls] + 1) < min_keep:
            continue
        remove_mask[idx] = True
        removed_by_class[cls] += 1
        removed += 1
        if removed >= target_remove:
            break
    return remove_mask, base


def add_residual_feature(train_df, val_df, base_score):
    order = np.argsort(base_score)[::-1]
    z = base_score[order]
    z_smooth = savgol_filter(z, window_length=21, polyorder=6, mode="nearest")
    smooth = np.zeros_like(base_score)
    smooth[order] = z_smooth
    resid = base_score - smooth
    t = train_df.copy()
    v = val_df.copy()
    # low-weighted injection
    t["resid_injected"] = 0.2 * resid
    v["resid_injected"] = float(np.median(t["resid_injected"]))
    return t, v


def ae_recon_error(train_df):
    text = TfidfVectorizer(max_features=4000, min_df=2, max_df=0.95, stop_words="english")
    x_text = text.fit_transform(train_df["Description"])
    svd = TruncatedSVD(n_components=40, random_state=42)
    xt = svd.fit_transform(x_text)
    tab = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Amount_bucket", "Transaction Type", "Account Name"]),
            ("num", StandardScaler(), ["Amount"]),
        ]
    )
    x_tab = tab.fit_transform(train_df[["Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    x_tab = x_tab.toarray() if hasattr(x_tab, "toarray") else x_tab
    x = np.hstack([xt, x_tab])
    ae = MLPRegressor(hidden_layer_sizes=(64, 24, 64), max_iter=120, random_state=42)
    ae.fit(x, x)
    xh = ae.predict(x)
    err = np.mean((x - xh) ** 2, axis=1)
    return err


def minority_focal_weights(train_df):
    x = train_df[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]].reset_index(drop=True)
    y = train_df[TARGET].reset_index(drop=True)
    counts = y.value_counts().sort_values()
    minority = set(counts.head(max(1, int(len(counts) * 0.3))).index.tolist())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_true = np.zeros(len(train_df), dtype=float)
    for tr, te in cv.split(x, y):
        m = build_model()
        m.fit(x.iloc[tr], y.iloc[tr])
        p = m.predict_proba(x.iloc[te])
        cls = m.named_steps["clf"].classes_
        cmap = {c: i for i, c in enumerate(cls)}
        for pos, idx in enumerate(te):
            oof_true[idx] = p[pos, cmap[y.iloc[idx]]]

    gamma = 2.0
    base_w = np.ones(len(train_df), dtype=float)
    hard = np.power(1.0 - oof_true, gamma)
    is_minority = y.astype(str).isin([str(c) for c in minority]).to_numpy()
    base_w[is_minority] = 1.0 + 1.2 * hard[is_minority]
    return np.clip(base_w, 0.5, 3.0)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = add_features(pd.read_csv(TRAIN_PATH)).reset_index(drop=True)
    va = add_features(pd.read_csv(VAL_PATH)).reset_index(drop=True)
    nz = pd.read_csv(NOISE_PATH).reset_index(drop=True)

    results = []

    # Current winner baseline (for this train/noise pair)
    rm_mask, base_score = current_winner_mask(tr, nz)
    tr_keep = tr.loc[~rm_mask].copy()
    m0 = build_model()
    m0.fit(tr_keep[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]], tr_keep[TARGET])
    p0 = m0.predict(va[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    results.append({"method": "winner_baseline_classaware", "rows_removed": int(rm_mask.sum()), **metrics(va[TARGET], p0)})

    # 1) Residual-Injection on kept data
    tr_r, va_r = add_residual_feature(tr, va, base_score)
    tr_r_keep = tr_r.loc[~rm_mask].copy()
    mr = build_model(extra_num_cols=["resid_injected"])
    mr.fit(
        tr_r_keep[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount", "resid_injected"]],
        tr_r_keep[TARGET],
    )
    pr = mr.predict(va_r[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount", "resid_injected"]])
    results.append({"method": "residual_injection_low_weight", "rows_removed": int(rm_mask.sum()), **metrics(va[TARGET], pr)})

    # 2) Error-Gating with AE only on already-flagged points
    ae_err = ae_recon_error(tr)
    flagged_idx = np.where(rm_mask)[0]
    for q in [0.25, 0.4, 0.5]:
        thr = np.quantile(ae_err[flagged_idx], q)
        # remove only flagged points that AE says are high-error
        rm2 = np.zeros(len(tr), dtype=bool)
        rm2[flagged_idx] = ae_err[flagged_idx] >= thr
        tr2 = tr.loc[~rm2].copy()
        m2 = build_model()
        m2.fit(tr2[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]], tr2[TARGET])
        p2 = m2.predict(va[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
        results.append({"method": f"error_gating_ae_q{q}", "rows_removed": int(rm2.sum()), **metrics(va[TARGET], p2)})

    # 3) Focal-Class-Awareness on already-cleaned dataset
    w = minority_focal_weights(tr_keep)
    mf = build_model()
    mf.fit(
        tr_keep[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]],
        tr_keep[TARGET],
        clf__sample_weight=w,
    )
    pf = mf.predict(va[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    results.append({"method": "focal_minority_on_cleaned", "rows_removed": int(rm_mask.sum()), **metrics(va[TARGET], pf)})

    out = pd.DataFrame(results).sort_values(["weighted_f1", "macro_f1", "accuracy"], ascending=False)
    out.to_csv(OUT_DIR / "results.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
