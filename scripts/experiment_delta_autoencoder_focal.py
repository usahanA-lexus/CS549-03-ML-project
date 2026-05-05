import re
from pathlib import Path

import numpy as np
import pandas as pd
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
OUT_DIR = Path("processed_output/delta_autoencoder_focal_experiments")
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
    if "Account Name" not in df.columns:
        df["Account Name"] = "unknown"
    if "Amount" not in df.columns:
        df["Amount"] = 0.0
    df["Description"] = df["Description"].fillna("").astype(str).apply(clean_description)
    df["Transaction Type"] = df["Transaction Type"].fillna("unknown").astype(str)
    df["Account Name"] = df["Account Name"].fillna("unknown").astype(str)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Amount_bucket"] = df["Amount"].apply(amount_bucket)
    return df


def eval_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def build_model(use_delta=False, use_ae_error=False):
    cat_cols = ["Amount_bucket", "Transaction Type", "Account Name"]
    num_cols = ["Amount"]
    if use_delta:
        num_cols += ["noise_delta", "noise_delta_sq"]
    if use_ae_error:
        num_cols += ["ae_recon_err"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95, stop_words="english"), "Description"),
            (
                "cat",
                Pipeline(
                    [("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]
                ),
                cat_cols,
            ),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ]
    )
    return Pipeline(
        [("prep", preprocessor), ("clf", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))]
    )


def add_delta_sigma(df, noise_df):
    df = df.copy().reset_index(drop=True)
    noise = noise_df.copy().reset_index(drop=True)
    base = (1.0 - noise["true_label_prob"].to_numpy()) + noise["flagged_suspicious"].astype(float).to_numpy()
    order = np.argsort(base)[::-1]
    sorted_vals = base[order]
    d1 = np.diff(sorted_vals, prepend=sorted_vals[0])
    d1_sq = d1 ** 2
    back_d1 = np.zeros_like(base)
    back_d1_sq = np.zeros_like(base)
    back_d1[order] = d1
    back_d1_sq[order] = d1_sq
    df["noise_delta"] = back_d1
    df["noise_delta_sq"] = back_d1_sq
    return df


def build_dense_embedding(train_df, val_df):
    text_vec = TfidfVectorizer(max_features=4000, min_df=2, max_df=0.95, stop_words="english")
    xtr_text = text_vec.fit_transform(train_df["Description"])
    xva_text = text_vec.transform(val_df["Description"])
    svd = TruncatedSVD(n_components=40, random_state=42)
    xtr_t = svd.fit_transform(xtr_text)
    xva_t = svd.transform(xva_text)

    tab = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Amount_bucket", "Transaction Type", "Account Name"]),
            ("num", StandardScaler(), ["Amount"]),
        ]
    )
    xtr_tab = tab.fit_transform(train_df[["Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    xva_tab = tab.transform(val_df[["Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    xtr_tab = xtr_tab.toarray() if hasattr(xtr_tab, "toarray") else xtr_tab
    xva_tab = xva_tab.toarray() if hasattr(xva_tab, "toarray") else xva_tab
    return np.hstack([xtr_t, xtr_tab]), np.hstack([xva_t, xva_tab])


def add_autoencoder_error(train_df, val_df):
    xtr, xva = build_dense_embedding(train_df, val_df)
    ae = MLPRegressor(
        hidden_layer_sizes=(64, 24, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=120,
        random_state=42,
    )
    ae.fit(xtr, xtr)
    xtr_hat = ae.predict(xtr)
    xva_hat = ae.predict(xva)
    tr_err = np.mean((xtr - xtr_hat) ** 2, axis=1)
    va_err = np.mean((xva - xva_hat) ** 2, axis=1)
    t = train_df.copy()
    v = val_df.copy()
    t["ae_recon_err"] = tr_err
    v["ae_recon_err"] = va_err
    return t, v


def focal_style_weights(train_df):
    # Focal-like reweighting via OOF true-class probability from baseline model.
    x = train_df[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]].reset_index(drop=True)
    y = train_df[TARGET].reset_index(drop=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_true_prob = np.zeros(len(train_df), dtype=float)
    for tr, te in cv.split(x, y):
        m = build_model(use_delta=False, use_ae_error=False)
        m.fit(x.iloc[tr], y.iloc[tr])
        p = m.predict_proba(x.iloc[te])
        classes = m.named_steps["clf"].classes_
        cmap = {c: i for i, c in enumerate(classes)}
        for pos, idx in enumerate(te):
            oof_true_prob[idx] = p[pos, cmap[y.iloc[idx]]]
    gamma = 2.0
    alpha = 0.75
    w = alpha * np.power(1.0 - oof_true_prob, gamma)
    return np.clip(w, 0.05, 2.5)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = add_features(pd.read_csv(TRAIN_PATH))
    va = add_features(pd.read_csv(VAL_PATH))
    nz = pd.read_csv(NOISE_PATH)

    rows = []

    # Baseline on current train
    m0 = build_model()
    m0.fit(tr[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]], tr[TARGET])
    p0 = m0.predict(va[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    rows.append({"method": "baseline", "rows_removed": 0, **eval_metrics(va[TARGET], p0)})

    # 1) Delta-sigma temporal-context features
    tr_d = add_delta_sigma(tr, nz)
    va_d = va.copy()
    va_d["noise_delta"] = float(tr_d["noise_delta"].median())
    va_d["noise_delta_sq"] = float(tr_d["noise_delta_sq"].median())
    md = build_model(use_delta=True, use_ae_error=False)
    md.fit(
        tr_d[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount", "noise_delta", "noise_delta_sq"]],
        tr_d[TARGET],
    )
    pd_ = md.predict(
        va_d[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount", "noise_delta", "noise_delta_sq"]]
    )
    rows.append({"method": "delta_sigma_features", "rows_removed": 0, **eval_metrics(va[TARGET], pd_)})

    # 2) Autoencoder reconstruction error as feature
    tr_a, va_a = add_autoencoder_error(tr, va)
    ma = build_model(use_delta=False, use_ae_error=True)
    ma.fit(
        tr_a[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount", "ae_recon_err"]],
        tr_a[TARGET],
    )
    pa = ma.predict(va_a[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount", "ae_recon_err"]])
    rows.append({"method": "autoencoder_recon_error_feature", "rows_removed": 0, **eval_metrics(va[TARGET], pa)})

    # 2b) Autoencoder threshold cleaning
    q = np.quantile(tr_a["ae_recon_err"], 0.95)
    keep = tr_a["ae_recon_err"] <= q
    mt = build_model(use_delta=False, use_ae_error=False)
    mt.fit(
        tr_a.loc[keep, ["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]],
        tr_a.loc[keep, TARGET],
    )
    pt = mt.predict(va[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    rows.append(
        {"method": "autoencoder_threshold_95", "rows_removed": int((~keep).sum()), **eval_metrics(va[TARGET], pt)}
    )

    # 3) Focal-style reweighting (hard-example emphasis)
    w = focal_style_weights(tr)
    mf = build_model(use_delta=False, use_ae_error=False)
    mf.fit(
        tr[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]],
        tr[TARGET],
        clf__sample_weight=w,
    )
    pf = mf.predict(va[["Description", "Amount_bucket", "Transaction Type", "Account Name", "Amount"]])
    rows.append({"method": "focal_style_reweight_gamma2", "rows_removed": 0, **eval_metrics(va[TARGET], pf)})

    out = pd.DataFrame(rows).sort_values(["weighted_f1", "macro_f1", "accuracy"], ascending=False)
    out.to_csv(OUT_DIR / "results.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
