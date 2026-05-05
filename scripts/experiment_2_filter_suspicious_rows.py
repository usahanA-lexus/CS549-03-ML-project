# scripts/experiment_2_filter_suspicious_rows.py

import os
import re
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = os.getenv("TRAIN_PATH", "processed_output/shared_baseline/train_raw_split.csv")
VAL_PATH = os.getenv("VAL_PATH", "processed_output/shared_baseline/valid_raw_split.csv")
OUT_DIR = os.getenv("OUT_DIR", "processed_output/exp2_filter_suspicious")
SUSPICIOUS_PROB_THRESHOLD = float(os.getenv("SUSPICIOUS_PROB_THRESHOLD", "0.20"))
LR_C = float(os.getenv("LR_C", "2.0"))
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "10000"))

os.makedirs(OUT_DIR, exist_ok=True)

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
    except:
        return "unknown"
    a = abs(v)
    if a < 10:
        return "0_10"
    elif a < 25:
        return "10_25"
    elif a < 50:
        return "25_50"
    elif a < 100:
        return "50_100"
    elif a < 250:
        return "100_250"
    else:
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
            ("text", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=2, max_df=0.95, stop_words="english"), text_col),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=3000, C=LR_C, class_weight="balanced"))
    ])
    return model

train_df = add_features(pd.read_csv(TRAIN_PATH))
val_df = add_features(pd.read_csv(VAL_PATH))

feature_cols = ["Description", "Amount_bucket", "Transaction Type"]
X = train_df[feature_cols].reset_index(drop=True)
y = train_df[TARGET].reset_index(drop=True)

oof_pred = np.empty(len(train_df), dtype=object)
oof_true_prob = np.zeros(len(train_df), dtype=float)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
    model = build_model()
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

    proba = model.predict_proba(X.iloc[te_idx])
    preds = model.predict(X.iloc[te_idx])
    class_order = model.named_steps["clf"].classes_
    class_to_idx = {cls: i for i, cls in enumerate(class_order)}

    oof_pred[te_idx] = preds
    for pos, row_idx in enumerate(te_idx):
        true_label = y.iloc[row_idx]
        oof_true_prob[row_idx] = proba[pos, class_to_idx[true_label]]

suspicious = (oof_pred != y.values) & (oof_true_prob < SUSPICIOUS_PROB_THRESHOLD)

train_filtered = train_df.loc[~suspicious].copy()

final_model = build_model()
final_model.fit(train_filtered[feature_cols], train_filtered[TARGET])
pred = final_model.predict(val_df[feature_cols])

acc = accuracy_score(val_df[TARGET], pred)
wf1 = f1_score(val_df[TARGET], pred, average="weighted")
mf1 = f1_score(val_df[TARGET], pred, average="macro")

pd.DataFrame({
    "metric": ["accuracy", "weighted_f1", "macro_f1", "rows_removed"],
    "value": [acc, wf1, mf1, int(suspicious.sum())]
}).to_csv(f"{OUT_DIR}/metrics.csv", index=False)

diag = train_df.copy()
diag["cv_pred"] = oof_pred
diag["true_label_prob"] = oof_true_prob
diag["flagged_suspicious"] = suspicious
diag.to_csv(f"{OUT_DIR}/training_noise_scores.csv", index=False)

val_out = val_df.copy()
val_out["prediction"] = pred
val_out.to_csv(f"{OUT_DIR}/predictions.csv", index=False)

with open(f"{OUT_DIR}/report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Weighted F1: {wf1}\n")
    f.write(f"Macro F1: {mf1}\n")
    f.write(f"Rows removed: {int(suspicious.sum())}\n\n")
    f.write(classification_report(val_df[TARGET], pred))
