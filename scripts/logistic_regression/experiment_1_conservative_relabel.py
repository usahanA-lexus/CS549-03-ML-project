# scripts/logistic_regression/experiment_1_conservative_relabel.py

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

TRAIN_PATH = "processed_output/shared_baseline/train_raw_split.csv"
VAL_PATH = "processed_output/shared_baseline/valid_raw_split.csv"
OUT_DIR = "processed_output/exp1_conservative_relabel"

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

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

train_df = add_features(train_df)
val_df = add_features(val_df)

group_cols = ["Description", "Amount_bucket", "Transaction Type"]

train_group = (
    train_df.groupby(group_cols + [TARGET])
    .size()
    .reset_index(name="count")
)

totals = train_group.groupby(group_cols)["count"].sum().reset_index(name="total")
top = train_group.sort_values(group_cols + ["count"], ascending=[True, True, True, False]).drop_duplicates(group_cols)
top = top.merge(totals, on=group_cols, how="left")
top["dominance"] = top["count"] / top["total"]

strong_map = top[top["dominance"] >= 0.90][group_cols + [TARGET, "dominance"]].copy()
strong_map = strong_map.rename(columns={TARGET: "new_label"})

train_relabeled = train_df.merge(strong_map, on=group_cols, how="left")
train_relabeled["Category_original"] = train_relabeled[TARGET]
mask = train_relabeled["new_label"].notna() & (train_relabeled[TARGET] != train_relabeled["new_label"])
train_relabeled.loc[mask, TARGET] = train_relabeled.loc[mask, "new_label"]

text_col = "Description"
cat_cols = ["Amount_bucket", "Transaction Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95, stop_words="english"), text_col),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))
])

X_train = train_relabeled[[text_col] + cat_cols]
y_train = train_relabeled[TARGET]

X_val = val_df[[text_col] + cat_cols]
y_val = val_df[TARGET]

model.fit(X_train, y_train)
pred = model.predict(X_val)

acc = accuracy_score(y_val, pred)
wf1 = f1_score(y_val, pred, average="weighted")
mf1 = f1_score(y_val, pred, average="macro")

pd.DataFrame({
    "metric": ["accuracy", "weighted_f1", "macro_f1", "rows_relabeled"],
    "value": [acc, wf1, mf1, int(mask.sum())]
}).to_csv(f"{OUT_DIR}/metrics.csv", index=False)

val_out = val_df.copy()
val_out["prediction"] = pred
val_out.to_csv(f"{OUT_DIR}/predictions.csv", index=False)

with open(f"{OUT_DIR}/report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Weighted F1: {wf1}\n")
    f.write(f"Macro F1: {mf1}\n")
    f.write(f"Rows relabeled: {int(mask.sum())}\n\n")
    f.write(classification_report(y_val, pred))
