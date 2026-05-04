import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# CONFIGURATION SWITCH
# =========================
# Options:
# "one_hot"
# "one_hot_balanced"
# "tfidf"
# "no_desc"
MODE = "one_hot_balanced"

# Load data
train_df = pd.read_csv("processed_output/shared_baseline/train_raw_split.csv")
valid_df = pd.read_csv("processed_output/shared_baseline/valid_raw_split.csv")

target_col = "Category"

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_valid = valid_df.drop(columns=[target_col])
y_valid = valid_df[target_col]

# Preprocessing
if MODE == "one_hot" or MODE == "one_hot_balanced":
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             ["Description", "Transaction Type", "Account Name"]),
            ("num", StandardScaler(), ["Amount"]),
        ]
    )

elif MODE == "tfidf":
    text_preprocess = Pipeline([
        ("selector", FunctionTransformer(lambda x: x.squeeze(), validate=False)),
        ("tfidf", TfidfVectorizer())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("desc", text_preprocess, ["Description"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             ["Transaction Type", "Account Name"]),
            ("num", StandardScaler(), ["Amount"]),
        ]
    )

elif MODE == "no_desc":
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             ["Transaction Type", "Account Name"]),
            ("num", StandardScaler(), ["Amount"]),
        ]
    )

else:
    raise ValueError(f"Invalid MODE: {MODE}")

# Model
if MODE == "one_hot":
    svm = LinearSVC(max_iter=5000)
else:
    svm = LinearSVC(max_iter=5000, class_weight="balanced")

# Full pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", svm)
])

# Hyperparameter tuning
param_grid = {
    "model__C": [0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1
)

# Train
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

print(f"\n=== MODE: {MODE} ===")
print("Best parameters:", grid_search.best_params_)

# Evaluate
y_pred = best_model.predict(X_valid)

print("\nAccuracy:", accuracy_score(y_valid, y_pred))

print("\nClassification Report:")
print(classification_report(y_valid, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, y_pred))