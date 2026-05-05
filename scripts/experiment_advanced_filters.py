import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import filtfilt, firwin, medfilt, savgol_filter
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
OUT_DIR = Path("processed_output/advanced_filter_experiments")
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
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95, stop_words="english"), "Description"),
            (
                "cat",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
                ),
                ["Amount_bucket", "Transaction Type"],
            ),
        ]
    )
    return Pipeline(
        [("prep", preprocessor), ("clf", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"))]
    )


def eval_model(train_df, val_df, remove_mask):
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
    }


def trimmed_mean_filter(x, k=7, trim=0.2):
    half = k // 2
    y = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        w = np.sort(x[lo:hi])
        t = int(len(w) * trim)
        if t * 2 >= len(w):
            y[i] = float(np.mean(w))
        else:
            y[i] = float(np.mean(w[t : len(w) - t]))
    return y


def build_scores(base):
    scores = {}
    order = np.argsort(base)[::-1]
    z = base[order]

    # 1) Adjusted Savitzky-Golay
    scores["savgol_w21_p6"] = np.zeros_like(base, dtype=float)
    scores["savgol_w21_p6"][order] = savgol_filter(z, window_length=21, polyorder=6, mode="nearest")

    # 2) Forward-reverse FIR (zero phase)
    b = firwin(numtaps=17, cutoff=0.15)
    fr = filtfilt(b, [1.0], z, method="gust")
    scores["filtfilt_fir17_c015"] = np.zeros_like(base, dtype=float)
    scores["filtfilt_fir17_c015"][order] = fr

    # 3) Trimmed mean
    tm = trimmed_mean_filter(z, k=9, trim=0.2)
    scores["trimmed_mean_k9_t02"] = np.zeros_like(base, dtype=float)
    scores["trimmed_mean_k9_t02"][order] = tm

    # 4) Filter-then-threshold (gaussian smoothed score as ranking signal)
    gf = gaussian_filter1d(z, sigma=2.0, mode="nearest")
    scores["gauss_sigma2"] = np.zeros_like(base, dtype=float)
    scores["gauss_sigma2"][order] = gf

    # 5) Median reference
    md = medfilt(z, kernel_size=7)
    scores["median_k7"] = np.zeros_like(base, dtype=float)
    scores["median_k7"][order] = md

    # 6) Standardized-first variant
    zstd = (base - base.mean()) / (base.std() + 1e-12)
    scores["zscore_raw"] = zstd

    # 7) Optional methods when libs exist
    try:
        import pywt  # type: ignore

        c = pywt.wavedec(z, "db4", level=3)
        # soft-threshold high-frequency details
        for i in range(1, len(c)):
            thr = np.median(np.abs(c[i])) / 0.6745 * np.sqrt(2 * np.log(len(z)))
            c[i] = pywt.threshold(c[i], thr, mode="soft")
        wz = pywt.waverec(c, "db4")[: len(z)]
        scores["wavelet_db4_l3"] = np.zeros_like(base, dtype=float)
        scores["wavelet_db4_l3"][order] = wz
    except Exception:
        pass

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore

        idx = np.arange(len(z))
        lw = lowess(z, idx, frac=0.05, it=2, return_sorted=False)
        scores["lowess_f005_it2"] = np.zeros_like(base, dtype=float)
        scores["lowess_f005_it2"][order] = lw
    except Exception:
        pass

    try:
        from PyEMD import EMD  # type: ignore

        emd = EMD()
        imfs = emd(z)
        if imfs.ndim == 2:
            if imfs.shape[0] >= 2:
                # drop highest-frequency IMF
                recon = np.sum(imfs[1:, :], axis=0)
            else:
                recon = imfs[0]
            scores["emd_drop_imf1"] = np.zeros_like(base, dtype=float)
            scores["emd_drop_imf1"][order] = recon
    except Exception:
        pass

    return scores


def classaware_mask(train_df, candidate_idx, score, frac=0.75, min_keep=40):
    remove_mask = np.zeros(len(train_df), dtype=bool)
    class_counts = train_df[TARGET].value_counts().to_dict()
    removed_by_class = {k: 0 for k in class_counts}
    target_remove = int(len(candidate_idx) * frac)
    ranked = candidate_idx[np.argsort(score[candidate_idx])[::-1]]
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
    return remove_mask


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = add_features(pd.read_csv(TRAIN_PATH))
    val_df = add_features(pd.read_csv(VAL_PATH))
    noise_df = pd.read_csv(NOISE_PATH)

    base = (1.0 - noise_df["true_label_prob"].to_numpy()) + noise_df["flagged_suspicious"].astype(float).to_numpy()
    candidate = np.where((noise_df["flagged_suspicious"].astype(bool).to_numpy()) & (noise_df["true_label_prob"].to_numpy() < 0.15))[0]

    score_map = build_scores(base)
    results = []

    for name, score in score_map.items():
        score_norm = (score - score.min()) / (score.max() - score.min() + 1e-12)

        # Filter-then-threshold at quantiles over candidate pool
        for q in [0.90, 0.95, 0.975]:
            thr = np.quantile(score_norm[candidate], q)
            remove = np.zeros(len(train_df), dtype=bool)
            remove[candidate] = score_norm[candidate] >= thr
            m = eval_model(train_df, val_df, remove)
            results.append({"method": name, "strategy": f"threshold_q{q}", **m})

        # Class-aware ranked removal for macro stability
        for frac in [0.6, 0.75, 0.9]:
            for keep in [30, 40, 60]:
                remove = classaware_mask(train_df, candidate, score_norm, frac=frac, min_keep=keep)
                m = eval_model(train_df, val_df, remove)
                results.append({"method": name, "strategy": f"classaware_f{frac}_k{keep}", **m})

    out = pd.DataFrame(results).sort_values(["weighted_f1", "macro_f1", "accuracy"], ascending=False)
    out.to_csv(OUT_DIR / "advanced_results.csv", index=False)
    out.head(1).to_csv(OUT_DIR / "best_weighted.csv", index=False)
    out.sort_values(["macro_f1", "weighted_f1", "accuracy"], ascending=False).head(1).to_csv(
        OUT_DIR / "best_macro.csv", index=False
    )
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
