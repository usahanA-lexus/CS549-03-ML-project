from pathlib import Path
import pandas as pd


IN_PATH = Path("processed_output/final_capped_filter/training_noise_scores.csv")
OUT_DIR = Path("processed_output/shared_baseline")
TOP_N = 300

# Conservative, high-precision description rules for obvious mislabels.
KEYWORD_LABEL_RULES = [
    ("paycheck", "Paycheck"),
    ("salary", "Paycheck"),
    ("netflix", "Television"),
    ("hulu", "Television"),
    ("spotify", "Music"),
    ("apple music", "Music"),
    ("movie theater", "Movies & DVDs"),
    ("cinema", "Movies & DVDs"),
    ("grocery", "Groceries"),
    ("supermarket", "Groceries"),
    ("mortgage", "Mortgage & Rent"),
    ("rent", "Mortgage & Rent"),
    ("hardware store", "Home Improvement"),
    ("home depot", "Home Improvement"),
    ("lowes", "Home Improvement"),
    ("gas station", "Gas & Fuel"),
    ("shell", "Gas & Fuel"),
    ("chevron", "Gas & Fuel"),
    ("at&t", "Mobile Phone"),
    ("verizon", "Mobile Phone"),
    ("t-mobile", "Mobile Phone"),
]


def corrected_label_for_row(row):
    desc = str(row.get("Description", "")).strip().lower()
    current = str(row.get("Category", "")).strip()
    model_pred = str(row.get("cv_pred", "")).strip()
    for keyword, target_label in KEYWORD_LABEL_RULES:
        if keyword in desc:
            if current == target_label:
                return current, ""
            # Require model agreement to keep corrections conservative.
            if model_pred == target_label:
                return target_label, f"keyword='{keyword}' + model_agrees"
            return current, ""
    return current, ""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    noise_df = pd.read_csv(IN_PATH)

    # Rank by suspicion: low true_label_prob, low rank_score, high wrong_flag.
    noise_df["suspicion_score"] = (
        (1 - noise_df["true_label_prob"])
        + noise_df["wrong_flag"]
        + (1 - noise_df["rank_score"])
    )

    noise_df = noise_df.reset_index().rename(columns={"index": "source_row_index"})
    top_suspicious = noise_df.sort_values("suspicion_score", ascending=False).head(TOP_N).copy()

    corrections = top_suspicious.apply(corrected_label_for_row, axis=1, result_type="expand")
    corrections.columns = ["Category_corrected", "review_note"]
    top_suspicious = pd.concat([top_suspicious, corrections], axis=1)
    top_suspicious["label_changed"] = (
        top_suspicious["Category_corrected"].astype(str) != top_suspicious["Category"].astype(str)
    )

    top_out = OUT_DIR / "top_suspicious_rows_reviewed.csv"
    changed_out = OUT_DIR / "top_suspicious_rows_corrected_only.csv"
    likely_out = OUT_DIR / "likely_mislabeled.csv"

    top_suspicious.to_csv(top_out, index=False)
    top_suspicious[top_suspicious["label_changed"]].to_csv(changed_out, index=False)

    likely_mislabeled = noise_df[
        (noise_df["true_label_prob"] < 0.05) & (noise_df["wrong_flag"] == 1)
    ]
    likely_mislabeled.to_csv(likely_out, index=False)

    print(f"Saved reviewed top-{TOP_N} rows: {top_out}")
    print(f"Obvious label corrections: {int(top_suspicious['label_changed'].sum())}")
    print(f"Saved corrected-only subset: {changed_out}")
    print(f"Flagged likely mislabeled rows: {len(likely_mislabeled)}")


if __name__ == "__main__":
    main()
