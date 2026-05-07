from pathlib import Path
import pandas as pd


TRAIN_PATH = Path("processed_output/shared_baseline/train_raw_split.csv")
REVIEWED_PATH = Path("processed_output/shared_baseline/top_suspicious_rows_corrected_only.csv")
OUT_PATH = Path("processed_output/shared_baseline/train_raw_split_cleaned.csv")
LOG_PATH = Path("processed_output/shared_baseline/train_label_corrections_applied.csv")


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    reviewed_df = pd.read_csv(REVIEWED_PATH)

    required_cols = {"source_row_index", "Category_corrected", "Category"}
    missing = required_cols - set(reviewed_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in reviewed file: {sorted(missing)}")

    corrections = reviewed_df[
        reviewed_df["source_row_index"].notna()
        & reviewed_df["Category_corrected"].notna()
        & (reviewed_df["Category_corrected"].astype(str) != reviewed_df["Category"].astype(str))
    ][["source_row_index", "Category", "Category_corrected", "Description", "review_note"]].copy()

    corrections["source_row_index"] = corrections["source_row_index"].astype(int)
    corrections = corrections.drop_duplicates(subset=["source_row_index"], keep="first")

    max_idx = len(train_df) - 1
    bad_idx = corrections[
        (corrections["source_row_index"] < 0) | (corrections["source_row_index"] > max_idx)
    ]
    if not bad_idx.empty:
        raise ValueError("Found out-of-range source_row_index values in reviewed corrections.")

    applied = []
    for row in corrections.itertuples(index=False):
        idx = int(row.source_row_index)
        before = str(train_df.at[idx, "Category"])
        after = str(row.Category_corrected)
        if before == after:
            continue
        train_df.at[idx, "Category"] = after
        applied.append(
            {
                "source_row_index": idx,
                "Category_before": before,
                "Category_after": after,
                "Description": row.Description,
                "review_note": row.review_note,
            }
        )

    train_df.to_csv(OUT_PATH, index=False)
    pd.DataFrame(applied).to_csv(LOG_PATH, index=False)

    print(f"Cleaned training split saved to: {OUT_PATH}")
    print(f"Applied label corrections: {len(applied)}")
    print(f"Correction log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
