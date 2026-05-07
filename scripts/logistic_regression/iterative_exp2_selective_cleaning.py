from pathlib import Path
import json
import os
import shutil
import subprocess

import pandas as pd


PROJECT_ROOT = Path(".")
VAL_PATH = PROJECT_ROOT / "processed_output/shared_baseline/valid_raw_split.csv"
BASE_TRAIN_PATH = PROJECT_ROOT / "processed_output/shared_baseline/train_raw_split_cleaned.csv"
OUT_ROOT = PROJECT_ROOT / "processed_output/iterative_exp2_selective_cleaning"
TOP_N = 300
MAX_ROUNDS = 10
TARGET_WEIGHTED_F1 = float(os.getenv("TARGET_WEIGHTED_F1", "0.20"))
TARGET_MACRO_F1 = float(os.getenv("TARGET_MACRO_F1", "0.12"))
MIN_MERCHANT_COUNT = int(os.getenv("MIN_MERCHANT_COUNT", "5"))
MIN_MERCHANT_PURITY = float(os.getenv("MIN_MERCHANT_PURITY", "0.95"))

MERCHANT_WHITELIST = {
    "netflix",
    "hulu",
    "spotify",
    "apple music",
    "shell",
    "chevron",
    "at&t",
    "verizon",
    "t-mobile",
    "home depot",
    "lowes",
}

# Try different cleaning settings each round and keep the best-improving one.
ROUND_CONFIGS = [
    {"top_n": 150, "min_purity": 0.98},
    {"top_n": 300, "min_purity": 0.98},
    {"top_n": 300, "min_purity": 0.95},
    {"top_n": 500, "min_purity": 0.95},
]


def build_merchant_rules_from_noise(noise_df: pd.DataFrame, min_count: int, min_purity: float):
    df = noise_df.copy()
    df["Description_norm"] = df["Description"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["Description_norm"] != ""]
    counts = (
        df.groupby(["Description_norm", "cv_pred"])
        .size()
        .reset_index(name="n")
    )
    totals = counts.groupby("Description_norm")["n"].sum().reset_index(name="total_n")
    top = counts.sort_values(["Description_norm", "n"], ascending=[True, False]).drop_duplicates("Description_norm")
    merged = top.merge(totals, on="Description_norm", how="left")
    merged["purity"] = merged["n"] / merged["total_n"]
    merged = merged[
        (merged["Description_norm"].isin(MERCHANT_WHITELIST))
        & (merged["total_n"] >= min_count)
        & (merged["purity"] >= min_purity)
    ]
    # desc -> dominant model-predicted category
    return {row["Description_norm"]: row["cv_pred"] for _, row in merged.iterrows()}


def correction_for_row(row, merchant_rules):
    desc = str(row.get("Description", "")).strip().lower()
    current = str(row.get("Category", "")).strip()
    model_pred = str(row.get("cv_pred", "")).strip()
    target = merchant_rules.get(desc)
    if target is None:
        return current, ""
    if current == target:
        return current, ""
    if model_pred == target:
        return target, "merchant_rule + model_agrees"
    return current, ""


def run_exp2(train_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TRAIN_PATH"] = str(train_path).replace("\\", "/")
    env["VAL_PATH"] = str(VAL_PATH).replace("\\", "/")
    env["OUT_DIR"] = str(out_dir).replace("\\", "/")

    cmd = [str(PROJECT_ROOT / ".venv/Scripts/python.exe"), "scripts/logistic_regression/experiment_2_filter_suspicious_rows.py"]
    subprocess.run(cmd, check=True, env=env)


def read_metrics(metrics_path: Path):
    m = pd.read_csv(metrics_path)
    metric_map = {row["metric"]: float(row["value"]) for _, row in m.iterrows()}
    return {
        "accuracy": metric_map["accuracy"],
        "weighted_f1": metric_map["weighted_f1"],
        "macro_f1": metric_map["macro_f1"],
        "rows_removed": metric_map["rows_removed"],
    }


def build_corrections(noise_scores_path: Path, review_out_path: Path, min_count: int, min_purity: float, top_n: int):
    noise_df = pd.read_csv(noise_scores_path).reset_index().rename(columns={"index": "source_row_index"})
    merchant_rules = build_merchant_rules_from_noise(
        noise_df=noise_df,
        min_count=min_count,
        min_purity=min_purity,
    )
    wrong_col = "wrong_flag" if "wrong_flag" in noise_df.columns else "flagged_suspicious"
    rank_term = (1 - noise_df["rank_score"]) if "rank_score" in noise_df.columns else 0.0
    noise_df["suspicion_score"] = (1 - noise_df["true_label_prob"]) + noise_df[wrong_col].astype(float) + rank_term
    top = noise_df.sort_values("suspicion_score", ascending=False).head(top_n).copy()

    corrections = top.apply(lambda r: correction_for_row(r, merchant_rules), axis=1, result_type="expand")
    corrections.columns = ["Category_corrected", "review_note"]
    top = pd.concat([top, corrections], axis=1)
    top["label_changed"] = top["Category"].astype(str) != top["Category_corrected"].astype(str)
    top.to_csv(review_out_path, index=False)

    changed = top[top["label_changed"]].copy()
    return changed


def apply_corrections(train_df: pd.DataFrame, changed_df: pd.DataFrame):
    applied_rows = []
    if changed_df.empty:
        return train_df.copy(), pd.DataFrame(applied_rows)

    next_df = train_df.copy()
    changed_df = changed_df.drop_duplicates(subset=["source_row_index"], keep="first")
    for row in changed_df.itertuples(index=False):
        idx = int(row.source_row_index)
        if idx < 0 or idx >= len(next_df):
            continue
        before = str(next_df.at[idx, "Category"])
        after = str(row.Category_corrected)
        if before == after:
            continue
        next_df.at[idx, "Category"] = after
        applied_rows.append(
            {
                "source_row_index": idx,
                "Category_before": before,
                "Category_after": after,
                "Description": row.Description,
                "review_note": row.review_note,
            }
        )
    return next_df, pd.DataFrame(applied_rows)


def is_improved(prev_metrics, new_metrics):
    return (
        new_metrics["weighted_f1"] > prev_metrics["weighted_f1"]
        and new_metrics["macro_f1"] >= prev_metrics["macro_f1"]
    )


def reached_target(metrics):
    return (
        metrics["weighted_f1"] >= TARGET_WEIGHTED_F1
        and metrics["macro_f1"] >= TARGET_MACRO_F1
    )


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_current = pd.read_csv(BASE_TRAIN_PATH)

    # Round 0 baseline using starting cleaned data.
    round0_dir = OUT_ROOT / "round_0"
    run_exp2(BASE_TRAIN_PATH, round0_dir)
    best_metrics = read_metrics(round0_dir / "metrics.csv")
    history = [{
        "round": 0,
        "applied_corrections": 0,
        **best_metrics,
        "accepted": True,
        "target_reached": reached_target(best_metrics),
    }]
    current_train_path = OUT_ROOT / "train_round_0.csv"
    train_current.to_csv(current_train_path, index=False)

    if reached_target(best_metrics):
        final_train_path = OUT_ROOT / "train_best_iterative_cleaned.csv"
        shutil.copyfile(current_train_path, final_train_path)
        pd.DataFrame(history).to_csv(OUT_ROOT / "iterative_metrics_history.csv", index=False)
        with open(OUT_ROOT / "iterative_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_train_path": str(BASE_TRAIN_PATH),
                    "final_train_path": str(final_train_path),
                    "rounds_recorded": int(len(history)),
                    "best_metrics": best_metrics,
                    "target_weighted_f1": TARGET_WEIGHTED_F1,
                    "target_macro_f1": TARGET_MACRO_F1,
                    "stop_reason": "target_reached_at_round_0",
                },
                f,
                indent=2,
            )
        return

    for rnd in range(1, MAX_ROUNDS + 1):
        noise_path = round0_dir / "training_noise_scores.csv" if rnd == 1 else OUT_ROOT / f"round_{rnd-1}/training_noise_scores.csv"
        best_trial = None

        for cfg_idx, cfg in enumerate(ROUND_CONFIGS, start=1):
            review_path = OUT_ROOT / f"round_{rnd}_cfg_{cfg_idx}_top_suspicious_reviewed.csv"
            changed = build_corrections(
                noise_scores_path=noise_path,
                review_out_path=review_path,
                min_count=MIN_MERCHANT_COUNT,
                min_purity=cfg["min_purity"],
                top_n=cfg["top_n"],
            )
            updated_train, applied = apply_corrections(train_current, changed)
            applied_count = len(applied)
            if applied_count == 0:
                continue

            trial_train_path = OUT_ROOT / f"train_round_{rnd}_cfg_{cfg_idx}.csv"
            applied_log_path = OUT_ROOT / f"round_{rnd}_cfg_{cfg_idx}_applied_corrections.csv"
            updated_train.to_csv(trial_train_path, index=False)
            applied.to_csv(applied_log_path, index=False)

            round_dir = OUT_ROOT / f"round_{rnd}_cfg_{cfg_idx}"
            run_exp2(trial_train_path, round_dir)
            trial_metrics = read_metrics(round_dir / "metrics.csv")

            if (best_trial is None) or (trial_metrics["weighted_f1"] > best_trial["metrics"]["weighted_f1"]):
                best_trial = {
                    "cfg_idx": cfg_idx,
                    "cfg": cfg,
                    "updated_train": updated_train,
                    "trial_train_path": trial_train_path,
                    "applied_count": applied_count,
                    "metrics": trial_metrics,
                }

        if best_trial is None:
            history.append(
                {
                    "round": rnd,
                    "applied_corrections": 0,
                    **best_metrics,
                    "accepted": False,
                    "target_reached": reached_target(best_metrics),
                    "stop_reason": "no_new_obvious_corrections_any_config",
                }
            )
            break

        trial_metrics = best_trial["metrics"]
        accepted = is_improved(best_metrics, trial_metrics)
        history.append(
            {
                "round": rnd,
                "config_idx": best_trial["cfg_idx"],
                "top_n": best_trial["cfg"]["top_n"],
                "min_purity": best_trial["cfg"]["min_purity"],
                "applied_corrections": best_trial["applied_count"],
                **trial_metrics,
                "accepted": accepted,
                "target_reached": reached_target(trial_metrics),
            }
        )

        if accepted:
            train_current = best_trial["updated_train"]
            best_metrics = trial_metrics
            current_train_path = best_trial["trial_train_path"]
            selected_round_dir = OUT_ROOT / f"round_{rnd}"
            selected_round_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                OUT_ROOT / f"round_{rnd}_cfg_{best_trial['cfg_idx']}" / "training_noise_scores.csv",
                selected_round_dir / "training_noise_scores.csv",
            )
            if reached_target(best_metrics):
                break
        else:
            break

    final_train_path = OUT_ROOT / "train_best_iterative_cleaned.csv"
    shutil.copyfile(current_train_path, final_train_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT_ROOT / "iterative_metrics_history.csv", index=False)

    summary = {
        "base_train_path": str(BASE_TRAIN_PATH),
        "final_train_path": str(final_train_path),
        "rounds_recorded": int(len(history)),
        "best_metrics": best_metrics,
        "target_weighted_f1": TARGET_WEIGHTED_F1,
        "target_macro_f1": TARGET_MACRO_F1,
        "target_reached": reached_target(best_metrics),
    }
    with open(OUT_ROOT / "iterative_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved iterative history to:", OUT_ROOT / "iterative_metrics_history.csv")
    print("Saved best cleaned training file to:", final_train_path)
    print("Best metrics:", best_metrics)


if __name__ == "__main__":
    main()

