# CS549-03-ML-project

## Shared Baseline

`scripts/preprocess.py` is for the team's baseline. It standardizes the raw dataset, removes exact duplicates, drops shared non-model columns, applies the same missing-value handling for everyone, and creates one shared train/validation split for model comparison.

The official handoff lives in `processed_output/shared_baseline/`:

- `train_raw_split.csv`
- `valid_raw_split.csv`
- `preprocessing_summary.json`

These files are the starting point for logistic regression, SVM, CatBoost, and any other model. Model-specific encoding, scaling, text handling, and feature engineering should happen only after loading these baseline splits.

## Baseline Workflow

Install the baseline dependencies:

```bash
pip install -r requirements.txt
```

Generate the official shared baseline:

```bash
python scripts/preprocess.py --input data/transactions.csv
```

Generate an optional controlled experiment without `Description`:

```bash
python scripts/preprocess.py \
  --input data/transactions.csv \
  --run-name baseline_no_description \
  --drop-description
```

If you want an explicitly named run that still includes `Description`, use:

```bash
python scripts/preprocess.py \
  --input data/transactions.csv \
  --run-name baseline_with_description
```

## What The Summary Be About

`preprocessing_summary.json` records the shared preprocessing info:

- raw dataset source
- duplicate-removal rule
- dropped columns
- selected feature columns
- missing-value handling and training-derived imputation values
- train/validation split size and random seed
- whether `Description` was included
- shared evaluation setup for the held-out validation split

## Logistic Regression Experiments

All logistic-regression runs in this repo start from:

- `processed_output/shared_baseline/train_raw_split.csv`
- `processed_output/shared_baseline/valid_raw_split.csv`

If an experiment requires the cleaned train file, use:

- `processed_output/shared_baseline/train_raw_split_cleaned.csv`

Compiled logistic-regression outputs and scorecards are stored in:

- `processed_output/logistic_regression_suite/`
- `processed_output/logistic_regression_suite/logistic_regression_compiled_scores.txt`
- `processed_output/logistic_regression_suite/logistic_regression_compiled_scores.csv`

Current best run from the compiled scorecard:

- `iterative_exp2_selective_cleaning_best`

## Logistic Regression Scripts And Purpose

- `scripts/logistic_regression/experiment_1_conservative_relabel.py`: conservative relabeling pass and retrain.
- `scripts/logistic_regression/experiment_2_filter_suspicious_rows.py`: suspicious-row scoring/filtering baseline.
- `scripts/logistic_regression/experiment_3_repeated_cv_filter.py`: repeated CV-based filtering check.
- `scripts/logistic_regression/iterative_exp2_selective_cleaning.py`: iterative correction/filter loop (current best overall).
- `scripts/logistic_regression/experiment_filter_methods.py`: classic smoothing/filtering variants.
- `scripts/logistic_regression/experiment_hybrid_and_classaware.py`: hybrid + class-aware removal search.
- `scripts/logistic_regression/experiment_advanced_filters.py`: advanced filtering benchmark variants.
- `scripts/logistic_regression/experiment_signal_and_label_noise_methods.py`: residual/label-noise alternatives.
- `scripts/logistic_regression/experiment_delta_autoencoder_focal.py`: delta/autoencoder/focal-style variants.
- `scripts/logistic_regression/experiment_hybrid_second_stage.py`: second-stage hybrids on top of winner.
- `scripts/logistic_regression/final_capped_filter_experiment.py`: capped suspicious-row removal sweep.

## How To Run Logistic Regression Experiments

Run preprocess first:

```bash
python scripts/preprocess.py --input data/transactions.csv
```

If needed, reset cleaned train to the 9-category baseline copy:

```powershell
Copy-Item processed_output/shared_baseline/train_raw_split.csv processed_output/shared_baseline/train_raw_split_cleaned.csv -Force
```

Run individual experiments:

```bash
python scripts/logistic_regression/experiment_1_conservative_relabel.py
python scripts/logistic_regression/experiment_2_filter_suspicious_rows.py
python scripts/logistic_regression/experiment_3_repeated_cv_filter.py
python scripts/logistic_regression/iterative_exp2_selective_cleaning.py
python scripts/logistic_regression/experiment_filter_methods.py
python scripts/logistic_regression/experiment_hybrid_and_classaware.py
python scripts/logistic_regression/experiment_advanced_filters.py
python scripts/logistic_regression/experiment_signal_and_label_noise_methods.py
python scripts/logistic_regression/experiment_delta_autoencoder_focal.py
python scripts/logistic_regression/experiment_hybrid_second_stage.py
python scripts/logistic_regression/final_capped_filter_experiment.py
```

Main output locations:

- `processed_output/exp1_conservative_relabel/`
- `processed_output/exp2_filter_suspicious/`
- `processed_output/exp3_repeated_cv_filter/`
- `processed_output/iterative_exp2_selective_cleaning/`
- `processed_output/hybrid_classaware_experiments/`
- `processed_output/advanced_filter_experiments/`
- `processed_output/signal_labelnoise_experiments/`
- `processed_output/delta_autoencoder_focal_experiments/`
- `processed_output/hybrid_second_stage_experiments/`
- `processed_output/final_capped_filter/`

