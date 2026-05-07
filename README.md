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

## Logistic Regression Experiments (Alexus Aguirre Arias)

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

### Logistic Regression Scripts And Purpose

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

### How To Run Logistic Regression Experiments

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

## CatBoost Model (Nicholas Nguyen)

The CatBoost gradient boosting model is implemented in:

```
scripts/train_catboost.py
```

CatBoost handles `Description`, `Transaction Type`, and `Account Name` natively as categorical features (no manual one-hot encoding) and `Amount` as a numeric feature. The script does both hyperparameter tuning and the 2x2 ablation in one run.

### Tuning protocol

Hyperparameters are tuned via 5-fold stratified cross-validation on the training split only; `valid_raw_split.csv` is reserved for the single final reported number per ablation cell. Otherwise the held-out validation set leaks into model selection and cross-model comparisons become invalid.

- Grid: `learning_rate` in {0.03, 0.05, 0.1} x `depth` in {4, 6, 8} (9 configurations)
- Iterations capped at 1000 with early stopping after 50 rounds without improvement
- Selection metric: weighted F1
- Tuning randomness comes from the fold split (`StratifiedKFold(random_state=42)`); the CatBoost `random_seed` is held fixed at 0 inside CV to keep per-config rankings stable

### Ablation cells

The chosen hyperparameters are reused across all 4 cells so the comparison isolates feature/weight effects rather than re-tuning artifacts. Each cell is trained with five seeds (0-4); reported metrics are mean +/- std across those seeds.

| cell | Description | class weights | training data |
|---|---|---|---|
| `baseline` | included | none | `processed_output/shared_baseline/` |
| `class_weighted` | included | balanced | `processed_output/shared_baseline/` |
| `no_description` | excluded | none | `processed_output/baseline_no_description/` |
| `no_description_weighted` | excluded | balanced | `processed_output/baseline_no_description/` |

Class weights come from `suggested_class_weights_from_training_split` in `preprocessing_summary.json`, which is the sklearn `compute_class_weight(class_weight="balanced", ...)` value computed on the training split.

### How to run

From the project root:

```bash
# Regenerate the no-Description baseline (only needed once or after preprocess.py changes)
python scripts/preprocess.py \
  --input data/transactions.csv \
  --run-name baseline_no_description \
  --drop-description

# Full 2x2 ablation (tunes once, runs all 4 cells)
python scripts/train_catboost.py --all-ablations

# Or a single cell
python scripts/train_catboost.py --run-name baseline
python scripts/train_catboost.py --run-name class_weighted --use-class-weights

# CV grid heatmap (run after train_catboost.py has produced cv_tuning_results.json)
python scripts/cv_heatmap.py
```

### Output

Per ablation cell (under `processed_output/models/catboost/<cell>/`):

- `metrics.json` - accuracy / weighted F1 / macro F1 (mean +/- std across seeds), per-class precision/recall/F1 from seed 0, per-seed values, training time
- `confusion_matrix.png` - seed 0, on validation split
- `feature_importance.csv` and `feature_importance.png` - `PredictionValuesChange` from seed 0
- `best_params.json` - hyperparameters used

Top-level (under `processed_output/models/catboost/`):

- `cv_tuning_results.json` - full grid trace + best params
- `ablation_summary.json` - 2x2 summary written after `--all-ablations`
- `per_class_f1_delta.png` - F1(class_weighted) - F1(baseline) per class, sorted by support
- `cv_heatmap.png` - 3x3 heatmap of mean weighted F1 across the grid (from `cv_heatmap.py`)

### Notes

- The validation split is ~2160 rows (20% of ~10,800). Differences smaller than 1-2% F1 are within noise; treat the ablation as directional, not significance-tested.
- The shared baseline currently contains label noise: rows without a `User ID` (the majority of the dataset) have effectively random Description->Category mappings. CatBoost reflects this by reporting modest weighted F1; the model is fitting the marginal class distribution rather than a deterministic feature->label mapping. This is a property of the data, not the model.

## SVM Model (Deryn Cabana)

The SVM model is implemented in:

```
scripts/train_svm.py
```

This script uses the shared baseline splits and evaluates different modeling configurations through a single configuration switch.

### How to Run

From the project root:

```bash
python scripts/train_svm.py
```

### Configuration Modes

Edit the `MODE` variable inside `train_svm.py` to select the experiment:

- `one_hot`
  Uses one-hot encoding for all categorical features, including Description. No class weighting.

- `one_hot_balanced`
  Same as above, but uses class weighting to address class imbalance.

- `tfidf`
  Uses TF-IDF vectorization for the Description field and one-hot encoding for other categorical features. Includes class weighting.

- `no_desc`
  Excludes the Description feature entirely. Uses one-hot encoding for remaining categorical features with class weighting.

### Hyperparameter Tuning

All configurations use GridSearchCV to tune the SVM regularization parameter `C` over:

```
[0.01, 0.1, 1, 10]
```

using 3-fold cross-validation and weighted F1 score.

### Output

Each run prints:

- Best hyperparameters
- Accuracy on validation set
- Classification report (precision, recall, F1-score)
- Confusion matrix

### Notes

- All models use the same preprocessing baseline for fair comparison.
- Experiments are designed to evaluate the impact of:
  - class imbalance handling
  - text feature representation
  - feature inclusion/exclusion

## Three-model comparison (CatBoost vs SVM vs LogReg)

`scripts/compare_models.py` runs all three models against the same shared baseline across the same 4 ablation cells and produces a unified comparison.

```bash
python scripts/compare_models.py
```

CatBoost results are read from existing artifacts under `processed_output/models/catboost/` (multi-seed mean across seeds 0-4 produced by `scripts/train_catboost.py --all-ablations`). SVM and LogReg are trained inline against the same shared baseline; they're effectively deterministic given fixed `C` and `random_state`, so they run with a single seed (the comparison CSV records `seeds_run = 1` for them and `seeds_run = 5` for CatBoost so the asymmetry is explicit).

The SVM configuration mirrors `scripts/train_svm.py` (Deryn's script) - `LinearSVC` with `OneHotEncoder` + `StandardScaler` and `GridSearchCV` over `C in {0.01, 0.1, 1, 10}`. The 4 ablation cells map to {with/without Description} x {with/without `class_weight="balanced"`}. The LogReg configuration is a sklearn `LogisticRegression(solver="lbfgs", max_iter=2000)` with the same one-hot/scaler preprocessor and `C in {0.01, 0.1, 1, 10, 100}`.

Output (under `processed_output/comparison/`):

- `comparison_summary.csv` - flat table: model x cell x {weighted_f1, macro_f1, accuracy, training_time_seconds, seeds_run}
- `comparison_full.json` - same plus per-class precision/recall/F1
- `weighted_f1_bar.png`, `macro_f1_bar.png`, `accuracy_bar.png` - grouped bar charts (4 cells x 3 models)
- `per_class_f1_baseline.png` - per-class F1 for the baseline cell, all three models side by side
