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
