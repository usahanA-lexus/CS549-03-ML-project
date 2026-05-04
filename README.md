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