# Predicting Trademark Registration Failure in Israel

Data Mining 2026 course project, Ariel University.

**Team:** Nir Avrahamoff, Itamar Rot

Each team member works on a personal branch (`nir`, `itamar`). Finished work is merged into `main` through a pull request.

## Goal

Predict, from the details of a trademark application alone, whether the application will fail to be registered in Israel. The project has two parts:

1. **Classification:** an end-to-end pipeline comparing at least five classification models.
2. **Unsupervised analysis:** at least three clustering models on the same data.

## Dataset

**Source:** Israel Patent Office, "סימני מסחר" (Trademark Database), data.gov.il
https://data.gov.il/dataset/simaneymisahr

The main file has 364,238 applications and 31 columns. It covers applications from 1920 to today. The file in `data/raw/trademarks.csv.gz` is the original CSV, compressed with gzip.

Known data issues in the original file:

- 42 invalid UTF-8 bytes. Decode with `errors="replace"`.
- 123 malformed rows with the wrong number of fields (the file has 31 columns). These rows are dropped on load, leaving 364,115 rows.

## Target definition

An application counts as **failed** if it was decided and never got a registration date.

- The `status` column can't be used as the label directly. Most applications with status "closed" (141,354 of 196,066) were registered and lapsed later, and all "non-payment" applications were registered.
- Only decided applications are kept (status registered, closed or non-payment). The 7,481 applications still in process have no outcome yet.
- Only applications filed in 2000-2022 are kept. Older records look inconsistent, and from 2023 many applications are still pending.
- Columns filled after the decision are never used as features: status, registration date, expiry date, last update.

Result: 227,489 applications, 15.0% failed. (Two "registered" rows without a registration date count as registered.)

| Split | Filing years | Applications | Failure rate |
|---|---|---|---|
| Train | 2000-2018 | 180,442 | 15.3% |
| Test | 2019-2022 | 47,047 | 13.9% |

## Results so far

Classification (test set 2019-2022). The model is chosen and the threshold is set on the validation years 2017-2018;
the test set is used only for this table.

| Model | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.419 | 0.728 | 0.348 | 0.339 | 0.399 | 0.366 |
| LightGBM | 0.416 | 0.723 | 0.343 | 0.315 | 0.422 | 0.360 |
| MLP | 0.350 | 0.705 | 0.295 | 0.254 | 0.533 | 0.344 |
| Logistic Regression | 0.348 | 0.704 | 0.281 | 0.280 | 0.423 | 0.337 |
| Decision Tree | 0.324 | 0.675 | 0.281 | 0.265 | 0.402 | 0.320 |
| KNN | 0.299 | 0.652 | 0.259 | 0.248 | 0.386 | 0.302 |
| Baseline (majority) | 0.144 | 0.500 | 0.139 | 0 | 0 | 0 |

Clustering: K-Means++ and Agglomerative (Ward) agree closely (ARI 0.83) with 30 clusters. Service clusters
(telecom, computer services, restaurants, advertising) fail the most, around 20-23%. Software filed in both classes 9 and 42
fails at 12%, versus 23% for computer services filed alone.

## Repository structure

```
data/
  raw/          original data from data.gov.il (gzip)
  processed/    cleaned data produced by the code
notebooks/      01_data_exploration.ipynb   data checks, target definition, EDA
                02_feature_engineering.ipynb   features, leakage check
                03_classification.ipynb   6 models, tuning, imbalance, variants, importance, errors
                04_clustering.ipynb   K-Means++, Agglomerative, GMM, DBSCAN
src/            load_data.py (raw file), prepare.py (cleaning and target),
                features.py (feature engineering), modeling.py (pipeline, models, metrics)
figures/        plots used in the report
outputs/        model results and metrics
```

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run the notebooks in order. Notebook 02 writes `data/processed/features.parquet` (not in git, about 50 MB),
which notebooks 03 and 04 read. Notebook 03 has `FAST_MODE = True` in its settings cell: it loads the tuned parameters from
`outputs/best_params.json` and runs in about 25 minutes on a 2-core machine. With `FAST_MODE = False` it runs the
full hyperparameter search again (about an hour more).

The versions in `requirements.txt` are pinned so the notebooks reproduce the reported numbers (Python 3.11 or newer).

## Team contributions

[to be completed]
