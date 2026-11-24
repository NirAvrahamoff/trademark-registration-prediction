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

| Model | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| **LightGBM** | 0.439 | 0.728 | 0.353 | 0.331 | 0.416 | 0.368 | 0.801 |
| Random Forest | 0.418 | 0.724 | 0.344 | 0.309 | 0.463 | 0.370 | 0.781 |
| MLP | 0.355 | 0.693 | 0.290 | 0.287 | 0.411 | 0.338 | 0.775 |
| Logistic Regression | 0.349 | 0.702 | 0.281 | 0.302 | 0.378 | 0.336 | 0.792 |
| Decision Tree | 0.330 | 0.695 | 0.289 | 0.264 | 0.488 | 0.343 | 0.739 |
| KNN | 0.298 | 0.646 | 0.256 | 0.234 | 0.429 | 0.303 | 0.724 |
| Baseline (majority) | 0.144 | 0.500 | 0.139 | 0 | 0 | 0 | 0.861 |

- Checking the 10% highest-risk applications finds 28% of the failures (2.8 times a random 10%).
- A random 80/20 split would report PR-AUC 0.60 instead of 0.35 (notebook 05).
- Even with yearly retraining, later years are harder to predict (notebook 05).

Clustering: K-Means++ and Agglomerative (Ward) agree closely (ARI 0.83) with 30 clusters. Service clusters
(telecom, computer services, restaurants, advertising) fail the most, around 20-23%. Inside the large classes, the goods text
splits applications into sub-groups whose failure rates differ three- to eightfold (notebook 06).

## Repository structure

```
data/
  raw/          original data from data.gov.il (gzip)
  processed/    cleaned data produced by the code
notebooks/      01_data_exploration.ipynb   data checks, target definition, EDA
                02_feature_engineering.ipynb   features, leakage check
                03_classification.ipynb   6 models, tuning, imbalance, variants, importance, errors
                04_clustering.ipynb   K-Means++, Agglomerative, GMM, DBSCAN
                05_time_evaluation.ipynb   random vs time split, year-by-year evaluation
                06_within_class_clustering.ipynb   sub-groups inside the large Nice classes
                07_self_training.ipynb   semi-supervised experiment
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
which notebooks 03 and 04 read. Notebook 03 has a `FAST_MODE` setting at the top. The saved notebook was run with `FAST_MODE = False`
(full hyperparameter search, about 1.5 hours on a 2-core machine). Set it to `True` to load the tuned parameters
from `outputs/best_params.json` instead; the run then takes about 15 minutes. Notebooks 04, 05 and 07 read the same
parameter file, so run notebook 03 first.

The versions in `requirements.txt` are pinned so the notebooks reproduce the reported numbers (Python 3.11 or newer).

## Team contributions

[to be completed]
