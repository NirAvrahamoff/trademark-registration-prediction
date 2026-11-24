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

Result: 227,489 applications, 15.0% failed.

| Split | Filing years | Applications | Failure rate |
|---|---|---|---|
| Train | 2000-2018 | 180,442 | 15.3% |
| Test | 2019-2022 | 47,047 | 13.9% |

## Repository structure

```
data/
  raw/          original data from data.gov.il (gzip)
  processed/    cleaned data produced by the code
notebooks/      01_data_exploration.ipynb: data checks, target definition, EDA
src/            reusable code: load_data.py (raw file), prepare.py (cleaning and target)
figures/        plots used in the report
outputs/        model results and metrics
```

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook 01_data_exploration.ipynb
```

## Team contributions

[to be completed]
