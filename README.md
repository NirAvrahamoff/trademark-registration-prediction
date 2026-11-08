# Predicting Trademark Registration Failure in Israel

Data Mining 2026 course project, Ariel University.

**Team:** Nir Avrahamoff, [partner name]

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

## Repository structure

```
data/
  raw/          original data from data.gov.il (gzip)
  processed/    cleaned data produced by the code
notebooks/      analysis notebooks
src/            reusable code (data loading, features, models)
figures/        plots used in the report
outputs/        model results and metrics
```

## How to run

```bash
pip install -r requirements.txt
```

## Team contributions

[to be completed]
