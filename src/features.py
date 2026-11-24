"""Feature engineering for the trademark failure model.

All features use only information available when the application is filed.
History features (prior filings, past failure rates) look only at earlier
filing dates, and the failure-rate features use training labels only.
"""
import re

import numpy as np
import pandas as pd

TRAIN_LAST_YEAR = 2018
SMOOTHING = 20  # weight of the global rate in the smoothed failure rates

COMPANY_PATTERN = re.compile(
    r"\b(?:ltd|limited|inc|corp|corporation|co\.|company|gmbh|ag|s\.a\.|s\.p\.a|llc|plc|b\.v\.|kg|sas|sarl)\b|בע\"?מ|בעמ",
    re.IGNORECASE,
)


def _name_key(primary, fallback):
    """Normalized name used to match the same owner / attorney across rows."""
    name = primary.fillna(fallback)
    return name.str.lower().str.replace(r"[^\w֐-׿]+", " ", regex=True).str.strip()


def _script(text):
    if pd.isna(text):
        return "none"
    has_he = any("֐" <= ch <= "׿" for ch in text)
    has_latin = any(ch.isascii() and ch.isalpha() for ch in text)
    if has_he and has_latin:
        return "both"
    if has_he:
        return "hebrew"
    if has_latin:
        return "latin"
    return "other"


def _parse_codes(series, top_level=False):
    """'9; 35; 42' -> [9, 35, 42]. For Vienna codes '26.4' -> 26 when top_level."""
    out = []
    for value in series.fillna(""):
        codes = set()
        for part in value.split(";"):
            part = part.strip()
            if not part:
                continue
            head = part.split(".")[0] if top_level else part
            if head.isdigit():
                codes.add(int(head))
        out.append(sorted(codes))
    return out


def prior_count(df_all, key_col, target_index):
    """Number of applications with the same key filed strictly earlier.

    Counted over all applications in the file (any year, any status), since
    earlier filings are public at filing time.
    """
    keys = df_all[key_col]
    valid = keys.notna() & (keys != "")
    ranks = (df_all.loc[valid]
             .groupby(key_col)["filing_date"]
             .rank(method="min") - 1)
    counts = pd.Series(0.0, index=df_all.index)
    counts.loc[ranks.index] = ranks
    return counts.loc[target_index].to_numpy()


def history_failure_rate(data, key_col, is_train):
    """Smoothed past failure rate of the owner / attorney.

    Train rows: failures among the same key's training applications filed
    strictly earlier. Test rows: rate over all training applications of the key.
    Applications without a key get the global training rate.
    """
    global_rate = data.loc[is_train, "failed"].mean()
    rate = pd.Series(global_rate, index=data.index)

    train = data.loc[is_train & data[key_col].ne("") & data[key_col].notna(),
                     [key_col, "filing_date", "failed"]]
    # per key and day, then cumulative sums of earlier days only
    daily = train.groupby([key_col, "filing_date"])["failed"].agg(["sum", "count"]).reset_index()
    daily = daily.sort_values([key_col, "filing_date"])
    grp = daily.groupby(key_col)
    daily["prior_fail"] = grp["sum"].cumsum() - daily["sum"]
    daily["prior_n"] = grp["count"].cumsum() - daily["count"]
    merged = train.reset_index().merge(daily[[key_col, "filing_date", "prior_fail", "prior_n"]],
                                       on=[key_col, "filing_date"], how="left").set_index("index")
    rate.loc[merged.index] = ((merged["prior_fail"] + SMOOTHING * global_rate)
                              / (merged["prior_n"] + SMOOTHING))

    totals = train.groupby(key_col)["failed"].agg(["sum", "count"])
    test = data.loc[~is_train, key_col]
    test_sum = test.map(totals["sum"])
    test_n = test.map(totals["count"])
    known = test_n.notna()
    rate.loc[test[known].index] = ((test_sum[known] + SMOOTHING * global_rate)
                                   / (test_n[known] + SMOOTHING))
    return rate.to_numpy()


def build_features(df_all, data, train_last_year=TRAIN_LAST_YEAR, train_mask=None):
    """Return the feature table for `data` (the filtered dataset).

    df_all: the full cleaned file (used for prior-filing counts).
    data:   the decided 2000-2022 applications with the `failed` target.
    train_last_year: last filing year treated as training data. The failure-rate
        features use labels up to this year only. Notebook 03 also builds a
        version with 2016 so that the validation years (2017-2018) are treated
        exactly like the test years.
    train_mask: optional boolean array that overrides the year rule (used in
        notebook 05 to build a random train / test split for comparison).
    """
    df_all = df_all.copy()
    df_all["owner_key"] = _name_key(df_all["owner_en"], df_all["owner_he"])
    df_all["attorney_key"] = _name_key(df_all["attorney_en"], df_all["attorney_he"])

    data = data.copy()
    data = data.merge(df_all[["row_id", "owner_key", "attorney_key"]], on="row_id", how="left")

    f = pd.DataFrame(index=data.index)
    f["row_id"] = data["row_id"]
    f["filing_date"] = data["filing_date"]
    f["filing_year"] = data["filing_year"]
    f["filing_month"] = data["filing_date"].dt.month
    f["is_train"] = (data["filing_year"] <= train_last_year) if train_mask is None else np.asarray(train_mask)

    # categorical flags
    f["route"] = data["route"].map({"לאומי": "national", "בינלאומי": "international",
                                    "בסיס לבינלאומי": "base_for_international"})
    f["registration_type"] = data["registration_type"].map(
        {"סימן מסחר/שירות": "trade_service", "קיבוצי": "collective", "מאשר": "certification",
         "מאשר/קיבוצי - (זמני)": "cert_collective_temp"}).fillna("other")
    f["is_figurative"] = (data["is_figurative"] == "כן").astype(int)
    f["priority_claim"] = (data["priority_claim"] == "כן").astype(int)
    f["external_registration"] = (data["external_registration"].fillna("לא") != "לא").astype(int)

    # Nice classes
    nice = _parse_codes(data["nice_classes"])
    f["n_classes"] = [len(c) for c in nice]
    for k in range(1, 46):
        f[f"nice_{k}"] = [int(k in c) for c in nice]

    # Vienna codes (figurative elements), top-level categories
    vienna = _parse_codes(data["vienna_codes"], top_level=True)
    f["n_vienna"] = data["vienna_codes"].fillna("").str.count(";") + data["vienna_codes"].notna().astype(int)
    f["has_vienna"] = data["vienna_codes"].notna().astype(int)
    for k in range(1, 30):
        f[f"vienna_{k}"] = [int(k in c) for c in vienna]

    # mark text
    mark = (data["mark_text_en"].fillna("") + " " + data["mark_text_he"].fillna("")).str.strip().replace("", pd.NA)
    f["mark_script"] = mark.apply(_script)
    f["mark_len"] = mark.str.len().fillna(0)
    f["mark_words"] = mark.str.split().str.len().fillna(0)
    f["mark_has_digit"] = mark.str.contains(r"\d", regex=True).fillna(False).astype(int)

    # goods
    f["goods_en_len"] = data["goods_en"].fillna("").str.len()
    f["goods_he_len"] = data["goods_he"].fillna("").str.len()
    f["has_goods_he"] = data["goods_he"].notna().astype(int)
    f["goods_text"] = data["goods_en"].fillna("").str.replace("~", ",", regex=False)

    # owner and attorney
    owner_name = data["owner_en"].fillna(data["owner_he"]).fillna("")
    f["owner_is_company"] = owner_name.str.contains(COMPANY_PATTERN).astype(int)
    f["owner_has_he_name"] = data["owner_he"].notna().astype(int)
    f["has_attorney"] = (data["attorney_key"].fillna("") != "").astype(int)

    f["owner_prior_filings"] = np.log1p(prior_count(df_all.set_index("row_id", drop=False), "owner_key", data["row_id"]))
    f["attorney_prior_filings"] = np.log1p(prior_count(df_all.set_index("row_id", drop=False), "attorney_key", data["row_id"]))
    f["owner_fail_rate"] = history_failure_rate(data, "owner_key", f["is_train"])
    f["attorney_fail_rate"] = history_failure_rate(data, "attorney_key", f["is_train"])

    f["attorney_name"] = data["attorney_key"].fillna("").replace("", "none")
    f["failed"] = data["failed"].to_numpy()
    return f


CATEGORICAL = ["route", "registration_type", "mark_script"]
TEXT = "goods_text"
# filing_year is kept for splitting but not used as a feature: the test years are
# outside the training range, so a model can't learn anything valid from them
ID_COLUMNS = ["row_id", "filing_date", "filing_year", "is_train", "failed", "attorney_name"]


# Columns built but not used by the models.
# external_registration: its meaning isn't documented, and applications still waiting for
# examination almost never have it (0.2%) while decided ones do (2-3%), so it may be filled
# later in the process. It is excluded to be safe.
EXCLUDED = ["external_registration"]


def feature_groups(columns):
    """Split feature columns into numeric / categorical / text for the pipeline."""
    numeric = [c for c in columns if c not in CATEGORICAL + [TEXT] + ID_COLUMNS + EXCLUDED]
    return numeric, CATEGORICAL, TEXT
