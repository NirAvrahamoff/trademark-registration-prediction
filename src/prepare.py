### CODE BY CLAUDE ###
"""Clean the raw trademark data and build the classification target.

Target definition
-----------------
An application "failed" if it was decided and never got a registration date.
The status column alone is not a valid label: most applications with status
"closed" were registered and later lapsed (they have a registration date).

Only decided applications are kept (status registered, closed or non-payment).
Applications still in examination have no outcome yet and are removed.
"""
import pandas as pd

from load_data import load_raw

# Hebrew column names in the source file -> short English names
COLUMNS = {
    "מספור רץ": "row_id",
    "מספר הסימן": "mark_number",
    "דמות": "is_figurative",
    "קישור לנסח": "link",
    "מצב": "status",
    "לאומי/בינלאומי": "route",
    "מספר הסימן הבינלאומי": "intl_number",
    "תיאור סימן עברית": "mark_text_he",
    "תיאור סימן אנגלית": "mark_text_en",
    "תאריך בקשה": "filing_date",
    "תאריך רישום": "registration_date",
    "סוג רישום": "registration_type",
    "סוגי ניצה לבקשה": "nice_classes",
    "סוגי וינה לבקשה": "vienna_codes",
    "רשימת סחורות בעברית": "goods_he",
    "רשימת סחורות באנגלית": "goods_en",
    "דין קדימה": "priority_claim",
    "תאריך תום תוקף": "expiry_date",
    "שם מיופה כח": "attorney_he",
    "שם מיופה כח אנגלית": "attorney_en",
    "שם בעל הסימן בעברית": "owner_he",
    "שם בעל הסימן באנגלית": "owner_en",
    "שם בעל הרשות בעברית": "licensee_he",
    "שם בעל הרשות באנגלית": "licensee_en",
    "שם מייצג לעניין אכיפת מכס": "customs_rep",
    "מען למכס תאריך התחלה": "customs_start_date",
    "כתובת אימייל מייצג מכס": "customs_email",
    "טלפון מייצג מכס": "customs_phone",
    "רישום חיצוני": "external_registration",
    "תאריך עדכון אחרון": "last_update",
}

# English translation of the status values
STATUS_EN = {
    "רשום": "registered",
    "סגור": "closed",
    "אי תשלום": "non_payment",
    "בחינה ראשונה": "first_exam",
    "בחינה שנייה": "second_exam",
    "ממתין לבחינה": "awaiting_exam",
    "קובל": "opposed",
    "במיון": "sorting",
    "ממתין לקבלת החלטה בתיק אחר": "awaiting_other_case",
}
DECIDED = {"registered", "closed", "non_payment"}

# Columns that are filled only after the decision, or are identifiers / constants.
# They must never be used as features.
LEAKAGE_COLUMNS = ["status", "registration_date", "expiry_date", "last_update"]
DROP_COLUMNS = ["row_id", "mark_number", "link"]

DATE_COLUMNS = ["filing_date", "registration_date", "expiry_date", "last_update", "customs_start_date"]

FIRST_YEAR, LAST_YEAR = 2000, 2022


def clean(df_raw):
    """Rename columns to English, strip whitespace, parse dates and translate status."""
    df = df_raw.rename(columns=COLUMNS).copy()
    for col in df.columns:
        df[col] = df[col].str.strip().replace("", pd.NA)
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")
    df["status"] = df["status"].map(STATUS_EN)
    df["filing_year"] = df["filing_date"].dt.year
    df["is_decided"] = df["status"].isin(DECIDED)
    df["failed"] = df["registration_date"].isna().astype(int)
    return df


def build_dataset(df_clean, first_year=FIRST_YEAR, last_year=LAST_YEAR):
    """Keep decided applications filed in [first_year, last_year]."""
    mask = df_clean["is_decided"] & df_clean["filing_year"].between(first_year, last_year)
    return df_clean.loc[mask].reset_index(drop=True)


def load_dataset():
    """Raw file -> cleaned, filtered dataset with the `failed` target."""
    return build_dataset(clean(load_raw()))
####
