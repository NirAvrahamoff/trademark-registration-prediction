### CODE BY CLAUDE ###
"""Load the raw trademark CSV from data.gov.il, handling its known issues."""
import csv
import gzip
import io
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "trademarks.csv.gz"
N_COLUMNS = 31


def load_raw(path=RAW_PATH):
    """Return the raw data as a DataFrame of strings.

    The source file contains a few invalid UTF-8 bytes and some malformed
    rows (wrong number of fields). Invalid bytes are replaced, and rows
    that do not have exactly 31 fields are dropped.
    """
    with gzip.open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace").lstrip("﻿")
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    good = [r for r in body if len(r) == N_COLUMNS]
    print(f"Loaded {len(good):,} rows, dropped {len(body) - len(good)} malformed rows")
    return pd.DataFrame(good, columns=header)


if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
####
