"""
clean_covid_data.py
-------------------
Run once:

    python scripts/clean_covid_data.py

• Reads every *-street.csv in data_covid/**/*
• Keeps only rows where `Falls within` == "Metropolitan Police Service"
• Re-saves the same filename (overwriting the bulky original)
• If a file ends up empty, it is deleted.
"""

import os
import pandas as pd
from pathlib import Path

BASE = Path("data_covid")          # where you unzipped 2020-2022 months
KEPT_ROWS, DROPPED_ROWS, DELETED_FILES = 0, 0, 0

for root, _, files in os.walk(BASE):
    for f in files:
        if f.endswith(".csv") and "street" in f.lower():
            fp = Path(root) / f
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"⚠️  Could not read {fp}: {e}")
                continue

            original_len = len(df)
            df = df[df["Falls within"] == "Metropolitan Police Service"]
            kept_len = len(df)

            if kept_len == 0:
                fp.unlink()               # delete empty file
                DELETED_FILES += 1
                print(f"🗑️  {fp} — removed (no MPS rows)")
            else:
                df.to_csv(fp, index=False)
                KEPT_ROWS   += kept_len
                DROPPED_ROWS += (original_len - kept_len)
                print(f"✅ {fp} — kept {kept_len:,} / {original_len:,}")

print("\nSummary")
print("Rows kept   :", f"{KEPT_ROWS:,}")
print("Rows dropped:", f"{DROPPED_ROWS:,}")
print("Files deleted (all rows non-MPS):", DELETED_FILES)
