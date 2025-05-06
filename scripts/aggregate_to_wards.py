"""
Aggregate LSOA-level forecasts to ward level
-------------------------------------------
Input 1: model_outputs/burglary_forecast.parquet
          columns: lsoa, month, pred_count
Input 2: external/lsoa_to_ward.csv
          columns: LSOA11CD, WD22CD, WD22NM
Output : data_ready/ward_forecast.parquet
"""

import pandas as pd
from pathlib import Path

FORECAST_IN = Path("model_outputs/burglary_forecast.parquet")
LOOKUP_IN   = Path("external/lsoa_to_ward.csv")
OUT_FILE    = Path("data_ready/ward_forecast.parquet")

# 1. load forecast
fcast = pd.read_parquet(FORECAST_IN)

# 2. load lookup (keep only needed cols, rename for clarity)
lookup = (pd.read_csv(LOOKUP_IN, dtype=str)
            .rename(columns={
                 "LSOA11CD": "lsoa",
                 "WD22CD":  "ward_code",
                 "WD22NM":  "ward_name"
             })[["lsoa","ward_code","ward_name"]])

# 3. merge & aggregate
merged = fcast.merge(lookup, on="lsoa", how="left")
ward_month = (merged
              .groupby(["ward_code","ward_name","month"], as_index=False)
              .agg(pred_burglaries=("pred_count","sum")))

ward_month.to_parquet(OUT_FILE, index=False)
print("✅ saved ward forecasts →", OUT_FILE)
