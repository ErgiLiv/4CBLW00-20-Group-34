import os, pandas as pd
from pathlib import Path

DATA_CLEAN   = Path("data_cache/burglary_cleaned.parquet")   # already created by eda.py
DATA_READY   = Path("data_ready")            # new folder to save modelling tables
IMD_RAW_FILE = Path("external/imd_2019_london.csv")  # download once (see step 3)
WEATHER_RAW  = Path("external/weather_daily_london.csv")     # optional

DATA_READY.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------
# 1.  LOAD the cleaned parquet produced by eda.py
# ---------------------------------------------------------------------
df = pd.read_parquet(DATA_CLEAN)

# ── choose modelling granularity ─────────────────────────────────────
df["month"] = df["Month"].dt.to_period("M")     # monthly
df["lsoa"]  = df["LSOA code"]

# ---------------------------------------------------------------------
# 2.  CREATE the TARGET table  (burglary counts per month-LSOA)
# ---------------------------------------------------------------------
target = (
    df.groupby(["lsoa", "month"])
      .size()
      .rename("burglary_count")
      .reset_index()
)

target.to_parquet(DATA_READY / "burglary_counts.parquet", index=False)
print("✅ saved burglary_counts.parquet   →", len(target), "rows")

# ---------------------------------------------------------------------
# 3.  LOAD IMD data  (download from gov.uk once, CSV ≈ 500 KB)
#     Source: https://assets.publishing.service.gov.uk/.../IMD_2019.csv
# ---------------------------------------------------------------------
if IMD_RAW_FILE.exists():
    imd = pd.read_csv(IMD_RAW_FILE, usecols=["LSOA code (2011)", "IMD Decile"])
    imd.columns = ["lsoa", "imd_decile"]
    imd.to_parquet(DATA_READY / "features_imd.parquet", index=False)
    print("✅ saved features_imd.parquet     →", len(imd), "rows")
else:
    print("⚠️  IMD file not found – skip for now")

# ---------------------------------------------------------------------
# 4.  OPTIONAL: add WEATHER features (daily → monthly avg)
#     You can grab Met Office or Meteostat data; here’s the aggregation stub.
# ---------------------------------------------------------------------
if WEATHER_RAW.exists():
    wx = pd.read_csv(WEATHER_RAW, parse_dates=["date"])
    wx["month"] = wx["date"].dt.to_period("M")
    wx_monthly  = (wx.groupby("month")
                     .agg(temp_mean=("tavg", "mean"),
                          rain_sum =("prcp", "sum"))
                     .reset_index())
    wx_monthly.to_parquet(DATA_READY / "features_weather.parquet", index=False)
    print("✅ saved features_weather.parquet →", len(wx_monthly), "rows")
else:
    print("ℹ️  No weather file provided – skipping")

print("\n🎉 Dataset build finished")
