"""
Baseline forecast: seasonal-naïve
--------------------------------
For every LSOA, the forecast for month *t* is simply the observed
value 12 months earlier.  We evaluate MAE on a 2-year test window.
"""

import pandas as pd
from pathlib import Path

DATA = Path("data_ready/burglary_counts.parquet")

# ------------------- 1.  Load target table ----------------------------
df = pd.read_parquet(DATA)

# pivot to wide: index = Period('YYYY-MM'), columns = lsoa
y = (
    df.pivot(index="month", columns="lsoa", values="burglary_count")
      .sort_index()                # ensure chronological order
)

# ------------------- 2.  Train / test split ---------------------------
# customise these dates if you want a different split
TRAIN_END  = "2017-12"      # everything through 2017 goes in the model
TEST_START = "2018-01"      # evaluate on 2018-19
TEST_END   = "2019-12"


y_train = y.loc[:TRAIN_END]
y_test  = y.loc[TEST_START:TEST_END]

# ------------------- 3.  Forecast: 12-month seasonal naïve -------------
y_pred = y.shift(12)               # forecast = value 12 months ago
y_pred_test = y_pred.loc[TEST_START:TEST_END]

from pathlib import Path
forecast_path = Path("model_outputs/baseline_forecast.parquet")

(  # pivot-wide back to long and save
    y_pred        # DataFrame, index = month, cols = lsoa
      .stack()    # → long Series with MultiIndex (month, lsoa)
      .reset_index(name="pred_count")   # columns: month, lsoa, pred_count
      .to_parquet(forecast_path, index=False)
)
print("✅  saved baseline forecast →", forecast_path)

# 4. Evaluate MAE  -----------------------------------------------------
diff = (y_test - y_pred_test).abs()

# long form -> mean per LSOA
mae_per_lsoa = (
    diff
      .stack(dropna=False)        # keep NaN for later drop
      .groupby(level="lsoa")
      .mean()
      .dropna()                   # drop LSOAs with no valid rows
)

overall_mae = mae_per_lsoa.mean()
print(f"Seasonal-naïve MAE (2020-21): {overall_mae:,.2f}")

mean_level = y_test.mean().mean()
print("Average monthly burglaries per LSOA:", mean_level)
print("MAE / mean level:", 1.16 / mean_level)



