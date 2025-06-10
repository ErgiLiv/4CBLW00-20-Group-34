# London Residential Burglary Analysis
## TU/e - Addressing real-world crime and security problems with data science (4CBLW00-20) - Group 34

This repository contains a comprehensive analysis of residential burglary patterns across London wards and LSOAs, using data from the Metropolitan Police Service. The project focuses on processing and analyzing crime data to understand spatial and temporal patterns of residential burglaries.

The aim of this project is to aid the Metropolitan Police combat burglaries by providing a prediction and visualization tool for burglaries and police force allocation in Greater London.

## Project Overview
Our analysis pipeline processes raw police data into ward-level and LSOA-level aggregations, enabling both broad-scale and granular analysis of burglary patterns across London. The project combines crime data with geographic boundaries to create detailed spatiotemporal visualizations and analysis.

## Data Pipeline
**Core Data Processing Pipeline** (`scripts/ingest_burglary.py`):
- Ingests monthly CSV files manually downloaded from [data.police.uk](https://data.police.uk/data/).
- Filters records for Metropolitan Police residential burglaries.
- Maps crimes to current ward (2024) and LSOA (2021) boundaries.
- Aggregates the data into monthly counts by area.
- Generates three outputs:
   - `lsoa_month_burglary.parquet`: Monthly burglary counts by LSOA.
   - `ward_month_burglary.parquet`: Monthly burglary counts by ward.
   - `ward_month_burglary.geojson`: Same data with ward geometries for mapping.

###

**XGBoost Machine Learning Analysis** (`scripts/xgboost_predictions_12m.py`):
- Data Loading: Imports and prepares the monthly ward/LSOA burglary datasets.
- Data Processing: Transforms data into a format suitable for model training and testing.
- Model Training: Configures and trains an XGBoost model using the processed data.
- Predictions and Evaluation: Generates predictions and evaluates model performance.
- Generates four outputs:
   - `residuals_analysis_Ward.png`: Residuals analysis plot for test Ward predictions.
   - `ward_burglary_predictions_12m.csv`: Ward burglary prediction results (12 months).
   - `residuals_analysis_LSOA.png`: Residuals analysis plot for test LSOA predictions.
   - `lsoa_burglary_predictions_12m.csv`: LSOA burglary prediction results (12 months).
###

**Interactive Visualization** (`scripts/streamlit_app_12m.py`):
- Launches a Streamlit web application to display interactive visualizations.
- Presents monthly burglary counts and model predictions in dynamic charts and maps.
- Presents police force resource allocation for each ward and LSOA based on the predictions.

## Features Used for XGBoost Burglary Predictions
The XGBoost model leverages a robust set of features to forecast monthly burglary counts. The main categories of features include:

- Time components:
   - year - The calendar year extracted from the Month timestamp.
   - month - The numerical month (1-12) extracted from the Month column.
   - quarter - The quarter of the year (1-4) corresponding to the Month.
   - is_summer - A binary flag (1/0) indicating if the month falls in summer (typically June, July, August).
   - is_winter - A binary flag (1/0) that is set when the month is in winter (December, January, February).

- Lagged values (historical burglary counts):
   - burglaries_lag_1 - The number of burglaries in the previous month (1‐month lag).
   - burglaries_lag_2 - The burglary count two months ago.
   - burglaries_lag_3 - The burglary count three months earlier.
   - burglaries_lag_6 - The burglary count from six months ago.
   - burglaries_lag_12 - The value from 12 months ago (year‐ago comparison).

- Rolling statistics (aggregated measures over a window):
   - burglaries_rollmean_3 - The 3‑month rolling mean of burglaries, which smooths short‐term fluctuations.
   - burglaries_rollmean_6 - The 6‑month rolling average of burglaries.
   - burglaries_rollmean_12 - The 12‑month (yearly) rolling average.
   - burglaries_rollstd_3 - The rolling standard deviation over 3 months, used to capture short‐term variability.
   - burglaries_rollstd_6 - The 6‑month rolling standard deviation of burglary counts.
   - burglaries_rollstd_12 - The 12‑month rolling standard deviation, summarizing annual variability.

- Trend indicators:
   - trend_3m - Measures the short-term trend by comparing the 3‑month rolling mean with the value 3 months ago.
   - trend_6m - The medium-term trend using the 6‑month rolling average against the corresponding lag.
   - trend_12m - The long-term trend by comparing the 12‑month rolling mean to the burglary count from 12 months ago.
   - yoy_change - The year-over-year change in burglaries (current value minus the value 12 months earlier).

- Socio-economic indicators:
   - population - The ward population taken from the ward population dataset.
   - burglary_rate - A derived rate (burglaries per 1,000 people), calculated from the burglary count and ward population.
   - imd_score - The Index of Multiple Deprivation score reflects socioeconomic deprivation at ward level.
   - income_score - A measure of income deprivation (often given as a rate).
   - employment_score - Reflects the level of employment deprivation.
   - crime_score - A score indicating crime deprivation (not to be confused with the burglaries count).
   - health_score - Indicates health-related deprivation in the area.
   - housing_score - Captures housing-related deprivation.
   - environment_score - A score describing the quality of the local living environment, such as air quality and green space.

These features combine to help the model understand temporal patterns and local contextual factors, ensuring more accurate burglary predictions across Greater London.
## Project Structure
```
├── data/                                # Contains raw police monthly data (December 2010 - February 2025)
│   └── YYYY-MM/                            # Folder for each month's data
│       └── YYYY-MM-metropolitan-street.csv    # Street-level crime data for Metropolitan Police for that month
├── data_cache/                
│   ├── lookups/                            # Required reference files
│   │   ├── imd2019_lsoa.csv                   # Index of Multiple Deprivation 2019 dataset per LSOA
│   │   ├── lsoa_pop2022.csv                   # LSOA 2022 population estimates
│   │   ├── LSOA21_Boundaries.geojson          # LSOA 2021 geographic boundaries
│   │   ├── LSOA21_WD24_Lookup.csv             # Mapping of LSOA 2021 to Ward 2024
│   │   ├── ward_pop2022.csv                   # Ward 2022 population estimates
│   │   └── wards_2024.geojson                 # Ward 2024 geographic boundaries
│   └── processed/                          # Pipeline outputs
│       ├── lsoa_month_burglary.parquet        # Monthly burglary counts by LSOA
│       ├── ward_month_burglary.geojson        # Same data with ward geometries for mapping
│       └── ward_month_burglary.parquet        # Monthly burglary counts by ward in tabular format
├── notebooks/                 
│   └── EDA.ipynb                           # Exploratory Data Analysis notebook
├── predictions/
│   ├── lsoa_burglary_predictions_12m.csv   # LSOA burglary prediction results (12 months)
│   ├── residuals_analysis_LSOA.png         # Residuals analysis plot for test LSOA predictions
│   ├── residuals_analysis_Ward.png         # Residuals analysis plot for test Ward predictions
│   └── ward_burglary_predictions_12m.csv   # Ward burglary prediction results (12 months)
├── scripts/                   
│   ├── ingest_burglary.py                  # Data ingestion script for processing burglary records
│   ├── streamlit_app_12m.py                 # Launches Streamlit web app for interactive visualizations
│   └── xgboost_predictions_12m.py          # Executes XGBoost model to predict monthly burglary trends
├── .gitattributes                       # Git attributes configuration
├── .gitignore                           # Files and directories to be ignored by Git
├── feature_columns.txt                  # List of feature columns used in modeling
├── README.md                            # Project overview and documentation
└── requirements.txt                     # Python dependencies for the project
```

## Setup & Usage
1. **Prerequisites**:
   - Python 3.11+
   - Clone the repository

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:
   ```bash
   python scripts/ingest_burglary.py   #Run time: ~3 minutes (depending on device performance)
   ```
   ```bash            
   python scripts/xgboost_predictions_12m.py    #Run time: ~3 hours (depending on device performance)
   ```
   ```bash            
   streamlit run scripts/streamlit_app_12m.py    #Loading time: ~30 seconds
   ```

## Data Requirements
- **Raw Data**: Monthly crime data from [data.police.uk](https://data.police.uk/data/).
  - Already in the repository: Raw data December 2010 - February 2025.
  - If you want to add future data, follow these steps:
      - Download street-level crime data for Metropolitan Police
      - Place in `data/YYYY-MM/` folders
      - Files should be named `YYYY-MM-metropolitan-street.csv`

- **Data lookups** (located in this repository in `data_cache/lookups`):
   - Index of Multiple Deprivation 2019 dataset per LSOA (`imd2019_lsoa.csv`)
   - LSOA 2022 population estimates (`lsoa_pop2022.csv`)
   - LSOA 2021 geographic boundaries (`LSOA21_Boundaries.geojson`)
   - Mapping of LSOA 2021 to Ward 2024 (`LSOA21_WD24_Lookup.csv`)
   - Ward 2022 population estimates (`ward_pop2022.csv`)
   - Ward 2024 geographic boundaries (`wards_2024.geojson`)

## Outputs
- The pipeline generates three files in `data_cache/processed/`:
   1. **ward_month_burglary.parquet**
      - Monthly burglary counts by ward
      - Fast tabular format for analysis

   2. **ward_month_burglary.geojson**
      - Same data with ward geometries
      - Suitable for mapping

   3. **lsoa_month_burglary.parquet**
      - Monthly burglary counts by LSOA
      - Enables granular analysis

- Four files in `predictions/`:

   1. **residuals_analysis_Ward.png**
      - Residuals analysis plot for test Ward predictions
   2. **ward_burglary_predictions_12m.csv**
      - Ward burglary prediction results (12 months)
   3. **residuals_analysis_LSOA.png**
      - Residuals analysis plot for test LSOA predictions
   4. **lsoa_burglary_predictions_12m.csv**
      - LSOA burglary prediction results (12 months)

- As well as a Streamlit web application to display interactive visualizations, showing:
   - Monthly burglary counts and model predictions per ward and LSOA in dynamic charts and maps
   - Police force resource allocation for each ward and LSOA based on the predictions

## Authors
Group 34 - TU/e - Addressing real-world crime and security problems with data science (4CBLW00-20) course

## License
This project is part of the TU/e Addressing real-world crime and security problems with data science (4CBLW00-20) course and is intended for educational purposes.