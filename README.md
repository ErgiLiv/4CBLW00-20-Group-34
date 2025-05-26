# London Residential Burglary Analysis
## TU/e Data Challenge 2 - Group 34

This repository contains a comprehensive analysis of residential burglary data in London, using data from the Metropolitan Police Service from 2010 to 2025.

## Project Structure

```
├── data/                       # Raw data from police.uk (monthly folders)
├── data_cache/                 # Processed data and lookups
│   ├── lookups/               # Reference tables & shapefiles
│   └── processed/             # Generated data artifacts
├── notebooks/                  # Analysis notebooks
│   ├── 02_eda.ipynb          # Exploratory Data Analysis
│   ├── 03_baseline_models.ipynb   # Baseline forecasting models
│   ├── 04_ml_models.ipynb        # Advanced ML models
│   └── figures/               # Generated visualizations
├── scripts/                    # Processing scripts
│   ├── build_features.py      # Feature engineering
│   ├── ingest_burglary.py     # Data ingestion pipeline
│   └── streamlit_app.py       # Interactive dashboard
└── requirements.txt           # Project dependencies
```

## Key Features

1. **Data Processing Pipeline**: Automated ingestion and processing of Metropolitan Police burglary data
2. **Exploratory Analysis**: Comprehensive EDA including:
   - London-wide monthly burglary totals
   - Top-20 wards by burglary incidents
   - Interactive choropleth maps
3. **Predictive Models**:
   - Seasonal Naïve baseline
   - SARIMA models
   - Advanced ML models
4. **Interactive Dashboard**: Streamlit-based visualization with:
   - Month selector
   - Interactive choropleth maps
   - Ward/LSOA level analysis
   - Actual vs Forecast comparison

## Setup & Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Processing
Run the ingestion pipeline:
```bash
python scripts/ingest_burglary.py
```

### Analysis Notebooks
Navigate to the `notebooks/` directory and run Jupyter notebooks in sequence:
1. `02_eda.ipynb` - Exploratory analysis
2. `03_baseline_models.ipynb` - Baseline forecasting
3. `04_ml_models.ipynb` - Advanced modeling

### Interactive Dashboard
Launch the Streamlit dashboard:
```bash
streamlit run scripts/streamlit_app.py
```

## Data Sources
- Primary data: [Metropolitan Police Service crime data](https://data.police.uk/)
- Geographic data: London Ward and LSOA boundaries

## Dependencies
Key packages required:
- pandas
- geopandas
- matplotlib
- seaborn
- streamlit
- statsmodels
- numpy

See `requirements.txt` for complete list.

## Authors
Group 34 - TU/e Data Challenge 2

## License
This project is part of the TU/e Data Challenge 2 course and is intended for educational purposes.