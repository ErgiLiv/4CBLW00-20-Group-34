"""
XGBoost model for predicting ward-level burglaries in London
==========================================================

This script:
1. Loads and preprocesses burglary data at ward level
2. Merges IMD scores (aggregated from LSOA to ward level) 
3. Adds ward population data
4. Creates time-based and socioeconomic features
5. Trains XGBoost model using temporal split
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# Paths setup
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_cache"
PROCESSED = DATA / "processed"
LOOKUPS = DATA / "lookups"

def load_ward_population(ward_pop_file: Path) -> pd.DataFrame:
    """Load and aggregate ward population data."""
    pop_df = pd.read_csv(ward_pop_file)
    # Sum population across age/sex groups for each ward
    ward_pop = pop_df.groupby('WD22CD').agg({
        'population_2022': 'sum'  # Using 2022 population
    }).reset_index()
    ward_pop = ward_pop.rename(columns={
        'WD22CD': 'WD24CD',  # Match with current ward codes
        'population_2022': 'population'
    })
    return ward_pop

def calculate_ward_imd(imd_file: Path, lookup_file: Path) -> pd.DataFrame:
    """
    Calculate ward-level IMD scores by population-weighted averaging of LSOA scores.
    """
    imd_df = pd.read_csv(imd_file)
    lookup_df = pd.read_csv(lookup_file)
    
    # Merge IMD with lookup
    merged = pd.merge(
        imd_df,
        lookup_df[['LSOA21CD', 'WD24CD', 'WD24NM']],
        left_on='LSOA code (2011)',
        right_on='LSOA21CD',
        how='inner'
    )
    
    # Calculate ward-level metrics (mean of LSOA values)
    ward_imd = merged.groupby('WD24CD').agg({
        'Index of Multiple Deprivation (IMD) Score': 'mean',
        'Income Score (rate)': 'mean',
        'Employment Score (rate)': 'mean',
        'Crime Score': 'mean',
        'Health Deprivation and Disability Score': 'mean',
        'Barriers to Housing and Services Score': 'mean',
        'Living Environment Score': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    ward_imd.columns = ['WD24CD', 'imd_score', 'income_score', 'employment_score', 
                       'crime_score', 'health_score', 'housing_score', 'environment_score']
    
    return ward_imd

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from the Month column."""
    df = df.copy()
    
    # Basic time components
    df['year'] = df['Month'].dt.year
    df['month'] = df['Month'].dt.month
    df['quarter'] = df['Month'].dt.quarter
    df['day_of_year'] = df['Month'].dt.dayofyear
    
    # Seasonal indicators
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    # Lagged features
    for lag in [1, 2, 3, 6, 12]:  # Multiple lag periods
        df[f'burglaries_lag_{lag}'] = df.groupby('WD24CD')['burglaries'].shift(lag)
        
    # Rolling means
    for window in [3, 6, 12]:
        df[f'burglaries_roll_mean_3'] = (
            df.groupby('WD24CD')['burglaries']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    return df

def prepare_features(burglary_file: Path, 
                    ward_pop_file: Path,
                    imd_file: Path, 
                    lookup_file: Path) -> Tuple[pd.DataFrame, list]:
    """Prepare all features for modeling."""
    
    # Load base burglary data
    df = pd.read_parquet(burglary_file)
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Add population data
    ward_pop = load_ward_population(ward_pop_file)
    df = pd.merge(df, ward_pop, on='WD24CD', how='left')
    
    # Add IMD data
    ward_imd = calculate_ward_imd(imd_file, lookup_file)
    df = pd.merge(df, ward_imd, on='WD24CD', how='left')
    
    # Create time features
    df = create_time_features(df)
    
    # Calculate rate-based features
    df['burglary_rate'] = (df['burglaries'] * 1000) / df['population']
      # List of features for modeling
    feature_cols = [
        # Time features
        'year', 'month', 'quarter', 
        'is_summer', 'is_winter',
        
        # Lagged features (reduced to avoid leakage)
        'burglaries_lag_1', 'burglaries_lag_2', 'burglaries_lag_3',
        
        # Rolling means (reduced window)
        'burglaries_roll_mean_3',
        
        # Socioeconomic indicators
        'population',  # Removed burglary_rate as it might cause leakage
        'imd_score', 'income_score', 'employment_score',
        'crime_score', 'health_score', 'housing_score',
        'environment_score'
    ]
    
    return df.dropna(), feature_cols

def train_test_split(df: pd.DataFrame, test_months: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets based on time."""
    split_date = df['Month'].max() - pd.DateOffset(months=test_months)
    train = df[df['Month'] < split_date]
    test = df[df['Month'] >= split_date]
    return train, test

def train_model(df: pd.DataFrame, feature_cols: list) -> None:
    """Train and evaluate XGBoost model."""
    
    # Split data
    train_df, test_df = train_test_split(df, test_months=12)
    
    # Prepare features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    
    y_train = train_df['burglaries']
    y_test = test_df['burglaries']
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    )

    # Fit model and get training history
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )
    
    # Make predictions for both train and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate various metrics
    def calculate_metrics(y_true, y_pred, set_name=""):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"\n{set_name} Set Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.3f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return rmse, r2, mae, mape
    
    # Calculate metrics for both sets
    train_metrics = calculate_metrics(y_train, y_pred_train, "Training")
    test_metrics = calculate_metrics(y_test, y_pred_test, "Test")
    
    # Compare train vs test performance
    train_rmse, train_r2, train_mae, train_mape = train_metrics
    test_rmse, test_r2, test_mae, test_mape = test_metrics
    
    print("\nOverfit Analysis:")
    print(f"RMSE Difference (Test - Train): {test_rmse - train_rmse:.2f}")
    print(f"R² Difference (Train - Test): {train_r2 - test_r2:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    # Calculate ward-level metrics
    ward_metrics = test_df.groupby('WD24NM').agg({
        'burglaries': ['mean', 'std', 'count'],
        'population': 'first'
    }).round(2)
    
    print("\nWard-level Performance Summary:")
    print(ward_metrics.head())
      # Analyze prediction errors by ward
    test_df = test_df.copy()  # Create a copy to avoid warnings
    test_df.loc[:, 'predicted'] = y_pred_test
    test_df.loc[:, 'abs_error'] = np.abs(test_df['burglaries'] - test_df['predicted'])
    test_df.loc[:, 'pct_error'] = (test_df['abs_error'] / test_df['burglaries']) * 100
    
    ward_errors = test_df.groupby('WD24NM').agg({
        'abs_error': 'mean',
        'pct_error': 'mean'
    }).round(2)
    
    print("\nWorst Performing Wards (by absolute error):")
    print(ward_errors.nlargest(5, 'abs_error'))
    
    return model, scaler  # Return model and scaler for future predictions

def predict_next_month(model, df: pd.DataFrame, feature_cols: list, scaler) -> pd.DataFrame:
    """Generate predictions for next month for all wards."""
    
    # Get the most recent month's data
    latest_date = df['Month'].max()
    next_month = latest_date + pd.DateOffset(months=1)
    
    # Prepare features for prediction
    prediction_df = df[df['Month'] == latest_date].copy()
    prediction_df['Month'] = next_month
    prediction_df = create_time_features(prediction_df)
    
    # Scale features
    X_pred = scaler.transform(prediction_df[feature_cols])
    
    # Make predictions
    predictions = model.predict(X_pred)
    
    # Create output dataframe
    results = pd.DataFrame({
        'Ward': prediction_df['WD24NM'],
        'Predicted_Burglaries': predictions.round(1),
        'Previous_Month_Actual': prediction_df['burglaries'],
        'Population': prediction_df['population']
    })
    
    return results.sort_values('Predicted_Burglaries', ascending=False)

def predict_with_confidence(model, X_pred: np.ndarray, n_iterations: int = 100) -> tuple:
    """Generate predictions with confidence intervals using bootstrap."""
    predictions = []
    
    for _ in range(n_iterations):
        # Randomly sample with replacement
        indices = np.random.choice(len(X_pred), size=len(X_pred))
        X_bootstrap = X_pred[indices]
        
        # Get predictions
        pred = model.predict(X_bootstrap)
        predictions.append(pred)
    
    # Calculate mean and confidence intervals
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    lower_ci = np.percentile(predictions, 5, axis=0)
    upper_ci = np.percentile(predictions, 95, axis=0)
    
    return mean_pred, lower_ci, upper_ci

def main():
    print("Loading and preparing data...")
    df, feature_cols = prepare_features(
        burglary_file=PROCESSED / "ward_month_burglary.parquet",
        ward_pop_file=LOOKUPS / "ward_pop2022.csv",
        imd_file=LOOKUPS / "imd2019_lsoa.csv",
        lookup_file=LOOKUPS / "LSOA21_WD24_Lookup.csv"
    )
    
    print("\nTraining model...")
    model, scaler = train_model(df, feature_cols)
    
    print("\nPredicting next month's burglaries...")
    predictions = predict_next_month(model, df, feature_cols, scaler)
    print(predictions.head(10))

if __name__ == "__main__":
    main()