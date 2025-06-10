"""
XGBoost model for predicting ward-level and LSOA-level burglaries in London

================================================================
Run this file directly
================================================================

This script:
1. Loads and preprocesses burglary data at ward and LSOA level
2. Merges IMD scores (aggregated from LSOA to ward level) 
3. Adds ward and LSOA population data
4. Creates time-based and socioeconomic features
5. Trains XGBoost model and predicts next 12 months

"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from tqdm.auto import tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

#paths setup
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_cache"
PROCESSED = DATA / "processed"
LOOKUPS = DATA / "lookups"
PREDICTIONS = ROOT / "predictions"

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from the Month column."""
    df = df.copy()
    
    #basic time components
    df['year'] = df['Month'].dt.year
    df['month'] = df['Month'].dt.month
    df['quarter'] = df['Month'].dt.quarter
    
    #seasonal indicators
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    #lagged features (essential for multi-month forecasting)
    for lag in [1, 2, 3, 6, 12]:
        df[f'burglaries_lag_{lag}'] = df.groupby('WD24CD')['burglaries'].shift(lag)
    
    #rolling statistics
    for window in [3, 6, 12]:
        #mean
        df[f'burglaries_rollmean_{window}'] = df.groupby('WD24CD')['burglaries'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        #standard deviation for uncertainty
        df[f'burglaries_rollstd_{window}'] = df.groupby('WD24CD')['burglaries'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    #trend indicators
    df['trend_3m'] = df['burglaries_rollmean_3'] - df['burglaries_lag_3']
    df['trend_6m'] = df['burglaries_rollmean_6'] - df['burglaries_lag_6']
    df['trend_12m'] = df['burglaries_rollmean_12'] - df['burglaries_lag_12']
    
    #year-over-year change
    df['yoy_change'] = df['burglaries'] - df['burglaries_lag_12']
    
    return df

def load_ward_population(ward_pop_file: Path) -> pd.DataFrame:
    """Load and aggregate ward population data."""
    pop_df = pd.read_csv(ward_pop_file)
    #sum population across age/sex groups for each ward
    ward_pop = pop_df.groupby('WD22CD').agg({
        'population_2022': 'sum'  #using 2022 population
    }).reset_index()
    ward_pop = ward_pop.rename(columns={
        'WD22CD': 'WD24CD',  #match with current ward codes
        'population_2022': 'population'
    })
    return ward_pop

#new function: load LSOA population data (assumes file lsoa_pop2022.csv exists)
def load_lsoa_population(lsoa_pop_file: Path) -> pd.DataFrame:
    pop_df = pd.read_csv(lsoa_pop_file)
    pop_df = pop_df.rename(columns={'LSOA 2021 Code': 'LSOA21CD', 'Total': 'population'})
    #remove commas from the 'population' column and convert to float
    pop_df['population'] = pop_df['population'].replace({',': ''}, regex=True).astype(float)
    return pop_df

def calculate_ward_imd(imd_file: Path, lookup_file: Path) -> pd.DataFrame:
    """
    Calculate ward-level IMD scores by population-weighted averaging of LSOA scores.
    """
    imd_df = pd.read_csv(imd_file)
    lookup_df = pd.read_csv(lookup_file)
    
    #merge IMD with lookup
    merged = pd.merge(
        imd_df,
        lookup_df[['LSOA21CD', 'WD24CD', 'WD24NM']],
        left_on='LSOA code (2011)',
        right_on='LSOA21CD',
        how='inner'
    )
    
    #calculate ward-level metrics (mean of LSOA values)
    ward_imd = merged.groupby('WD24CD').agg({
        'Index of Multiple Deprivation (IMD) Score': 'mean',
        'Income Score (rate)': 'mean',
        'Employment Score (rate)': 'mean',
        'Crime Score': 'mean',
        'Health Deprivation and Disability Score': 'mean',
        'Barriers to Housing and Services Score': 'mean',
        'Living Environment Score': 'mean'
    }).reset_index()
    
    #rename columns for clarity
    ward_imd.columns = ['WD24CD', 'imd_score', 'income_score', 'employment_score', 
                       'crime_score', 'health_score', 'housing_score', 'environment_score']
    
    return ward_imd

#new function: get LSOA-level IMD data (no aggregation)
def get_lsoa_imd(imd_file: Path, lookup_file: Path) -> pd.DataFrame:
    imd_df = pd.read_csv(imd_file)
    lookup_df = pd.read_csv(lookup_file)[['LSOA21CD', 'WD24CD', 'WD24NM']]
    merged = pd.merge(
        imd_df,
        lookup_df,
        left_on='LSOA code (2011)',
        right_on='LSOA21CD',
        how='inner'
    )
    #rename columns for clarity
    merged = merged.rename(columns={
        'Index of Multiple Deprivation (IMD) Score': 'imd_score',
        'Income Score (rate)': 'income_score',
        'Employment Score (rate)': 'employment_score',
        'Crime Score': 'crime_score',
        'Health Deprivation and Disability Score': 'health_score',
        'Barriers to Housing and Services Score': 'housing_score',
        'Living Environment Score': 'environment_score'
    })
    return merged

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Prepare features for modeling."""
    df = df.copy()
    df['Month'] = pd.to_datetime(df['Month'])

    #ensure records are chronologically sorted per ward
    df = df.sort_values(['WD24CD', 'Month'])
    
    #load and merge socioeconomic data
    ward_pop = load_ward_population(LOOKUPS / "ward_pop2022.csv")
    ward_imd = calculate_ward_imd(LOOKUPS / "imd2019_lsoa.csv", LOOKUPS / "LSOA21_WD24_Lookup.csv")
    
    #merge population and IMD data
    df = pd.merge(df, ward_pop, on='WD24CD', how='left')
    df = pd.merge(df, ward_imd, on='WD24CD', how='left')
    
    #calculate rate-based features
    df['burglary_rate'] = (df['burglaries'] * 1000) / df['population']
    
    #create time features
    df = create_time_features(df)
    
    #list of features for modeling
    feature_cols = [
        #time components
        'year', 'month', 'quarter', 'is_summer', 'is_winter',
        
        #lagged values (essential for multi-month forecasting)
        'burglaries_lag_1', 'burglaries_lag_2', 'burglaries_lag_3',
        'burglaries_lag_6', 'burglaries_lag_12',
        
        #rolling statistics
        'burglaries_rollmean_3', 'burglaries_rollmean_6', 'burglaries_rollmean_12',
        'burglaries_rollstd_3', 'burglaries_rollstd_6', 'burglaries_rollstd_12',
        
        #trend indicators
        'trend_3m', 'trend_6m', 'trend_12m', 'yoy_change',
        
        #socioeconomic indicators
        'population', 'burglary_rate',
        'imd_score', 'income_score', 'employment_score',
        'crime_score', 'health_score', 'housing_score',
        'environment_score'
    ]
    
    #drop rows with NaN values (first year will have NaNs due to lags)
    df = df.dropna(subset=feature_cols)
    
    return df, feature_cols

#new function: Prepare features for LSOA-level modeling (expects input file with LSOA21CD)
def prepare_features_lsoa(df: pd.DataFrame) -> tuple:
    df = df.copy()
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values(['LSOA21CD', 'Month'])
    
    #load and merge LSOA-specific socioeconomic data
    lsoa_pop = load_lsoa_population(LOOKUPS / "lsoa_pop2022.csv")
    lsoa_imd = get_lsoa_imd(LOOKUPS / "imd2019_lsoa.csv", LOOKUPS / "LSOA21_WD24_Lookup.csv")
    #merge on LSOA21CD
    df = pd.merge(df, lsoa_pop, on='LSOA21CD', how='left')
    df = pd.merge(df, lsoa_imd, on='LSOA21CD', how='left')
    
    df['burglary_rate'] = (df['burglaries'] * 1000) / df['population']
    df = create_time_features(df)
    
    feature_cols = [
        #time components, lags, rolling stats, trends, and socioeconomic features
        'year', 'month', 'quarter', 'is_summer', 'is_winter',
        'burglaries_lag_1', 'burglaries_lag_2', 'burglaries_lag_3',
        'burglaries_lag_6', 'burglaries_lag_12',
        'burglaries_rollmean_3', 'burglaries_rollmean_6', 'burglaries_rollmean_12',
        'burglaries_rollstd_3', 'burglaries_rollstd_6', 'burglaries_rollstd_12',
        'trend_3m', 'trend_6m', 'trend_12m', 'yoy_change',
        'population', 'burglary_rate',
        'imd_score', 'income_score', 'employment_score',
        'crime_score', 'health_score', 'housing_score',
        'environment_score'
    ]
    df = df.dropna(subset=feature_cols)
    return df, feature_cols

def train_model(df: pd.DataFrame, feature_cols: list) -> Tuple[xgb.XGBRegressor, StandardScaler]:
    """Train XGBoost model with carefully tuned parameters."""
    
    #split data keeping most recent data for testing
    train_date = df['Month'].max() - pd.DateOffset(months=12)
    train_df = df[df['Month'] <= train_date]
    test_df = df[df['Month'] > train_date]
    
    #prepare features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_train = train_df['burglaries']
    y_test = test_df['burglaries']
    
    #initialize model with tuned parameters
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric=['rmse', 'mae'],
        early_stopping_rounds=20
    )
    
    #train model with evaluation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # --- Statistics ---
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # --- Additional Statistics ---
    y_pred_train = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    mae_test = np.mean(np.abs(y_test - y_pred))
    mae_train = np.mean(np.abs(y_train - y_pred_train))
    mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if (y_test != 0).all() else np.nan
    
    print("\nModel Performance:")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.3f}")
    print(f"Test MAE: {mae_test:.2f}")
    print(f"Test MAPE: {mape_test:.2f}%")

    print("\nTraining Set Metrics:")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Train MAE: {mae_train:.2f}")
    
    print("\nOverfit Analysis:")
    print(f"RMSE Difference (Test - Train): {rmse - train_rmse:.2f}")
    print(f"R² Difference (Train - Test): {train_r2 - r2:.3f}")

    
    # --- Residual Analysis and Prediction Error Distribution ---
    residuals = y_test - y_pred
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.axhline(0, color='red', linestyle='--')
    
    plt.subplot(1,2,2)
    plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel('Residuals')
    plt.title('Prediction Error Distribution')
    plt.tight_layout()
    
    #save the plot to the predictions folder
    PREDICTIONS.mkdir(exist_ok=True)
    plot_path = PREDICTIONS / "residuals_analysis.png"
    plt.savefig(plot_path, dpi=300)
    

    # --- Feature Importance ---
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
    
    # Ward-level Performance and Error Analysis (if available)
    if 'WD24NM' in test_df.columns:
        ward_metrics = test_df.groupby('WD24NM').agg({
            'burglaries': ['mean', 'std', 'count']
        }).round(2)
        print("\nWard-level Performance Summary:")
        print(ward_metrics.head().to_string())
        
        test_df = test_df.copy()
        test_df['predicted'] = y_pred
        test_df['abs_error'] = abs(test_df['burglaries'] - test_df['predicted'])
        test_df['pct_error'] = (test_df['abs_error'] / test_df['burglaries']) * 100
        ward_errors = test_df.groupby('WD24NM').agg({
            'abs_error': 'mean',
            'pct_error': 'mean'
        }).round(2)
        print("\nWorst Performing Wards (by absolute error):")
        print(ward_errors.nlargest(5, 'abs_error').to_string())
    
    return model, scaler

#new function: Recursive forecasting per LSOA
def predict_next_n_months_lsoa(df: pd.DataFrame, model: xgb.XGBRegressor, 
                               scaler: StandardScaler, feature_cols: list, 
                               n_months: int = 12) -> pd.DataFrame:
    latest_date = df['Month'].max()
    predictions_list = []
    
    current_df = df.copy()
    unique_lsoas = df['LSOA21CD'].unique()
    #for LSOA naming, we simply use the LSOA code (or augment with lookup if needed)
    print(f"\nGenerating LSOA-level predictions for {n_months} months across {len(unique_lsoas)} LSOAs...")
    
    for i in tqdm(range(1, n_months + 1), desc="Predicting months"):
        next_month = latest_date + pd.DateOffset(months=i)
        pred_rows = []
        for lsoa in tqdm(unique_lsoas, desc=f"Processing LSOAs for {next_month.strftime('%B %Y')}", leave=False):
            lsoa_data = current_df[current_df['LSOA21CD'] == lsoa].copy()
            #update time features
            lsoa_data['Month'] = next_month
            lsoa_data = create_time_features(lsoa_data)
            pred_row = lsoa_data.iloc[-1:]
            pred_row['LSOA'] = lsoa  #add LSOA identifier column
            pred_rows.append(pred_row)
        pred_df = pd.concat(pred_rows, ignore_index=True)
        X_pred = scaler.transform(pred_df[feature_cols])
        predictions = model.predict(X_pred)
        
        #enforce predictions within 2 std deviations of historical mean per LSOA
        last_year_same_month = latest_date - pd.DateOffset(months=12-i)
        historical_stats = df[df['Month'] == last_year_same_month].groupby('LSOA21CD')['burglaries'].agg(['mean', 'std'])
        for idx, lsoa in enumerate(unique_lsoas):
            if lsoa in historical_stats.index:
                mean = historical_stats.loc[lsoa, 'mean']
                std = max(1, historical_stats.loc[lsoa, 'std'])
                predictions[idx] = np.clip(predictions[idx], mean - 2*std, mean + 2*std)
                
        pred_std = pred_df[['burglaries_rollstd_3', 'burglaries_rollstd_6', 'burglaries_rollstd_12']].mean(axis=1)
        lower_ci = np.maximum(predictions - 1.96 * pred_std, 0)
        upper_ci = np.minimum(predictions + 1.96 * pred_std, predictions * 2)
        
        results = pd.DataFrame({
            'LSOA': pred_df['LSOA'],
            'Month': next_month,
            'Predicted_Burglaries': np.nan_to_num(np.round(predictions, 0)).astype(int),
            'Lower_CI': np.nan_to_num(np.round(lower_ci, 0)).astype(int),
            'Upper_CI': np.nan_to_num(np.round(upper_ci, 0)).astype(int),
            'Previous_Month_Actual': np.nan_to_num(np.round(pred_df['burglaries'].values, 0)).astype(int)
        })
        predictions_list.append(results)
        
        for idx, lsoa in enumerate(unique_lsoas):
            mask = current_df['LSOA21CD'] == lsoa
            new_row = current_df[mask].iloc[-1:].copy()
            new_row['Month'] = next_month
            new_row['burglaries'] = predictions[idx]
            current_df = pd.concat([current_df, new_row], ignore_index=True)
    
    final_predictions = pd.concat(predictions_list, ignore_index=True)
    return final_predictions.sort_values(['Month', 'LSOA'])

def predict_next_n_months(df: pd.DataFrame, model: xgb.XGBRegressor, 
                         scaler: StandardScaler, feature_cols: list, 
                         n_months: int = 12) -> pd.DataFrame:
    """Predict burglaries for next n months using recursive forecasting."""
    
    #start with the most recent data
    latest_date = df['Month'].max()
    predictions_list = []
    
    #create a copy of the latest data to update iteratively
    current_df = df.copy()
    
    #get unique wards once
    unique_wards = df['WD24CD'].unique()
    ward_names = df[['WD24CD', 'WD24NM']].drop_duplicates().set_index('WD24CD')['WD24NM']
    
    print(f"\nGenerating predictions for {n_months} months across {len(unique_wards)} wards...")
    
    #progress bar for months
    for i in tqdm(range(1, n_months + 1), desc="Predicting months"):
        #calculate next month
        next_month = latest_date + pd.DateOffset(months=i)
        
        #create prediction data for each ward
        pred_rows = []
        #progress bar for wards (nested, leave=False to keep it clean)
        for ward_code in tqdm(unique_wards, desc=f"Processing wards for {next_month.strftime('%B %Y')}", leave=False):
            #get ward's data
            ward_data = current_df[current_df['WD24CD'] == ward_code].copy()
            ward_name = ward_names[ward_code]
            
            #update time features for next month
            ward_data['Month'] = next_month
            ward_data = create_time_features(ward_data)
            
            #select most recent row
            pred_row = ward_data.iloc[-1:]
            
            #include ward identifier columns
            pred_row['Ward'] = ward_name
            pred_rows.append(pred_row)
        
        #combine all wards
        pred_df = pd.concat(pred_rows, ignore_index=True)
        
        #make predictions
        X_pred = scaler.transform(pred_df[feature_cols])
        predictions = model.predict(X_pred)
        
        #ensure predictions stay within reasonable bounds
        last_year_same_month = latest_date - pd.DateOffset(months=12-i)
        historical_stats = df[df['Month'] == last_year_same_month].groupby('WD24CD')['burglaries'].agg(['mean', 'std'])
        
        #for each ward, ensure prediction is within 2 standard deviations of historical mean
        for idx, ward_code in enumerate(unique_wards):
            if ward_code in historical_stats.index:
                mean = historical_stats.loc[ward_code, 'mean']
                std = max(1, historical_stats.loc[ward_code, 'std'])  #minimum std of 1
                predictions[idx] = np.clip(predictions[idx], mean - 2*std, mean + 2*std)
        
        #calculate prediction intervals using historical variability
        pred_std = pred_df[['burglaries_rollstd_3', 'burglaries_rollstd_6', 'burglaries_rollstd_12']].mean(axis=1)
        lower_ci = np.maximum(predictions - 1.96 * pred_std, 0)  #ensure non-negative
        upper_ci = np.minimum(predictions + 1.96 * pred_std, predictions * 2)  #cap at double the prediction
        
        #create results dataframe
        results = pd.DataFrame({
            'Ward': pred_df['Ward'],
            'Month': next_month,
            'Predicted_Burglaries': np.nan_to_num(np.round(predictions, 0)).astype(int),
            'Lower_CI': np.nan_to_num(np.round(lower_ci, 0)).astype(int),
            'Upper_CI': np.nan_to_num(np.round(upper_ci, 0)).astype(int),
            'Previous_Month_Actual': np.nan_to_num(np.round(pred_df['burglaries'].values, 0)).astype(int)
        })
        
        predictions_list.append(results)
        
        #update current_df with new predictions for next iteration
        for idx, ward_code in enumerate(unique_wards):
            mask = current_df['WD24CD'] == ward_code
            new_row = current_df[mask].iloc[-1:].copy()
            new_row['Month'] = next_month
            new_row['burglaries'] = predictions[idx]
            current_df = pd.concat([current_df, new_row], ignore_index=True)
    
    #combine all months
    final_predictions = pd.concat(predictions_list, ignore_index=True)
    return final_predictions.sort_values(['Month', 'Ward'])

def main():
    print("Loading ward-level data...")
    df = pd.read_parquet(PROCESSED / "ward_month_burglary.parquet")
    print("\nPreparing ward-level features...")
    df, feature_cols = prepare_features(df)
    print(f"Created {len(feature_cols)} features for wards")
    
    print("\nTraining ward-level model...")
    model, scaler = train_model(df, feature_cols)
    
    ward_predictions = predict_next_n_months(df, model, scaler, feature_cols, n_months=12)
    output_path = PREDICTIONS / "ward_burglary_predictions_12m.csv"
    ward_predictions.to_csv(output_path, index=False, date_format='%Y-%m-%d')
    print(f"Ward predictions saved to {output_path}")
    print("\nSample ward predictions:")
    first_month = ward_predictions['Month'].min()
    print(ward_predictions[ward_predictions['Month'] == first_month][['Ward', 'Month', 'Predicted_Burglaries', 'Lower_CI', 'Upper_CI']])
    
    # --- LSOA-level processing ---
    print("\nLoading LSOA-level data...")
    df_lsoa = pd.read_parquet(PROCESSED / "lsoa_month_burglary.parquet")
    print("\nPreparing LSOA-level features...")
    df_lsoa, feature_cols_lsoa = prepare_features_lsoa(df_lsoa)
    print(f"Created {len(feature_cols_lsoa)} features for LSOAs")
    
    print("\nTraining LSOA-level model...")
    model_lsoa, scaler_lsoa = train_model(df_lsoa, feature_cols_lsoa)
    
    lsoa_predictions = predict_next_n_months_lsoa(df_lsoa, model_lsoa, scaler_lsoa, feature_cols_lsoa, n_months=12)
    output_lsoa_path = PREDICTIONS / "lsoa_burglary_predictions_12m.csv"
    lsoa_predictions.to_csv(output_lsoa_path, index=False, date_format='%Y-%m-%d')
    print(f"LSOA predictions saved to {output_lsoa_path}")
    print("\nSample LSOA predictions:")
    first_month_lsoa = lsoa_predictions['Month'].min()
    print(lsoa_predictions[lsoa_predictions['Month'] == first_month_lsoa][['LSOA', 'Month', 'Predicted_Burglaries', 'Lower_CI', 'Upper_CI']])
    
if __name__ == "__main__":
    main()
