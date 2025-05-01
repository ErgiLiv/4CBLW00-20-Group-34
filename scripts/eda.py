import os
import pandas as pd
import matplotlib.pyplot as plt

# --- STEP 1: LOAD CLEANED DATA (with caching) ---
def load_all_crime_data_from_folders(base_folder):
    all_dfs = []
    total_loaded = 0

    for folder in sorted(os.listdir(base_folder)):
        folder_path = os.path.join(base_folder, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.csv') and "street" in file.lower():
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        df = df[df['Falls within'] == 'Metropolitan Police Service']

                        if not df.empty:
                            all_dfs.append(df)
                            total_loaded += 1
                            print(f"[{total_loaded:03}] Loaded: {file_path} ({len(df)} rows)")
                        else:
                            print(f"[---] Skipped {file_path} (no MPS data)")

                    except Exception as e:
                        print(f"⚠️ Skipped {file_path} due to error: {e}")

    print(f"\n✅ Finished loading {total_loaded} files with Metropolitan Police data.")
    return pd.concat(all_dfs, ignore_index=True)

# --- STEP 2: CLEAN AND FILTER FOR BURGLARY ---
def clean_and_filter(df):
    df = df[df['Crime type'] == 'Burglary'].copy()
    df.loc[:, 'Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
    df = df.dropna(subset=['Longitude', 'Latitude'])
    df = df[['Month', 'Longitude', 'Latitude', 'Location', 'LSOA code', 'LSOA name']]
    return df

# --- STEP 3: CACHE UTILS ---
def save_cache(df, path='data_cache/burglary_cleaned.parquet'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Cached cleaned data to: {path}")

def load_or_prepare_data(base_folder='data', cache_path='data_cache/burglary_cleaned.parquet'):
    if os.path.exists(cache_path):
        print(f"✅ Loaded cached data from: {cache_path}")
        return pd.read_parquet(cache_path)
    else:
        df = load_all_crime_data_from_folders(base_folder)
        df = clean_and_filter(df)
        save_cache(df, cache_path)
        return df

# --- STEP 4: VISUALIZATION: AVERAGE SEASONALITY WITH STD ---
def plot_seasonal_average(df):
    df['MonthNum'] = df['Month'].dt.month
    grouped = df.groupby(['MonthNum', df['Month'].dt.year]).size().unstack()

    monthly_avg = grouped.mean(axis=1)
    monthly_std = grouped.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', label='Average (2010–2025)', linewidth=2)
    plt.fill_between(monthly_avg.index,
                     monthly_avg - monthly_std,
                     monthly_avg + monthly_std,
                     alpha=0.2, label='±1 Std. Dev.')

    plt.title('Average Monthly Burglary Pattern (2010–2025)')
    plt.xlabel('Month')
    plt.ylabel('Average Burglaries')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df = load_or_prepare_data()
    print(f"\nTotal residential burglaries: {len(df)}")
    plot_seasonal_average(df)