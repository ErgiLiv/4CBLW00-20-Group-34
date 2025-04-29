# Example placeholder (eda.py)
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import os

def load_all_crime_data_from_folders(base_folder):
    all_dfs = []

    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)

        # Only process directories
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def clean_and_filter(df):
    df = df[df['Crime type'] == 'Burglary']
    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
    df = df.dropna(subset=['Longitude', 'Latitude'])
    df = df[['Month', 'Longitude', 'Latitude', 'Location', 'LSOA code', 'LSOA name']]
    return df


def plot_monthly_trends(df):
    monthly_counts = df.groupby(df['Month']).size()
    plt.figure(figsize=(12, 6))
    monthly_counts.plot(marker='o')
    plt.title('Monthly Residential Burglary Trends')
    plt.xlabel('Month')
    plt.ylabel('Number of Burglaries')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_all_crime_data_from_folders('data')
    df = clean_and_filter(df)
    print(f"Total residential burglaries: {len(df)}")
    plot_monthly_trends(df)




