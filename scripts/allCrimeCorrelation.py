import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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

def sampleCorelation(X, Y):
    Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
    Sum_x_squared = sum((X-X.mean())**2)
    Sum_y_squared = sum((Y-Y.mean())**2)       
    corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr

if __name__ == "__main__":
    #Load all crime data since 2022
    crimedf = load_all_crime_data_from_folders("allCrimeData")

    #Seperate burglary data from other crimes
    burglarydf = crimedf[crimedf['Crime type'] == 'Burglary']
    crimedf = crimedf[crimedf['Crime type'] != 'Burglary']

    crimeTypes = crimedf['Crime type'].unique().tolist()
    crimeTypes.remove('Anti-social behaviour')
    print(crimeTypes)

    burglarydf = burglarydf.groupby(['LSOA name'])["Crime ID"].count().to_frame()

    #Calculate correlation between burglary and other crime types
    for crimeType in crimeTypes:
        tempdf = crimedf[crimedf['Crime type'] == crimeType]
        tempdf = tempdf.groupby(['LSOA name'])["Crime ID"].count().to_frame()
        tempdf.columns = [crimeType]
        tempdf = tempdf.join(burglarydf, how='outer')
        tempdf = tempdf.fillna(0)
        correlation = sampleCorelation(tempdf[crimeType], tempdf['Crime ID'])
        print(f"Correlation between {crimeType} and Burglary: {correlation}")

    