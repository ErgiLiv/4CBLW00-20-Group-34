import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree

def load_all_crime_data_from_folders(base_folder):
    all_dfs = []
    total_loaded = 0

    for folder in sorted(os.listdir(base_folder)):
        folder_path = os.path.join(base_folder, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.csv') and "stop" in file.lower():
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        #df = df[df['Falls within'] == 'Metropolitan Police Service']

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

def assign_nearest_lsoa(df1, df2):
    # Ensure we are working with float types for lat/lon
    df1[['Latitude', 'Longitude']] = df1[['Latitude', 'Longitude']].astype(float)
    df2[['Latitude', 'Longitude']] = df2[['Latitude', 'Longitude']].astype(float)

    # Build KDTree for fast nearest-neighbor lookup
    tree = cKDTree(df2[['Latitude', 'Longitude']].values)

    # Query nearest neighbor for each point in df1
    distances, indices = tree.query(df1[['Latitude', 'Longitude']].values, k=1)

    # Assign LSOA name from df2 to df1 based on nearest neighbor index
    df1['LSOA name'] = df2.iloc[indices]['LSOA name'].values

    return df1

def sampleCorelation(X, Y):
    Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
    Sum_x_squared = sum((X-X.mean())**2)
    Sum_y_squared = sum((Y-Y.mean())**2)       
    corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr

if __name__ == "__main__":
    #Load data to assign LSOAs
    df = pd.read_parquet("Repo/4CBLW00-20-Group-34/data_cache/burglary_cleaned.parquet")
    helperdf = df.drop_duplicates(subset=['LSOA code'])
    helperdf = helperdf[["LSOA name", "LSOA code", "Latitude", "Longitude"]]

    #Assign LSOA to Stop and Seach leading to arrest
    dfStop = load_all_crime_data_from_folders("allCrimeData")
    dfStop = dfStop[dfStop["Outcome"] == "Arrest"]
    dfStop = dfStop[dfStop['Latitude'].notna() & dfStop['Longitude'].notna()]
    dfStop = assign_nearest_lsoa(dfStop, helperdf)
    dfStop.to_parquet("StopAndSearchLSOA.parquet", index=False)

    #Group data by LSOA name and count occurrences
    dfStop = dfStop.groupby(['LSOA name'])["Date"].count().to_frame()
    df = df.groupby(['LSOA name'])["Month"].count().to_frame()
    
    #Join the two dataframes
    df = df.join(dfStop, how='outer')
    df = df.fillna(0)

    #Calculate correlation
    correlation = sampleCorelation(df['Month'], df['Date'])
    print(f"Correlation between Burglary and Stop and Search leading to arrest: {correlation}")

    #Plot correlation
    plt.scatter(df['Month'], df['Date'])
    plt.xlabel("Burglary")
    plt.ylabel("Stop and Search leading to arrest")
    plt.title("Correlation between Burglary and Stop and Search leading to arrest")
    plt.show()