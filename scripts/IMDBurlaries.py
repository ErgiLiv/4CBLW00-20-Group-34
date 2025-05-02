import matplotlib.pyplot as plt
import pandas as pd
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

if __name__ == "__main__":
    # Load all crime data from December 2019
    maindf = load_all_crime_data_from_folders('data')

    # Filter for burglary crimes only
    maindf = maindf[maindf['Crime type'] == 'Burglary']

    # Count the number of burglaries per LSOA (Lower Layer Super Output Area)
    lsoadf = maindf.groupby(['LSOA name'])['Crime ID'].count()

    # Calculate z-scores for burglary counts
    meanCrime = lsoadf.mean()
    stdCrime = lsoadf.std()
    zScores = (lsoadf - meanCrime) / stdCrime
    zScores = zScores.reset_index()
    zScores.rename(columns={'Crime ID': 'Burglary z-score'}, inplace=True)

    # Load the Index of Multiple Deprivation (IMD) scores
    IMDdata = pd.read_csv('societal-wellbeing_imd2019_indices.csv')
    IMDdeciles = pd.read_csv('societal-wellbeing_imd2019_indices_decile.csv')

    # Keep only the relevant columns and rename them for consistency
    IMDdata = IMDdata[['Reference area', '2019']]
    IMDdata.rename(columns={'Reference area': 'LSOA name', '2019': 'IMD score'}, inplace=True)
    IMDdeciles = IMDdeciles[['Reference area', '2019']]
    IMDdeciles.rename(columns={'Reference area': 'LSOA name', '2019': 'IMD decile'}, inplace=True)

    # Merge IMD data with burglary z-scores based on LSOA name
    merged = IMDdata.merge(lsoadf, on='LSOA name', how='inner')
    mergedeciles = IMDdeciles.merge(zScores, on='LSOA name', how='inner')

    # Create a scatter plot of IMD score vs. burglary z-score
    plt.scatter(merged['IMD score'], merged['Crime ID'])
    plt.xlabel('IMD score')
    plt.ylabel('Number of burglaries')
    plt.title('Burglary vs IMD score')

    #Create boxplot of IMD deciles vs. number of burglaries
    mergedeciles.boxplot(column='Burglary z-score', by='IMD decile', grid=True)
    plt.xlabel('IMD Decile (1 = Most Deprived, 10 = Least Deprived)')
    plt.ylabel('Burglary z-score')
    plt.title('Distribution of Burglaries by IMD Decile')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



