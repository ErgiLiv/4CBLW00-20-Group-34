import pandas as pd
import matplotlib.pyplot as plt
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
    warddf = pd.read_csv('LSOA21_WD24_Lookup.csv')
    warddf = warddf[['LSOA21NM', 'WD24NM','LAD24NM']]
    warddf.rename(columns={'LSOA21NM': 'LSOA name', 'WD24NM': 'Ward name', 'LAD24NM': 'Borough name'}, inplace=True)

    # Filter for burglary crimes only
    maindf = maindf[maindf['Crime type'] == 'Burglary']

    # Count the number of burglaries per LSOA (Lower Layer Super Output Area)
    lsoadf = maindf.groupby(['LSOA name'])['Crime ID'].count().to_frame()

    #merge with wards
    lsoadf = lsoadf.merge(warddf, on='LSOA name', how='left')
    lsoadf = lsoadf[['Ward name', 'LSOA name']]

    #merge with IMD scores
    names = ["crime", "income", "employment", "housing", "environment", "education", "IMD"]#"crime", "income", "employment", "housing", "environment", "education"
    for i in range(len(names)):
        df = pd.read_csv(f"data/DeprivationStats2019/{names[i]}DeprivationScores.csv")
        df = df[["LSOA name", "Score"]]
        df.rename(columns={"Score": names[i]}, inplace=True)
        lsoadf = lsoadf.merge(df, on='LSOA name', how='outer')
    lsoadf = lsoadf[['Ward name', 'crime', 'income', 'employment', 'housing', 'environment', 'education', 'IMD']]

    #Get mean depravation scores for each ward
    lsoadf = lsoadf.groupby(['Ward name'], as_index=False).mean().round(5)
    lsoadf = lsoadf.dropna()

    #Get deciles for each score
    for i in range(len(names)):
        lsoadf["decile"] = pd.qcut(lsoadf[names[i]], 10, labels=False)
        lsoadf.rename(columns={"decile": f"{names[i]} decile"}, inplace=True)

    #Print deciles and scores to csv file
    deciledf = lsoadf[['Ward name', 'crime decile', 'income decile', 'employment decile', 'housing decile', 'environment decile', 'education decile', 'IMD decile']]
    scroredf = lsoadf[['Ward name', 'crime', 'income', 'employment', 'housing', 'environment', 'education', 'IMD']]
    deciledf.to_csv('wardsDeprevationDeciles.csv', index=False)
    scroredf.to_csv('wardsDeprevationScores.csv', index=False)
    