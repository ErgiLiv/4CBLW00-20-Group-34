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
    # Load all crime data from December 2019
    maindf = load_all_crime_data_from_folders('data')

    # Filter for burglary crimes only
    maindf = maindf[maindf['Crime type'] == 'Burglary']

    # Count the number of burglaries per LSOA (Lower Layer Super Output Area)
    lsoadf = maindf.groupby(['LSOA name'])['Crime ID'].count()

    #Calculate z-scores for burglary counts
    meanCrime = lsoadf.mean()
    stdCrime = lsoadf.std()
    zScores = (lsoadf - meanCrime) / stdCrime

    #Plot each deprivation score against the number of burglaries
    names = ["crime", "income", "employment", "housing", "environment", "education"]#"crime", "income", "employment", "housing", "environment", "education"
    for i in range(len(names)):
        df = pd.read_csv(f"data/DeprivationStats2019/{names[i]}DeprivationScores.csv")
        df = df[["LSOA name", "Score"]]
        df = df.merge(zScores, on='LSOA name', how='inner')
        plt.subplot(3, 2, i+1)
        plt.scatter(df["Score"], df["Crime ID"])
        plt.title(names[i])
        #Calculate correlation
        correlation = round(sampleCorelation(df["Score"], df["Crime ID"]), 2)
        plt.xlabel("Correlation: " + str(correlation))
        if(i == 2):
            plt.ylabel("Number of Burglaries")
    plt.tight_layout()
    plt.show()




