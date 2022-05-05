import pandas as pd
import os
"""
Creates a file with number of positive/swapped/negative in the dataset and saves it to file
"""

def main():
    sample = 5
    DIR = "/Users/christianjohansen/Desktop/speciale/processing/data/datasets"
    DATA_FILE = os.path.join(DIR, f"train_data_subsample{sample}.csv")
    META_FILE = os.path.join(DIR, "metadata_subsamples.csv")
    df = pd.read_csv(DATA_FILE)
    df.loc[df["origin"].str.startswith("swapped"), ["origin"]] = "swapped"
    meta_data = pd.DataFrame(df.value_counts(["pep", "origin"]))
    meta_data.rename(columns={0 : "counts"}, inplace=True)
    meta_data["sampling"] = sample
    meta_data.to_csv(META_FILE, mode="a", header=False)

if __name__ == "__main__":
    main()