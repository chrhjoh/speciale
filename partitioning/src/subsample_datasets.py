import pandas as pd
import os

def sample_swapped(df, ids):
    """
    Returns the swapped negatives, that were swapped from sampled positives
    """
    ids = list(ids)
    swap_idx = pd.to_numeric(df["origin"].str.split("_").str[1])
    return df[swap_idx.isin(ids)]

def main():
    FRAC = 0.05
    SEED = 42
    DATA_DIR = "/Users/christianjohansen/Desktop/speciale/modeling/data"
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, "datasets/train_data_all.csv")
    OUTFILE = os.path.join(DATA_DIR, f"datasets/train_data_subsample{FRAC}.csv")
    
    df = pd.read_csv(TRAIN_DATA_FILE)
    pos_samples = df[df["origin"] == "positive"].groupby("pep").sample(frac=FRAC, random_state=SEED)
    neg_samples = df[df["origin"] == "10x"].groupby("pep").sample(frac=FRAC, random_state=SEED)
    swapped_samples = sample_swapped(df[df["origin"].str.startswith("swapped")], pos_samples["ID"])
    sampled_df = pd.concat([pos_samples, neg_samples, swapped_samples])
    sampled_df.to_csv(OUTFILE)

    print("Sample fraction:", FRAC)
    print("Original DataFrame shape:", df.shape)
    print("New shape:", sampled_df.shape)
    print("Fraction actually sampled:", sampled_df.shape[0] / df.shape[0])

if __name__ == "__main__":
    main()