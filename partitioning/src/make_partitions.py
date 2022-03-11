import pandas as pd
import numpy as np
import os

DIR = "/Users/christianjohansen/Desktop/speciale"
NEG_FILE = "data/hobohm/out_hob_neg_98_unique.dat"
POS_FILE = "data/hobohm/out_hob_pos_98_unique.dat"
DATA_FILE = "data/datasets/data_cleaned.csv"
TRAIN_FILE = "data/datasets/train_data_98neg_98pos.csv"

def map_swapped(x, df):
    _, idx = x["origin"].split("_")
    idx = int(idx)
    if (df["ID"].isin([idx])).any():
        idx = df.index[df["ID"] == idx]
        return df.loc[idx].squeeze().at["partition"]
    else:
        return np.NaN

data = pd.read_csv(os.path.join(DIR, DATA_FILE))
neg_subset = pd.read_csv(os.path.join(DIR, NEG_FILE), delim_whitespace=True, names=["hobohm_idx","cdr3a","cdr3b"], index_col=0)
pos_subset = pd.read_csv(os.path.join(DIR, POS_FILE), delim_whitespace=True, names=["hobohm_idx","cdr3a","cdr3b"], index_col=0)

# Sort data so it is sorted as the hobohm output (by sum of cdr3 len)
pos_data = data[data["origin"] == "positive"].reset_index()
neg_data = data[data["origin"] == "10x"].reset_index()
pos_data.index.name = "hobohm_idx"
neg_data.index.name = "hobohm_idx"
swapped_data = data[data["origin"].str.startswith("swapped")]

# Remove non unique tcrs from the positives identified by hobohm
pos_data = pd.merge(pos_data, pos_subset, left_index=True, right_index=True, how="inner")
pos_data = pos_data.drop(columns=['cdr3a_y','cdr3b_y', 'index'])
pos_data = pos_data.rename(columns={'cdr3a_x' : "cdr3a",'cdr3b_x' : "cdr3b"})

# Remove non unique tcrs from the negatives identified by hobohm
neg_data = pd.merge(neg_data, neg_subset, left_index=True, right_index=True, how="inner")
neg_data = neg_data.drop(columns=['cdr3a_x','cdr3b_x', 'index'])
neg_data = neg_data.rename(columns={'cdr3a_y' : "cdr3a",'cdr3b_y' : "cdr3b"})

# Randomly repartition the data
pos_data['partition'] = np.random.randint(low=1, high=6, size=len(pos_data))
neg_data['partition'] = np.random.randint(low=1, high=6, size=len(neg_data))

# Map swapped negatives back to corresponding positive tcrs
swapped_data["partition"] = swapped_data.apply(lambda x : map_swapped(x, pos_data), axis=1)
# NAs are refering to positives that were removed during hobohm, so we remove them
swapped_data = swapped_data.dropna()

train_data = pd.concat([pos_data, neg_data, swapped_data])

train_data.to_csv(os.path.join(DIR, TRAIN_FILE))