import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

def read_file(dir: str, file: str, use_header: bool = False):
    if use_header:
        return pd.read_csv(os.path.join(dir, file))
    else:
        return pd.read_csv(os.path.join(dir, file), names=["ID", "peptide", "origin", "partition", "score", "label"])

def sample_scores(df: pd.DataFrame):
    return df.sample(frac=1, replace=True)

def calc_auc(df: pd.DataFrame):
    return roc_auc_score(df["label"], df["score"])

def bootstrap_once(df):
    aucs = []
    df = sample_scores(df)
    aucs.append(roc_auc_score(df["label_x"], df["score_x"]))
    aucs.append(roc_auc_score(df["label_y"], df["score_y"]))
    return aucs, np.argmax(aucs)

def report_results(results: np.ndarray, wincounter):
    print(f"Number of times df1 won: {wincounter} out of {results.shape[0]} bootstraps")
    print(f"Average AUC across all bootstraps: {results.mean(axis=0)}")
    print(f"Significance from bootstrapping: {1 - wincounter / results.shape[0]}")

    
def main():

    DIR = "/Users/christianjohansen/Desktop/speciale/modeling/results"
    FILE1 = "cnn_sequences_cdr3bpep_scores.csv"
    FILE2 = "cnn_sequences_cdr3apep_scores.csv"
    N = 10000

    df1 = read_file(DIR, FILE1, use_header=False)
    df2 = read_file(DIR, FILE2, use_header=False)
    df = pd.merge(df1, df2, how="inner", on="ID")
    print(df1.shape, df2.shape, df.shape)
    results = np.zeros((N, 2))
    df1_wincount = 0
    for i in range(N):
        results[i, :], winner = bootstrap_once(df)
        if winner == 0:
            df1_wincount += 1
        if i % 500 == 0:
            print(f"Done {i} out of {N} bootstraps")
    report_results(results, df1_wincount)
    

if __name__ == "__main__":
    main()