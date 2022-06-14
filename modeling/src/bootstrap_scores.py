import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

def read_file(dir: str, file: str, use_header: bool = False):
    if use_header:
        return pd.read_csv(os.path.join(dir, file))
    else:
        return pd.read_csv(os.path.join(dir, file), names=["ID", "peptide", "origin", "partition", "score", "label"])

def read_baseline(dir: str, file: str):
    df=pd.read_csv(os.path.join(dir, file), names=["cdr3a", "cdr3b", "label", "cdr3a_q", "cdr3b_q", "score", "peptide"])
    return df

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


def read_files(dir, prefix, suffix):
    dfs = []
    for suf in suffix:
        dfs.append(pd.read_csv(os.path.join(dir, prefix+suf), names=["ID", "peptide", "origin", "partition", "score", "label"]))
    df = pd.concat(dfs)
    df = df.groupby("ID").agg({"peptide" : pd.Series.mode,
                          "origin" : pd.Series.mode,
                          "partition" : pd.Series.mode,
                          "score" : "mean",
                          "label" : pd.Series.mode})
    return df

    
def main():

    DIR = "/Users/christianjohansen/Desktop/speciale"
    RES_DIR = os.path.join(DIR, "modeling/results/subsampling")
    BASELINE_DIR = os.path.join(DIR, "baseline/out")
    FILE1 = "attlstmpan_GIL359"
    FILE2 = "pretrained_attlstmsingle_GIL359"
    SUFFIXES = [".csv", "_1.csv","_2.csv","_3.csv","_4.csv", ]
    N = 10000

    df1 = read_files(RES_DIR, FILE1, SUFFIXES)
    df2 = read_files(RES_DIR, FILE2, SUFFIXES)
    df = pd.merge(df1, df2, how="inner", on="ID")
    df = df[df.peptide_x == "GILGFVFTL"]
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