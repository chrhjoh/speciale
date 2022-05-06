import pandas as pd
from sklearn import metrics
import os


def do_file(filename):
    """
    Calculates AUC per peptide for the 10 most present peptides
    """
    result = pd.read_csv(filename, names=["ID", "peptide", "origin", "partition", "score", "label"])
    # subset to 10 most frequent peptides
    result = result[result["peptide"].isin(result["peptide"].value_counts(ascending=False).head(5).index)]

    auc_df = result.groupby(["peptide"]).apply(lambda x: pd.Series({
                                        "auc" : metrics.roc_auc_score(x["label"], x["score"]),
                                        "auc_0.1" : metrics.roc_auc_score(x["label"], x["score"], max_fpr=0.1),

                                        "auc_swapped" : metrics.roc_auc_score(x.loc[x.origin != "10x","label"], 
                                                                              x.loc[x.origin != "10x", "score"]),
                                        "auc_swapped_0.1" :  metrics.roc_auc_score(x.loc[x.origin != "10x", "label"], 
                                                                                   x.loc[x.origin != "10x", "score"], max_fpr = 0.1),

                                        "auc_10x" : metrics.roc_auc_score(x.loc[~x.origin.str.startswith("swapped"),"label"], 
                                                                          x.loc[~x.origin.str.startswith("swapped"), "score"]),
                                        "auc_10x_0.1" : metrics.roc_auc_score(x.loc[~x.origin.str.startswith("swapped"),"label"], 
                                                                              x.loc[~x.origin.str.startswith("swapped"), "score"], max_fpr = 0.1),
                                        "count" : x.shape[0]}))

    auc_df.loc["total"] = pd.Series({"auc" : metrics.roc_auc_score(result["label"], result["score"]),
                                     "auc_0.1" : metrics.roc_auc_score(result["label"], result["score"], max_fpr=0.1),

                                     "auc_swapped" : metrics.roc_auc_score(result.loc[result.origin != "10x","label"], 
                                                                           result.loc[result.origin != "10x", "score"]),
                                     "auc_swapped_0.1" :  metrics.roc_auc_score(result.loc[result.origin != "10x","label"], 
                                                                                result.loc[result.origin != "10x", "score"], max_fpr = 0.1),

                                     "auc_10x" : metrics.roc_auc_score(result.loc[~result.origin.str.startswith("swapped"),"label"], 
                                                                       result.loc[~result.origin.str.startswith("swapped"), "score"]),
                                     "auc_10x_0.1" : metrics.roc_auc_score(result.loc[~result.origin.str.startswith("swapped"),"label"], 
                                                                           result.loc[~result.origin.str.startswith("swapped"), "score"], max_fpr = 0.1),
                                     "count" : result.shape[0]})
    return auc_df


def main():
    DIR = "/Users/christianjohansen/Desktop/speciale/modeling"
    RES_DIR = os.path.join(DIR, "results/subsampling")
    PREFIX = "attlstmsingle_GIL"
    RES_FILES = [os.path.join(RES_DIR, PREFIX+"15.csv"),
                 os.path.join(RES_DIR, PREFIX+"35.csv"),
                 os.path.join(RES_DIR, PREFIX+"65.csv"),
                 os.path.join(RES_DIR, PREFIX+"83.csv"),
                  os.path.join(RES_DIR, PREFIX+"144.csv"),
                  os.path.join(RES_DIR, PREFIX+"212.csv"),
                  os.path.join(RES_DIR, PREFIX+"359.csv"),
                  os.path.join(RES_DIR, PREFIX+"550.csv"),
                  os.path.join(RES_DIR, PREFIX+"724.csv")]
    OUT_FILE = os.path.join(RES_DIR, PREFIX+"_auc.csv")
    labels = ["15",
              "35",
              "65",
              "83",
              "144",
              "212",
              "359",
              "550",
              "724"]

    outputs = []
    for i, file in enumerate(RES_FILES):
        df = do_file(file)
        df["n_positives"] = labels[i]
        outputs.append(df)
    df = pd.concat(outputs)
    df = df.set_index([df.index, "n_positives"])

    df.to_csv(OUT_FILE)
if __name__ == "__main__":
    main()