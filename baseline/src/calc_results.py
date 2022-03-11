import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os

def calc_auc(x):
    d = {}
    d["auc"] = metrics.roc_auc_score(x["label"],x["score"])
    d["counts"] = len(x)
    return pd.Series(d, index=["auc", "counts"])
    
def do_file(filename):
    result = pd.read_csv(filename, names=["cdr3a", "cdr3b", "label", "cdr3a_db", "cdr3b_db", "score", "peptide"])
    # subset to 10 most frequent peptides
    result = result[result["peptide"].isin(result["peptide"].value_counts(ascending=False).head(10).index)]
    auc_df = result.groupby("peptide").apply(calc_auc)
    auc_df.loc["total"] = calc_auc(result)
    return auc_df


def main():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, palette="pastel")
    DIR = "/Users/christianjohansen/Desktop/speciale/baseline"
    RES_DIR = os.path.join(DIR, "results")
    OUT_DIR = os.path.join(DIR, "out")
    PLOT_FILE = os.path.join(RES_DIR, "partition_baseline_auc.png")
    res_files = [os.path.join(OUT_DIR, "baseline_85neg_90pos.csv"),
                 os.path.join(OUT_DIR, "baseline_90neg_90pos.csv"),
                 os.path.join(OUT_DIR, "baseline_90neg_95pos.csv"),
                 os.path.join(OUT_DIR, "baseline_95neg_95pos.csv"),
                 os.path.join(OUT_DIR, "baseline_98neg_98pos.csv")]
    labels = ["0.85, 0.90",
              "0.9",
              "0.9, 0.95",
              "0.95",
              "0.98"]

    outputs = []
    for i, file in enumerate(res_files):
        df = do_file(file)
        df["label"] = labels[i]
        outputs.append(df)
    df = pd.concat(outputs)

    fig, ax = plt.subplots(figsize=(8,7))
    df.loc["total"].plot.bar(x="label",y="auc", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
if __name__ == "__main__":
    main()