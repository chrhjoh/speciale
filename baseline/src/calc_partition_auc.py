import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os

def calc_auc(x):
    d = {}
    d["auc"] = metrics.roc_auc_score(x["label"],x["score"])
    d["pos_count"] = (x.label == 1).sum()
    d["neg_count"] = (x.label == 0).sum()
    return pd.Series(d, index=["auc", "pos_count", "neg_count"])
    
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
    PLOT_FILE = os.path.join(RES_DIR, "partition_baseline_auc_per_peptide.png")
    res_files = [os.path.join(OUT_DIR, "baseline_90neg_90pos.csv"),
                 os.path.join(OUT_DIR, "baseline_95neg_95pos.csv"),
                 os.path.join(OUT_DIR, "baseline_98neg_98pos.csv"),
                  os.path.join(OUT_DIR, "baseline_all.csv")]
    labels = ["0.9",
              "0.95",
              "0.98",
              "1.0"]

    outputs = []
    for i, file in enumerate(res_files):
        df = do_file(file)
        df["label"] = labels[i]
        outputs.append(df)
    df = pd.concat(outputs)
    df = df.set_index([df.index, "label"])


    fig, ax = plt.subplots(figsize=(8,7))
    df = df.sort_values(["pos_count", "neg_count"],ascending=False)
    df.unstack().sort_values([("pos_count", "1.0")], ascending=False).auc.plot.bar(ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, rotation=90)
    ax.set_ylabel("AUC")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
if __name__ == "__main__":
    main()