import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

def calc_auc(x):
    d = {}
    d["auc"] = metrics.roc_auc_score(x[3],x[6])
    d["counts"] = len(x)
    return pd.Series(d, index=["auc", "counts"])
    

def main():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, palette="pastel")
    result_dir = "results/baseline/"
    filenames = [result_dir + "tcr_baseline.res", result_dir + "cdrs_baseline.res", result_dir + "cdr3_baseline.res", result_dir + "cdr3b_baseline.res"]
    peptide_file = result_dir + "peptide_labels.txt" 
    inpt = ["TCRs", "CDRs", "CDR3s", "CDR3b"]
    results = pd.DataFrame()
    for i, filename in enumerate(filenames):
        result = pd.read_csv(filename, delimiter=" ", header=None)
        peptides = pd.read_csv(peptide_file, header=None)
        result["peptide"] = peptides[0]
        result = result[result["peptide"].isin(result["peptide"].value_counts(ascending=False).head(10).index)]
        auc_df = result.groupby("peptide").apply(calc_auc)
        auc_df["type"] = inpt[i]
        results = pd.concat([auc_df, results])

    results = results.pivot(columns="type")
    results = results.sort_values(("counts","TCRs"),ascending=False)
    ax = results.plot.bar(y="auc")
    ax.axhline(0.5, ls="--", c="black")
    plt.tight_layout()
    plt.savefig(result_dir+"auc_score.png")
    plt.show()

if __name__ == "__main__":
    main()