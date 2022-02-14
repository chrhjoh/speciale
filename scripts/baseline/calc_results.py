import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_roc(y_true, y_pred):
        # ROC
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)

        # plot ROC
        plt.figure()
        plt.title("Receiver Operating Characteristic")
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")


def main():
    result_dir = "results/baseline/"
    filenames = [result_dir + "tcr_baseline.res", result_dir + "cdrs_baseline.res", result_dir + "cdr3_baseline.res", result_dir + "cdr3b_baseline.res"] 
    for filename in filenames:
        result = pd.read_csv(filename, delimiter=" ", header=None)
        plot_roc(result[3],result[6])
        plt.title(filename)
        plt.show()

if __name__ == "__main__":
    main()