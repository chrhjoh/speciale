import pandas as pd
import os

def main():
    DIR = "/Users/christianjohansen/Desktop/speciale/modeling/results"
    FILE_TO_CLEAN = os.path.join(DIR, "attlstm_98_cv_scores.csv")
    df = pd.read_csv(FILE_TO_CLEAN, names=["ID", "peptide", "origin", "partition", "score", "label"])
    grouped_df = df.groupby("ID").agg({"peptide" : pd.Series.mode,
                                       "origin" : pd.Series.mode,
                                       "partition" : pd.Series.mode,
                                       "score" : "mean",
                                       "label" : pd.Series.mode})
    grouped_df.to_csv(FILE_TO_CLEAN)



if __name__ == "__main__":
    main()