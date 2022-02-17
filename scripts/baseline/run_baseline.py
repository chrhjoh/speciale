import pandas as pd
import subprocess
import sys


def main():

    filename = sys.argv[1]
    data = pd.read_csv(filename, delimiter="\t",names=["peptide", "alpha", "beta", "label"], index_col=0)
    for i in range(1,6):

        test = data[data.index.str.startswith(f"P{i}")]
        db = data[~data.index.str.startswith(f"P{i}")]  #db is other partitions
        peptides = test["peptide"].unique() # get all peptides that should be tested
        db = db[db["label"] == 1]   # subset database to only contain positives

        for peptide in peptides:
            test_pep = test[test["peptide"] == peptide]
            db_pep = db[db["peptide"] == peptide]
            
            if len(db_pep) > 0:
                test_pep.to_csv(f"tmp/P{i}_{peptide}_test", sep=" ", columns=["alpha", "beta", "label"], index=False,header=False)
                db_pep.to_csv(f"tmp/P{i}_{peptide}_db", sep=" ", columns=["alpha", "beta", "label"], index=False,header=False)
                subprocess.run(["../../morni/bin/pairlistscore_db_kernel", 
                               f"tmp/P{i}_{peptide}_test",
                               f"tmp/P{i}_{peptide}_db"])

                test_pep.to_csv("results/baseline/peptide_labels2.txt", columns=["peptide"], index=False, header=None, mode="a")

        

if __name__ == "__main__":
    main()



