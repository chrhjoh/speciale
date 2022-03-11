import pandas as pd
import subprocess
import sys
import os

def main():
    DIR = "/home/projects/vaccine/people/chrhol/speciale/baseline"
    KERNEL_PATH = "/home/projects/vaccine/people/morni/bin/pairlistscore_db_kernel"
    TMP_DIR = os.path.join(DIR, "data/tmp")
    
    filename = sys.argv[1]
    data = pd.read_csv(filename, index_col=0)
    
    for i in range(1,6):
        test = data[data["partition"] == i]
        db = data[data["partition"] != i]  #db is other partitions
        peptides = test["pep"].unique() # get all peptides that should be tested
        db = db[db["label"] == 1]   # subset database to only contain positives

        for peptide in peptides:
            test_pep = test[test["pep"] == peptide]
            db_pep = db[db["pep"] == peptide]
            
            if len(db_pep) > 0:
                test_pep.to_csv(os.path.join(TMP_DIR, f"P{i}_{peptide}_test"), sep=" ", columns=["cdr3a", "cdr3b", "label"], index=False,header=False)
                db_pep.to_csv(os.path.join(TMP_DIR, f"P{i}_{peptide}_db"), sep=" ", columns=["cdr3a", "cdr3b", "label"], index=False,header=False)

                subprocess.run([KERNEL_PATH,
                                os.path.join(TMP_DIR, f"P{i}_{peptide}_test"),
                                os.path.join(TMP_DIR, f"data/tmp/P{i}_{peptide}_db")])

if __name__ == "__main__":
    main()



