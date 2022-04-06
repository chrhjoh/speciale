import pandas as pd
import os


def main():
    DIR = "/Users/christianjohansen/Desktop/speciale/processing"
    DATA_FILE = os.path.join(DIR, "data/raw/raw_10x_specificity.csv")
    OUT_FILE = os.path.join(DIR, "out/10x_200000sample.csv")
    df = pd.read_csv(DATA_FILE)
    mask = (df.check_clono == "unique") & (df.HLA.str.startswith("A02"))
    df = df[mask]
    df[["cdr3a", "cdr3b"]] = pd.DataFrame(df.cell_clono_cdr3_aa.str.split(";").to_list(), index= df.index, columns=["cdr3a", "cdr3b"])
    df.cdr3a = df.cdr3a.str[4:]
    df.cdr3b = df.cdr3b.str[4:]

    df_sampled = df[df.binder == True]
    df_sampled = pd.concat([
        df[df.binder == False].sample(190000,random_state=42),
        df_sampled])
    print(df_sampled)
    df_sampled.to_csv(OUT_FILE,index=False)

if __name__ == '__main__':
    main()
