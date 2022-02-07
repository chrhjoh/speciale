from functions import load_sequences
# Get CDR1, CDR2 and CDR3 start from igblast
# All index should be -1 (they are 1 based and not 0 based)


def parse_igblast(filename):
    features = dict()
    with open(filename, "r") as fh:
        for line in fh:
            if line.startswith("# Query"):
                id = line.split()[-1]
                features[id] = dict()

            elif line.startswith("CDR1") or line.startswith("CDR2"):
                line = line.split()
                feat = line[0][:4]

                features[id][feat+"-start"] = int(line[1])
                features[id][feat+"-stop"] = int(line[2])

            elif line.startswith("FR3"):
                line = line.split()
                features[id]["CDR3-start"] = int(line[2])+1

    return features


def find_cdr3_stop(sequences, features):
    i=0
    j=0
    for id in sequences:

        seq = sequences[id]
        cdr3_start = features[id]["CDR3-start"]
        cdr3_end = seq[cdr3_start-1:].find("FG", 7)
        if cdr3_end == -1:
            cdr3_end = seq[cdr3_start-1:].find("WG", 7)

        if cdr3_end ==-1:
            print("Error in",seq[cdr3_start-1:])
        
        features[id]["CDR3-end"] = cdr3_start + cdr3_end


def extract_features(sequences, features):
    pass

def main():
    data_dir = "data/"
    alpha_igblast = data_dir + "alpha_igblast.txt"
    beta_igblast = data_dir + "beta_igblast.txt"
    alpha_file = data_dir + "alpha_chains.fa"
    beta_file = data_dir + "beta_chains.fa"

    alpha_sequences = load_sequences(alpha_file)
    beta_sequences = load_sequences(beta_file)
    alpha_annotation = parse_igblast(alpha_igblast)
    beta_annotation = parse_igblast(beta_igblast)
    find_cdr3_stop(alpha_sequences, alpha_annotation)
    find_cdr3_stop(beta_sequences, beta_annotation)

    


if __name__ == "__main__":
    main()