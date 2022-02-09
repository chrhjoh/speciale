from functions import load_sequences
# Get CDR1, CDR2 and CDR3 start from igblast
# All index should be -1 (they are 1 based and not 0 based)

def find_cdr3_stop(seq, feat):
    """
    Finds the missing CDR3 stop, by using that the CDR3 allways ends on either FG or WG
    and is atleast 8 amino acids long 
    """
        
    cdr3_start = feat["CDR3-start"]
    cdr3_end = seq[cdr3_start-1:].find("FG", 7)
    if cdr3_end == -1:
        cdr3_end = seq[cdr3_start-1:].find("WG", 7)

    if cdr3_end ==-1:
        print("Error in",seq[cdr3_start-1:])
    
    feat["CDR3-stop"] = cdr3_start + cdr3_end
    return feat

def parse_igblast(filename):
    """
    Parses the output from igblast and returns a dict with CDR idxs
    Key will be the query name
    """
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



def extract_features(seq : str, feat : dict) -> dict:
    """
    Extracts CDRS from a sequence by simple slicing
    """
    cdrs = dict()

    cdrs["CDR1"] = seq[feat["CDR1-start"]-1:feat["CDR1-stop"]]
    cdrs["CDR2"] = seq[feat["CDR2-start"]-1:feat["CDR2-stop"]]
    cdrs["CDR3"] = seq[feat["CDR3-start"]-1:feat["CDR3-stop"]]

    return cdrs

def write_cdrs(cdrs: dict, filename : str) -> None:
    """
    Writes CDRs with tabs in a fasta format
    Tabs are included for easier later extraction
    """
    with open(filename, "w") as fh:
        for id in cdrs:
            fh.write(">"+id+"\n")
            fh.write(cdrs[id]["CDR1"]+"\t")
            fh.write(cdrs[id]["CDR2"]+"\t")
            fh.write(cdrs[id]["CDR3"]+"\n")



def main():
    data_dir = "data/"
    alpha_igblast_file = data_dir + "igblast/alpha_igblast.txt"
    beta_igblast_file = data_dir + "igblast/beta_igblast.txt"
    alpha_seq_file = data_dir + "fasta/alpha_chains.fa"
    beta_seq_file = data_dir + "fasta/beta_chains.fa"

    alpha_sequences = load_sequences(alpha_seq_file)
    beta_sequences = load_sequences(beta_seq_file)
    alpha_igblast = parse_igblast(alpha_igblast_file)
    beta_igblast = parse_igblast(beta_igblast_file)

    alpha_cdrs = dict()
    beta_cdrs = dict()
    for id in alpha_sequences:
        alpha_seq = alpha_sequences[id]
        beta_seq = beta_sequences[id]
        alpha_annotation =  alpha_igblast[id]
        beta_annotation = beta_igblast[id]


        alpha_annotation = find_cdr3_stop(alpha_seq, alpha_annotation)
        beta_annotation = find_cdr3_stop(beta_seq, beta_annotation)
        alpha_cdrs[id] = extract_features(alpha_seq, alpha_annotation)
        beta_cdrs[id] = extract_features(beta_seq, beta_annotation)

    write_cdrs(alpha_cdrs, data_dir + "fasta/alpha_cdrs.fa")
    write_cdrs(beta_cdrs, data_dir + "fasta/beta_cdrs.fa")


    


if __name__ == "__main__":
    main()