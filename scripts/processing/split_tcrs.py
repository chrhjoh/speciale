import re
from functions import load_sequences, write_fasta


def parse_hmm(file_path) -> dict:
    """
    Loads output from hmmalign (simply a Fasta format)
    """
    return load_sequences(file_path)

def split_tcrs(sequences : dict, anno : dict):
    """
    Splits TCRS into alpha and beta
    Captial letters indicate alpha chain. 
    Finds last area with alpha chain [-1] and uses it for slicing
    """

    splt = re.compile(r"(?s:.*)[A-Z]+")

    res = splt.search(anno)
    match = res.group(0).split('-')[-1]

    idx = sequences.find(match)
    alpha = sequences[:idx+ len(match)]
    beta = sequences[idx + len(match):]
    assert (len(alpha) > 90) & (len(beta) > 90), "Chains are too short"

    return alpha, beta


def main():
    data_dir = "data/"
    seq_file = data_dir + "fasta/concat_tcrs.fa"
    hmm_file = data_dir + "hmm/hmm_alpha.txt"
    sequences = load_sequences(seq_file)
    anno = parse_hmm(hmm_file)

    alphas, betas = dict(), dict()
    for id in sequences:
        alphas[id], betas[id] = split_tcrs(sequences[id], anno[id])

    write_fasta(alphas, data_dir + "fasta/alpha_chains.fa")
    write_fasta(betas, data_dir + "fasta/beta_chains.fa")
    

if __name__ == "__main__":
    main()