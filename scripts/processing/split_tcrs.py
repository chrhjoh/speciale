import re
from functions import load_sequences, write_fasta


def parse_hmm(file_path) -> dict:
    return load_sequences(file_path)

def split_tcrs(sequences : dict, anno : dict):
    alphas, betas = dict(), dict()
    splt = re.compile(r"(?s:.*)[A-Z]+")

    for id in sequences:

        res = splt.search(anno[id])
        match = res.group(0).split('-')[-1]

        idx = sequences[id].find(match)
        alpha = sequences[id][:idx+ len(match)]
        beta = sequences[id][idx + len(match):]
        assert (len(alpha) > 90) & (len(beta) > 90), "Chains are too short"

        alphas[id] = alpha
        betas[id] = beta
    
    return alphas, betas


def main():
    data_dir = "data/"
    seq_file = data_dir + "concat_tcrs.fa"
    hmm_file = data_dir + "hmm_alpha.txt"
    sequences = load_sequences(seq_file)
    anno = parse_hmm(hmm_file)
    alphas, betas = split_tcrs(sequences, anno)

    write_fasta(alphas, data_dir + "alpha_chains.fa")
    write_fasta(betas, data_dir + "beta_chains.fa")
    

if __name__ == "__main__":
    main()