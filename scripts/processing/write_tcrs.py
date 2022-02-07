import numpy as np
import pandas as pd



def load_data(files):
    return [np.load(file)["arr_0"] for file in files]

def reverseOneHot(encoding):
    """
    Converts one-hot encoded array back to string sequence
    """
    mapping = dict(zip(range(20), "ACDEFGHIKLMNPQRSTVWY"))
    seq = ""
    for i in range(len(encoding)):
        if np.max(encoding[i]) > 0:
            seq += mapping[np.argmax(encoding[i])]
    return seq


def reverse_one_hot(arr, start_idx, stop_idx):
    sequences = []
    mapping = dict(zip(range(20), "ACDEFGHIKLMNPQRSTVWY"))
    for encoding in arr:
        seq = ""
        for pos in range(start_idx, stop_idx):
            if np.any((encoding[pos] == 1)):
                seq += mapping[np.argmax(encoding[pos])]

        sequences.append(seq)
    return sequences


def write_fasta(sequences, filename, header_prefix, mode="w"):
    with open(filename, mode) as outfile:
        for i, seq in enumerate(sequences, 1):
            outfile.write(">"+header_prefix+f"{i}\n")  # Print Header

            for j in range(0, len(seq), 60):
                outfile.write(seq[j:j+60]+"\n")


def main():
    data_path = "data/"
    filenames = [data_path+f"P{i}_input.npz" for i in range(1,6)]
    start_idx = 190
    stop_idx = 420
    partitions = load_data(filenames)

    for i, partition in enumerate(partitions, 1):
        partition = partition[:,:,:20]
        sequences = reverse_one_hot(partition, start_idx, stop_idx)
        write_fasta(sequences, data_path+f"concat_tcrs.fa", f"P{i}_tcr_seq_", "a")
        

        
# alphas = []
        # betas = []
        # for sequence in sequences:
        #     alpha, beta = split_tcrs(sequence)
        #     if alpha == None:
        #         write_fasta(alphas, data_path+f"alpha_chain.fa", f"P{i}_tcr_seq_", "a")
        #         write_fasta(betas, data_path+f"beta_chain.fa", f"P{i}_tcr_seq_", "a")
        #     alphas.append(alpha)
        #     betas.append(beta)
def split_tcrs(sequence: str):
    j_genes = [
        "VTQTPRHKV",
        "GITQSPKYL",
        "GVTQTPRHL",
        "GVSQSPRYK",
        "AGVTQTPKF",
        "TQTPSHQVT",
        "VSQHPSWVI",
        "VTQNPRYLI",
        "IHQWPATLV",
        "GAGTQVVVT",
        "ISQKPSRDI",
        "AGITQAPTS",
        "AGVIQSPRH",
        "VSQTPKYLV",
        "GVTQTPKHL",
        "GVTQFPSHS",
        "GVSQNPRHK",
        "VTQSSRYLV",
        "GVTQTPKFQ",
        "GVTQSPTHL",
        "AGVSQTPSN",
        "VTQTPKHLV",
        "IYQTPRYLV",
        "GVMQNPRHL",
        "AGVAQSPRY",
        "GVTQTPRYL",
        "GVIQSPRHL",
        "AGITQSPRY",
        "VIQNPRYQV",
        "GVSQSPSNK"
    ]
    
    found_gene = False
    for j_gene in j_genes:
        
        idx = sequence.find(j_gene)
        if idx != -1:
            found_gene = True
            sequences = (sequence[:idx], sequence[idx:])
            print(j_gene, "Found at:", idx)
            print("Alpha:", sequences[0])
            print("Beta:", sequences[1])
            assert len(sequences[0]) > 90, "Problem with alpha chain length"
            assert len(sequences[1]) > 90, "Problem with beta chain length"

            return sequences

    if not found_gene:
        print("Sequence not found:")
        print(sequence)
        return (None, None)


if __name__ == "__main__":
    main()
