import numpy as np

def load_sequences(file_path) -> dict:
    """
    Loads a Fasta file into a dict with header as key and sequences as values
    """
    headers = []
    sequences = []
    seq_lst = []
    with open(file_path, 'r') as fh:
        for line in fh:
            if line.startswith('>'):
                headers.append(line[1:].strip())

                if len(seq_lst) != 0:
                    sequences.append("".join(seq_lst))

                seq_lst = []

            else:
                seq_lst.append(line.strip())
    sequences.append("".join(seq_lst))

    return dict(zip(headers, sequences))

def write_fasta(sequences, filename):
    """
    Writes a Fasta file from a dict id will become header, and value will be the sequence 
    """
    with open(filename, "w") as fh:
        for id in sequences:
            fh.write(">"+id+"\n")
            for j in range(0, len(sequences[id]), 60):
                fh.write(sequences[id][j:j+60]+"\n")

def reverse_one_hot(arr, pad=""):
    """
    Takes a one hot encoding and reverses it to an amino acid string
    If the padding is needed it can be gotten through the pad parameter
    """
    mapping = dict(zip(range(20), "ACDEFGHIKLMNPQRSTVWY"))
    seq = ""
    for pos in range(len(arr)):
        if np.any((arr[pos] == 1)):
            seq += mapping[np.argmax(arr[pos])]
        else:
            seq += pad

    return seq
