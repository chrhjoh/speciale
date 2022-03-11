import numpy as np
import os
from functions import reverse_one_hot



def load_data(files):
    return [np.load(file)["arr_0"] for file in files]



def write_fasta(sequences, filename, header_prefix, mode="w"):
    with open(filename, mode) as fh:
        for i, seq in enumerate(sequences, 1):
            fh.write(">"+header_prefix+f"{i}\n")  # Print Header

            for j in range(0, len(seq), 60):
                fh.write(seq[j:j+60]+"\n")

def write_targets(targets, filename):
    with open(filename, "w") as fh:
        for i, partition in enumerate(targets, 1):
            for j, target in enumerate(partition, 1):
                fh.write(f"P{i}_target_{j}\t"+str(int(target))+"\n") 


def main():
    DIR = "/Users/christianjohansen/Desktop/speciale"
    input_files = [os.path.join(DIR,f"modeling/data/datasets/original_partitions/P{i}_input.npz") for i in range(1,6)]
    start_idx = 190
    stop_idx = 420
    partitions = load_data(input_files)

    for i, partition in enumerate(partitions, 1):
        partition = partition[:,:,:20]  # Only take sequence
        sequences = []
        for arr in partition:
            sequences.append(reverse_one_hot(arr, start_idx, stop_idx))
        write_fasta(sequences, os.path.join(DIR,f"processing/data/fasta/concat_tcrs.fa"), f"P{i}_tcr_seq_", "a")

if __name__ == "__main__":
    main()
