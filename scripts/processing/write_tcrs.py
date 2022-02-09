import numpy as np
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
    data_path = "data/"
    input_files = [data_path+f"hackathon_data/P{i}_input.npz" for i in range(1,6)]
    target_files = [data_path+f"hackathon_data/P{i}_labels.npz" for i in range(1,6)]
    start_idx = 190
    stop_idx = 420
    partitions = load_data(input_files)
    targets = load_data(target_files)

    for i, partition in enumerate(partitions, 1):
        partition = partition[:,:,:20]  # Only take sequence
        sequences = []
        for arr in partition:
            sequences.append(reverse_one_hot(arr, start_idx, stop_idx))
        write_fasta(sequences, data_path+f"fasta/concat_tcrs.fa", f"P{i}_tcr_seq_", "a")

    write_targets(targets, data_path + "targets.txt")


if __name__ == "__main__":
    main()
