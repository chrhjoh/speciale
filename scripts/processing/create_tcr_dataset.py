from functions import load_sequences, reverse_one_hot
import numpy as np
import re


# Load the numpy array data
def load_arrs(filenames):
    return [np.load(file)["arr_0"] for file in filenames]

# Slice the numpy array into the individual tcrs and peptides
def find_tcr_b_idx(seq, anno):
    """
    Find the start index of the tcr beta chain using the output of hmmalign
    """
    splt = re.compile(r"(?s:.*)[A-Z]+")

    res = splt.search(anno)
    match = res.group(0).split('-')[-1]
    idx = seq.find(match)
    return idx + len(match)

def find_pad_length(arrs, axis=0):
    pad_len = 0
    for arr, _ in arrs:
        if arr.shape[axis] > pad_len:
            pad_len = arr.shape[axis]

    return pad_len


def pad_array(arr, pad_length):
    """
    Pad an array so that the sequence dimension will have length pad_length
    """
    pad_amount = pad_length - arr.shape[0]
    
    assert pad_amount >= 0, "Pad amount should be non negative"
    if pad_amount > 0:
        pad_arr = np.zeros((pad_amount, arr.shape[1]))
        arr = np.concatenate([arr, pad_arr])
    return arr


def main():
    data_dir = "data/"
    input_files = [data_dir+f"datasets/P{i}_input.npz" for i in range(1,6)]
    hmm_file = data_dir+"hmm/hmm_alpha.txt"

    data = load_arrs(input_files)
    tcr_anno = load_sequences(hmm_file)

    arr_dict = {"mhc_pep" : [], "tcr_a" : [], "tcr_b" : []}
    for i, partition in enumerate(data, 1):
        for j, arr in enumerate(partition, 1):  
            seq = arr[:,:20]    # Grap only one hot encoded sequence
            id = f"P{i}_tcr_seq_{j}"

            seq = reverse_one_hot(seq, pad="*") # Get sequences with padding so slicing can be done on array
            seq_list = seq.strip("*").split("*")
            tcr_a_start = len(seq_list) - 1 + len(seq_list[0])  # Length og MHC+pep -1 + number of pads till tcr start
            tcr_b_start = find_tcr_b_idx(seq[tcr_a_start:], tcr_anno[id]) + tcr_a_start
            tcr_b_end = seq.find("*", tcr_b_start)

            
            # Append the arrays with energy terms, so they can be included when reconstructing
            arr_dict["mhc_pep"].append([arr[:188],i])
            # idx works by tcr start + cdr start - 1 to tcr start + cdr end (-1 is because of igblast vs python indexing)
            # i indicates what partition to redistribute the array to after padding
            arr_dict["tcr_a"].append([arr[tcr_a_start:tcr_b_start],i])
            arr_dict["tcr_b"].append([arr[tcr_b_start:tcr_b_end],i])

    
    pad_lengths = [find_pad_length(arr_dict[feat]) for feat in arr_dict ]

    for i, feat in enumerate(arr_dict):
        for j, arr in enumerate(arr_dict[feat]):
            arr_dict[feat][j][0] = pad_array(arr[0], pad_lengths[i])

    partitions = [[], [], [], [] ,[]]
    for i in range(len(arr_dict["mhc_pep"])): # Just loop through any of the features
        # concatenate arrays corresponding to one observation
        arr_list = [arr_dict["mhc_pep"][i][0], arr_dict["tcr_a"][i][0], arr_dict["tcr_b"][i][0]]
        padded_arr = np.concatenate(arr_list)
        partitions[arr_dict["mhc_pep"][i][1]-1].append(padded_arr) 

    # stack arrays for each observation
    for i, partition in enumerate(partitions,1):
        np.stack(partition)
        np.savez(data_dir+f"datasets/P{i}_input_tcrs", partition)
    print("Indexes for different features")
    print("MHC_pep", "tcr_a", "tcr_b", sep="\t")
    print(*pad_lengths,sep="\t")

if __name__ == "__main__":
    main()