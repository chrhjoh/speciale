from get_cdrs import parse_igblast, find_cdr3_stop
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

#TODO: Find a way to introduce padding into numpy arrays, so all arrays in the list has same length
# Put back all CDRs and pad zeroes so all CDRs start at same position
# Zeroes should also be padded in energy terms
# Then stack on the length so one array per sequence
# After that divide into partitions, and stack those partitions
# Write the new numpy arrays out


def main():
    data_dir = "data/"
    input_files = [data_dir+f"hackathon_data/P{i}_input.npz" for i in range(1,6)]
    hmm_file = data_dir+"hmm/hmm_alpha.txt"
    igblast_file_a = data_dir+"igblast/alpha_igblast.txt"
    igblast_file_b = data_dir+"igblast/beta_igblast.txt"

    data = load_arrs(input_files)
    tcr_anno = load_sequences(hmm_file)
    cdr_anno_a = parse_igblast(igblast_file_a)
    cdr_anno_b = parse_igblast(igblast_file_b)

    arr_dict = {"mhc_pep" : [], "cdr1_a" : [], "cdr2_a" : [], "cdr3_a" : [],"cdr1_b" : [],"cdr2_b" : [], "cdr3_b" : []}
    for i, partition in enumerate(data, 1):
        for j, arr in enumerate(partition, 1):  
            seq = arr[:,:20]    # Grap only one hot encoded sequence
            id = f"P{i}_tcr_seq_{j}"

            seq = reverse_one_hot(seq, pad="*") # Get sequences with padding so slicing can be done on array
            seq_list = seq.strip("*").split("*")
            tcr_a_start = len(seq_list) - 1 + len(seq_list[0])  # Length og MHC+pep -1 + number of pads till tcr start
            tcr_b_start = find_tcr_b_idx(seq[tcr_a_start:], tcr_anno[id]) + tcr_a_start

            cdr_anno_a[id] = find_cdr3_stop(seq[tcr_a_start:tcr_b_start],cdr_anno_a[id])
            cdr_anno_b[id] = find_cdr3_stop(seq[tcr_b_start:],cdr_anno_b[id])
            
            # Append the arrays with energy terms, so they can be included when reconstructing
            arr_dict["mhc_pep"].append((arr[:188],i))
            # idx works by tcr start + cdr start - 1 to tcr start + cdr end (-1 is because of igblast vs python indexing)
            # i indicates what partition to redistribute the array to after padding
            arr_dict["cdr1_a"].append((arr[tcr_a_start+cdr_anno_a[id]["CDR1-start"]-1:tcr_a_start+cdr_anno_a[id]["CDR1-stop"]],i))
            arr_dict["cdr2_a"].append((arr[tcr_a_start+cdr_anno_a[id]["CDR2-start"]-1:tcr_a_start+cdr_anno_a[id]["CDR2-stop"]],i))
            arr_dict["cdr3_a"].append((arr[tcr_a_start+cdr_anno_a[id]["CDR3-start"]-1:tcr_a_start+cdr_anno_a[id]["CDR3-stop"]],i))
            arr_dict["cdr1_b"].append((arr[tcr_b_start+cdr_anno_b[id]["CDR1-start"]-1:tcr_b_start+cdr_anno_b[id]["CDR1-stop"]],i))
            arr_dict["cdr2_b"].append((arr[tcr_b_start+cdr_anno_b[id]["CDR2-start"]-1:tcr_b_start+cdr_anno_b[id]["CDR2-stop"]],i))
            arr_dict["cdr3_b"].append((arr[tcr_b_start+cdr_anno_b[id]["CDR3-start"]-1:tcr_b_start+cdr_anno_b[id]["CDR3-stop"]],i))
    
    pad_lengths = []
    for feat in arr_dict:
        pad_len = 0
        for arr, _ in arr_dict[feat]:
            if arr.shape[0] > pad_len:
                pad_len = arr.shape[0]

        pad_lengths.append(pad_len)
        




if __name__ == "__main__":
    main()