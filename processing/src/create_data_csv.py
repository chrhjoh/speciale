from get_cdrs import parse_igblast, find_cdr3_stop
from functions import load_sequences, reverse_one_hot
import numpy as np
import re
import pandas as pd
import os


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

def annotate_negatives(df):
    res_check = dict()
    for index, row in df.iterrows():
        if row["tcra"] + row["tcrb"] not in res_check:
            res_check[row["tcra"] + row["tcrb"]] = [[row["partition"], row.name, row["pep"], row["label"]]]
        else:
            res_check[row["tcra"] + row["tcrb"]].append([row["partition"], row.name, row["pep"], row["label"]])

    unique_pos = 0
    unique_neg = 0
    not_two = 0
    one_pos_one_neg = 0
    two_neg = 0
    multiple_pos = 0
    total = 0
    ten_x_neg = set()
    swapped_neg = set()
    to_drop = set()
    for key in res_check:
        if len(res_check[key]) > 2: # Is there more than one negative and one positive (unexpected)
            one_pos = False
            two_pos = False
            print(res_check[key])
            not_two += 1
            for i, element in enumerate(res_check[key]):
                if element[-1] == 1:
                    pos_idx = i
                    if one_pos:
                        multiple_pos   += 1
                        two_pos = True
                    one_pos = True
            if two_pos: # If theres two positives then they should be dropped (impossible to know truth)
                to_drop.update([element[1] for element in res_check[key]])
            elif one_pos:
                # If they are in same partition then they are swapped negative, else they are ten x negative
                swapped_neg.update([(element[1], res_check[key][pos_idx][1])for element in res_check[key] if (element[0] == res_check[key][pos_idx][0]) & (element[-1] == 0)])
                ten_x_neg.update([element[1] for element in res_check[key] if (element[0] != res_check[key][pos_idx][0]) & (element[-1] == 0)])
            else:
                ten_x_neg.update([element[1] for element in res_check[key]])
        # if only one positive
        elif (len(res_check[key]) == 1) & (res_check[key][0][-1] == 1):
            unique_pos += 1
        # if only one negative
        elif (len(res_check[key]) == 1) & (res_check[key][0][-1] == 0):
            unique_neg += 1
            ten_x_neg.add(res_check[key][0][1])
        else:
            # Are there one negative and one positive or two negatives, there cant be two positives, since 1 negative per positive
            if (res_check[key][0][-1] == 0) & (res_check[key][1][-1] == 1):
                one_pos_one_neg += 1
                # If they are in same partition add to swapped, else the negative is 10x
                if res_check[key][0][0] == res_check[key][1][0]:
                    swapped_neg.add((res_check[key][0][1], res_check[key][1][1]))
                else:
                    ten_x_neg.add(res_check[key][0][1])

            elif (res_check[key][1][-1] == 0) & (res_check[key][0][-1] == 1):
                one_pos_one_neg += 1
                # If they are in same partition add to swapped, else the negative is 10x
                if res_check[key][0][0] == res_check[key][1][0]:
                    swapped_neg.add((res_check[key][1][1], res_check[key][0][1]))
                else:
                    ten_x_neg.add(res_check[key][1][1])
            else:
                # if both are negatives then add to 10x
                two_neg += 1
                ten_x_neg.add(res_check[key][0][1])
                ten_x_neg.add(res_check[key][1][1])

        total += 1
    print("More than two sequences:", not_two)
    print("Unique positives:", unique_pos)
    print("Unique negatives:", unique_neg)
    print("One positive, One negative:", one_pos_one_neg)
    print("Two negatives:", two_neg)
    print("multiple positives:", multiple_pos)
    print("total:", total)
    return swapped_neg, ten_x_neg, to_drop

def mark_swapped(df : pd.DataFrame, idxs):
    for idx in idxs:
        df.loc[idx[0], "origin"] = f"swapped_{idx[1]}"

def main():
    DIR = "/Users/christianjohansen/Desktop/speciale/"
    DATASET_DIR = os.path.join(DIR, "modeling/data/datasets")
    ANNOTATION_DIR = os.path.join(DIR, "processing/data")
    PARTITION_DIR = os.path.join(DIR, "partitioning/data")

    input_files = [os.path.join(DATASET_DIR, f"datasets/original_partitions/P{i}_input.npz") for i in range(1,6)]
    label_files = [os.path.join(DATASET_DIR, f"datasets/original_partitions/P{i}_labels.npz") for i in range(1,6)]

    hmm_file = os.path.join(ANNOTATION_DIR, "hmm/hmm_alpha.txt")
    igblast_file_a = os.path.join(ANNOTATION_DIR, "igblast/alpha_igblast.txt")
    igblast_file_b = os.path.join(ANNOTATION_DIR, "igblast/beta_igblast.txt")

    data = load_arrs(input_files)
    labels = load_arrs(label_files)
    tcr_anno = load_sequences(hmm_file)
    cdr_anno_a = parse_igblast(igblast_file_a)
    cdr_anno_b = parse_igblast(igblast_file_b)

    arr_dict = { "pep" : [], "cdr1a" : [], "cdr2a" : [], "cdr3a" : [],"cdr1b" : [],"cdr2b" : [], "cdr3b" : [],
                "fa_tot" : [], "fa_atr" : [], "fa_rep" : [], "fa_sol" : [], "fa_elec" : [], "fa_dun" : [],"p_aa_pp" : [],
                "global_interactions" : [], "partition" : [], "label" : [], "mhc" : [], "tcra" : [], "tcrb" : [] }
    for i, (partition, label_partition) in enumerate(zip(data,labels), 1):
        for j, (arr, label) in enumerate(zip(partition, label_partition), 1):  
            seq = arr[:,:20]    # Grap only one hot encoded sequence
            id = f"P{i}_tcr_seq_{j}"

            seq = reverse_one_hot(seq, pad="*") # Get sequences with padding so slicing can be done on array
            seq_list = seq.strip("*").split("*")
            tcr_a_start = len(seq_list) - 1 + len(seq_list[0])  # Length og MHC+pep -1 + number of pads till tcr start
            tcr_b_start = find_tcr_b_idx(seq[tcr_a_start:], tcr_anno[id]) + tcr_a_start

            cdr_anno_a[id] = find_cdr3_stop(seq[tcr_a_start:tcr_b_start],cdr_anno_a[id])
            cdr_anno_b[id] = find_cdr3_stop(seq[tcr_b_start:],cdr_anno_b[id])
            
            # Append the arrays with energy terms, so they can be included when reconstructing
            arr_dict["mhc"].append(reverse_one_hot(arr[:179, :20]))
            arr_dict["tcra"].append(reverse_one_hot(arr[tcr_a_start:tcr_b_start, :20]))
            arr_dict["tcrb"].append(reverse_one_hot(arr[tcr_b_start:, :20]))
            arr_dict["pep"].append(reverse_one_hot(arr[179:188, :20]))
            # idx works by tcr start + cdr start - 1 to tcr start + cdr end (-1 is because of igblast vs python indexing)
            # i indicates what partition to redistribute the array to after padding
            arr_dict["cdr1a"].append(reverse_one_hot(arr[tcr_a_start+cdr_anno_a[id]["CDR1-start"]-1:tcr_a_start+cdr_anno_a[id]["CDR1-stop"], :20]))
            arr_dict["cdr2a"].append(reverse_one_hot(arr[tcr_a_start+cdr_anno_a[id]["CDR2-start"]-1:tcr_a_start+cdr_anno_a[id]["CDR2-stop"], :20]))
            arr_dict["cdr3a"].append(reverse_one_hot(arr[tcr_a_start+cdr_anno_a[id]["CDR3-start"]-1:tcr_a_start+cdr_anno_a[id]["CDR3-stop"], :20]))
            arr_dict["cdr1b"].append(reverse_one_hot(arr[tcr_b_start+cdr_anno_b[id]["CDR1-start"]-1:tcr_b_start+cdr_anno_b[id]["CDR1-stop"], :20]))
            arr_dict["cdr2b"].append(reverse_one_hot(arr[tcr_b_start+cdr_anno_b[id]["CDR2-start"]-1:tcr_b_start+cdr_anno_b[id]["CDR2-stop"], :20]))
            arr_dict["cdr3b"].append(reverse_one_hot(arr[tcr_b_start+cdr_anno_b[id]["CDR3-start"]-1:tcr_b_start+cdr_anno_b[id]["CDR3-stop"], :20]))
            arr_dict["fa_tot"].append(list(arr[:, 20]))
            arr_dict["fa_atr"].append(list(arr[:, 21]))
            arr_dict["fa_rep"].append(list(arr[:, 22]))
            arr_dict["fa_sol"].append(list(arr[:, 23]))
            arr_dict["fa_elec"].append(list(arr[:, 24]))
            arr_dict["fa_dun"].append(list(arr[:, 25]))
            arr_dict["p_aa_pp"].append(list(arr[:, 26]))
            arr_dict["global_interactions"].append(list(arr[0, 27:]))
            arr_dict["partition"].append(i)
            arr_dict["label"].append(label)
            
    df = pd.DataFrame(arr_dict)
    df.index.name = "ID"
    swapped_neg, ten_x_neg, to_drop = annotate_negatives(df)
    df["origin"] = np.NaN
    df.loc[df["label"] == 1,"origin"] = "positive"
    mark_swapped(df, swapped_neg)
    df.loc[ten_x_neg,"origin"] = "10x"
    df = df.drop(to_drop)
    df["cdr3_len"] = df["cdr3a"].str.len() + df["cdr3b"].str.len()
    df = df.sort_values("cdr3_len", ascending=False)

    df.to_csv(os.path.join(DATASET_DIR, "data_cleaned.csv"))

    positives = df[df["origin"] == "positive"].reset_index()
    positives.index.name = "hobohm_idx"
    positives.to_csv(os.path.join(PARTITION_DIR, "positive_list.txt"), sep=" ", columns=["cdr3a", "cdr3b"],
                     index=False, header=False)

    negatives = df[df["origin"] == "10x"].reset_index()
    negatives.index.name = "hobohm_idx"
    negatives.to_csv(os.path.join(PARTITION_DIR, "negative_list.txt"), sep=" ", columns=["cdr3a", "cdr3b"],
                     index=False, header=False)

if __name__ == "__main__":
    main()
