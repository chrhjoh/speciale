# score positive TCR peptides and a large sample of swapped negatives. This can be used to calculate AUC per TCR afterwards.
import os
import pandas as pd
import torch
from torch import cuda
from torch import nn
from gensim.models import KeyedVectors
from attention_net import AttentionNet, EmbedAttentionNet
from utils import Runner, AttentionDataset
from torch.utils.data import DataLoader


def make_swapped_tcrs(positive, peptides, seed):
    """
    Creates all possible swapped tcrs from the single positive TCR
    """
    negatives = positive.copy()
    peptides = set(peptides)
    peptides.remove(positive["pep"])

    negatives["pep"] = peptides
    negatives["label"] = 0
    negatives["origin"] = "swapped"

    negatives = negatives.to_frame().T.explode("pep")
    negatives = negatives.sample(10, random_state=seed)
    return pd.concat([negatives, positive.to_frame().T])


def create_dataframe(filename, seed):
    """
    Loads a csv files and creates all possible swapped from the tcrs
    """
    df = pd.read_csv(filename, index_col=0)
    df = df.loc[df.label == 1]
    peptides = pd.unique(df["pep"])
    df = df.loc[df.pep == "NLVPMVATV"] # GLCTLVAML NLVPMVATV GILGFVFTL
    swapped_dfs = []
    for index, row in df.iterrows():
        swapped_dfs.append(make_swapped_tcrs(row, peptides, seed))
    return pd.concat(swapped_dfs)




def do_predictions(data, net, device, score_file):
    sequences = ["pep", 
                 "cdr1a", "cdr2a", "cdr3a",
                 "cdr1b", "cdr2b", "cdr3b",]
    ENCODING = "encode"

    # Loader parameters
    BATCH_SIZE = 64
    
    test_dl = DataLoader(
        AttentionDataset(data, seq_features=sequences, partitions=[5], shuffle=True, encode_type=ENCODING), 
        batch_size=BATCH_SIZE
        )

    criterion = nn.BCELoss(reduction='none')
    loss_weight = sum(test_dl.dataset.labels) / len(test_dl.dataset.labels)
    
    test_runner = Runner(
        test_dl, 
        net, 
        criterion, 
        loss_weight, 
        device)
    

    test_runner.run_epoch()
    test_runner.scores_to_file(score_file)


def main():
    SEED = 42

    DIR = "/Users/christianjohansen/Desktop/speciale/modeling/"
    DATA_DIR = os.path.join(DIR, "data")
    DATA_FILE = os.path.join(DATA_DIR, "datasets/train_data_all.csv")
    SCORE_FILE = os.path.join(DIR, "results/NLVattlstm_positive_NLV_scores.csv")
    MODEL_FILE = os.path.join(DATA_DIR, "models/NLVattlstm_model.pt")

    data = create_dataframe(DATA_FILE, SEED)
    net = AttentionNet()
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    net.to(device)
    net.load_state_dict(torch.load(MODEL_FILE))
    do_predictions(data=data, net=net, device=device, score_file=SCORE_FILE)




if __name__ == "__main__":
    main()