import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
from torch import nn, optim, cuda
from torch.utils.data.dataloader import DataLoader
from utils import Runner, EarlyStopping, AttentionDataset, setup_seed
from attention_net import LSTMNet, AttentionNet, EmbedAttentionNet
from gensim.models import KeyedVectors

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, palette="pastel")

def main():

    ############ PARAMETERS ##############
    DIR = "/Users/christianjohansen/Desktop/speciale/modeling"
    DATA_DIR = os.path.join(DIR,"data")
    DATA_FILE = os.path.join(DATA_DIR, "datasets/train_data_all.csv")
    MODEL_FILE = os.path.join(DATA_DIR, "models/lstm_cdr_model.pt")
    ATTENTION_FILE = os.path.join(DIR, 'results/lstm_attention_partition5.csv')

    CLI=False
    # Data parameters
    if CLI:
        test_partition = [int(sys.argv[1]) % 5 + 1]
        train_partition = [(int(sys.argv[1]) + i) % 5 + 1  for i in range(1,4)]
        val_partition = [(int(sys.argv[1]) + 4) % 5 + 1]
    else:   
        train_partition = [1,2,3]
        val_partition = [4]
        test_partition = [5]

    sequences = ["pep", 
                 "cdr1a", "cdr2a", "cdr3a",
                 "cdr1b", "cdr2b", "cdr3b",]
    ENCODING = "tokenize"
    cdr3a_embedding = KeyedVectors.load(os.path.join(DATA_DIR, "encoding/embeddings_cdr3a.wv"))
    cdr3b_embedding = KeyedVectors.load(os.path.join(DATA_DIR, "encoding/embeddings_cdr3b.wv"))

    # Loader parameters
    batch_size = 64
    seed= 42
    setup_seed(seed)

    # Hyperparameters
    epochs = 300
    patience = 20
    lr = 0.005
    weight_decay = 0

    ################ Load Data ####################

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print("Using device:", device)

    data = pd.read_csv(DATA_FILE, index_col=0)
    train_data = AttentionDataset(data, train_partition, sequences, shuffle=True, encode_type=ENCODING)
    val_data = AttentionDataset(data, val_partition, sequences, shuffle=True, encode_type=ENCODING)
    test_data = AttentionDataset(data, test_partition, sequences, shuffle=True, encode_type=ENCODING)

    train_dl = DataLoader(train_data, batch_size, drop_last=True)
    val_dl = DataLoader(val_data, batch_size)
    test_dl = DataLoader(test_data, batch_size)

    ############### DEFINE NETWORK ################
    # Define loss and optimizer
    criterion = nn.BCELoss(reduction='none')
    loss_weight = sum(train_data.labels) / len(train_data.labels)
    stopper = EarlyStopping(patience, filename=MODEL_FILE,delta=0)

    # Define network
    net = EmbedAttentionNet(cdr3a_wv = cdr3a_embedding, cdr3b_wv = cdr3b_embedding)
    net.to(device)
 
    optimizer = optim.Adam(net.parameters(), lr=lr,
        weight_decay=weight_decay,
        amsgrad=True
    )
    ############# TRAIN ################
    # Define runners
    train_runner = Runner(train_dl, net, criterion, loss_weight, device, optimizer)
    val_runner = Runner(val_dl, net, criterion, loss_weight, device)
    test_runner = Runner(test_dl, net, criterion, loss_weight, device)

    # Training Loop
    train_loss, val_loss, train_auc, val_auc = [], [], [], []

    for epoch in range(1, epochs+1):
        train_runner.run_epoch()
        val_runner.run_epoch()
        
        train_runner.follow_performance(epoch)
        val_runner.follow_performance(epoch)
        stopper.evaluate_epoch(val_runner.loss, net, epoch)
        
        train_loss.append(train_runner.loss)
        val_loss.append(val_runner.loss)
        train_auc.append(train_runner.auc)
        val_auc.append(val_runner.auc)

        train_runner.reset()
        val_runner.reset()

        if stopper.stop:
            break

    ################ EVALUATE ##################
    #Plots of training epochs
    epoch = np.arange(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epoch, train_loss, "r", epoch, val_loss, "b", linewidth=3)
    plt.vlines(stopper.best_epoch, ymin=0, ymax=0.3, colors="black", linestyles='dashed')
    plt.legend(["Train Loss", "Validation Loss", "Best Epoch"])
    plt.xlabel("Epoch"), plt.ylabel("Loss")
    plt.show()

    epoch = np.arange(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epoch, train_auc, "r", epoch, val_auc, "b", linewidth=3)
    plt.vlines(stopper.best_epoch, ymin=0, ymax=1, colors="black", linestyles='dashed')
    plt.legend(["Train AUC", "Validation AUC", "Best Epoch"])
    plt.xlabel("Epoch"), plt.ylabel("AUC")
    plt.show()

    net.load_state_dict(torch.load(MODEL_FILE))

    train_runner.model = net
    val_runner.model = net
    test_runner.model = net

    train_runner.reset()
    val_runner.reset()
    test_runner.reset()

    train_runner.evaluation_mode() # Set to validation to avoid more training
    train_runner.run_epoch()
    val_runner.run_epoch()
    test_runner.run_epoch()

    print("Evaluation on Training Data:")
    train_runner.evaluate_model()
    plt.title("Training Data")
    plt.show()

    print("Evaluation on Validation Data:")
    val_runner.evaluate_model()
    plt.title("Evaluation Data")
    plt.show()

    print("Evaluation on Test Data:")
    test_runner.evaluate_model()
    plt.title("Test Data")
    plt.show()

    test_runner.save_attention_weights(ATTENTION_FILE)
    print("Final model saved at:", MODEL_FILE)
    print("Attention weights saved at:", ATTENTION_FILE)
    
if __name__ == "__main__":
    main() 