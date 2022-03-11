import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import pickle
from torch import nn, optim, cuda
from torch.utils.data.dataloader import DataLoader
from utils import Runner, EarlyStopping, TcrDataset, setup_seed
from cdr_network import CdrCNN

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, palette="pastel")

def read_data(filename):
    df = pd.read_csv(filename, index_col=0)
    return df

def main():

    ############ PARAMETERS ##############
    DIR = "/Users/christianjohansen/Desktop/speciale/modeling"
    DATA_DIR = os.path.join(DIR,"data")
    DATA_FILE = os.path.join(DATA_DIR, "datasets/train_data_all.csv")
    MODEL_FILE = os.path.join(DATA_DIR, "models/cdr_model.pt")
    ENCODING_FILE = os.path.join(DATA_DIR, "blosum/blosum.pkl")

    # Data parameters
    train_partition = [1,2,3]
    val_partition = [4]
    test_partition = [5]

    sequences = ["pep", 
                 "cdr1a", "cdr2a", "cdr3a",
                 "cdr1b", "cdr2b", "cdr3b"]

    local_features = np.arange(20)
    global_features = None
    use_global_features = False

    # Loader parameters
    batch_size = 64
    seed= 42

    # Hyperparameters
    epochs = 300
    patience = 20
    lr = 0.005
    weight_decay = 0.0005

    # Layer parameters
    cnn_channels = 30
    hidden_neurons = 64
    dropout = 0.3
    cnn_kernel = 3

    ################ Load Data ####################
    setup_seed(seed)
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load encoding
    with open(ENCODING_FILE, "rb" ) as fh:
            encoding = pickle.load(fh)

    data = read_data(DATA_FILE)
    train_data = TcrDataset(data, train_partition)
    val_data = TcrDataset(data, val_partition)
    test_data = TcrDataset(data, test_partition)
    
    train_data.add_sequence_features(sequences, encoding)
    val_data.add_sequence_features(sequences, encoding)
    test_data.add_sequence_features(sequences, encoding)

    # Shuffle data randomly is needed
    train_data.shuffle_data()
    val_data.shuffle_data()
    test_data.shuffle_data()

    train_dl = DataLoader(train_data, batch_size, drop_last=True)
    val_dl = DataLoader(val_data, batch_size)
    test_dl = DataLoader(test_data, batch_size)

    ############### DEFINE NETWORK ################
    # Define loss and optimizer
    criterion = nn.BCELoss(reduction='none')
    loss_weight = sum(train_data.labels) / len(train_data.labels)
    stopper = EarlyStopping(patience, MODEL_FILE)

    # Define network
    net = CdrCNN(local_features, global_features, use_global_features, cnn_channels=cnn_channels, dropout=dropout, cnn_kernel_size=cnn_kernel, dense_neurons=hidden_neurons)
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
    # Plots of training epochs
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

    final_model = CdrCNN(local_features, global_features, use_global_features, cnn_channels=cnn_channels, dropout=dropout, cnn_kernel_size=cnn_kernel, dense_neurons=hidden_neurons)
    final_model.load_state_dict(torch.load(MODEL_FILE))
    final_model.to(device)

    train_runner.model = final_model
    val_runner.model = final_model
    test_runner.model = final_model

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

    print("Final model saved at:", MODEL_FILE)

if __name__ == "__main__":
    main() 