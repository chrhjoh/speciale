import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch import nn, optim, cuda
from torch.utils.data.dataloader import DataLoader
from utils import Runner, EarlyStopping, TcrDataset, setup_seed
from tcr_network import TcrCNN

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, palette="pastel")

def main():
    data_dir = "data/"
    data_files = [data_dir+f"datasets/P{i}_input_tcrs.npz" for i in range(1,6)]
    label_files = [data_dir+f"datasets/P{i}_labels.npz" for i in range(1,6)]
    model_name = "tcr_model.pt"
    model_path = "stored_models/"

    # Loader parameters
    batch_size = 64
    seed= 42

    # Hyperparameters
    epochs = 100
    patience = 20
    lr = 0.005
    weight_decay = 0.0005

    # Layer parameters
    cnn_channels = 20
    hidden_neurons = 64
    dropout = 0.4
    cnn_kernel = 3

    setup_seed(seed)
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print("Using device:", device)
    
    ########### Select Indexes TCR Data ############

    # Full model
    # idx = np.arange(417)

    # Peptide + both TCRs
    idx = np.arange(179,417)

    ########### Select Features ###########

    # All features
    #local_features = np.arange(27)
    #global_features = np.arange(27, 54)
    #use_global_features = True

    # Sequence
    local_features = np.arange(20)
    global_features = None
    use_global_features = False

    # Energy terms
    #local_features = np.arange(20, 27)
    #global_features = np.arange(27, 54)
    #use_global_features = True

    # Sequence and global energy
    #local_features = np.arange(20)
    #global_features = np.arange(27, 54)
    #use_global_features = True

    # Load data partitions
    train_data = TcrDataset(data_files[0], label_files[0])
    train_data.add_partition(data_files[1], label_files[1])
    train_data.add_partition(data_files[2], label_files[2])
    val_data = TcrDataset(data_files[3], label_files[3])
    test_data = TcrDataset(data_files[4], label_files[4])


    # Shuffle data randomly is needed
    train_data.shuffle_data()
    val_data.shuffle_data()
    test_data.shuffle_data()

    # slicing sequence dimension
    train_data.slice_sequences(idx)
    val_data.slice_sequences(idx)
    test_data.slice_sequences(idx)

    train_data.to_blossum("data/blosum/blosum.pkl")
    val_data.to_blossum("data/blosum/blosum.pkl")
    test_data.to_blossum("data/blosum/blosum.pkl")

    train_dl = DataLoader(train_data, batch_size)
    val_dl = DataLoader(val_data, batch_size)
    test_dl = DataLoader(test_data, batch_size)

    # Define loss and optimizer
    criterion = nn.BCELoss(reduction='none')
    loss_weight = sum(train_data.labels) / len(train_data.labels)
    stopper = EarlyStopping(patience, model_name, model_path)

    # Define network
    net = TcrCNN(local_features, global_features, use_global_features, cnn_channels=cnn_channels, dropout=dropout, cnn_kernel_size=cnn_kernel, dense_neurons=hidden_neurons)
    net.to(device)
 
    optimizer = optim.Adam(net.parameters(), lr=lr,
        weight_decay=weight_decay,
        amsgrad=True
    )

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

    final_model = TcrCNN(local_features, global_features, use_global_features, cnn_channels=cnn_channels, dropout=dropout, cnn_kernel_size=cnn_kernel, dense_neurons=hidden_neurons)
    final_model.load_state_dict(torch.load(model_path + model_name))
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

    print("Evaluation on Validation Data:")
    val_runner.evaluate_model()
    plt.title("Evaluation Data")

    print("Evaluation on Test Data:")
    test_runner.evaluate_model()
    plt.title("Test Data")

    print("Final model saved at:", model_path+model_name)


if __name__ == "__main__":
    main()