import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

def maxpool_length(length, kernel_size, stride=1, padding=0, dilation=1):
    return int((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

def conv_length(length, kernel_size, stride=1, padding=0, dilation=1):
    return int((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)



class CdrCNN(nn.Module):
    def __init__(self,
                local_features,
                global_features,
                use_global_features,
                cnn_channels = 30,
                cnn_kernel_size = 3,
                cnn_stride = 1,
                cnn_padding = "same",
                dense_neurons = 64,
                dropout = 0.3):

        super(CdrCNN, self).__init__()  

        # Define which features to use for CNN
        self.local_features = local_features
        self.n_local_feat = len(local_features)

        self.use_global_features = use_global_features
        if use_global_features: 
            self.global_features = global_features
            self.n_global_feat = len(global_features)
    
        # Peptide indexes
        self.pep = np.arange(9)

        # CDR indexes
        self.cdr1a = np.arange(9, 16)
        self.cdr2a = np.arange(16, 24)
        self.cdr3a = np.arange(24, 39)

        self.cdr1b = np.arange(39, 45)
        self.cdr2b = np.arange(45, 52)
        self.cdr3b = np.arange(52, 69)

        # CDR1a
        self.cdr1a_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                    out_channels = cnn_channels,
                                    kernel_size  = cnn_kernel_size,
                                    stride       = cnn_stride,
                                    padding      = cnn_padding)
        # CDR2a
        self.cdr2a_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                    out_channels = cnn_channels,
                                    kernel_size  = cnn_kernel_size,
                                    stride       = cnn_stride,
                                    padding      = cnn_padding)

        # CDR3a
        self.cdr3a_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                    out_channels = cnn_channels,
                                    kernel_size  = cnn_kernel_size,
                                    stride       = cnn_stride,
                                    padding      = cnn_padding)

        # CDR1b
        self.cdr1b_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                    out_channels = cnn_channels,
                                    kernel_size  = cnn_kernel_size,
                                    stride       = cnn_stride,
                                    padding      = cnn_padding)

        # CDR2b
        self.cdr2b_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                    out_channels = cnn_channels,
                                    kernel_size  = cnn_kernel_size,
                                    stride       = cnn_stride,
                                    padding      = cnn_padding)

        # CDR3b
        self.cdr3b_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                    out_channels = cnn_channels,
                                    kernel_size  = cnn_kernel_size,
                                    stride       = cnn_stride,
                                    padding      = cnn_padding)

        # Peptide
        self.pep_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                  out_channels = cnn_channels,
                                  kernel_size  = cnn_kernel_size,
                                  stride       = cnn_stride,
                                  padding      = cnn_padding)

        if cnn_padding == "same":
            self.cnn_padding = int((cnn_kernel_size - 1) / 2)
        else:
            self.cnn_padding = cnn_padding

        # Max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.max_pool_index = nn.AdaptiveMaxPool1d(1, return_indices=True)
        self.return_pool = False
    

        # Dense layers
        n_convs = 3
        if self.use_global_features:
            self.dense1 = nn.Linear(cnn_channels * n_convs + self.n_global_feat, dense_neurons)
        else:
            self.dense1 = nn.Linear(cnn_channels * n_convs, dense_neurons) 
        
        self.dense_out = nn.Linear(dense_neurons, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Batchnorm before dense
        if self.use_global_features:
            self.bn_dense = nn.BatchNorm1d(cnn_channels * n_convs + self.n_global_feat)
        else:
            self.bn_dense = nn.BatchNorm1d(cnn_channels * n_convs) 
        
        self.bn_start = nn.BatchNorm1d(self.n_local_feat)

        # Init weights
        init.kaiming_normal_(self.cdr1a_conv.weight)
        init.kaiming_normal_(self.cdr2a_conv.weight)
        init.kaiming_normal_(self.cdr3a_conv.weight)

        init.kaiming_normal_(self.cdr1b_conv.weight)
        init.kaiming_normal_(self.cdr2b_conv.weight)
        init.kaiming_normal_(self.cdr3b_conv.weight)

        init.kaiming_normal_(self.pep_conv.weight)

        init.kaiming_normal_(self.dense1.weight)
        init.kaiming_normal_(self.dense_out.weight)


    def forward(self, x):
        # Initial dropout
        x = self.dropout(x)
        # global features are the same for the whole sequence -> take first value
        if self.use_global_features:
            global_features = x[:, self.global_features, 0]

        # Get all local features divided
        local_features = x[:, self.local_features, :]
        x = self.bn_start(local_features)

        pep = local_features[:, :, self.pep]
        
        #cdr1a = local_features[:, :, self.cdr1a]
        #cdr2a = local_features[:, :, self.cdr2a]
        cdr3a = local_features[:, :, self.cdr3a]

        #cdr1b = local_features[:, :, self.cdr1b]
        #cdr2b = local_features[:, :, self.cdr2b]
        cdr3b = local_features[:, :, self.cdr3b]
        

        # Do all the convolutions
        pep_conved = torch.tanh(self.pep_conv(pep))

        #cdr1a_conved = torch.tanh(self.cdr1a_conv(cdr1a))
        #cdr2a_conved = torch.tanh(self.cdr2a_conv(cdr2a))
        cdr3a_conved = torch.tanh(self.cdr3a_conv(cdr3a))

        #cdr1b_conved = torch.tanh(self.cdr1b_conv(cdr1b))
        #cdr2b_conved = torch.tanh(self.cdr2b_conv(cdr2b))
        cdr3b_conved = torch.tanh(self.cdr3b_conv(cdr3b))

        # If we want to adjust for paddings. Do so here
        #pep_conved = self.adjust_padding_activation(pep_conved, pep)
        #cdr3a_conved = self.adjust_padding_activation(cdr3a_conved, cdr3a)
        #cdr3b_conved = self.adjust_padding_activation(cdr3b_conved, cdr3b)
        # Do all poolings
        pep_pool = self.max_pool(pep_conved)

        #cdr1a_pool = self.max_pool(cdr1a_conved)
        #cdr2a_pool = self.max_pool(cdr2a_conved)
        cdr3a_pool = self.max_pool(cdr3a_conved)

        #cdr1b_pool = self.max_pool(cdr1b_conved)
        #cdr2b_pool = self.max_pool(cdr2b_conved)
        cdr3b_pool = self.max_pool(cdr3b_conved)
        
        if self.return_pool != False:
            if self.return_pool == "pep":
                return self.max_pool_index(pep_conved)
            elif self.return_pool == "cdr3b":
                return self.max_pool_index(cdr3b_conved)
            elif self.return_pool == "cdr3a":
                return self.max_pool_index(cdr3a_conved)
            else:
                print("Unknown type")
                return
        ############################################

        # Combine convolutions
        x = torch.cat((pep_pool, cdr3a_pool, cdr3b_pool), dim=1)
        x = torch.flatten(x, 1)

        if self.use_global_features:
            x = torch.cat((x, global_features), dim=1) # add global features
        
        x = self.bn_dense(x)
        x = self.dropout(x)
        # Dense
        x = torch.tanh(self.dense1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.dense_out(x))
        return x

    def adjust_padding_activation(self, x_conv, x):
        """
        Takes a batch of arrays, and subtracts 2 from any convoluted position, which contains padding.
        This is to ensure that the following maxpooling selects a convoluted position containing full information.
        Returns a batch of adjusted arrays
        Only works in simple cases (stride = 1, dilation = 1, kernel_size = 3)
        """
        adjust_tensor = torch.zeros(x_conv.shape)
        for i, arr in enumerate(x): # loop through batch
            
            for j in reversed(range(arr.shape[1])): # loop reverse through seq length
                pos = arr[:, j]
                if not torch.all(pos == 0):
                    if j == arr.shape[1] - 1:
                        last_pos = None
                    else:
                        last_pos = j
                    break

            # adjust any intial and end padding made by CNN
            if self.cnn_padding > 0:
                adjust_tensor[i, :, :self.cnn_padding] = -2
                adjust_tensor[i, :, -self.cnn_padding:] = -2

            if last_pos is not None:
                adjust_tensor[i, :, last_pos - self.cnn_padding + 1:] = -2
        return x_conv + adjust_tensor

class TcrCNN(nn.Module):

    def __init__(self,
            local_features,
            global_features,
            use_global_features,
            cnn_channels = 30,
            cnn_kernel_size = 3,
            cnn_stride = 1,
            cnn_padding = "same",
            dense_neurons = 64,
            dropout = 0.3):

        super(TcrCNN, self).__init__()  

        # Define which features to use for CNN
        self.local_features = local_features
        self.n_local_feat = len(local_features)

        self.use_global_features = use_global_features
        if use_global_features: 
            self.global_features = global_features
            self.n_global_feat = len(global_features)
    
        # Peptide indexes
        self.pep = np.arange(9)

        # TCR indexes
        self.tcr_a = np.arange(9, 125)
        self.tcr_b = np.arange(125, 238)

        # TCR alpha
        self.tcr_a_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                    out_channels = cnn_channels,
                                    kernel_size  = cnn_kernel_size,
                                    stride       = cnn_stride,
                                    padding      = cnn_padding)
        # TCR beta
        self.tcr_b_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                    out_channels = cnn_channels,
                                    kernel_size  = cnn_kernel_size,
                                    stride       = cnn_stride,
                                    padding      = cnn_padding)

        # Peptide
        self.pep_conv = nn.Conv1d(in_channels  = self.n_local_feat,
                                  out_channels = cnn_channels,
                                  kernel_size  = cnn_kernel_size,
                                  stride       = cnn_stride,
                                  padding      = cnn_padding)

        if cnn_padding == "same":
            self.cnn_padding = int((cnn_kernel_size - 1) / 2)
        else:
            self.cnn_padding = cnn_padding

        # Max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.max_pool_index = nn.AdaptiveMaxPool1d(1, return_indices=True)
        self.return_pool = False
    

        # Dense layers
        n_convs = 3
        if self.use_global_features:
            self.dense1 = nn.Linear(cnn_channels * n_convs + self.n_global_feat, dense_neurons)
        else:
            self.dense1 = nn.Linear(cnn_channels * n_convs, dense_neurons) 
        
        self.dense_out = nn.Linear(dense_neurons, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Batchnorm before dense
        if self.use_global_features:
            self.bn_dense = nn.BatchNorm1d(cnn_channels * n_convs + self.n_global_feat)
        else:
            self.bn_dense = nn.BatchNorm1d(cnn_channels * n_convs) 
        
        self.bn_start = nn.BatchNorm1d(self.n_local_feat)

        # Init weights
        init.kaiming_normal_(self.tcr_a_conv.weight)
        init.kaiming_normal_(self.tcr_b_conv.weight)

        init.kaiming_normal_(self.pep_conv.weight)

        init.kaiming_normal_(self.dense1.weight)
        init.kaiming_normal_(self.dense_out.weight)

    def forward(self, x):
        # Initial dropout
        x = self.dropout(x)
        # global features are the same for the whole sequence -> take first value
        if self.use_global_features:
            global_features = x[:, self.global_features, 0]

        # Get all local features divided
        local_features = x[:, self.local_features, :]
        x = self.bn_start(local_features)

        pep = local_features[:, :, self.pep]
        
        tcr_a = local_features[:, :, self.tcr_a]
        tcr_b = local_features[:, :, self.tcr_b]
        
        # Do all the convolutions
        pep_conved = torch.tanh(self.pep_conv(pep))

        tcr_a_conved = torch.tanh(self.tcr_a_conv(tcr_a))
        tcr_b_conved = torch.tanh(self.tcr_b_conv(tcr_b))

        # If we want to adjust for paddings. Do so here
        #pep_conved = self.adjust_padding_activation(pep_conved, pep)
        #tcr_a_conved = self.adjust_padding_activation(tcr_a_conved, tcr_a)
        #tcr_b_conved = self.adjust_padding_activation(tcr_b_conved, tcr_b)
        # Do all poolings
        pep_pool = self.max_pool(pep_conved)

        tcr_a_pool = self.max_pool(tcr_a_conved)
        tcr_b_pool = self.max_pool(tcr_b_conved)

        if self.return_pool != False:
            if self.return_pool == "pep":
                return self.max_pool_index(pep_conved)
            elif self.return_pool == "tcra":
                return self.max_pool_index(tcr_a_conved)
            elif self.return_pool == "tcrb":
                return self.max_pool_index(tcr_b_conved)
            else:
                print("Unknown type")
                return
        ############################################

        # Combine convolutions
        x = torch.cat((pep_pool,tcr_a_pool, tcr_b_pool), dim=1)
        x = torch.flatten(x, 1)

        if self.use_global_features:
            x = torch.cat((x, global_features), dim=1) # add global features
        
        x = self.bn_dense(x)
        x = self.dropout(x)
        # Dense
        x = torch.tanh(self.dense1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.dense_out(x))
        return x

    def adjust_padding_activation(self, x_conv, x):
        """
        Takes a batch of arrays, and subtracts 2 from any convoluted position, which contains padding.
        This is to ensure that the following maxpooling selects a convoluted position containing full information.
        Returns a batch of adjusted arrays
        Only works in simple cases (stride = 1, dilation = 1, kernel_size = 3)
        """
        adjust_tensor = torch.zeros(x_conv.shape)
        for i, arr in enumerate(x): # loop through batch
            
            for j in reversed(range(arr.shape[1])): # loop reverse through seq length
                pos = arr[:, j]
                if not torch.all(pos == 0):
                    if j == arr.shape[1] - 1:
                        last_pos = None
                    else:
                        last_pos = j
                    break

            # adjust any intial and end padding made by CNN
            if self.cnn_padding > 0:
                adjust_tensor[i, :, :self.cnn_padding] = -2
                adjust_tensor[i, :, -self.cnn_padding:] = -2

            if last_pos is not None:
                adjust_tensor[i, :, last_pos - self.cnn_padding + 1:] = -2
        return x_conv + adjust_tensor









if __name__ == "__main__":
    # Only for testing purposes
    import torch
    import numpy as np
    from torch import nn, optim, cuda
    from torch.utils.data.dataloader import DataLoader
    from utils import Runner, EarlyStopping, TcrDataset, setup_seed
    from cnn_network import SimpleCNN

    # General parameters
    data_dir = "data/"
    data_files = [data_dir+f"datasets/P{i}_input_cdrs.npz" for i in range(1,6)]
    label_files = [data_dir+f"datasets/P{i}_labels.npz" for i in range(1,6)]
    model_name = "cnn_model.pt"
    model_path = "stored_models/"

    batch_size = 64
    seed= 123

    setup_seed(seed)
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    #device= torch.device("cpu")
    idx = np.arange(179,248)
    local_features = np.arange(20)
    global_features = None
    use_global_features = False

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
    train_data.slice_data(idx)
    val_data.slice_data(idx)
    test_data.slice_data(idx)

    train_data.to_blossum("data/blosum/blosum.pkl")
    val_data.to_blossum("data/blosum/blosum.pkl")
    test_data.to_blossum("data/blosum/blosum.pkl")

    input_len = train_data.data.shape[2]

    train_dl = DataLoader(train_data, batch_size)
    val_dl = DataLoader(val_data, batch_size)
    test_dl = DataLoader(test_data, batch_size)

    epochs = 100
    patience = 20
    lr = 0.005
    loss_weight = sum(train_data.labels) / len(train_data.labels)
    weight_decay = 0.0005

    # Layer parameters
    cnn_channels = 30
    hidden_neurons = 64
    dropout = 0.4
    cnn_kernel = 3

    # Loss and optimizer
    criterion = nn.BCELoss(reduction='none')
    stopper = EarlyStopping(patience, model_name, model_path)

    net = SimpleCNN(local_features, global_features, use_global_features, cnn_channels=cnn_channels, dropout=dropout, cnn_kernel_size=cnn_kernel, dense_neurons=hidden_neurons)
    net.to(device)
    print(net)
    print("Using Device:", device)
    optimizer = optim.Adam(net.parameters(), lr=lr,
        weight_decay=weight_decay,
        amsgrad=True
    )
    train_runner = Runner(train_dl, net, criterion, loss_weight, device, optimizer)
    val_runner = Runner(val_dl, net, criterion, loss_weight, device)
    test_runner = Runner(test_dl, net, criterion, loss_weight, device)

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
