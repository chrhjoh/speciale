from torch import nn
import torch
import numpy as np
from torch.nn import init
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

