import torch
from torch import nn
from torch.nn import init
import numpy as np
import pandas as pd

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
                dropout = 0.3,
                use_all_cdrs=True):

        super(CdrCNN, self).__init__()  

        # Define which features to use for CNN
        self.use_all_cdrs = use_all_cdrs
        self.local_features = local_features
        self.n_local_feat = len(local_features)

        self.use_global_features = use_global_features
        if use_global_features: 
            self.global_features = global_features
            self.n_global_feat = len(global_features)
                
        if cnn_padding == "same":
            self.cnn_padding = int((cnn_kernel_size - 1) / 2)
        else:
            self.cnn_padding = cnn_padding
    
        ########## LAYERS FOR PEPTIDE AND CDR3s ###########
        self.pep = np.arange(9)
        self.cdr3a = np.arange(24, 39)
        self.cdr3b = np.arange(52, 69)

        # CDR3a
        self.cdr3a_conv = nn.Conv1d(in_channels  = self.n_local_feat,
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
        
        ############# LAYERS FOR OTHER CDRS ###############
        if use_all_cdrs:
            self.cdr1a = np.arange(9, 16)
            self.cdr2a = np.arange(16, 24)
            self.cdr1b = np.arange(39, 45)
            self.cdr2b = np.arange(45, 52)

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
            n_convs = 7
        else:
            n_convs = 3

        ############ DENSE LAYERS #############
        if self.use_global_features:
            self.dense1 = nn.Linear(cnn_channels * n_convs + self.n_global_feat, dense_neurons)
        else:
            self.dense1 = nn.Linear(cnn_channels * n_convs, dense_neurons)
        self.dense_out = nn.Linear(dense_neurons, 1)


        ############ REGULARIZATION ############
        # Max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.max_pool_index = nn.AdaptiveMaxPool1d(1, return_indices=True)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Batchnorm before dense
        if self.use_global_features:
            self.bn_dense = nn.BatchNorm1d(cnn_channels * n_convs + self.n_global_feat)
        else:
            self.bn_dense = nn.BatchNorm1d(cnn_channels * n_convs) 
        
        self.bn_start = nn.BatchNorm1d(self.n_local_feat)

        # Init weights
        init.kaiming_uniform_(self.cdr3a_conv.weight)
        init.kaiming_uniform_(self.cdr3b_conv.weight)
        init.kaiming_uniform_(self.pep_conv.weight)
        init.kaiming_uniform_(self.dense1.weight)
        init.kaiming_uniform_(self.dense_out.weight)
        if use_all_cdrs:
            init.kaiming_uniform_(self.cdr1a_conv.weight)
            init.kaiming_uniform_(self.cdr2a_conv.weight)
            init.kaiming_uniform_(self.cdr1b_conv.weight)
            init.kaiming_uniform_(self.cdr2b_conv.weight)

    def _forward_seq_feat(self, x, idxs, conv_layer, pooler):
        """
        Runs a sequence feature described by idxs through conv and pooler
        """
        sequence = x[:, :, idxs]
        conv_out = torch.sigmoid(conv_layer(sequence))
        pooled = pooler(conv_out)
        return pooled

    def forward(self, x, return_attention=False):
        # Initial dropout
        x = self.dropout(x)
        # global features are the same for the whole sequence -> take first value
        if self.use_global_features:
            global_features = x[:, self.global_features, 0]

        # Get all local features divided
        local_features = x[:, self.local_features, :]
        x = self.bn_start(local_features)

        # Decide what pooler to use
        max_pooler = self.max_pool if not return_attention else self.max_pool_index

        # Do all convolutions and maxpools
        pep_pooled = self._forward_seq_feat(x, self.pep, self.pep_conv, max_pooler)
        cdr3a_pooled = self._forward_seq_feat(x, self.cdr3a, self.cdr3a_conv, max_pooler)
        cdr3b_pooled = self._forward_seq_feat(x, self.cdr3b, self.cdr3b_conv, max_pooler)

        if self.use_all_cdrs:
            cdr1a_pooled = self._forward_seq_feat(x, self.cdr1a, self.cdr1a_conv, max_pooler)
            cdr2a_pooled = self._forward_seq_feat(x, self.cdr2a, self.cdr2a_conv, max_pooler)
            cdr1b_pooled = self._forward_seq_feat(x, self.cdr1b, self.cdr1b_conv, max_pooler)
            cdr2b_pooled = self._forward_seq_feat(x, self.cdr2b, self.cdr2b_conv, max_pooler)

            features = [pep_pooled, cdr1a_pooled, cdr2a_pooled, cdr3a_pooled,
                        cdr1b_pooled, cdr2b_pooled, cdr3b_pooled]
        else:
            features = [pep_pooled, cdr3a_pooled, cdr3b_pooled]

        if return_attention:
            return self.clean_maxpool_output(features)

        # Combine poolings
        x = torch.cat(features, dim=1)
        x = torch.flatten(x, 1)

        if self.use_global_features:
            x = torch.cat((x, global_features), dim=1) # add global features
        
        x = self.bn_dense(x)
        x = self.dropout(x)
        # Dense
        x = torch.sigmoid(self.dense1(x))
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

    def clean_maxpool_output(self, features):
        # features : List[Tuple[Tensor]]
        # List indicates which feature (ie. cdr3b_forward)
        # Tuple indicates whether attention or output
        # Tensor contains either the weighted hidden state (output) (batch, hidden_dim) 
        # or attention (batch, seq_len)

        if len(features) == 7: # if using all cdrs
            names = ["pep", "cdr1a", "cdr2a",  "cdr3a",  "cdr1b",  "cdr2b", "cdr3b"]

        else:   # if using only cdr3s
            names = ["pep", "cdr3a", "cdr3b", ]

        n_filters = features[0][0].shape[1]

        maxpool_value = pd.DataFrame(features[0][0].squeeze().detach().numpy(),
                                     columns=[f"{names[0]}_{i}" for i in range(1, n_filters + 1)])

        maxpool_index = pd.DataFrame(features[0][1].squeeze().detach().numpy(),
                                     columns=[f"{names[0]}_{i}" for i in range(1, n_filters + 1)])

        for name, feature in zip(names[1:], features[1:]):
            tmp_value = pd.DataFrame(feature[0].squeeze().detach().numpy(),
                                     columns=[f"{name}_{j}" for j in range(1, n_filters + 1)])
            
            tmp_index = pd.DataFrame(feature[1].squeeze().detach().numpy(),
                                     columns=[f"{name}_{j}" for j in range(1, n_filters + 1)])

            maxpool_index = pd.concat([maxpool_index, tmp_index], axis=1)
            maxpool_value = pd.concat([maxpool_value, tmp_value], axis=1)
        return maxpool_value, maxpool_index
