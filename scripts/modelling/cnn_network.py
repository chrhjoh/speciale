import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

def maxpool_length(length, kernel_size, stride=1, padding=0, dilation=1):
    return int((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

def conv_length(length, kernel_size, stride=1, padding=0, dilation=1):
    return int((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

class SimpleCNN(nn.Module):
    def __init__(self,
                local_features,
                global_features,
                use_global_features,
                cnn_channels = 30,
                cnn_kernel_size = 3,
                cnn_stride = 1,
                cnn_padding = 1,
                dense_neurons = 64,
                dropout = 0.3):

        super(SimpleCNN, self).__init__()  

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

        # Max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Dense layers
        n_convs = 7
        if self.use_global_features:
            self.dense1 = nn.Linear(cnn_channels * n_convs + self.n_global_feat, dense_neurons)
        else:
            self.dense1 = nn.Linear(cnn_channels * n_convs, dense_neurons) 
        
        self.dense_out = nn.Linear(dense_neurons, 1)

        # Batchnorm before dense
        if self.use_global_features:
            self.bn_dense = nn.BatchNorm1d(cnn_channels * n_convs + self.n_global_feat)
        else:
            self.bn_dense = nn.BatchNorm1d(cnn_channels * n_convs) 
        


        # Init weights
        init.kaiming_uniform_(self.cdr1a_conv.weight)
        init.kaiming_uniform_(self.cdr2a_conv.weight)
        init.kaiming_uniform_(self.cdr3a_conv.weight)

        init.kaiming_uniform_(self.cdr1b_conv.weight)
        init.kaiming_uniform_(self.cdr2b_conv.weight)
        init.kaiming_uniform_(self.cdr3b_conv.weight)

        init.kaiming_uniform_(self.pep_conv.weight)

        init.kaiming_uniform_(self.dense1.weight)
        init.kaiming_uniform_(self.dense_out.weight)


    def forward(self, x):
        # global features are the same for the whole sequence -> take first value
        if self.use_global_features:
            global_features = x[:, self.global_features, 0]

        # Get all local features divided
        local_features = x[:, self.local_features, :]

        pep = local_features[:, :, self.pep]
        
        cdr1a = local_features[:, :, self.cdr1a]
        cdr2a = local_features[:, :, self.cdr2a]
        cdr3a = local_features[:, :, self.cdr3a]

        cdr1b = local_features[:, :, self.cdr1b]
        cdr2b = local_features[:, :, self.cdr2b]
        cdr3b = local_features[:, :, self.cdr3b]
        
        # Do all the convolutions and poolings

        pep_pool = self.max_pool(torch.tanh(self.cdr1a_conv(pep)))
        cdr1a_pool = self.max_pool(torch.tanh(self.cdr1a_conv(cdr1a)))
        cdr2a_pool = self.max_pool(torch.tanh(self.cdr1a_conv(cdr2a)))
        cdr3a_pool = self.max_pool(torch.tanh(self.cdr1a_conv(cdr3a)))

        cdr1b_pool = self.max_pool(torch.tanh(self.cdr1a_conv(cdr1b)))
        cdr2b_pool = self.max_pool(torch.tanh(self.cdr1a_conv(cdr2b)))
        cdr3b_pool = self.max_pool(torch.tanh(self.cdr1a_conv(cdr3b)))

        # Combine convolutions
        x = torch.cat((pep_pool, cdr1a_pool, cdr2a_pool, cdr3a_pool, cdr1b_pool, cdr2b_pool, cdr3b_pool), dim=1)
        x = torch.flatten(x, 1)

        if self.use_global_features:
            x = torch.cat((x, global_features), dim=1) # add global features
        
        x = self.bn_dense(x)
        # Dense
        x = torch.tanh(self.dense1(x))
        x = torch.sigmoid(self.dense_out(x))
        return x
