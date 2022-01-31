from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import init

class Net(nn.Module):
    def __init__(self,
                n_local_feat,
                n_global_feat, 
                cnn_one_channel = 100,
                cnn_two_channel = 100,
                cnn_kernel_size = 3,
                cnn_stride = 2,
                cnn_padding = 1,
                pool_kernel = 2,
                pool_stride = 2,
                rnn_hidden = 26,
                dropout = 0.5,
                dense_one = 60,
                dense_two = 60):

        super(Net, self).__init__()  

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels  = n_local_feat,
                               out_channels = cnn_one_channel,
                               kernel_size  = cnn_kernel_size,
                               stride       = cnn_stride,
                               padding      = cnn_padding)
        
        self.conv2 = nn.Conv1d(in_channels  = cnn_one_channel, 
                               out_channels = cnn_two_channel,
                               kernel_size  = cnn_kernel_size,
                               stride       = cnn_stride, 
                               padding      = cnn_padding)
        

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel,
                                 stride=pool_stride)

        # LSTM layers
        self.rnn1 = nn.LSTM(input_size    = cnn_two_channel,
                            hidden_size   = rnn_hidden, 
                            batch_first   = True, 
                            bidirectional = True)

        self.rnn2 = nn.LSTM(input_size   = rnn_hidden*2,
                            hidden_size   = rnn_hidden, 
                            batch_first   = True, 
                            bidirectional = True)

        self.rnn3 = nn.LSTM(input_size   = rnn_hidden*2,
                            hidden_size   = rnn_hidden, 
                            batch_first   = True, 
                            bidirectional = True)

        # Dense layers
        self.dense1 = nn.Linear(rnn_hidden*2 + n_global_feat, dense_one)
        self.dense2 = nn.Linear(dense_one, dense_two)
        self.dense3 = nn.Linear(dense_two, 1)
        

        # Batch normalization
        self.bn_start = nn.BatchNorm1d(n_local_feat)
        self.bn_cnn1 = nn.BatchNorm1d(cnn_one_channel)
        self.bn_cnn2 = nn.BatchNorm1d(cnn_two_channel)
        self.bn_dense = nn.BatchNorm1d(rnn_hidden*2 + n_global_feat)

        # Dropout
        self.drop = nn.Dropout(p = dropout)

        
        # Init weights
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.conv2.weight)
        init.kaiming_uniform_(self.dense1.weight)
        init.kaiming_uniform_(self.dense2.weight)
        init.kaiming_uniform_(self.dense3.weight)
         
    def forward(self, x):
        # global features are the same for the whole sequence -> take first value
        global_features = x[:, 27:, 0]
        local_features = x[:, :27, :]
        x = self.bn_start(local_features)

        # Convolutional 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn_cnn1(x)
        x = self.drop(x)

        # Convolutional 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.bn_cnn2(x)
        x = self.drop(x)
        x = x.transpose_(2, 1)

        ### LSTM layers ###
        x, (h, c) = self.rnn1(x)
        x = self.drop(x)
        x, (h, c) = self.rnn2(x)
        x = self.drop(x)
        x, (h, c) = self.rnn3(x)
        ###################

        # combine outputs
        x = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)  # concatenate bidirectional output of last layer
        x = torch.cat((x, global_features), dim=1) # add global features
        
        # Dense
        x = self.drop(x)
        x = self.bn_dense(x)
        x = F.relu(self.dense1(x))
        x = self.drop(x)
        x = F.relu(self.dense2(x))
        x = self.drop(x)
        x = torch.sigmoid(self.dense3(x))
        return x