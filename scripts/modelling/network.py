from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import init

class Net(nn.Module):
    def __init__(self,
                local_features,
                global_features,
                use_global_features,
                cnn_one_channel = 100,
                cnn_two_channel = 100,
                cnn_kernel_size = 3,
                cnn_stride = 2,
                cnn_padding = 1,
                pool_kernel = 2,
                pool_stride = 2,
                rnn_hidden = 26,
                dropout = 0.3,
                dense_one = 60,
                dense_two = 60):

        super(Net, self).__init__()  

        # Assign which features to use
        self.local_features = local_features
        self.n_local_feat = len(local_features)
        if use_global_features: 
            self.global_features = global_features
            self.n_global_feat = len(global_features)
            
        self.use_global_features = use_global_features
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels  = self.n_local_feat,
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
        if self.use_global_features:
            self.dense1 = nn.Linear(rnn_hidden*2 + self.n_global_feat, dense_one)
        else:
            self.dense1 = nn.Linear(rnn_hidden*2, dense_one)
        self.dense2 = nn.Linear(dense_one, dense_two)
        self.dense3 = nn.Linear(dense_two, 1)
        

        # Batch normalization
        self.bn_start = nn.BatchNorm1d(self.n_local_feat)
        self.bn_cnn1 = nn.BatchNorm1d(cnn_one_channel)
        self.bn_cnn2 = nn.BatchNorm1d(cnn_two_channel)

        if self.use_global_features:
            self.bn_dense = nn.BatchNorm1d(rnn_hidden*2 + self.n_global_feat)
        else:
            self.bn_dense = nn.BatchNorm1d(rnn_hidden*2)

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
        if self.use_global_features:
            global_features = x[:, self.global_features, 0]

        local_features = x[:, self.local_features, :]
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

        if self.use_global_features:
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



class Test(nn.Module):
    def __init__(self,
                local_features,
                global_features,
                use_global_features,
                cnn_one_channel = 100,
                cnn_two_channel = 100,
                cnn_kernel_size = 3,
                cnn_stride = 2,
                cnn_padding = 1,
                pool_kernel = 2,
                pool_stride = 2,
                rnn_hidden = 50,
                dropout = 0.5,
                dense_one = 60,
                dense_two = 60):

        super(Test, self).__init__()  

        # Assign which features to use
        self.local_features = local_features
        self.n_local_feat = len(local_features)
        if use_global_features: 
            self.global_features = global_features
            self.n_global_feat = len(global_features)
            
        self.use_global_features = use_global_features
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels  = self.n_local_feat,
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

        # self.rnn2 = nn.LSTM(input_size   = rnn_hidden*2,
        #                     hidden_size   = rnn_hidden, 
        #                     batch_first   = True, 
        #                     bidirectional = True)

        # self.rnn3 = nn.LSTM(input_size   = rnn_hidden*2,
        #                     hidden_size   = rnn_hidden, 
        #                     batch_first   = True, 
        #                     bidirectional = True)

        # Dense layers
        if self.use_global_features:
            self.dense1 = nn.Linear(rnn_hidden*2 + self.n_global_feat, dense_one)
        else:
            self.dense1 = nn.Linear(rnn_hidden*2, dense_one)
        self.dense2 = nn.Linear(dense_one, dense_two)
        self.dense3 = nn.Linear(dense_two, 1)
        

        # Batch normalization
        self.bn_start = nn.BatchNorm1d(self.n_local_feat)
        self.bn_cnn1 = nn.BatchNorm1d(cnn_one_channel)
        self.bn_cnn2 = nn.BatchNorm1d(cnn_two_channel)

        if self.use_global_features:
            self.bn_dense = nn.BatchNorm1d(rnn_hidden*2 + self.n_global_feat)
        else:
            self.bn_dense = nn.BatchNorm1d(rnn_hidden*2)

        # Dropout
        self.drop = nn.Dropout(p = dropout)
        self.heavy_drop = nn.Dropout(0.7)

        
        # Init weights
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.conv2.weight)
        init.kaiming_uniform_(self.dense1.weight)
        init.kaiming_uniform_(self.dense2.weight)
        init.kaiming_uniform_(self.dense3.weight)
         
    def forward(self, x):
        # global features are the same for the whole sequence -> take first value
        if self.use_global_features:
            global_features = x[:, self.global_features, 0]

        local_features = x[:, self.local_features, :]
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
        #x = self.drop(x)
        #x, (h, c) = self.rnn2(x)
        #x = self.drop(x)
        #x, (h, c) = self.rnn3(x)
        ###################

        # combine outputs
        x = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)  # concatenate bidirectional output of last layer

        if self.use_global_features:
            x = torch.cat((x, global_features), dim=1) # add global features
        
        # Dense
        x = self.heavy_drop(x)
        x = self.bn_dense(x)
        x = F.relu(self.dense1(x))
        x = self.heavy_drop(x)
        x = F.relu(self.dense2(x))
        x = self.heavy_drop(x)
        x = torch.sigmoid(self.dense3(x))
        return x

class SimpleCNN(nn.Module):
    def __init__(self,
                local_features,
                global_features,
                use_global_features,
                input_length,
                cnn_one_channel = 100,
                cnn_kernel_size = 5,
                cnn_stride = 1,
                cnn_padding = 2,
                dense_neurons = 1,
                dropout = 0.):

        super(SimpleCNN, self).__init__()  

        # Assign which features to use
        self.input_length = input_length
        self.local_features = local_features
        self.n_local_feat = len(local_features)
        if use_global_features: 
            self.global_features = global_features
            self.n_global_feat = len(global_features)
            
        self.use_global_features = use_global_features
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels  = self.n_local_feat,
                               out_channels = cnn_one_channel,
                               kernel_size  = cnn_kernel_size,
                               stride       = cnn_stride,
                               padding      = cnn_padding)

        self.conv1_len = int(((self.input_length -  cnn_kernel_size +2 *cnn_padding)/cnn_stride)+1)
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=self.conv1_len,
                                 stride=1)

        # Dense layers
        if self.use_global_features:
            self.dense1 = nn.Linear(cnn_one_channel + self.n_global_feat, dense_neurons)
        else:
            self.dense1 = nn.Linear(cnn_one_channel, dense_neurons)       

        # Batch normalization
        self.bn_start = nn.BatchNorm1d(self.n_local_feat)
        self.bn_cnn1 = nn.BatchNorm1d(cnn_one_channel)

        if self.use_global_features:
            self.bn_dense = nn.BatchNorm1d(cnn_one_channel + self.n_global_feat)
        else:
            self.bn_dense = nn.BatchNorm1d(cnn_one_channel)

        # Dropout
        self.drop = nn.Dropout(p = dropout)

        
        # Init weights
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.dense1.weight)


    def forward(self, x):
        # global features are the same for the whole sequence -> take first value
        if self.use_global_features:
            global_features = x[:, self.global_features, 0]

        local_features = x[:, self.local_features, :]
        x = self.bn_start(local_features)

        # Convolutional 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.bn_cnn1(x)
        x = self.drop(x)

        if self.use_global_features:
            x = torch.cat((x, global_features), dim=1) # add global features
        
        # Dense
        x = self.drop(x)
        x = self.bn_dense(x)
        x = torch.sigmoid(self.dense1(x))
        return x
