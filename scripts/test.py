import torch
from torch import nn
import torch.nn.functional as F
# Hyperparameters
# input_size = 420

n_local_feat = 27
n_global_feat = 27
num_classes = 1
# learning_rate = 0.01
learning_rate = 0.001


class Net(nn.Module):
    def __init__(self,  num_classes):
        super(Net, self).__init__()   
        self.bn0 = nn.BatchNorm1d(n_local_feat)
        self.conv1 = nn.Conv1d(in_channels=n_local_feat, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(100)
        
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(100)
        
        ######## code from master thesis 
        self.rnn1 = nn.LSTM(input_size=100,hidden_size=26,num_layers=1, batch_first=True, bidirectional = True)
        self.rnn2 = nn.LSTM(input_size=26*2,hidden_size=26,num_layers=1, batch_first=True, bidirectional = True)
        self.rnn3 = nn.LSTM(input_size=26*2,hidden_size=26,num_layers=1, batch_first=True, bidirectional = True)
        self.drop = nn.Dropout(p = 0.5) # Dunno if dropout should be even higher?? - Christian
        self.fc1 = nn.Linear(26*2 + n_global_feat, 26*2 + n_global_feat)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        ########
        
        # since we add new features in this step, we have to use batch normalization again
        self.bn1 = nn.BatchNorm1d(26*2 + n_global_feat)
        # if we pipe the global terms innto the fc, we should have more than just 1
        self.fc2 = nn.Linear(26*2 + n_global_feat, 26*2 + n_global_feat)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(26*2 + n_global_feat, num_classes)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)
    def forward(self, x):
        
        local_features = x[:, :27, :]
        # global features are the same for the whole sequence -> take first value
        global_features = x[:, 27:, 0]
        ######## code from master thesis
        x = self.bn0(local_features)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.drop(x)
        x = x.transpose_(2, 1)
        x, (h, c) = self.rnn1(x)
        x = self.drop(x)
        x, (h, c) = self.rnn2(x)
        x = self.drop(x)
        x, (h, c) = self.rnn3(x)
        # concatenate bidirectional output of last layer
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        # add global features
        x = torch.cat((cat, global_features), dim=1)
        x = self.drop(x)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc3(x))
        ########
        
        return x