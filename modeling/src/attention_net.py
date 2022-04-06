import pandas as pd
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

class ContextLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=24):
        super(ContextLayer, self).__init__()  
        self.to_hidden = nn.Linear(input_dim, hidden_dim)
        self.context_vec = nn.Linear(hidden_dim, 1, bias=False)

        init.kaiming_uniform_(self.to_hidden.weight)
        init.kaiming_uniform_(self.context_vec.weight)

    def forward(self, x : torch.Tensor):
        # x (batch, input_dim)
        hidden = torch.tanh(self.to_hidden(x))
        # hidden (batch, hidden)
        context = self.context_vec(hidden).squeeze()
        # context (batch)
        return context

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=24):
        super(AttentionLayer, self).__init__()  
        self.seq_len = seq_len
        self.context = ContextLayer(input_dim, hidden_dim)

    def forward(self, x, return_attention=False):
        # x (batch, seq_len, hidden_dim)
        contexts = []
        for i in range(self.seq_len):
            contexts.append(self.context(x[:,i,:]))

        contexts = torch.stack(contexts).transpose(0, 1)
        attention = F.softmax(contexts,1)
        self.attention = attention
        output = torch.bmm(x.transpose(1,2), attention.unsqueeze(2)).squeeze()
        if return_attention:
            return output, attention
        else:
            return output



class AttentionNet(nn.Module):
    def __init__(self, hidden_lstm=24, hidden_attention=24, hidden_dense=64, all_cdrs=False):
        super(AttentionNet, self).__init__()  
        self.hidden_lstm = hidden_lstm
        self.all_cdrs = all_cdrs

        self.pep_index = np.arange(9)
        self.pep_len = len(self.pep_index)

        self.cdr3a_index = np.arange(24, 39)
        self.cdr3a_len = len(self.cdr3a_index)

        self.cdr3b_index = np.arange(52, 69)
        self.cdr3b_len = len(self.cdr3b_index)

        self.tcra_index = np.arange(9, 125)
        self.tcra_len = len(self.tcra_index)

        self.tcrb_index = np.arange(125, 238)
        self.tcrb_len = len(self.tcrb_index)

        ########### LAYERS FOR PEP AND CDR3s #############
        self.pep_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.cdr3a_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.cdr3b_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.pep_attention_f = AttentionLayer(hidden_lstm, self.pep_len, hidden_attention)
        self.cdr3a_attention_f = AttentionLayer(hidden_lstm, self.cdr3a_len, hidden_attention)
        self.cdr3b_attention_f = AttentionLayer(hidden_lstm, self.cdr3b_len, hidden_attention)
        self.pep_attention_r = AttentionLayer(hidden_lstm, self.pep_len, hidden_attention)
        self.cdr3a_attention_r = AttentionLayer(hidden_lstm, self.cdr3a_len, hidden_attention)
        self.cdr3b_attention_r = AttentionLayer(hidden_lstm, self.cdr3b_len, hidden_attention)

        ######### LAYERS FOR TCR TEST ###########
        self.tcra_attention_f = AttentionLayer(hidden_lstm, self.tcra_len, hidden_attention)
        self.tcrb_attention_f = AttentionLayer(hidden_lstm, self.tcrb_len, hidden_attention)
        self.tcra_attention_r = AttentionLayer(hidden_lstm, self.tcra_len, hidden_attention)
        self.tcrb_attention_r = AttentionLayer(hidden_lstm, self.tcrb_len, hidden_attention)
        
        if not all_cdrs:
            self.dense_in = nn.Linear(in_features=hidden_lstm*3*2,
                                      out_features=hidden_dense)
        ########### LAYERS FOR OTHER 4 CDRS ##############
        else:
            self.cdr1a_index = np.arange(9, 16)
            self.cdr2a_index = np.arange(16, 24)
            self.cdr1b_index = np.arange(39, 45)
            self.cdr2b_index = np.arange(45, 52)

            self.cdr1a_len = len(self.cdr1a_index)
            self.cdr2a_len = len(self.cdr2a_index)
            self.cdr1b_len = len(self.cdr1b_index)
            self.cdr2b_len = len(self.cdr2b_index)

            self.cdr1a_lstm = nn.LSTM(input_size=20,
                                    hidden_size=hidden_lstm,
                                    batch_first=True,
                                    bidirectional=True)

            self.cdr2a_lstm = nn.LSTM(input_size=20,
                                    hidden_size=hidden_lstm,
                                    batch_first=True,
                                    bidirectional=True)

            self.cdr1b_lstm = nn.LSTM(input_size=20,
                                    hidden_size=hidden_lstm,
                                    batch_first=True,
                                    bidirectional=True)

            self.cdr2b_lstm = nn.LSTM(input_size=20,
                                    hidden_size=hidden_lstm,
                                    batch_first=True,
                                    bidirectional=True)

            self.cdr1a_attention_f = AttentionLayer(hidden_lstm, self.cdr1a_len, hidden_attention)
            self.cdr2a_attention_f = AttentionLayer(hidden_lstm, self.cdr2a_len, hidden_attention)
            self.cdr1b_attention_f = AttentionLayer(hidden_lstm, self.cdr1b_len, hidden_attention)
            self.cdr2b_attention_f = AttentionLayer(hidden_lstm, self.cdr2b_len, hidden_attention)

            self.cdr1a_attention_r = AttentionLayer(hidden_lstm, self.cdr1a_len, hidden_attention)
            self.cdr2a_attention_r = AttentionLayer(hidden_lstm, self.cdr2a_len, hidden_attention)
            self.cdr1b_attention_r = AttentionLayer(hidden_lstm, self.cdr1b_len, hidden_attention)
            self.cdr2b_attention_r = AttentionLayer(hidden_lstm, self.cdr2b_len, hidden_attention)
        

            # Number of features * number of LSTMs * 2 for bidirectional
            self.dense_in = nn.Linear(in_features=hidden_lstm*7*2,
                                      out_features=hidden_dense)


        self.dense_out = nn.Linear(in_features=hidden_dense,
                                   out_features=1)
        
        ########### REGULARIZATION ############
        self.dropout = nn.Dropout(p=0.3)
    def _forward_seq_feat(self, 
                         x: torch.Tensor,
                         idxs: np.ndarray,
                         lstm_layer: nn.Module,
                         attention_layer_f: nn.Module,
                         attention_layer_r: nn.Module,
                         return_attention=False):
        """
        Takes a matrix and indexes to extract a specific sequence feature and runs it through the layers given
        """

        sequence = x[:,idxs,:]
       
        lstm_out, _ = lstm_layer(sequence)
        lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_lstm)
        # lstm_out (batch_size, seq_len, direction, hidden_dim)

        lstm_out = self.dropout(lstm_out)
        avg_hidden_rep_f = attention_layer_f(lstm_out[:,:,0,:], return_attention)
        avg_hidden_rep_r = attention_layer_r(lstm_out[:,:,1,:], return_attention)

        return avg_hidden_rep_f, avg_hidden_rep_r

    def forward(self, x, return_attention=False):
        # x dimensions (batch_size, seq_encoding, seq_len) -> transpose -> (batch_size, seq_len, seq_encoding)
        x = torch.transpose(x, 1, 2)
        x = self.dropout(x)
        ########## FORWARD PEPTIDE AND CDR3S ##############
        hidden_pep_f, hidden_pep_r = self._forward_seq_feat(x, 
                                                            self.pep_index, 
                                                            self.pep_lstm, 
                                                            self.pep_attention_f, 
                                                            self.pep_attention_r,
                                                            return_attention=return_attention)
        hidden_cdr3a_f, hidden_cdr3a_r = self._forward_seq_feat(x, 
                                                                self.cdr3a_index,
                                                                self.cdr3a_lstm, 
                                                                self.cdr3a_attention_f,
                                                                self.cdr3a_attention_r,
                                                                return_attention=return_attention)
        hidden_cdr3b_f, hidden_cdr3b_r = self._forward_seq_feat(x, 
                                                                self.cdr3b_index,
                                                                self.cdr3b_lstm, 
                                                                self.cdr3b_attention_f,
                                                                self.cdr3b_attention_r, 
                                                                return_attention=return_attention)
        ################# FORWARD OTHER CDRS
        if self.all_cdrs:
            hidden_cdr1a_f, hidden_cdr1a_r = self._forward_seq_feat(x, 
                                                                    self.cdr1a_index, 
                                                                    self.cdr1a_lstm, 
                                                                    self.cdr1a_attention_f, 
                                                                    self.cdr1a_attention_r,
                                                                    return_attention=return_attention)
            hidden_cdr2a_f, hidden_cdr2a_r = self._forward_seq_feat(x, 
                                                                    self.cdr2a_index, 
                                                                    self.cdr2a_lstm, 
                                                                    self.cdr2a_attention_f, 
                                                                    self.cdr2a_attention_r,
                                                                    return_attention=return_attention)
            hidden_cdr1b_f, hidden_cdr1b_r = self._forward_seq_feat(x, 
                                                                    self.cdr1b_index, 
                                                                    self.cdr1b_lstm, 
                                                                    self.cdr1b_attention_f, 
                                                                    self.cdr1b_attention_r,
                                                                    return_attention=return_attention)
            hidden_cdr2b_f, hidden_cdr2b_r = self._forward_seq_feat(x, 
                                                                    self.cdr2b_index, 
                                                                    self.cdr2b_lstm, 
                                                                    self.cdr2b_attention_f, 
                                                                    self.cdr2b_attention_r,
                                                                    return_attention=return_attention)

            features = [hidden_cdr1a_f, hidden_cdr1a_r, hidden_cdr2a_f, hidden_cdr2a_r,
                        hidden_cdr3a_f, hidden_cdr3a_r, hidden_cdr1b_f, hidden_cdr1b_r,
                        hidden_cdr2b_f, hidden_cdr2b_r, hidden_cdr3b_f, hidden_cdr3b_r,
                        hidden_pep_f, hidden_pep_r]
        else:
            features = [hidden_cdr3a_f, hidden_cdr3a_r, hidden_cdr3b_f, 
                        hidden_cdr3b_r, hidden_pep_f, hidden_pep_r]

        if return_attention:
               return self.clean_attention_output(features)

        dense_input = torch.cat(features, 1)

        dense_input = self.dropout(dense_input)
        hidden_dense = torch.relu(self.dense_in(dense_input))
        hidden_dense = self.dropout(hidden_dense)
        output = torch.sigmoid(self.dense_out(hidden_dense))
        return output

    def clean_attention_output(self, features):
        # features : List[Tuple[Tensor]]
        # List indicates which feature (ie. cdr3b_forward)
        # Tuple indicates whether attention or output
        # Tensor contains either the weighted hidden state (output) (batch, hidden_dim) 
        # or attention (batch, seq_len)

        if len(features) == 14: # if using all cdrs
            names = ["cdr1a_forward", "cdr1a_reverse", "cdr2a_forward", "cdr2a_reverse",
                     "cdr3a_forward", "cdr3a_reverse", "cdr1b_forward", "cdr1b_reverse",
                     "cdr2b_forward", "cdr2b_reverse", "cdr3b_forward", "cdr3b_reverse",
                     "pep_forward", "pep_reverse"]

        else:   # if using only cdr3s
            names = ["cdr3a_forward", "cdr3a_reverse", "cdr3b_forward", "cdr3b_reverse",
                     "pep_forward", "pep_reverse"]


        hidden_dim = features[0][0].shape[1]
        lengths = [feature[1].shape[1] for feature in features]

        output = pd.DataFrame(features[0][0].detach().numpy(),
                              columns=[f"{names[0]}_{i}" for i in range(1, hidden_dim + 1)])

        attention = pd.DataFrame(features[0][1].detach().numpy(),
                                 columns=[f"{names[0]}_{i}" for i in range(1, lengths[0] + 1)])

        for i, (name, feature) in enumerate(zip(names[1:], features[1:]),1):
            tmp_out = pd.DataFrame(feature[0].detach().numpy(),
                                  columns=[f"{name}_{j}" for j in range(1, hidden_dim + 1)])
            
            tmp_att = pd.DataFrame(feature[1].detach().numpy(),
                                  columns=[f"{name}_{j}" for j in range(1, lengths[i] + 1)])

            output = pd.concat([output, tmp_out], axis=1)
            attention = pd.concat([attention, tmp_att], axis=1)
        return output, attention
    

class LSTMNet(nn.Module):
    def __init__(self, hidden_lstm=24, hidden_dense=64, all_cdrs=False):
        super(LSTMNet, self).__init__()  
        self.all_cdrs = all_cdrs

        self.pep_index = np.arange(9)
        self.cdr3a_index = np.arange(24, 39)
        self.cdr3b_index = np.arange(52, 69)

        self.tcra_index = np.arange(9, 125)
        self.tcrb_index = np.arange(125, 238)

        ########### LAYERS FOR PEP AND CDR3s #############
        self.pep_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.cdr3a_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.cdr3b_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)
        if not all_cdrs:
            self.dense_in = nn.Linear(in_features=hidden_lstm*3*2,
                                      out_features=hidden_dense)
        ########### LAYERS FOR OTHER 4 CDRS ##############
        else:
            self.cdr1a_index = np.arange(9, 16)
            self.cdr2a_index = np.arange(16, 24)
            self.cdr1b_index = np.arange(39, 45)
            self.cdr2b_index = np.arange(45, 52)

            self.cdr1a_lstm = nn.LSTM(input_size=20,
                                    hidden_size=hidden_lstm,
                                    batch_first=True,
                                    bidirectional=True)

            self.cdr2a_lstm = nn.LSTM(input_size=20,
                                    hidden_size=hidden_lstm,
                                    batch_first=True,
                                    bidirectional=True)

            self.cdr1b_lstm = nn.LSTM(input_size=20,
                                    hidden_size=hidden_lstm,
                                    batch_first=True,
                                    bidirectional=True)

            self.cdr2b_lstm = nn.LSTM(input_size=20,
                                    hidden_size=hidden_lstm,
                                    batch_first=True,
                                    bidirectional=True)

            # Number of features * number of LSTMs * 2 for bidirectional
            self.dense_in = nn.Linear(in_features=hidden_lstm*7*2,
                                      out_features=hidden_dense)


        self.dense_out = nn.Linear(in_features=hidden_dense,
                                   out_features=1)


    def _forward_seq(self, x, idx, lstm):
        x = x[:,idx,:]
        _ , (x, _) = lstm(x)
        return x

    def forward(self, x):
        # x dimensions (batch_size, seq_encoding, seq_len) -> transpose -> (batch_size, seq_len, seq_encoding)
        x = torch.transpose(x, 1, 2)

        ###### RUN LSTMS #########
        lstm_pep = self._forward_seq(x, self.pep_index, self.pep_lstm)
        lstm_cdr3a = self._forward_seq(x, self.cdr3a_index, self.cdr3a_lstm)
        lstm_cdr3b = self._forward_seq(x, self.cdr3b_index, self.cdr3b_lstm)
       
            # stack output
        if self.all_cdrs:
            lstm_cdr1a = self._forward_seq(x, self.cdr1a_index, self.cdr1a_lstm)
            lstm_cdr2a = self._forward_seq(x, self.cdr2a_index, self.cdr2a_lstm)
            lstm_cdr1b = self._forward_seq(x, self.cdr1b_index, self.cdr1b_lstm)
            lstm_cdr2b = self._forward_seq(x, self.cdr2b_index, self.cdr2b_lstm)

            stacked_lstm = torch.cat([torch.transpose(x, 0, 1).flatten(1) for x in [lstm_pep, lstm_cdr1a, lstm_cdr2a,
                                                                                    lstm_cdr3a, lstm_cdr1b, lstm_cdr2b,
                                                                                    lstm_cdr3b,]], -1)
        else:
            stacked_lstm = torch.cat([torch.transpose(x, 0, 1).flatten(1) for x in [lstm_pep, lstm_cdr3a, lstm_cdr3b]], -1)
        

        # Dense layers to output
        x = torch.relu(self.dense_in(stacked_lstm))
        x = torch.sigmoid(self.dense_out(x))
        return x

class EmbedAttentionNet(nn.Module):
    def __init__(self, hidden_lstm=24, hidden_attention=24, hidden_dense=64):
        super(EmbedAttentionNet, self).__init__()  
        self.hidden_lstm = hidden_lstm

        self.pep_index = np.arange(13)
        self.pep_len = len(self.pep_index)
        self.pep_feat_dim= 20

        self.cdr_feat_dim= 100
        self.cdr3a_index = np.arange(13, 33)
        self.cdr3a_len = len(self.cdr3a_index)

        self.cdr3b_index = np.arange(33, 53)
        self.cdr3b_len = len(self.cdr3b_index)

        self.tcra_index = np.arange(9, 125)
        self.tcra_len = len(self.tcra_index)

        self.tcrb_index = np.arange(125, 238)
        self.tcrb_len = len(self.tcrb_index)

        ########### LAYERS FOR PEP AND CDR3s #############
        self.pep_lstm = nn.LSTM(input_size=self.pep_feat_dim,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.cdr3a_lstm = nn.LSTM(input_size=self.cdr_feat_dim,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.cdr3b_lstm = nn.LSTM(input_size=self.cdr_feat_dim,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.pep_attention_f = AttentionLayer(hidden_lstm, self.pep_len, hidden_attention)
        self.cdr3a_attention_f = AttentionLayer(hidden_lstm, self.cdr3a_len, hidden_attention)
        self.cdr3b_attention_f = AttentionLayer(hidden_lstm, self.cdr3b_len, hidden_attention)
        self.pep_attention_r = AttentionLayer(hidden_lstm, self.pep_len, hidden_attention)
        self.cdr3a_attention_r = AttentionLayer(hidden_lstm, self.cdr3a_len, hidden_attention)
        self.cdr3b_attention_r = AttentionLayer(hidden_lstm, self.cdr3b_len, hidden_attention)

        ######### LAYERS FOR TCR TEST ###########
        self.tcra_attention_f = AttentionLayer(hidden_lstm, self.tcra_len, hidden_attention)
        self.tcrb_attention_f = AttentionLayer(hidden_lstm, self.tcrb_len, hidden_attention)
        self.tcra_attention_r = AttentionLayer(hidden_lstm, self.tcra_len, hidden_attention)
        self.tcrb_attention_r = AttentionLayer(hidden_lstm, self.tcrb_len, hidden_attention)
        
        self.dense_in = nn.Linear(in_features=hidden_lstm*3*2,
                                  out_features=hidden_dense)


        self.dense_out = nn.Linear(in_features=hidden_dense,
                                   out_features=1)
        
        ########### REGULARIZATION ############
        self.dropout = nn.Dropout(p=0.4)
    def _forward_seq_feat(self, 
                         x: torch.Tensor,
                         idxs: np.ndarray,
                         embed_features: int,
                         lstm_layer: nn.Module,
                         attention_layer_f: nn.Module,
                         attention_layer_r: nn.Module,
                         return_attention=False):
        """
        Takes a matrix and indexes to extract a specific sequence feature and runs it through the layers given
        """

        embedding = x[:, idxs, :embed_features]
        embedding = self.dropout(embedding)
        lstm_out, _ = lstm_layer(embedding)
        lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_lstm)
        # lstm_out (batch_size, seq_len, direction, hidden_dim)

        lstm_out = self.dropout(lstm_out)
        avg_hidden_rep_f = attention_layer_f(lstm_out[:,:,0,:], return_attention)
        avg_hidden_rep_r = attention_layer_r(lstm_out[:,:,1,:], return_attention)

        return avg_hidden_rep_f, avg_hidden_rep_r

    def forward(self, x, return_attention=False):
        x = torch.transpose(x, 1, 2)
         # x dimensions (batch_size, seq_len, embedding)
        ########## FORWARD PEPTIDE AND CDR3S ##############
        hidden_pep_f, hidden_pep_r = self._forward_seq_feat(x, 
                                                            self.pep_index,
                                                            self.pep_feat_dim, 
                                                            self.pep_lstm, 
                                                            self.pep_attention_f, 
                                                            self.pep_attention_r,
                                                            return_attention=return_attention)
        hidden_cdr3a_f, hidden_cdr3a_r = self._forward_seq_feat(x, 
                                                                self.cdr3a_index,
                                                                self.cdr_feat_dim,
                                                                self.cdr3a_lstm, 
                                                                self.cdr3a_attention_f,
                                                                self.cdr3a_attention_r,
                                                                return_attention=return_attention)
        hidden_cdr3b_f, hidden_cdr3b_r = self._forward_seq_feat(x, 
                                                                self.cdr3b_index,
                                                                self.cdr_feat_dim,
                                                                self.cdr3b_lstm, 
                                                                self.cdr3b_attention_f,
                                                                self.cdr3b_attention_r, 
                                                                return_attention=return_attention)
        
        features = [hidden_cdr3a_f, hidden_cdr3a_r, hidden_cdr3b_f, 
                    hidden_cdr3b_r, hidden_pep_f, hidden_pep_r]

        if return_attention:
               return self.clean_attention_output(features)

        dense_input = torch.cat(features, 1)

        dense_input = self.dropout(dense_input)
        hidden_dense = torch.relu(self.dense_in(dense_input))
        hidden_dense = self.dropout(hidden_dense)
        output = torch.sigmoid(self.dense_out(hidden_dense))
        return output

    def clean_attention_output(self, features):
        # features : List[Tuple[Tensor]]
        # List indicates which feature (ie. cdr3b_forward)
        # Tuple indicates whether attention or output
        # Tensor contains either the weighted hidden state (output) (batch, hidden_dim) 
        # or attention (batch, seq_len)

        names = ["cdr3a_forward", "cdr3a_reverse", "cdr3b_forward", "cdr3b_reverse",
                 "pep_forward", "pep_reverse"]


        hidden_dim = features[0][0].shape[1]
        lengths = [feature[1].shape[1] for feature in features]

        output = pd.DataFrame(features[0][0].detach().numpy(),
                              columns=[f"{names[0]}_{i}" for i in range(1, hidden_dim + 1)])

        attention = pd.DataFrame(features[0][1].detach().numpy(),
                                 columns=[f"{names[0]}_{i}" for i in range(1, lengths[0] + 1)])

        for i, (name, feature) in enumerate(zip(names[1:], features[1:]),1):
            tmp_out = pd.DataFrame(feature[0].detach().numpy(),
                                  columns=[f"{name}_{j}" for j in range(1, hidden_dim + 1)])
            
            tmp_att = pd.DataFrame(feature[1].detach().numpy(),
                                  columns=[f"{name}_{j}" for j in range(1, lengths[i] + 1)])

            output = pd.concat([output, tmp_out], axis=1)
            attention = pd.concat([attention, tmp_att], axis=1)
        return output, attention