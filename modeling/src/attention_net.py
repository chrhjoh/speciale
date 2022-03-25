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

    def forward(self, x : torch.Tensor):
        # x (batch, input_dim)
        hidden = self.to_hidden(x)
        # hidden (batch, hidden)
        context = self.context_vec(hidden).squeeze()
        # context (batch)
        return context

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, input_len, hidden_dim=24):
        super(AttentionLayer, self).__init__()  
        self.input_len = input_len
        self.context = ContextLayer(input_dim, hidden_dim)


    def forward(self, x, return_attention=False):
        # x (batch, seq_len, hidden_dim)
        contexts = []
        for i in range(self.input_len):
            contexts.append(self.context(x[:,i,:]))

        contexts = torch.stack(contexts).transpose(0, 1)
        attention = F.softmax(contexts,1)
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

        if all_cdrs:
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
        else:
            self.dense_in = nn.Linear(in_features=hidden_lstm*3*2,
                                      out_features=hidden_dense)

        self.dense_out = nn.Linear(in_features=hidden_dense,
                                   out_features=1)
    def _forward_seq_feat(self, 
                         x: torch.Tensor,
                         idxs: np.ndarray,
                         lstm_layer: nn.Module,
                         attention_layer_f: nn.Module,
                         attention_layer_r: nn.Module):
        """
        Takes a matrix and indexes to extract a specific sequence feature and runs it through the layers given
        """

        sequence = x[:,idxs,:]
       
        lstm_out, _ = lstm_layer(sequence)
        lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_lstm)
        # lstm_out (batch_size, seq_len, direction, hidden_dim)

        avg_hidden_rep_f = attention_layer_f(lstm_out[:,:,0,:])
        avg_hidden_rep_r = attention_layer_r(lstm_out[:,:,1,:])

        return avg_hidden_rep_f, avg_hidden_rep_r

    def forward(self, x):
        # x dimensions (batch_size, seq_encoding, seq_len) -> transpose -> (batch_size, seq_len, seq_encoding)
        x = torch.transpose(x, 1, 2)
        hidden_pep_f, hidden_pep_r = self._forward_seq_feat(x, 
                                                            self.pep_index, 
                                                            self.pep_lstm, 
                                                            self.pep_attention_f, 
                                                            self.pep_attention_r)
        hidden_cdr3a_f, hidden_cdr3a_r = self._forward_seq_feat(x, 
                                                                self.cdr3a_index, 
                                                                self.cdr3a_lstm, 
                                                                self.cdr3a_attention_f, 
                                                                self.cdr3a_attention_r)
        hidden_cdr3b_f, hidden_cdr3b_r = self._forward_seq_feat(x, 
                                                                self.cdr3b_index, 
                                                                self.cdr3b_lstm, 
                                                                self.cdr3b_attention_f, 
                                                                self.cdr3b_attention_r)

        if self.all_cdrs:
            hidden_cdr1a_f, hidden_cdr1a_r = self._forward_seq_feat(x, 
                                                                    self.cdr1a_index, 
                                                                    self.cdr1a_lstm, 
                                                                    self.cdr1a_attention_f, 
                                                                    self.cdr1a_attention_r)
            hidden_cdr2a_f, hidden_cdr2a_r = self._forward_seq_feat(x, 
                                                                    self.cdr2a_index, 
                                                                    self.cdr2a_lstm, 
                                                                    self.cdr2a_attention_f, 
                                                                    self.cdr2a_attention_r)
            hidden_cdr1b_f, hidden_cdr1b_r = self._forward_seq_feat(x, 
                                                                    self.cdr1b_index, 
                                                                    self.cdr1b_lstm, 
                                                                    self.cdr1b_attention_f, 
                                                                    self.cdr1b_attention_r)
            hidden_cdr2b_f, hidden_cdr2b_r = self._forward_seq_feat(x, 
                                                                    self.cdr2b_index, 
                                                                    self.cdr2b_lstm, 
                                                                    self.cdr2b_attention_f, 
                                                                    self.cdr2b_attention_r)
            

            dense_input = torch.cat([hidden_pep_f, hidden_cdr3a_f, hidden_cdr3b_f,
                                     hidden_pep_r, hidden_cdr3a_r, hidden_cdr3b_r,
                                     hidden_cdr1a_f, hidden_cdr2a_f, hidden_cdr1b_f, hidden_cdr2b_f,
                                     hidden_cdr1a_r, hidden_cdr2a_r, hidden_cdr1b_r, hidden_cdr2b_r], 1)
            
        else:

            dense_input = torch.cat([hidden_pep_f, hidden_cdr3a_f, hidden_cdr3b_f,
                                     hidden_pep_r, hidden_cdr3a_r, hidden_cdr3b_r], 1)

        hidden_dense = torch.relu(self.dense_in(dense_input))
        output = torch.sigmoid(self.dense_out(hidden_dense))
        return output


class LSTMNet(nn.Module):
    def __init__(self, hidden_lstm=24, hidden_dense=64):

        super(LSTMNet, self).__init__()  
        self.hidden_lstm = hidden_lstm
        # Peptide indexes
        self.pep_index = np.arange(9)
        self.cdra_index = np.arange(24, 39)
        self.cdrb_index = np.arange(52, 69)

        self.pep_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.cdra_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)

        self.cdrb_lstm = nn.LSTM(input_size=20,
                                 hidden_size=hidden_lstm,
                                 batch_first=True,
                                 bidirectional=True)
        # (3 bidirectional LSTMs are concattenated 3*2*hidden)
        self.dense_in = nn.Linear(in_features=3 * hidden_lstm * 2,
                                  out_features=hidden_dense)
        self.dense_out = nn.Linear(in_features=hidden_dense,
                                   out_features=1)

    def forward(self, x):
        # x dimensions (batch_size, seq_encoding, seq_len) -> transpose -> (batch_size, seq_len, seq_encoding)
        x = torch.transpose(x, 1, 2)
        peptide = x[:,self.pep_index,:]
        cdra = x[:,self.cdra_index,:]
        cdrb = x[:,self.cdrb_index,:]

        # run lstms
        _ , (lstm_pep, _) = self.pep_lstm(peptide)
        _ , (lstm_cdra, _) = self.cdra_lstm(cdra)
        _ , (lstm_cdrb, _) = self.cdrb_lstm(cdrb)
       
        # stack output
        stacked_lstm = torch.cat([torch.transpose(x, 0, 1).flatten(1) for x in [lstm_pep, lstm_cdra, lstm_cdrb]], -1)

        # Dense layers to output
        x = torch.relu(self.dense_in(stacked_lstm))
        x = torch.sigmoid(self.dense_out(x))
        return x
