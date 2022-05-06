import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import os
import sys
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn import metrics
from gensim.models import KeyedVectors

class TcrDataset(Dataset):
    def __init__(self, df, partitions, seq_features, encoding_path="/Users/christianjohansen/Desktop/speciale/modeling/data/encoding/BLOSUM50"):
        self.df = df[df["partition"].isin(partitions)]
        self.labels = df.loc[df["partition"].isin(partitions), "label"].values
        self.is_reduced = False

        # Add sequence data
        print("Using sequence features:", ", ".join(seq_features))
        encoding = self.read_encoding(encoding_path)
        self.data = self.get_sequence_features(seq_features, encoding)
        
        # Add local energy terms
        local_energy = self.get_local_energies(seq_features)
        # Add global energy
        global_energy = self.get_global_energies()

        self.data = np.concatenate([self.data, local_energy, global_energy], axis=1)

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def shuffle_data(self):
        idx = np.random.permutation(self.data.shape[0])
        self.data = self.data[idx, : , :]
        self.labels = self.labels[idx]
        self.shuffle_idx = idx
        self.df = self.df.iloc[idx]

        
    def get_sequence_features(self, seq_features, encoding):
        data = []
        self.lengths = []
        for feat in seq_features:
            length = self.df[feat].str.len().max()
            encoded = self.df[feat].apply(lambda seq: self.encode(seq, encoding, length))
            data.append(np.stack(encoded.to_list()))
            self.lengths.append(length)

        return np.concatenate(data, axis=2)

    def encode(self, seq, encoding, length):
        arrs = []
        for i in range(length):
            if len(seq) > i:
                arrs.append(encoding[seq[i]])
            else:
                arrs.append(np.zeros(20))
        return np.transpose(np.stack(arrs))
    
    def get_local_energies(self, seq_features):
        energy_features = ['fa_tot', 'fa_atr', 'fa_rep', 'fa_sol', 'fa_elec', 'fa_dun', 'p_aa_pp']
        data = []
        # Find energy features, sliced to the sequence features used
        for length, seq_feat in zip(self.lengths, seq_features):
            feature_data = []
            start_idx = self.df[seq_feat+"_start"]
            stop_idx = self.df[seq_feat].str.len() + start_idx

            for energy_feat in energy_features:
                tmp_df = pd.concat([self.df[energy_feat].rename("feature"), start_idx.rename("start"), stop_idx.rename("stop")], axis=1)
                energies = tmp_df.apply(lambda x : self._slice_energies(x, length), axis=1)
                feature_data.append(np.array(energies.to_list(),dtype=float))
            data.append(np.stack(feature_data))
        data = np.concatenate(data,axis=2).transpose(1,0,2)
        return data

    def _slice_energies(self, x, length):
        """ Slices and pads energy features to the desired length"""
        data = x["feature"][x["start"]:x["stop"]]
        while len(data) < length:
            data.append(0)
        return data

    def get_global_energies(self):
        global_energy = np.zeros(shape=(self.data.shape[0], 
                                len(self.df["global_interactions"].iloc[0]), 
                                self.data.shape[2]))
        global_energy[:,:,0] = np.array(self.df["global_interactions"].tolist(),dtype=float)
        return global_energy
    
    def read_encoding(self, encoding_path):
        NORMALIZER = 5
        with open(encoding_path, "r") as fh:
            encoding = dict()
            for line in fh:
                if line.startswith("#"):
                    continue
                elif not line.startswith(" "):
                    target_aa = line.split()[0]
                    
                    scores = [int(x.strip()) / NORMALIZER for x in line.split()[1:]]
                    encoding[target_aa] = np.array(scores)
            
            return encoding


class AttentionDataset(Dataset):
    def __init__(self, df, partitions, seq_features, encode_type="encode", shuffle=True):
        self.df = df[df["partition"].isin(partitions)]
        self.labels = df.loc[df["partition"].isin(partitions), "label"].values
        self.ENCODE_PATH = os.path.join("/Users/christianjohansen/Desktop/speciale/modeling/", "data", "encoding")
        self.data = self.get_sequence_features(seq_features, encode_type)
        if shuffle:
            self.shuffle_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def shuffle_data(self):
        idx = np.random.permutation(self.data.shape[0])
        self.data = self.data[idx]
        self.labels = self.labels[idx]
        self.shuffle_idx = idx
        self.df = self.df.iloc[idx]

        
    def get_sequence_features(self, seq_features, encode_type):
        """
        Gets the sequence features with a specific encoding type
        Either make a blosum encoding, gets a simple token representation
        or get a high dimensional trained embedding from word2vec.
        """
        data = []
        self.lengths = [9, 7, 8, 15, 6, 7, 17]
        for length, feat in zip(self.lengths, seq_features):
            if encode_type == "encode":
                encoding = self.read_encoding(os.path.join(self.ENCODE_PATH, "BLOSUM50"))
                encoded = self.df[feat].apply(lambda seq: self.encode(seq, encoding, length))
                
            elif encode_type == "tokenize":
                ALL_AA = "ARNDCQEGHILKMFPSTWYV"
                if feat == "cdr3b":
                    embed_model = KeyedVectors.load(os.path.join(self.ENCODE_PATH, "embeddings_cdr3b.wv"))
                    tokenizer = embed_model.key_to_index
                elif feat == "cdr3a":
                    embed_model = KeyedVectors.load(os.path.join(self.ENCODE_PATH, "embeddings_cdr3a.wv"))
                    tokenizer = embed_model.key_to_index
                else:
                    tokenizer = dict(zip(ALL_AA, range(20)))
                encoded = self.df[feat].apply(lambda seq: self.tokenize(seq, length, tokenizer))

            else:
                print(encode_type, "not supported")
                print("Use: encode or tokenize")
                sys.exit(1)
                
            data.append(np.stack(encoded.to_list()))
            self.lengths.append(length)

        if encode_type == "tokenize":
            return np.concatenate(data, axis=1)
        else:
            return np.concatenate(data, axis=2)

    def encode(self, seq, encoding, length):
        arrs = []
        for i in range(length):
            if len(seq) > i:
                arrs.append(encoding[seq[i]])
            else:
                arrs.append(np.zeros(arrs[0].shape))
        return np.transpose(np.stack(arrs))
    
    def read_encoding(self, encoding_path):
        NORMALIZER = 5
        with open(encoding_path, "r") as fh:
            encoding = dict()
            for line in fh:
                if line.startswith("#"):
                    continue
                elif not line.startswith(" "):
                    target_aa = line.split()[0]
                    
                    scores = [int(x.strip()) / NORMALIZER for x in line.split()[1:]]
                    encoding[target_aa] = np.array(scores)
            
            return encoding
    
    def tokenize(self, seq, length, tokenizer):

        token_seq = []
        for i in range(length):
            if len(seq) > i:
                token_seq.append(tokenizer[seq[i]])
            else:
                token_seq.append(20)
        return np.array(token_seq)

class Runner:

    def __init__(self, loader: DataLoader, model: nn.Module, criterion, loss_weight, device="cpu", optimizer = None, cutoff = 0.5):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_weight = loss_weight
        self.cutoff = cutoff
        self.y_true_batches = []
        self.y_pred_batches = []
        self.y_score_batches = []
        self.loss = 0
        self.mode = "Train" if optimizer != None else "Validation"
        self.device = device

    def weighted_loss(self, pred, y):
        """
        Return average weighted loss in batch
        """
        loss = self.criterion(pred, y)
        weight = torch.abs(y - self.loss_weight)

        return torch.mean(weight * loss)
    
    def follow_performance(self, epoch):
        if self.mode == "Train" : print("Epoch:", epoch)
        print(self.mode ,"loss:", self.loss, 
             self.mode, "MCC:", self.mcc,
             self.mode, "AUC:", self.auc)
        
    def reset(self):
        self.y_pred_batches = []
        self.y_true_batches = []
        self.y_score_batches = []
        self.loss = 0

    def evaluation_mode(self):
        self.mode = "Validation"

    def run_epoch(self):
        if self.mode == "Train":
            self.model.train()
        else:
            self.model.eval()

        for x, y  in self.loader:
            x = x.float().to(self.device)
            y = y.float().unsqueeze(1).to(self.device)
            scores = self.model(x)

            # Calculate loss
            loss = self.weighted_loss(scores, y)
            #loss = torch.mean(self.criterion(scores, y))

            # Take step on gradient if training
            if self.mode == "Train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Round off predictions based on cutoff
            scores = scores.detach().to("cpu")
            pred = np.zeros(shape=scores.shape)
            pred[scores >= self.cutoff] = 1


            # Store results from batch
            self.y_true_batches += [y.detach().to("cpu").numpy()]
            self.y_pred_batches += [pred]
            self.y_score_batches += [scores.numpy()]
            self.loss += loss.detach()

        self.y_true_batches = np.vstack(self.y_true_batches)
        self.y_pred_batches = np.vstack(self.y_pred_batches)
        self.y_score_batches = np.vstack(self.y_score_batches)
        self.loss = self.loss.item() / len(self.loader) # Divide with batch numbers for average loss per data point
        
        self.auc = metrics.roc_auc_score(self.y_true_batches, self.y_score_batches)
        self.mcc = metrics.matthews_corrcoef(self.y_true_batches, self.y_pred_batches)

    def evaluate_model(self):
            # Loss
            print("Loss:", self.loss)
            # Mcc
            print("MCC:", metrics.matthews_corrcoef(self.y_true_batches, self.y_pred_batches))
            # Confusion matrix plot
            print("Confussion Matrix:\n", metrics.confusion_matrix(self.y_true_batches, self.y_pred_batches), "\n")
            # AUC curve
            self.plot_roc()
            
            
    def plot_roc(self):
        # ROC
        fpr, tpr, threshold = metrics.roc_curve(self.y_true_batches, self.y_score_batches)
        roc_auc = metrics.auc(fpr, tpr)

        # plot ROC
        plt.figure()
        plt.title("Receiver Operating Characteristic")
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
    
    def save_attention_weights(self, attention_filename, output_filename=None):
        """
        Gets the attention weights for all observations in dataset and saves it
        to csv
        """
        self.model.eval()
        outputs = []
        attentions = []
        for x, _  in self.loader:
            x = x.float().to(self.device)
            output, attention = self.model(x, return_attention=True)
            outputs.append(output)
            attentions.append(attention)

        self.combine_and_save_dfs(attentions, attention_filename)
        if output_filename is not None:
            self.combine_and_save_dfs(outputs, output_filename)

    def combine_and_save_dfs(self, output, filename):
        """
        Cleans a list of dfs, adds information about the output and then saves the resulting
        csv to filename.
        """
        output_df = pd.concat(output)
        output_df = pd.concat([output_df.set_index(self.loader.dataset.df.index), self.loader.dataset.df],axis=1)
        output_df.to_csv(filename, index=False)

    
    def scores_to_file(self, filename):
        df = self.loader.dataset.df.loc[:, ["ID", "pep", "origin", "partition"]]
        df["scores"] = self.y_score_batches
        df["labels"] = self.y_true_batches
        df.to_csv(filename,header=False, index=False, mode="a")
            


class EarlyStopping:
    def __init__(self, patience: int, delta: float = 0.003, filename = "early_stopping.pt", path : str = ""):
        self.patience = patience
        self.current_count = 0
        self.best_loss = np.inf
        self.stop = False
        self.path = path
        self.filename = filename
        self.delta = delta
    
    def evaluate_epoch(self, metric: float, model: nn.Module, epoch: int):

        if metric + self.delta < self.best_loss:
            self.best_loss = metric
            self.best_epoch = epoch
            self.current_count = 0
            torch.save(model.state_dict(), self.path + self.filename)
            print("Validation loss decreased. Counter reset")
        else:
            self.current_count += 1
            if self.current_count == self.patience:
                print("Early Stopping")
                self.stop = True
            else:
                print(f"Early Stopping Counter: {self.current_count} out of {self.patience}")
                
def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

def sample_swapped(df, ids):
    """
    Returns the swapped negatives, that were swapped from sampled positives
    """
    ids = list(ids)
    swap_idx = pd.to_numeric(df["origin"].str.split("_").str[1])
    return df[swap_idx.isin(ids)]



def downsample_peptide(df, peptide, frac):
    other_peps = df[df["pep"] != peptide]
    pos_samples = df[(df["origin"] == "positive") & (df["pep"] == peptide)].sample(frac=frac, random_state=42)
    neg_samples = df[(df["origin"] == "10x") & (df["pep"] == peptide)].sample(frac=frac, random_state=42)
    swapped_samples = df[(df["origin"].str.startswith("swapped")) & (df["pep"] == peptide)].sample(frac=frac, random_state=42)
    sampled_df = pd.concat([other_peps, pos_samples, neg_samples, swapped_samples])
    return sampled_df
    