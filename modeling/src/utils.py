import numpy as np
import sys
import torch
import random
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn import metrics
import pickle


class TcrDataset(Dataset):
    def __init__(self, df, partitions):
        self.df = df[df["partition"].isin(partitions)]
        self.labels = df.loc[df["partition"].isin(partitions), "label"].values
        self.is_reduced = False
        self.data = None

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def shuffle_data(self):
        try:
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx, : , :]
            self.labels = self.labels[idx]
            self.shuffle_idx = idx
        except AttributeError:
            print("Could not shuffle, Data should be added first")
            sys.exit(1)
        
    def add_sequence_features(self, features, encoding):
        print("Using sequence features:", ", ".join(features))
        data = []
        self.lengths = []
        for feat in features:
            length = self.df[feat].str.len().max()
            encoded = self.df[feat].apply(lambda seq: self.encode(seq, encoding, length))
            data.append(np.stack(encoded.to_list()))

            if len(self.lengths) == 0:
                self.lengths.append(length)
            else:
                self.lengths.append(length + self.lengths[-1])
        self.data = np.concatenate(data, axis=2)

    def encode(self, seq, encoding, length):
        arrs = []
        for i in range(length):
            if len(seq) > i:
                arrs.append(encoding[seq[i]])
            else:
                arrs.append(np.zeros(20))
        return np.transpose(np.stack(arrs))
    
    def add_local_energies(self):
        return NotImplementedError
    
    def add_global_energies(self):
        return NotImplementedError
    
    def subset_datapoints(self, idxs):
        if not self.is_reduced:
            self.total_data = self.data.copy()
            self.total_labels = self.labels.copy()

        self.data = self.total_data[idxs]
        self.labels = self.total_labels[idxs]
        self.is_reduced = True

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
    
    def get_pool_idxs(self, feat):
        self.model.return_pool = feat
        if self.mode == "Train":
            self.model.train()
        else:
            self.model.eval()
        max_output = []
        idxs = []
        conv_out = []

        for x, _  in self.loader:
            x = x.float().to(self.device)
            (max_out, idx), conv = self.model(x)
            max_output.append(max_out)
            idxs.extend(idx)
            conv_out.append(conv)

        max_output = torch.cat(max_output).flatten(1)
        conv_out = torch.cat(conv_out)
        idxs = torch.cat(idxs, 1)
        self.model.return_pool = False
        return max_output, idxs


class EarlyStopping:
    def __init__(self, patience: int, filename = "early_stopping.pt", path : str = ""):
        self.patience = patience
        self.current_count = 0
        self.best_loss = np.inf
        self.stop = False
        self.path = path
        self.filename = filename
    
    def evaluate_epoch(self, metric: float, model: nn.Module, epoch: int):

        if metric < self.best_loss:
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



    