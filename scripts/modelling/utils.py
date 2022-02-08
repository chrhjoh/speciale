import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from typing import Tuple
from torch import nn
from torch.utils.data.dataloader import DataLoader
from sklearn import metrics



def load_data(path : str, partitions, batch_size: int, seq_idx:np.ndarray=np.arange(420), shuffle=True) -> Tuple[DataLoader, np.ndarray] :
    if isinstance(partitions, int):
        data = np.load(path + f"P{partitions}_input.npz")["arr_0"]
        targets = np.load(path + f"P{partitions}_labels.npz")["arr_0"]
    
    else:
        data = np.concatenate([np.load(path + f"P{partition}_input.npz")["arr_0"] for partition in partitions])
        targets = np.concatenate([np.load(path + f"P{partition}_labels.npz")["arr_0"] for partition in partitions])
        
    if shuffle:
        idxs = np.random.permutation(data.shape[0])
        data = data[idxs, :, :] # Permute observations
        targets = targets[idxs]
    data = data[:,seq_idx, :] # Subset sequences
    data_list = [[np.transpose(data[i]), targets[i]] for i in range(len(data))]

    return DataLoader(data_list, batch_size,), data, targets

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
    

    
    