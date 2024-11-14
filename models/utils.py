"""
Author: Derek van Tilborg -- TU/e -- 23-05-2022

Utility functions that are used by several different models.

    - GNN():                 Base GNN parent class for all graph neural network models
    - NN():                  Base NN parent class for all basic neural network models (MLP and CNN)
    - graphs_to_loader():    Turn lists of molecular graphs and their bioactivities into a dataloader
    - plot_loss():           plot the losses of Torch models
    - scatter():             scatter plot of true/predicted for a Torch model using a dataloader
    - numpy_loader():        simple Torch dataloader from numpy arrays
    - squeeze_if_needed():   if the input is a squeezable tensor, squeeze it

"""
from const import CONFIG_PATH_SMILES
from new_unit import get_config, f1_acc
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Dict
import matplotlib.pyplot as plt
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset
import os
import pickle
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc


smiles_encoding = get_config(CONFIG_PATH_SMILES)


class GNN:
    """ Base GNN class that takes care of training, testing, predicting for all graph-based methods """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.f1s = []
        self.epoch = 0
        self.epochs = 100
        self.save_path = None

        self.model = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None
        self.lr_scheduler = None
        self.name = None
        self.herg_em = None
        self.batch_size = 32


    def train(self, model_save_path, x_train: List[Data], y_train: List[float], x_val: List[Data] = None, y_val: List[float] = None,
              early_stopping_patience: int = 50, epochs: int = 300, print_every_n: int = 10):
        """ Train a graph neural network.

        :param x_train: (List[Data]) a list of graph objects for training
        :param y_train: (List[float]) a list of bioactivites for training
        :param x_val: (List[Data]) a list of graph objects for validation
        :param y_val: (List[float]) a list of bioactivites for validation
        :param epochs: (int) train the model for n epochs
        :param print_every_n: (int) printout training progress every n epochs
        """
        if epochs is None:
            epochs = self.epochs
        train_loader = graphs_to_loader(x_train, y_train)
        patience = None if early_stopping_patience is None else 0
        f1=0
        acc=0
        roc_auc=0

        for epoch in range(epochs):

            # If we reached the end of our patience, load the best model and stop training
            if patience is not None and patience >= early_stopping_patience:

                if print_every_n < epochs:
                    print('Stopping training early')
                    
                # 这里重新load为了保存这个大类出去的时候model是最优的，而不是最后的
                try:
                    with open(model_save_path, 'rb') as handle:
                        self.model = pickle.load(handle)

                    # os.remove(self.save_path)
                except Warning:
                    print('Could not load best model, keeping the current weights instead')

                break

            # As long as the model is still improving, continue training
            else:
                loss = self._one_epoch(train_loader)
                if patience >10:
                    self.lr_scheduler.step()
                cur_lr = self.lr_scheduler.optimizer.state_dict()['param_groups'][0]['lr']
                self.train_losses.append(loss)

                val_loss = 0
                if x_val is not None:
                    val_prob = self.predict(x_val)
                    threshold = 0.5
                    val_pred = (val_prob >= threshold).float() 
                    f1_tmp, acc_tmp = f1_acc(y_val, val_pred)
                    
                    fpr, tpr, thresholds = roc_curve(np.array(y_val), val_prob)
                    roc_auc_tmp = auc(fpr, tpr)

                    val_loss = self.loss_fn(squeeze_if_needed(val_pred), torch.tensor(y_val))
                    
                self.val_losses.append(val_loss)
                self.f1s.append(f1_tmp)

                self.epoch += 1

                # Pickle model
                if f1 < f1_tmp:
                    with open(model_save_path, 'wb') as handle:
                        pickle.dump(self.model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    patience = 0
                    roc_auc, acc, f1 = roc_auc_tmp, acc_tmp, f1_tmp
                else:
                    patience += 1

                if self.epoch % print_every_n == 0:
                    print(f"Epoch {self.epoch} | Train Loss {loss} | Val Loss {val_loss/len(x_val)} | Acc_F1 {acc, f1} | patience {patience} | cur_lr {round(cur_lr,6)}")
        return roc_auc, acc, f1
    
    def _one_epoch(self, train_loader):
        """ Perform one forward pass of the train data through the model and perform backprop

        :param train_loader: Torch geometric data loader with training data
        :return: loss
        """
        # Enumerate over the data
        for idx, batch in enumerate(train_loader):

            # Move batch to gpu
            batch.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_hat = self.model(x=batch.x.float(), edge_index=batch.edge_index, edge_attr=batch.edge_attr.float(), batch=batch.batch, herg_em = self.herg_em) 
            # y_hat = torch.where(torch.isnan(y_hat), torch.full_like(y_hat, 0.00001), y_hat)

            # Calculating the loss and gradients
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y))
            if not loss > 0:
                print(idx)

            # Calculate gradients
            loss.backward()

            # # Update weights 
            self.optimizer.step() 

        return loss

    def test(self,  x_test: List[Data], y_test: List[float]):
        """ Perform testing

        :param x_test: (List[Data]) a list of graph objects for testing
        :param y_test: (List[float]) a list of bioactivites for testing
        :return: A tuple of two 1D-tensors (predicted, true)
        """
        data_loader = graphs_to_loader(x_test, y_test, shuffle=False)
        y_pred, y = [], []
        with torch.no_grad():
            for batch in data_loader:
                batch.to(self.device)
                if self.name=="AFP":
                    y_hat = self.model(x=batch.x.float(), edge_index=batch.edge_index, edge_attr=batch.edge_attr.float(), batch=batch.batch)
                else:
                    y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch, herg_em = self.herg_em)
                
                if self.name=="AFP":
                    y_hat=torch.nn.Sigmoid()(y_hat)

                y_hat = squeeze_if_needed(y_hat).tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                    y.extend(squeeze_if_needed(batch.y).tolist())
                else:
                    y_pred.append(y_hat)
                    y.append(squeeze_if_needed(batch.y).tolist())

        return torch.tensor(y_pred), torch.tensor(y)

    def predict(self, x):
        """ Predict bioactivity on molecular graphs

        :param x_test: (List[Data]) a list of graph objects for testing
        :param batch_size: (int) batch size for the prediction loader
        :return: A 1D-tensors of predicted values
        """
        loader = DataLoader(x, batch_size=self.batch_size, shuffle=False)
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)

                y_hat = self.model(x=batch.x.float(), edge_index=batch.edge_index, edge_attr=batch.edge_attr.float(), batch=batch.batch, herg_em = self.herg_em)
                y_hat = squeeze_if_needed(y_hat).tolist()

                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)


        return torch.tensor(y_pred)

    def __repr__(self):
        return 'Basic Graph Neural Network Class'


class NN:
    def __init__(self):

        self.train_losses = []
        self.val_losses = []
        self.f1s = []
        self.epoch = 0
        self.epochs = 100
        self.save_path = os.path.join('.', 'CNN_best_model.pkl')
        self.herg_em = None
        self.model = None
        self.device = None
        self.loss_fn = None
        self.optimizer = None
        self.lr_scheduler = None
        self.batch_size = 32
    def train(self, x_train: ArrayLike, y_train: List[float], x_val: ArrayLike = None, y_val: List[float] = None,
              early_stopping_patience: int = 50, epochs: int = 300, print_every_n: int = 10):

        if epochs is None:
            epochs = self.epochs
        train_loader = numpy_loader(x_train, y_train, batch_size = self.batch_size)
        patience = None if early_stopping_patience is None else 0

        for epoch in range(epochs):

            # If we reached the end of our patience, load the best model and stop training
            if patience is not None and patience >= early_stopping_patience:

                if print_every_n < epochs:
                    print('Stopping training early')
                try:
                    with open(self.save_path, 'rb') as handle:
                        self.model = pickle.load(handle)

                    # os.remove(self.save_path)
                except Warning:
                    print('Could not load best model, keeping the current weights instead')

                break

            # As long as the model is still improving, continue training
            else:
                
                loss = self._one_epoch(train_loader)
                if patience >10:
                    self.lr_scheduler.step()
                cur_lr = self.lr_scheduler.optimizer.state_dict()['param_groups'][0]['lr']
                    # print("Current lr : ", cur_lr)
                
                self.train_losses.append(loss)

                val_loss = 0
                if x_val is not None:
                    val_prob = self.predict(x_val)
                    threshold = 0.5
                    val_pred = (val_prob >= threshold).float() 
                    f1, acc = f1_acc(y_val, val_pred)

                    val_loss = self.loss_fn(squeeze_if_needed(val_pred), torch.tensor(y_val))

                    fpr, tpr, thresholds = roc_curve(np.array(y_val), val_prob)
                    roc_auc = auc(fpr, tpr)

                self.val_losses.append(val_loss)
                self.f1s.append(f1)
                
                # print(self.val_losses)

                self.epoch += 1

                # Pickle model if its the best
                if f1 >= max(self.f1s):
                # if val_loss <= min(self.val_losses):
                    with open(self.save_path, 'wb') as handle:
                        pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    patience = 0
                else:
                    patience += 1

                if self.epoch % print_every_n == 0:
                    print(f"Epoch {self.epoch} | Train Loss {loss} | Val Loss {val_loss} | ACC_F1 {acc,f1} | patience {patience} | cur_lr {round(cur_lr,6)}")
        return roc_auc, acc, f1 

    def _one_epoch(self, train_loader):
        """ Perform one forward pass of the train data through the model and perform backprop

        :param train_loader: Torch geometric data loader with training data
        :return: loss
        """
        # Enumerate over the data
        for idx, batch in enumerate(train_loader):

            # Move batch to gpu
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_hat = self.model(x.float(), self.herg_em)

            # Calculating the loss and gradients
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(y))
            if not loss > 0:
                print(idx)

            # Calculate gradients
            loss.backward()

            # Update weights
            self.optimizer.step()

        return loss

    def test(self, x_test: ArrayLike, y_test: List[float]):
        """ Perform testing

        :param x_test: (List[Data]) a list of graph objects for testing
        :param y_test: (List[float]) a list of bioactivites for testing
        :return: A tuple of two 1D-tensors (predicted, true)
        """
        data_loader = numpy_loader(x_test, y_test, self.batch_size)
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                y_hat = self.model(x.float(),self.herg_em)

                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                    y_true.extend(squeeze_if_needed(y).tolist())
                else:
                    y_pred.append(y_hat)
                    y_true.append(squeeze_if_needed(y).tolist())
                # y_pred.extend([i for i in squeeze_if_needed(pred).tolist()])
                # y_true.extend([i for i in squeeze_if_needed(y).tolist()])

        return torch.tensor(y_pred), torch.tensor(y)

    def predict(self, x):
        """ Predict bioactivity on molecular graphs

        :param x: (Array) a list of graph objects for testing
        :param batch_size: (int) batch size for the prediction loader
        :return: A 1D-tensors of predicted values
        """
        data_loader = numpy_loader(x, batch_size = self.batch_size)
        y_pred = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(self.device)
                y_hat = self.model(x.float(),self.herg_em)

                y_hat = squeeze_if_needed(y_hat).tolist()

                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)
                # y_pred.extend([i for i in squeeze_if_needed(y_hat).tolist()])

        return torch.tensor(y_pred)

    def __repr__(self):
        return f"Neural Network baseclass for NN taking numpy arrays"


class NumpyDataset(Dataset):
    def __init__(self, x: ArrayLike, y: List[float] = None):
        """ Create a dataset for the ChemBerta transformer using a pretrained tokenizer """
        super().__init__()

        if y is None:
            y = [0]*len(x)
        self.y = torch.tensor(y).unsqueeze(1)
        self.x = torch.tensor(x).float()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


def graphs_to_loader(x: List[Data], y: List[float], batch_size: int = 64, shuffle: bool = False):
    """ Turn a list of graph objects and a list of labels into a Dataloader """
    for graph, label in zip(x, y):
        graph.y = torch.tensor(label)

    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)


def plot_loss(model):
    train_losses_float = [float(loss.cpu().detach().numpy()) for loss in model.train_losses]
    val_losses_float = [float(loss) for loss in model.val_losses]
    loss_indices = [i for i, l in enumerate(train_losses_float)]

    plt.figure()
    plt.plot(loss_indices, train_losses_float, val_losses_float)
    plt.show()


def scatter(y_hat, y, min_axis_val: float = -5, max_axis_val: float = 1):

    plt.figure()
    plt.scatter(x=y_hat, y=y, alpha=0.5)
    plt.axline((0, 0), slope=1, color="black", linestyle=(1, (5, 5)))
    plt.xlim(min_axis_val, max_axis_val)
    plt.ylim(min_axis_val, max_axis_val)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.show()


def numpy_loader(x: ArrayLike, y: List[float] = None, batch_size: int = 32, shuffle: bool = False):
    dataset = NumpyDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def squeeze_if_needed(tensor):
    from torch import Tensor
    if len(tensor.shape) > 1 and tensor.shape[1] == 1 and type(tensor) is Tensor:
        tensor = tensor.squeeze()
    return tensor

def update_params(base_model, loss, update_lr):
    grads = torch.autograd.grad(loss, base_model.parameters(), allow_unused=True)

    # Replace None gradients with zeros
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, base_model.parameters())]

    return parameters_to_vector(grads), parameters_to_vector(
        base_model.parameters(
        )) - parameters_to_vector(grads) * update_lr