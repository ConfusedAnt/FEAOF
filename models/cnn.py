"""
Author: Derek van Tilborg -- TU/e -- 24-05-2022

Basic 1-D Convolutional Neural Network based on the architecture of [1]

[1] Kimber et al. (2021). Maxsmi: Maximizing molecular property prediction performance with confidence estimation using
    SMILES augmentation and deep learning

"""

import torch
import torch.nn.functional as F
from models.utils import NN


class CNN(NN):
    def __init__(self, herg_em, batch_size, nchar_in: int = 41, seq_len_in: int = 202, kernel_size: int = 10, hidden: int = 128,
                 lr: float = 0.0005, epochs: int = 300, lr_decay_ratio: float = 0.95, *args, **kwargs):
        super().__init__()

        self.model = CNNmodel(nchar_in=nchar_in, seq_len_in=seq_len_in, kernel_size=kernel_size, hidden=hidden)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'CNN'
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    # self.epochs,
                                                    step_size=3,
                                                    gamma=lr_decay_ratio,
                                                    last_epoch=-1,
                                                    verbose=False)
        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

        if herg_em is not None:
            self.herg_em = herg_em.to(self.device)
        self.batch_size = batch_size

    def __repr__(self):
        return f"1-D Convolutional Neural Network"


class CNNmodel(torch.nn.Module):
    def __init__(self, nchar_in: int = 41, seq_len_in: int = 202, kernel_size: int = 10, hidden: int = 128, *args,
                 **kwargs):
        """

        :param nchar_in: (int) number of unique characters in the SMILES sequence (default = 41)
        :param seq_len_in: (int) length of the SMILES sequence
        :param kernel_size: (int) convolution kernel size
        :param hidden: (int) number of neurons in the hidden layer
        """
        super().__init__()

        self.conv0 = torch.nn.Conv1d(in_channels=seq_len_in, out_channels=nchar_in, kernel_size=kernel_size)

        conv_out_size = (nchar_in-kernel_size+1)*nchar_in
        self.fc0 = torch.nn.Linear(conv_out_size, hidden)
        
        self.cat_liner = torch.nn.Linear(hidden+1280, hidden)

        self.out = torch.nn.Linear(hidden, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, herg_em):

        h = F.relu(self.conv0(x))
        h = torch.flatten(h, 1)
        h = F.relu(self.fc0(h))
        
        if herg_em is not None:
            herg_em = herg_em.expand(h.shape[0], -1)
            h = torch.cat((h, herg_em), dim=1)
            h = F.relu(self.cat_liner(h))
            # h = h + herg_em
            # h = self.dropout(h)
            
        out = self.out(h)
        out = self.sigmoid(out)
        return out
