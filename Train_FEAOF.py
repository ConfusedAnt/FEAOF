from models import FEAOF
# from models import LSTM
from const import Descriptors, datasets
from new_unit import Data, f1_acc, get_config
from tqdm import tqdm
import warnings
import pickle
import os
import pandas as pd

# from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Dict
import matplotlib.pyplot as plt
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset
import os
import numpy as np

from unit import graphs_to_loader_predict, graphs_to_loader,squeeze_if_needed,log_to_file

# hERG
herg_em = None
# from herg_em import fetch_sequence, cal_em
# uniprot_ID="Q12809"
# uni_id = fetch_sequence(uniprot_ID)
# herg_em = cal_em(uni_id)[0]

# Data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_pkl_path = "./data/IF_Data/Train_All_Features.pkl"
val_pkl_path = "./data/IF_Data/Val_All_Features.pkl"

Train_Val="./data/processed/Train_Val.csv"
Train_Val = Data(Train_Val)
# data.shuffle()
Train_Val(Descriptors.GRAPH)


assert all(label in [0, 1] for label in Train_Val.y_train)
assert all(label in [0, 1] for label in Train_Val.y_test)


# Model
lr = 5.0e-06
lr_decay_ratio = 0.9

config_path = "./configures/optimized/train_split/MPNN_GRAPH.yml"
hyperparameters = get_config(config_path)
hyperparameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyperparameters.items()}
model = FEAOF(**hyperparameters).model
name = FEAOF(**hyperparameters).name
    
save_path = f"0FC_hERG_{Train_Val}_{name}_loss.pkl"
log_file = f"0FC_hERG_{Train_Val}_{name}_loss.txt"

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                        # self.epochs,
                                        step_size=3,
                                        gamma=lr_decay_ratio,
                                        last_epoch=-1,
                                        verbose=False)


from unit import train_one_epoch, predict, validation
# Training loop
epochs = 1000
print_every_n = 2
early_stopping_patience = 50
patience = 0

best_f1 = 0
best_val_loss = 1000

for epoch in range(epochs):

    loss = train_one_epoch(Train_Val.x_train, Train_Val.y_train, model, optimizer, loss_fn, device, train_pkl_path, Train_Val.train_Ro5_idx, herg_em)

    cur_lr = lr
    if patience > 10:
        lr_scheduler.step()
        cur_lr = lr_scheduler.optimizer.state_dict()['param_groups'][0]['lr']

    val_pred = predict(model, Train_Val.x_test, val_pkl_path, Train_Val.test_Ro5_idx, herg_em, device)
    threshold = 0.5
    val_pred = (val_pred >= threshold).float()
    f1, acc = f1_acc(Train_Val.y_test, val_pred)
    val_loss = validation(Train_Val.x_test, Train_Val.y_test, model, optimizer, loss_fn, device, val_pkl_path, Train_Val.test_Ro5_idx, herg_em)

    message = f"Epoch {epoch+1} | Train Loss {loss:.4f} | Valid Loss {val_loss:.4f} | ACC {acc:.4f} | F1 {f1:.4f} | Patience {patience} | Lr {cur_lr}"
    if (epoch + 1) % print_every_n == 0:
        print(message) 

    if val_loss < best_val_loss:
        log_to_file(log_file, message)
        best_val_loss = val_loss
        with open(save_path, 'wb') as handle:
            pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        patience = 0

    elif f1 > best_f1:
        log_to_file(log_file.replace("loss","f1"), message)
        best_f1 = f1
        with open(save_path.replace("loss","f1"), 'wb') as handle:
            pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        patience = 0

    else:
        patience += 1

    # early stop
    if patience >= 100:
        print('Stopping training early')
        break
