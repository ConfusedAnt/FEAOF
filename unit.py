from PyBioMed.PyMolecule.fingerprint import (
    CalculatePubChemFingerprint,
    CalculateECFP2Fingerprint,
)

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from typing import List
from mordred import Calculator, descriptors
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score
from math import sqrt
import pickle
from new_unit import Data
import torch
from torch_geometric.loader import DataLoader

from torch.utils.data import Dataset
from numpy.typing import ArrayLike

# Cal Features
def compute_descriptor_features(smiles_list):
    """
    Compute 2D descriptor features for a list of SMILES strings

    Parameters
    ----------
    smiles_list: List[str]
        The list of SMILES strings.

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe, where each row corresponds
        to the descriptors of a SMILES strings in order.
    """
    descriptor_calc_2D = Calculator(descriptors, ignore_3D=False)
    molecular_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    descriptors_2D = descriptor_calc_2D.pandas(molecular_mols)
    # descriptors_2D = descriptors_2D.select_dtypes(include=[int, float])
    return descriptors_2D.values


def mols_from_smiles(smiles: List[str]):
    """ Create a list of RDkit mol objects from a list of SMILES strings """
    from rdkit.Chem import MolFromSmiles
    return [MolFromSmiles(m) for m in smiles]


def compute_physchem(smiles: List[str]):

    from rdkit.Chem import Descriptors
    from rdkit import Chem

    X = []
    for m in mols_from_smiles(smiles):
        weight = Descriptors.ExactMolWt(m)
        logp = Descriptors.MolLogP(m)
        h_bond_donor = Descriptors.NumHDonors(m)
        h_bond_acceptors = Descriptors.NumHAcceptors(m)
        rotatable_bonds = Descriptors.NumRotatableBonds(m)
        atoms = Chem.rdchem.Mol.GetNumAtoms(m)
        heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(m)
        molar_refractivity = Chem.Crippen.MolMR(m)
        topological_polar_surface_area = Chem.QED.properties(m).PSA
        formal_charge = Chem.rdmolops.GetFormalCharge(m)
        rings = Chem.rdMolDescriptors.CalcNumAromaticRings(m)

        X.append(np.array([weight, logp, h_bond_donor, h_bond_acceptors, rotatable_bonds, atoms, heavy_atoms,
                           molar_refractivity, topological_polar_surface_area, formal_charge, rings]))

    return np.array(X)

def compute_fingerprint_features(smiles_list: List[str]) -> np.ndarray:

    molecular_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    # Initialize an array to store ECFP2 & PubChem fingerprint features
    features = np.zeros((len(smiles_list), 1024 + 881), dtype=np.int32)

    for i, mol in enumerate(molecular_mols):
        ECFP2_mol_fingerprint = CalculateECFP2Fingerprint(mol)
        pubchem_mol_fingerprint = CalculatePubChemFingerprint(mol)
        numerical_representation = np.concatenate(
            (ECFP2_mol_fingerprint[0], pubchem_mol_fingerprint)
        )
        features[i] = numerical_representation
    return features

def merge_fp_des(smiles: List[str]) -> np.ndarray:
        
    from sklearn.preprocessing import StandardScaler
    fingerprints = compute_fingerprint_features(smiles)
    descriptors = compute_physchem(smiles)
    des_2d = compute_descriptor_features(smiles)
    # scaler_fp_des = StandardScaler().fit(descriptors)
    # descriptors = scaler_fp_des.transform(descriptors)
    molecular_features = np.concatenate((fingerprints, descriptors), axis=1)
    molecular_features = np.concatenate((molecular_features, des_2d), axis=1)

    
    return molecular_features

# Test
def compute_metrics(ground_truth: List[int], pre_pro: List[float], file_name) -> None:
    """
    Computes and prints binary classification performance metrics.

    Parameters:
        ground_truth: List[int]
            The list of true labels (ground truth).
        predicted: List[int]
            The list of predicted labels.

    Returns:
        None:
            This function does not return any value; it prints the computed metrics.

    Metrics Computed and Printed:
        - Confusion Matrix
        - True Positive (TP), False Negative (FN), True Negative (TN), False Positive (FP)
        - Accuracy (AC)
        - F1-score (f1)
        - Sensitivity (SN)
        - Specificity (SP)
        - Correct Classification Rate (CCR)
        - Matthews Correlation Coefficient (MCC)
    """
    print("Binary classification performace metrics:")
    
    auc_score = round(roc_auc_score(ground_truth, pre_pro),4)
    threshold = 0.5
    predicted = (pre_pro >= threshold)

    print(confusion_matrix(ground_truth, predicted))
    tn, fp, fn, tp = confusion_matrix(ground_truth, predicted).ravel()

    precision, recall, thresholds = precision_recall_curve(ground_truth, pre_pro)
    pr_auc = round(auc(recall, precision),4)
    
    print("TP, FN, TN, FP")
    print("{:02d}, {:02d}, {:02d}, {:02d}".format(tp, fn, tn, fp))
    AC = round((tp + tn) / (tp + tn + fn + fp),4)
    # print("AC: {0:.3f}".format(AC))

    f1 = round(f1_score(ground_truth, predicted),4)
    # print("f1: {0:.3f}".format(f1))

    
    SN = round((tp) / (tp + fn),4)
    # print("SN: {0:.3f}".format(SN))

    PR = round((tp) / (tp + fp),4)
    # print("PR: {0:.3f}".format(PR))

    SP = round((tn) / (tn + fp),4)
    # print("SP: {0:.3f}".format(SP))

    CCR=round((((tp) / (tp + fn)) + ((tn) / (tn + fp))) / 2,4)
    # print("CCR: {0:.3f}".format(CCR))

    MCC = round((tp * tn - fp * fn) / (sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))),4)
    # print(
    #     "MCC: {0:.3f}".format(
    #         MCC
    #     )
    # )


    with open(file_name,"a") as p:
        p.write(f"{auc_score}, {pr_auc}, {AC}, {f1}, {SN}, {PR}, {SP}, {CCR}, {MCC}\n")

    return auc_score, pr_auc, AC, f1, SN, PR, SP, CCR, MCC

def load_des_pkl(pkl_path):
    with open(pkl_path, 'rb') as handle:
        mols_rep = pickle.load(handle)
    return mols_rep

def squeeze_if_needed(tensor):
    from torch import Tensor
    if len(tensor.shape) >= 1 and tensor.shape[1] == 1 and type(tensor) is Tensor: #>=1
        tensor = tensor.squeeze()
    return tensor


def graphs_to_loader_predict(x: List[Data], pkl_path, Ro5_inx, batch_size: int = 30, shuffle: bool = False):
    """ Turn a list of graph objects and a list of labels into a Dataloader """
    mols_rep = load_des_pkl(pkl_path)[Ro5_inx]
    if len(x) != mols_rep.shape[0]:
        raise ValueError(f"Mismatch")
    
    for graph, rep in zip(x, mols_rep):
        graph.rep = torch.tensor(rep.reshape(1,-1), dtype=torch.float32)

    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)


def predict(model,x_test,pkl_path,Ro5_inx, herg_em,device):
    model.eval()
    y_pred = []
    hiden_features = []
    test_loader = graphs_to_loader_predict(x_test, pkl_path,Ro5_inx)
    with torch.no_grad():
        for batch in test_loader:

            batch.to(device)
            if herg_em is not None:
                herg_em = herg_em.to(device)
            y_hat, hiden_feature = model(herg_em, x=batch.x.float(), edge_index=batch.edge_index, 
                          edge_attr=batch.edge_attr.float(), batch=batch.batch, 
                          mol_rep=batch.rep)
        
            y_hat = squeeze_if_needed(y_hat).tolist()

            if isinstance(y_hat, list):
                y_pred.extend(y_hat)

            hiden_features.append(hiden_feature)
            
    return torch.tensor(y_pred), torch.cat(hiden_features, dim=0)

# FEAOF train
def graphs_to_loader(x: List[Data], y: List[float], pkl_path, Ro5_inx, batch_size: int = 32, shuffle: bool = False):
    """ Turn a list of graph objects and a list of labels into a Dataloader """
    mols_rep = load_des_pkl(pkl_path)[Ro5_inx]
    if len(x) != mols_rep.shape[0]:
        raise ValueError(f"Mismatch")
    for graph, label, rep in zip(x, y, mols_rep):
        graph.y = torch.tensor(label)
        graph.rep = torch.tensor(rep.reshape(1,-1), dtype=torch.float32)
    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)


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


def numpy_loader(x: ArrayLike, y: List[float] = None, batch_size: int = 32, shuffle: bool = False):
    dataset = NumpyDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def log_to_file(file_path, message):
    with open(file_path, 'w') as f:
        f.write(message + '\n')

def train_one_epoch(x_train, y_train, model, optimizer, loss_fn, device, pkl_path, Ro5_inx, herg_em):
    model.train()
    losses = 0.0
    train_loader = graphs_to_loader(x_train, y_train, pkl_path, Ro5_inx)
    for idx, batch in enumerate(train_loader):
        batch.to(device)
        optimizer.zero_grad()
        if herg_em is not None:
            herg_em = herg_em.to(device)
        
        y_hat, hiden_features = model(herg_em, x=batch.x.float(), edge_index=batch.edge_index, 
                      edge_attr=batch.edge_attr.float(), batch=batch.batch, 
                      mol_rep=batch.rep)

        loss = loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y))
        loss.backward()
        optimizer.step()

        losses += loss.item()
        # torch.cuda.empty_cache()

    return losses / len(train_loader)

def validation(x_test, y_test, model, optimizer, loss_fn, device, pkl_path,Ro5_inx,herg_em):
    model.eval()
    losses = 0.0
    test_loader = graphs_to_loader(x_test, y_test, pkl_path,Ro5_inx)

    for idx, batch in enumerate(test_loader):
        batch.to(device)
    
        if herg_em is not None:
            herg_em = herg_em.to(device)

        y_hat, hiden_features = model(herg_em, x=batch.x.float(), edge_index=batch.edge_index, 
                      edge_attr=batch.edge_attr.float(), batch=batch.batch, 
                      mol_rep=batch.rep)

        loss = loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y))
        losses += loss.item()

    return losses / len(test_loader)

def predict(model,x_test,pkl_path,Ro5_inx,herg_em,device):
    model.eval()
    y_pred = []
    test_loader = graphs_to_loader_predict(x_test, pkl_path,Ro5_inx)
    with torch.no_grad():
        for batch in test_loader:

            batch.to(device)
            if herg_em is not None:
                herg_em = herg_em.to(device)

            y_hat, hiden_features = model(herg_em, x=batch.x.float(), edge_index=batch.edge_index, 
                          edge_attr=batch.edge_attr.float(), batch=batch.batch, 
                          mol_rep=batch.rep)
            
            # y_hat, hiden_features = model(x=batch.x.float(), edge_index=batch.edge_index, 
            #               edge_attr=batch.edge_attr.float(), batch=batch.batch, herg_em=None)

            y_hat = squeeze_if_needed(y_hat).tolist()

            if isinstance(y_hat, list):
                y_pred.extend(y_hat)
    return torch.tensor(y_pred)

## Transformer Eval
from models.transformer import chemberta_loader
def transformer_predict(model,x,device):
    """ Predict bioactivity on molecular graphs

    :param x: (Array) a list of graph objects for testing
    :param batch_size: (int) batch size for the prediction loader
    :return: A 1D-tensors of predicted values
    """
    data_loader = chemberta_loader(x, batch_size = 32)
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to gpu
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            # Predict
            y_hat = model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            y_hat = y_hat.squeeze().tolist()
            if type(y_hat) is list:
                y_pred.extend(y_hat)
            else:
                y_pred.append(y_hat)

    return torch.tensor(y_pred)


# CNN Eval
def cnn_predict(model,x,batch_size,herg_em,device):
    """ Predict bioactivity on molecular graphs

    :param x: (Array) a list of graph objects for testing
    :param batch_size: (int) batch size for the prediction loader
    :return: A 1D-tensors of predicted values
    """
    data_loader = numpy_loader(x, batch_size = batch_size)
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            y_hat = model(x.float(),herg_em)

            y_hat = squeeze_if_needed(y_hat).tolist()

            if type(y_hat) is list:
                y_pred.extend(y_hat)
            else:
                y_pred.append(y_hat)
            # y_pred.extend([i for i in squeeze_if_needed(y_hat).tolist()])

    return torch.tensor(y_pred)

# GCN Eval
def gcn_predict(model,x,herg_em,device):
    """ Predict bioactivity on molecular graphs

    :param x_test: (List[Data]) a list of graph objects for testing
    :param batch_size: (int) batch size for the prediction loader
    :return: A 1D-tensors of predicted values
    """
    loader = DataLoader(x, batch_size=32, shuffle=False)
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            y_hat = model(herg_em = herg_em, x=batch.x.float(), edge_index=batch.edge_index, edge_attr=batch.edge_attr.float(), batch=batch.batch)
            y_hat = squeeze_if_needed(y_hat).tolist()
            if type(y_hat) is list:
                y_pred.extend(y_hat)
            else:
                y_pred.append(y_hat)

    return torch.tensor(y_pred)

import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(42)
