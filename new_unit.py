'''https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/benchmark/utils.py'''
from const import Descriptors, DATA_PATH, RANDOM_SEED, CONFIG_PATH_SMILES
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.model_selection import StratifiedKFold
from yaml import load, Loader, dump
from typing import List, Union
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import numpy as np
import pickle
import torch
import json
import os
import re
import random
import warnings
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
# import tensorflow as tf
from rdkit.Chem import Descriptors as Des

class Data:
    def __init__(self, file: Union[str, pd.DataFrame], Ro5=True):
        """ Data class to easily load and featurize molecular bioactivity data

        :param file: 1. (str) path to .csv file with the following columns; 'smiles': (str) SMILES strings
                                                                                'y': (float) bioactivity values
                                                                                'cliff_mol': (int) 1 if cliff else 0
                                                                                'split': (str) 'train' or 'test'
                     2. (str) name of a dataset provided with the benchmark; see MoleculeACE.benchmark.const.datasets
                     3. (pandas.DataFrame) pandas dataframe with columns similar to 1. (see above)
        """

        # Either load a .csv file or use a provided dataframe
        if type(file) is str:
            print(file)
            df = pd.read_csv(file)
            df['y'] = df['y'].astype(float)
            if Ro5:
                df["MW"]=df["smiles"].apply(lambda x : Des.ExactMolWt(Chem.MolFromSmiles(x)))


                df_train = df[df["split"]=="train"].reset_index()
                df_test = df[df["split"]=="test"].reset_index()
                train_mask = (df_train['MW'] >= 300) & (df_train['MW'] <= 500)
                test_mask = (df_test['MW'] >= 300) & (df_test['MW'] <= 500)

                self.train_Ro5_idx = df_train[train_mask].index
                self.test_Ro5_idx = df_test[test_mask].index
                # df = df.sample(50)
                df = df[(df['MW'] >= 300) & (df['MW'] <= 500)]
                
                print("data number : ", df.shape)
        else:
            df = file

        self.smiles_train = df[df["split"]=="train"]['smiles'].tolist()
        self.y_train = df[df["split"]=="train"]['y'].tolist()

        self.smiles_test = df[df["split"]=="test"]['smiles'].tolist()
        self.y_test = df[df["split"]=="test"]['y'].tolist()

        from featurization import Featurizer

        self.featurizer = Featurizer()

        self.x_train = None
        self.x_test = None

        self.featurized_as = 'Nothing'
        self.augmented = 0

    def featurize_data(self, descriptor: Descriptors, **kwargs):
        """ Encode your molecules with Descriptors.ECFP, Descriptors.MACCS,
        Descriptors.GRAPH, Descriptors.SMILES, Descriptors.TOKENS """

        try:
            self.x_train = self.featurizer(descriptor, smiles=self.smiles_train, **kwargs)
            self.x_test = self.featurizer(descriptor, smiles=self.smiles_test, **kwargs)
        except:
            self.x_test = self.featurizer(descriptor, smiles=self.smiles_test, **kwargs)
        self.featurized_as = descriptor.name

    def shuffle(self):
        """ Shuffle training data """
        c = list(zip(self.smiles_train, self.y_train))  # Shuffle all lists together
        random.shuffle(c)
        self.smiles_train, self.y_train = zip(*c)

        self.smiles_train = list(self.smiles_train)
        self.y_train = list(self.y_train)

    def augment(self, augment_factor: int = 10, max_smiles_len: int = 200):
        """ Augment training SMILES strings n times (Do this before featurizing them please)"""
        self.smiles_train, self.y_train = augment(self.smiles_train, self.y_train,
                                                                         augment_factor=augment_factor,
                                                                         max_smiles_len=max_smiles_len)
        if self.x_train is not None: 
            if len(self.y_train) > len(self.x_train):
                warnings.warn("DON'T FORGET TO RE-FEATURIZE YOUR AUGMENTED DATA")
        self.augmented = augment_factor

    def __call__(self, descriptor: Descriptors, **kwargs):
        self.featurize_data(descriptor, **kwargs)

    def __repr__(self):
        return f"Data object with molecules as: {self.featurized_as}. {len(self.y_train)} train/{len(self.y_test)} test"


def load_model(filename: str):
    """ Load a model """
    if filename.endswith('.h5'):
        from tensorflow.keras.models import load_model
        model = load_model(filename)
    else:
        with open(filename, 'rb') as handle:
            model = pickle.load(handle)
    return model


def get_config(file: str):
    """ Load a yml config file"""
    if file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, "r", encoding="utf-8") as read_file:
            config = load(read_file, Loader=Loader)
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    return config


def write_config(filename: str, args: dict):
    """ Write a dictionary to a .yml file"""
    args = {k: v.item() if isinstance(v, np.generic) else v for k, v in args.items()}
    with open(filename, 'w') as file:
        documents = dump(args, file)


def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))

def f1_acc(ground_truth: List[int], predicted: List[int]) -> None:
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
    # print("Binary classification performace metrics:")
    # print(confusion_matrix(ground_truth, predicted))
    tn, fp, fn, tp = confusion_matrix(ground_truth, predicted).ravel()
    # print("TP, FN, TN, FP")
    # print("{:02d}, {:02d}, {:02d}, {:02d}".format(tp, fn, tn, fp))
    # print("AC: {0:.3f}".format((tp + tn) / (tp + tn + fn + fp)))
    # print("f1: {0:.3f}".format(f1_score(ground_truth, predicted)))
    # print("SN: {0:.3f}".format((tp) / (tp + fn)))
    # print("SP: {0:.3f}".format((tn) / (tn + fp)))
    # print("CCR: {0:.3f}".format((((tp) / (tp + fn)) + ((tn) / (tn + fp))) / 2))
    # print(
    #     "MCC: {0:.3f}".format(
    #         (tp * tn - fp * fn) / (np.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp)))
    #     )
    # )
    return f1_score(ground_truth, predicted), (tp + tn) / (tp + tn + fn + fp)


def cross_validate(model, data, herg_em=None, n_folds: int = 5, early_stopping: int = 10, seed: int = RANDOM_SEED,
                   save_path: str = None, **hyperparameters):
    """

    :param model: a model that has a train(), test(), and predict() method and is initialized with its hyperparameters
    :param data: Moleculace.benchmark.utils.Data object
    :param n_folds: (int) n folds for cross-validation
    :param early_stopping: (int) stop training when not making progress for n epochs
    :param seed: (int) random seed
    :param save_path: (str) path to save trained models
    :param hyperparameters: (dict) dict of hyperparameters {name_of_param: value}

    :return: (list) rmse_scores, (list) cliff_rmse_scores
    """

    x_train = data.x_train
    y_train = data.y_train
    x_test = data.x_test
    y_test = data.y_test

    ### 这里之所以这么麻烦，是因为token表征的结果不是一维的向量！！！！！！！！！
    ### 所以不能直接用 splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(x_train, y_train)]
    ### 但可以用ss.split(y_train, y_train)]，因为只关注label的分布
    ss = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    # cutoff = np.median(y_train)
    # labels = [0 if i < cutoff else 1 for i in y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(y_train, y_train)]

    acc_scores = []
    f1_scores = []
    for i_split, split in enumerate(tqdm(splits)):

        # Convert numpy types to regular python type (this bug cost me ages)
        hyperparameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyperparameters.items()}

        f = model(herg_em, batch_size=256, **hyperparameters)
        # f = model()

        if type(x_train) is BatchEncoding:
            x_tr_fold = {'input_ids': x_train['input_ids'][split['train_idx']],
                         'attention_mask': x_train['attention_mask'][split['train_idx']]}
            x_val_fold = {'input_ids': x_train['input_ids'][split['val_idx']],
                          'attention_mask': x_train['attention_mask'][split['val_idx']]}
        else:
            x_tr_fold = [x_train[i] for i in split['train_idx']] if type(x_train) is list else x_train[
                split['train_idx']]
            x_val_fold = [x_train[i] for i in split['val_idx']] if type(x_train) is list else x_train[split['val_idx']]

        y_tr_fold = [y_train[i] for i in split['train_idx']] if type(y_train) is list else y_train[split['train_idx']]
        y_val_fold = [y_train[i] for i in split['val_idx']] if type(y_train) is list else y_train[split['val_idx']]

        f.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, early_stopping, epochs=1000)

        # Save model to "save_path+_{fold}.pkl"
        if save_path is not None:
            if save_path.endswith('.h5'):
                f.model.save(f"{save_path.split('.')[-2]}_{i_split}.{save_path.split('.')[-1]}")
            else:
                with open(f"{save_path.split('.')[-2]}_{i_split}.{save_path.split('.')[-1]}", 'wb') as handle:
                    pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)

        y_hat = f.predict(x_test)
                        
        # 将概率值转换为二分类预测结果
        threshold = 0.5
        y_hat = (y_hat >= threshold).float() 

        # print("view predict value : ", y_hat)
        acc = f1_acc(y_test, y_hat)[1]
        f1 = f1_acc(y_test, y_hat)[0]
        print(f"{f.name} Fold {i_split} acc and f1: ", acc, f1)

        acc_scores.append(acc)
        f1_scores.append(f1)

        del f.model
        del f
        torch.cuda.empty_cache()
        # tf.keras.backend.clear_session()
        if i_split > 0:
            return acc_scores, f1_scores
    # Return the rmse and cliff rmse for all folds
    return acc_scores, f1_scores

def smi_tokenizer(smi: str):
    """ Tokenize a SMILES """
    pattern = "(\[|\]|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Si|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    return tokens


def augment(smiles: List[str], *args, augment_factor: int = 10, max_smiles_len: int = 200, max_tries: int = 1000):
    """ Augment SMILES strings by adding non-canonical SMILES. Keeps corresponding activity values/CHEMBL IDs, etc """
    augmented_smiles = []
    augmented_args = [[] for _ in args]
    for i, smi in enumerate(tqdm(smiles)):
        generated = smile_augmentation(smi, augment_factor - 1, max_smiles_len, max_tries)
        augmented_smiles.append(smi)
        augmented_smiles.extend(generated)

        for a, arg in enumerate(args):
            for _ in range(len(generated) + 1):
                augmented_args[a].append(arg[i])

    return tuple([augmented_smiles], ) + tuple(augmented_args)


def random_smiles(mol):
    """ Generate a random non-canonical SMILES string from a molecule"""
    # https://github.com/michael1788/virtual_libraries/blob/master/experiments/do_data_processing.py
    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(mol)


smiles_encoding = get_config(CONFIG_PATH_SMILES)


def smile_augmentation(smile: str, augmentation: int, max_len: int = 200, max_tries: int = 1000):
    """Generate n random non-canonical SMILES strings from a SMILES string with length constraints"""
    # https://github.com/michael1788/virtual_libraries/blob/master/experiments/do_data_processing.py
    mol = Chem.MolFromSmiles(smile)
    s = set()
    for i in range(max_tries):
        if len(s) == augmentation:
            break

        smiles = random_smiles(mol)
        if len(smiles) <= max_len:
            tokens = smi_tokenizer(smiles)
            if all([tok in smiles_encoding['token_indices'] for tok in tokens]):
                s.add(smiles)

    return list(s)

def crate_results_file(filename):
    with open(filename, 'w') as f:
        f.write('dataset,algorithm,descriptor,augmentation,n_compounds,'
                'n_compounds_train,n_compounds_test, auc, acc, f1\n')
        
def write_results(filename, dataset, algo, descriptor, augmentation, data, auc, acc, f1):
    with open(filename, 'a') as f:
        f.write(f'{dataset},{algo},{descriptor},{augmentation},'
                f'{len(data.y_train) + len(data.y_test)},'
                f'{len(data.y_train)}, {len(data.y_test)}, {auc}, {acc}, {f1}\n')