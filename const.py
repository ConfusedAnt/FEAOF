'''https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/benchmark/const.py'''
import datetime
import os
from enum import Enum
from pathlib import Path



RANDOM_SEED = 42


class Algorithms(Enum):
    RF = 'RF'
    GBM = 'GBM'
    SVM = 'SVM'
    CNN = 'CNN'
    LSTM = 'LSTM'
    GCN = 'GCN'
    MPNN = 'MPNN'
    TRANSFORMER = 'TRANS'


class Descriptors(Enum):
    ECFP = 'ECFP'
    MACCS = 'MACCs'
    GRAPH = 'Graph'
    SMILES = 'SMILES'
    TOKENS = 'Tokens'


current_work_dir = os.getcwd()
WORKING_DIR=current_work_dir
DATA_PATH = os.path.join(WORKING_DIR, "data")
LOGS_PATH = os.path.join(WORKING_DIR, "logs")
CONFIG_PATH = os.path.join(WORKING_DIR, "configures")
RESULT_PATH = os.path.join(WORKING_DIR, "results")

CONFIG_PATH_SMILES = os.path.join(CONFIG_PATH, 'default', 'SMILES.yml')
CONFIG_PATH_TRANS = os.path.join(CONFIG_PATH, 'default', 'TRANSFORMER.yml')


def define_default_log_dir():
    dt = datetime.datetime.now()
    default_log_dir = os.path.join(LOGS_PATH, f"{dt.date()}_{dt.hour}_{dt.minute}_{dt.second}")
    os.makedirs(default_log_dir, exist_ok=True)
    return default_log_dir


def setup_working_dir(path):
    """ Setup a working directory if the user specifies a new one """

    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(os.path.join(path, 'logs')):
        os.mkdir(os.path.join(path, 'logs'))

    if not os.path.exists(os.path.join(path, 'pretrained_models')):
        os.mkdir(os.path.join(path, 'pretrained_models'))

    if not os.path.exists(os.path.join(path, 'data')):
        os.mkdir(os.path.join(path, 'data'))
        os.mkdir(os.path.join(path, 'data', 'train'))
        os.mkdir(os.path.join(path, 'data', 'test'))
        os.mkdir(os.path.join(path, 'data', 'raw'))
        os.mkdir(os.path.join(path, 'data', 'processed'))

    if not os.path.exists(os.path.join(path, 'configures')):
        os.mkdir(os.path.join(path, 'configures'))
        os.mkdir(os.path.join(path, 'configures', 'optimized'))
        
    if not os.path.exists(os.path.join(path, 'results')):
        os.mkdir(os.path.join(path, 'results'))
