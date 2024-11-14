RANDOM_SEED = 42
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd
import numpy as np

from typing import List, Callable, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdMMPA import FragmentMol
from Levenshtein import distance as levenshtein
from tqdm import tqdm
import os

from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def get_tanimoto_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024, hide: bool = False, top_n: int = None):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string"""

    # Make a fingerprint database
    db_fp = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]],
                                                     db_fp[smiles[j]])
    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m

def find_stereochemical_siblings(smiles: List[str]):
    """ Detects molecules that have different SMILES strings, but ecode for the same molecule with
    different stereochemistry. For racemic mixtures it is often unclear which one is measured/active

    Args:
        smiles: (lst) list of SMILES strings

    Returns: (lst) List of SMILES having a similar molecule with different stereochemistry

    """

    lower = np.tril(get_tanimoto_matrix(smiles, radius=4, nBits=4096), k=0)
    identical = np.where(lower == 1)
    identical_pairs = [[smiles[identical[0][i]], smiles[identical[1][i]]] for i, j in enumerate(identical[0])]

    return list(set(sum(identical_pairs, [])))

def check_matching(original_smiles, original_bioactivity, smiles, bioactivity):
    assert len(smiles) == len(bioactivity), "length doesn't match"
    for smi, label in zip(original_smiles, original_bioactivity):
        if smi in smiles:
            assert bioactivity[smiles.index(smi)] == label, f"{smi} doesn't match label {label}"

def split_data(df_raw, task_type, clusters: bool = False, test_size: float = 0.2,
               remove_stereo: bool = True):
    
    """ Split data into train/test according to activity cliffs and compounds characteristics.
    :df_raw: pic50,smiles,P_ID...
    :param smiles: (List[str]) list of SMILES strings
    :param bioactivity: (List[float]) list of bioactivity values
    :param n_clusters: (int) number of clusters the data is split into for getting homogeneous data splits
    :param test_size: (float) test split
    :param similarity:  (float) similarity threshold for calculating activity cliffs
    :param potency_fold: (float) potency difference threshold for calculating activity cliffs
    :param remove_stereo: (bool) Remove racemic mixtures altogether?

    :return: df[smiles, exp_mean [nM], y, cliff_mol, split]
    """

    original_smiles = df_raw["smiles"].tolist()
    original_bioactivity = df_raw["y"].tolist()

    if remove_stereo:
        stereo_smiles_idx = [original_smiles.index(i) for i in find_stereochemical_siblings(original_smiles)]
        smiles = [smi for i, smi in enumerate(original_smiles) if i not in stereo_smiles_idx]
        bioactivity = [act for i, act in enumerate(original_bioactivity) if i not in stereo_smiles_idx]
        if len(stereo_smiles_idx) > 0:
            print(f"Removed {len(stereo_smiles_idx)} stereoisomers")
    else:
        smiles = original_smiles
        bioactivity = original_bioactivity

    check_matching(original_smiles, original_bioactivity, smiles, bioactivity)

    if clusters:
        cls_list = df_raw["Cluster"]
        n_cluster = len(set(cls_list))

        train_idx, test_idx = [], []
        for cluster in range(n_cluster):
            cluster_idx = np.where(cls_list == cluster)[0]

            # Can only split stratiefied on cliffs if there are at least 2 cliffs present, else do it randomly
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                                random_state=RANDOM_SEED,
                                                                shuffle=True)

            train_idx.extend(clust_train_idx)
            test_idx.extend(clust_test_idx)
    else:
        train_idx, test_idx = train_test_split(range(len(smiles)), test_size=test_size,
                                                    random_state=RANDOM_SEED,
                                                    shuffle=True)
    train_test = []
    for i in range(len(smiles)):
        if i in train_idx:
            train_test.append('train')
        elif i in test_idx:
            train_test.append('test')
        else:
            raise ValueError(f"Can't find molecule {i} in train or test")

    # Check if there is any intersection between train and test molecules
    assert len(np.intersect1d(train_idx, test_idx)) == 0, 'train and test intersect'
    assert len(np.intersect1d(np.array(smiles)[np.where(np.array(train_test) == 'train')],
                              np.array(smiles)[np.where(np.array(train_test) == 'test')])) == 0, \
        'train and test intersect'

    df_out = pd.DataFrame({'smiles': smiles,
                        'y': bioactivity,
                        'split': train_test,
                        })
    return df_out


if __name__ == "__main__":
    df = pd.read_csv("./processed/Train_Val_Cluster.csv")

    df_out = split_data(df,clusters=True,test_size=0.1,remove_stereo=False)
    df_out.to_csv("./processed/Train_Val.csv",index=False)
    print(df_out.shape)
