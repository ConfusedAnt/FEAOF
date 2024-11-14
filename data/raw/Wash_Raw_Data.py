#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
import re
import pandas as pd
import multiprocessing as mp
# from const import DATA_PATH
import os
import math

symbol = ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te']
double_symbol = ["br", "ca", "cl", "na", "si", 'se', 'te']

def remove_components(smiles):
    components = smiles.split(".")
    if len(components) == 1:
        return smiles
    else:
        components.sort(key=lambda s: len(re.findall("[A-Za-z]",s)))
        return components[-1]

def initialise_reactions():
    patts= (
        # Imidazoles
        ('[n+;H]','n'),
        # Amines
        ('[N+;!H0]','N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]','O'),
        # Thiols
        ('[S-;X1]','S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]','N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]','N'),
        # Tetrazoles
        ('[n-]','[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]','S'),
        # Amides
        ('[$([N-]C=O)]','N'),
        )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

def neutralise_charges(mol, reactions=None):
    if reactions is None:
        _reactions=initialise_reactions()
        reactions=_reactions
        
    replaced = False
    for i,(reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol

def check_elements(mol, symbol):
    molecule_elements = set([atom.GetSymbol() for atom in mol.GetAtoms()])
    for ele in molecule_elements:
        if ele not in symbol:
            return False
    return True

def modified_strings(smi):
    for d_sym in double_symbol:
        if d_sym in smi:
            smi = smi.replace(d_sym, d_sym.capitalize())
    return smi

def wash_single_smiles(smi):
    remover = SaltRemover()
    try:
        smi = modified_strings(smi)
        m = Chem.MolFromSmiles(smi)
        gs_charge(m)
        # assert m is not None
        m = remover.StripMol(m,dontRemoveEverything=True)
        m = neutralise_charges(m)
        smi_new = remove_components(Chem.MolToSmiles(m))
        if check_elements(m, symbol):
            return smi_new
        else:
            return "invalid"
    except:
        print('Invalid smiles: {}'.format(smi))
        return "invalid"

def gs_charge(mol):
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    for atom in mol.GetAtoms():
        partial_charge = float(atom.GetProp('_GasteigerCharge'))
        if math.isnan(partial_charge):
            raise ValueError("Partial charge is NaN")

def main(file_path, save_path):
    df = pd.read_csv(file_path)
    print("data number before process: ", len(df))

    with mp.Pool(24) as pool:
        df["smiles"] = pool.map(wash_single_smiles,df["smiles"])
    df = df[df["smiles"]!="invalid"]
    df.drop_duplicates("smiles",inplace=True, keep="last")
    df=df.sample(frac=1)
    df.to_csv(save_path,index=False)
    print("data number after process: ", len(df))
        

if __name__ == "__main__":
    file_path = "./raw/herg_raw_0528.csv"
    save_path = "./raw/herg_raw_0528_clean.csv"
    main(file_path, save_path)