# FEAOF

This repository contains the code for the paper: [**FEAOF: A Transferable Framework Applied to Prediction of hERG-Related Cardiotoxicity**]. 

## Introduction

Inhibition of the hERG channel by drug molecules can lead to severe cardiac toxicity, resulting in the withdrawal of many approved drugs from the market or halting their development in later stages. This underscores the urgent need to address this issue. Therefore, evaluating hERG blocking activity during drug development is crucial. In this study, we propose a novel framework for **feature extraction and aggregation optimization (FEAOF)**, which primarily consists of a feature extraction module and an aggregation optimization module. The model integrates ligand features such as molecular fingerprints, molecular descriptors, and molecular graphs, along with interaction features of ligand-receptor complexes. Based on this integration, we further optimize the algorithmic framework to achieve precise predictions of compounds cardiac toxicity. We established two independent test sets with significant structural differences from the training data to rigorously assess the model's predictive capability. The results demonstrate that the FEAOF model exhibits strong robustness compared to seven baseline models, with AUC, F1, and SN values of approximately 83%, 67%, and 76%, respectively. Importantly, **this model can be easily adapted for other drug-target interaction prediction tasks**. It is made available as open source under the permissive MIT license at https://github.com/ConfusedAnt/FEAOF.

### FEAOF Architecture

![FEAOF](./docs/Architecture.png)

Overall architecture of the proposed FEAOF. It comprises two components: Feature Extraction and Aggregation Optimization. The feature extraction module focuses on characterizing the structures of both ligands and complexes. The aggregation optimization module is dedicated to integrating the characterizations of these two structural types, optimizing to obtain a comprehensive representation of the complex for property prediction.

### Raw and Split Data
```
---data
  ---raw
  # Wash Smiles
    ---- herg_raw_0528.csv
    ---- herg_raw_0528_clean.csv

  ---processed
  # Split Datasets
    ---- Train_Val.csv
    ---- Test_1.csv
    ---- Test_2.csv
```
### All Features Data

download the Data from:[Google Drive](https://drive.google.com/file/d/1vNyzwNYav4-BiDR-CxW4PrFIjaNRIJPY/view?usp=drive_link) and put it in the following path:

```
---data
  ---IF_Data
    # Docking Result
    ---- Docking_IF_1.csv
    ---- Docking_IF_2.csv
    ---- Docking_IF_3.csv

    # Merge All Features
    ---- Train_All_Features.pkl
    ---- Val_All_Features.pkl
    ---- Test_1_All_Features.pkl
    ---- Test_2_All_Features.pkl

    # Select Interaction Fingerprint
    ---- Train_Val_IF.csv
    ---- Test_1_IF.csv
    ---- Test_2_IF.csv
```

### Trained Models

- download the cpkts in the following link: [Google Drive](https://drive.google.com/file/d/1vNyzwNYav4-BiDR-CxW4PrFIjaNRIJPY/view?usp=drive_link)
- unzip the cpkts to the following path:
```
--- trained_models
    --- DL
        --- CNN-SMILES.pkl
        --- Transformer-TOKENS.pkl
        --- GCN-GRAPH.pkl
        --- MPNN-GRAPH.pkl
    --- ML
        --- GBM_2FP_11Des_1441IF.pkl
        --- RF_2FP_11Des_1441IF.pkl
        --- SVM_2FP_11Des_1441IF.pkl
    --- FEAOF
        --- 0FC_FEAOF.pkl
        --- 1FC_FEAOF.pkl
        --- 3FC_FEAOF.pkl
        --- 5FC_FEAOF.pkl
        --- Protein_0FC_FEAOF.pkl
        --- Protein_3FC_FEAOF.pkl
```

### Retrain and test models
```bash
python Train_FEAOF.py
python Train_DL.py
Train_Eval_ML.ipynb
```

### Model Performance
![Performance](./docs/Performance.png)


