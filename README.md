# FEAOF

This repository contains the code for the paper: [**FEAOF: A Transferable Framework Applied to Prediction of hERG-Related Cardiotoxicity**]. 

## Introduction

Inhibition of the hERG channel by drug molecules can lead to severe cardiac toxicity, resulting in the withdrawal of many approved drugs from the market or halting their development in later stages. This underscores the urgent need to address this issue. Therefore, evaluating hERG blocking activity during drug development is crucial. In this study, we propose a novel framework for **feature extraction and aggregation optimization (FEAOF)**, which primarily consists of a feature extraction module and an aggregation optimization module. The model integrates ligand features such as molecular fingerprints, molecular descriptors, and molecular graphs, along with interaction features of ligand-receptor complexes. Based on this integration, we further optimize the algorithmic framework to achieve precise predictions of compounds cardiac toxicity. We established two independent test sets with significant structural differences from the training data to rigorously assess the model's predictive capability. The results demonstrate that the FEAOF model exhibits strong robustness compared to seven baseline models, with AUC, F1, and SN values of approximately 83%, 67%, and 76%, respectively. Importantly, **this model can be easily adapted for other drug-target interaction prediction tasks**. It is made available as open source under the permissive MIT license at https://github.com/ConfusedAnt/FEAOF.

### FEAOF Architecture

![FEAOF](./docs/Architecture.tif)

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

1. download the Data from: 。。。 and put it in the following path:

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

### Further Pretrained Model Path (for downstream tasks in peptideeval)

- download the cpkts in the following link: [Google Drive](https://drive.google.com/file/d/15Ai_lOrsxQ11UlvHZcMbKvU9YMGRltYl/view?usp=drive_link)
- unzip the cpkts to the following path:
```
---data
    ---cpkt
        ---- af80_step_50: Multivew Gearnet + ESM; Further Pretrained on AF90
        ---- af890_step_50: Multivew Gearnet + ESM; Further Pretrained on AF80
        ---- pdb_step_50: Multivew Gearnet + ESM; Further Pretrained on PDB
```

### Use pretrained pepharmony model to extract peptide representation

```bash
python inference_sscp.py
```
