# FEAOF

This repository contains the code for the paper: [**FEAOF: A Transferable Framework Applied to Prediction of hERG-Related Cardiotoxicity**]. 

## Introduction

Inhibition of the hERG channel by drug molecules can lead to severe cardiac toxicity, resulting in the withdrawal of many approved drugs from the market or halting their development in later stages. This underscores the urgent need to address this issue. Therefore, evaluating hERG blocking activity during drug development is crucial. In this study, we propose a novel framework for **feature extraction and aggregation optimization (FEAOF)**, which primarily consists of a feature extraction module and an aggregation optimization module. The model integrates ligand features such as molecular fingerprints, molecular descriptors, and molecular graphs, along with interaction features of ligand-receptor complexes. Based on this integration, we further optimize the algorithmic framework to achieve precise predictions of compounds cardiac toxicity. We established two independent test sets with significant structural differences from the training data to rigorously assess the model's predictive capability. The results demonstrate that the FEAOF model exhibits strong robustness compared to seven baseline models, with AUC, F1, and SN values of approximately 83%, 67%, and 76%, respectively. Importantly, **this model can be easily adapted for other drug-target interaction prediction tasks**. It is made available as open source under the permissive MIT license at https://github.com/ConfusedAnt/FEAOF.

### FEAOF Architecture

![pepharmony](./doc/main.png)

Overall architecture of the proposed PepHarmony framework. The sequence encoder and structural encoder are trained together by contrastive or generative learning. The downstream prediction tasks will just use the sequence coder to extract peptide representation.

### Evaluation Dataset
```
---data
  ---eval
    ---- aff.csv
    ---- CPP.txt
    ---- Sol.txt
```
### Pretrained Model Path

1. download the esm_t12 from: https://github.com/facebookresearch/esm and put it in the following path:

```
---data
  ---pretrained
    ---- esm2_t12/
      ---- esm2_t12_35M_UR50D-contact-regression.pt
      ---- esm2_t12_35M_UR50D.pt
    ---- mc_gearnet_edge.pth
    ---- siamdiff_gearnet_res.pth
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
