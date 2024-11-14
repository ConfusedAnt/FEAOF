from models import GCN, MPNN, CNN, Transformer
from const import Descriptors, CONFIG_PATH, WORKING_DIR, RANDOM_SEED, RESULT_PATH
from new_unit import Data, get_config, cross_validate, f1_acc
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import warnings
import pickle
import os
import pandas as pd
import numpy as np
from new_unit import crate_results_file,write_results

combinations = {
                Descriptors.GRAPH: [GCN, 
                                    MPNN
                                    ],
                # Descriptors.SMILES: [CNN],
                # Descriptors.TOKENS: [Transformer]
}

AUGMENTATION_FACTOR = 10

Train_Val="./data/processed/Train_Val.csv"
herg_em = None

early_stopping_patience=50
epochs=500

def main(results_filename):

    results_path = os.path.join(RESULT_PATH, results_filename)
    crate_results_file(results_path)

    data = Data(Train_Val)

    for descriptor, algorithms in combinations.items():
        if descriptor in [Descriptors.SMILES, Descriptors.TOKENS]:
            data = Data(Train_Val)
            data.augment(AUGMENTATION_FACTOR)
        data.shuffle()

        data(descriptor)
        print(f"{descriptor} rep have been caled")

        for algo in algorithms:
            combi = f"{algo.__name__}_{descriptor.name}"
            print(f"now is at {combi}")

            config_path = os.path.join(CONFIG_PATH, 'optimized', "train_split", f"{combi}.yml")
            trained_models_dir = os.path.join(WORKING_DIR, 'trained_models', "DL")
            if not os.path.exists(trained_models_dir):
                os.mkdir(trained_models_dir)

            model_save_path = os.path.join(trained_models_dir, f"{combi}.pkl")
            hyperparameters = get_config(config_path)

            # Train model
            hyperparameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyperparameters.items()}
            f = algo(herg_em, batch_size=32, **hyperparameters)
            roc_auc, acc, f1 = f.train(model_save_path, data.x_train, data.y_train, data.x_test, data.y_test, early_stopping_patience=early_stopping_patience, epochs=epochs)
            print(f"{f.name} auc & acc & f1: {roc_auc}, {acc}, {f1}")
            
            # with open(model_save_path, 'wb') as handle:
            #     pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            write_results(filename=results_path, dataset=Train_Val,
                algo=algo.__name__, descriptor=descriptor.name,
                augmentation=data.augmented,
                data=data,auc=roc_auc,acc=acc,f1=f1)
                    

if __name__ == '__main__':
    main(results_filename="DL_Metrics.csv")
