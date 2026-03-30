import os
import random
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import timm
from data import LoadDataset
from trainbk import ModelTrainer
import utils
import modelbuilder as mb

# to reproduce
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True
os.environ['PYTHONHASHSEED'] = str(seed)
PROJECT_ROOT = Path(__file__).parent # Path to folder opened file

DB_REGISTRY = {
    "UTKFace" : {'root': Path(PROJECT_ROOT / 'skin/UTKFace/'), 'num_classes': 4, 'epochs': 50},
    "dermoscopic": {'root': Path(PROJECT_ROOT / 'skin/milk10k/dermoscopic/'), 'num_classes': 2, 'epochs': 50},
    "clinical": {'root': Path(PROJECT_ROOT / 'skin/milk10k/clinical/'), 'num_classes': 2, 'epochs': 50},
}

def main():

    #train_mode = 'reweight'
    #train_mode = 'baseline'
    train_mode = 'regularization'

    for db_name, db in DB_REGISTRY.items():
        root = DB_REGISTRY[db_name]['root']
        num_classes = DB_REGISTRY[db_name]['num_classes']
        epochs = DB_REGISTRY[db_name]['epochs']

        for model_reg, model_conf in mb.MODEL_REGISTRY.items():

            batch_size = mb.MODEL_REGISTRY[model_reg]['batch_size']
            df_train = pd.read_csv(f"{root}/df_train.csv", dtype={"label": str, "skin tone": str})
            df_valid = pd.read_csv(f"{root}/df_valid.csv", dtype={"label": str, "skin tone": str})
            df_test = pd.read_csv(f"{root}/df_test.csv", dtype={"label": str, "skin tone": str})

            le = LabelEncoder()
            train_loader = LoadDataset(df_train, "label", le, batch_size=batch_size, isTrain=True).get_dataloader()
            val_loader = LoadDataset(df_valid, "label", le, batch_size=batch_size, isTrain=False).get_dataloader()
            test_loader = LoadDataset(df_test, "label", le, batch_size=batch_size, isTrain=False).get_dataloader()

            model_save_directory = f"{root}/models/{train_mode}/{model_reg}/"
            utils.create_directory(model_save_directory)
            utils.delete_subfolders(model_save_directory)

            model = timm.create_model(model_conf['model_name'], pretrained=True, num_classes=num_classes)
            make_optim = model_conf["make_optim"]
            optimizer, scheduler, batch_scheduler = make_optim(model, epochs, len(train_loader))

            loaders = {"train": train_loader, "valid": val_loader}
            training_settings = {
                "optimizer": optimizer,
                "scheduler": scheduler,
                'batch_scheduler': batch_scheduler,
                "epochs": epochs,
                "model_save_directory": model_save_directory,
            }

            # train
            trainer = ModelTrainer(train_loader, train_mode=train_mode)
            best_model_path = trainer.train(model, loaders, training_settings)

            # test
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            pred = trainer.test(model, test_loader)
            df_test['pred'] = pred
            df_test.to_csv(f"{root}/results/{train_mode}_{model_reg}_test_results.csv", index=False)

            # valid by best model for fair train
            pred = trainer.test(model, val_loader)
            df_valid['pred'] = pred
            df_valid.to_csv(f"{root}/results/{train_mode}_{model_reg}_valid_results.csv", index=False)

if __name__== "__main__":

    main()
