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
import pickle

# to reproduce
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


PROJECT_ROOT = Path(__file__).parent # Path to folder opened file

DB_REGISTRY = {
    "UTKFace" : {'root': Path(PROJECT_ROOT / 'skin/UTKFace/'), 'num_classes': 4, 'epochs': 50, 'folder': 'UTKFace'},
    "dermoscopic": {'root': Path(PROJECT_ROOT / 'skin/milk10k/dermoscopic/'), 'num_classes': 2, 'epochs': 50, 'folder': 'milk10k/dermoscopic'},
    "clinical": {'root': Path(PROJECT_ROOT / 'skin/milk10k/clinical/'), 'num_classes': 2, 'epochs': 50, 'folder': 'milk10k/clinical'},
}

def load_brfair_regressors(db_name, model_name, num_classes):
    regressors = {}
    for cls in range(num_classes):
        br_filepath = PROJECT_ROOT / f"skin/{db_name}/bayes/{model_name}_{cls}.pkl"
        with open(br_filepath, "rb") as f:
            regressors[cls] = pickle.load(f)
    return regressors


def main():

    set_seed(seed=42)
    #train_mode = 'reweight'
    #train_mode = 'baseline'
    #train_mode = 'regularization'
    train_mode = 'individual'

    for db_name, db in DB_REGISTRY.items():
        root = DB_REGISTRY[db_name]['root']
        num_classes = DB_REGISTRY[db_name]['num_classes']
        epochs = DB_REGISTRY[db_name]['epochs']
        folder = DB_REGISTRY[db_name]['folder']

        for model_reg, model_conf in mb.MODEL_REGISTRY.items():

            batch_size = mb.MODEL_REGISTRY[model_reg]['batch_size']
            df_train = pd.read_csv(f"{root}/df_train.csv", dtype={"label": str, "skin tone": str})
            df_valid = pd.read_csv(f"{root}/df_valid.csv", dtype={"label": str, "skin tone": str})
            df_test = pd.read_csv(f"{root}/df_test.csv", dtype={"label": str, "skin tone": str})

            set_seed(42)
            le = LabelEncoder()
            train_loader = LoadDataset(df_train, "label", le, batch_size=batch_size, isTrain=True).get_dataloader()
            val_loader = LoadDataset(df_valid, "label", le, batch_size=batch_size, isTrain=False).get_dataloader()
            test_loader = LoadDataset(df_test, "label", le, batch_size=batch_size, isTrain=False).get_dataloader()

            model_save_directory = f"{root}/models/{train_mode}/{model_reg}/"
            utils.create_directory(model_save_directory)
            utils.delete_subfolders(model_save_directory)

            set_seed(seed=42)
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

            brfair_regressors = None
            if train_mode == "individual":
                brfair_regressors = load_brfair_regressors(
                    db_name=folder,
                    model_name=model_reg,
                    num_classes=num_classes,
                )

            # train
            trainer = ModelTrainer(train_loader, model_name=model_reg, train_mode=train_mode, brfair_regressors=brfair_regressors)
            best_model_path = trainer.train(model, loaders, training_settings)

            # test
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            pred, labels, pred_logits, label_logits, prob = trainer.test(model, test_loader)
            df_test['pred'] = pred
            df_test['pred_logits'] = pred_logits
            df_test['label_logits'] = label_logits
            for i in range(prob.shape[1]):
                df_test[f'prob_{i}'] = prob[:, i]
            df_test.to_csv(f"{root}/results/{train_mode}_{model_reg}_test_results.csv", index=False)

            # valid by best model for fair train
            pred, labels, pred_logits, label_logits, prob = trainer.test(model, val_loader)
            df_valid['pred'] = pred
            df_valid['pred_logits'] = pred_logits
            df_valid['label_logits'] = label_logits
            for i in range(prob.shape[1]):
                df_valid[f'prob_{i}'] = prob[:, i]
            df_valid.to_csv(f"{root}/results/{train_mode}_{model_reg}_valid_results.csv", index=False)

if __name__== "__main__":

    main()
