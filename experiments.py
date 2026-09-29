import torch
from pathlib import Path
import pandas as pd
import timm
from data import LoadDataset
from train import ModelTrainer
import utils
import random
import numpy as np
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import DATASETS, MODEL_REGISTRY, SEEDS

PROJECT_ROOT = Path(__file__).parent # Path to folder opened file
OUTPUT_DIR = PROJECT_ROOT / "results"

# training setting
EPOCHS = 50
BATCH_SIZE = 24

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():

    for seed in SEEDS:

        train_mode = 'reweight'
        #train_mode = 'baseline'
        #train_mode = 'regularization'
        train_mode = 'individual'

        for db_name in DATASETS:
            num_classes = 2
            root = PROJECT_ROOT / "skin" / db_name

            for model_name, model_id in MODEL_REGISTRY.items():

                set_seed(seed)

                df_train = pd.read_csv(f"{root}/df_train.csv")
                df_valid = pd.read_csv(f"{root}/df_valid.csv")
                df_test = pd.read_csv(f"{root}/df_test.csv")

                train_loader = LoadDataset(
                    df_train,
                    batch_size=BATCH_SIZE,
                    isTrain=True,
                    seed=seed
                ).get_dataloader()

                val_loader = LoadDataset(
                    df_valid,
                    batch_size=BATCH_SIZE,
                    isTrain=False,
                    seed=seed
                ).get_dataloader()

                test_loader = LoadDataset(
                    df_test,
                    batch_size=BATCH_SIZE,
                    isTrain=False,
                    seed=seed
                ).get_dataloader()

                model_save_directory = (PROJECT_ROOT / "models" / db_name / train_mode / model_name / f"seed_{seed}")
                utils.create_directory(model_save_directory)
                utils.delete_subfolders(model_save_directory)

                model = timm.create_model(model_id, pretrained=True, num_classes=num_classes)
                loaders = {"train": train_loader, "valid": val_loader}

                # train
                trainer = ModelTrainer(train_loader, model_name=model_name, db_name=db_name, train_mode=train_mode)
                best_model_path = trainer.train(model, loaders, epochs=EPOCHS, model_save_directory=model_save_directory)

                # test
                model.load_state_dict(torch.load(best_model_path, weights_only=True))
                pred, labels, pred_logits, label_logits, prob = trainer.test(model, test_loader)
                df_test['pred'] = pred
                df_test['pred_logits'] = pred_logits
                df_test['label_logits'] = label_logits
                for i in range(prob.shape[1]):
                    df_test[f'prob_{i}'] = prob[:, i]

                test_result_path = (
                        OUTPUT_DIR
                        / f"{db_name}_{train_mode}_{model_name}_seed{seed}_test.csv"
                )

                df_test.to_csv(
                    test_result_path,
                    index=False
                )

                # valid by best model for fair train
                pred, labels, pred_logits, label_logits, prob = trainer.test(model, val_loader)
                df_valid['pred'] = pred
                df_valid['pred_logits'] = pred_logits
                df_valid['label_logits'] = label_logits
                for i in range(prob.shape[1]):
                    df_valid[f'prob_{i}'] = prob[:, i]

                valid_result_path = (
                        OUTPUT_DIR
                        / f"{db_name}_{train_mode}_{model_name}_seed{seed}_valid.csv"
                )

                df_valid.to_csv(
                    valid_result_path,
                    index=False
                )

if __name__== "__main__":

    main()
