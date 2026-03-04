import os
import random
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import timm
from data import LoadDataset
from train import ModelTrainer
from transformers import get_cosine_schedule_with_warmup
import utils

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

def param_groups_weight_decay(model: torch.nn.Module, weight_decay: float = 0.01):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias") or param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def make_mobilenetv3(model, epochs, train_loader_len):
    opt = torch.optim.AdamW(param_groups_weight_decay(model, 1e-2), lr=1e-3)
    num_training_steps = train_loader_len * epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return opt, sch, True

def make_mobilevitv2(model, epochs, train_loader_len):
    no_decay = ["bias", "ln_1.weight", "ln_2.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5, amsgrad=True)
    num_training_steps = train_loader_len * epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return opt, sch, True

def make_swin_transformerv2(model, epochs, train_loader_len):
    no_decay = ["bias", "ln_1.weight", "ln_2.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-5, amsgrad=True)
    num_training_steps = train_loader_len * epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return opt, sch, True


MODEL_REGISTRY = {
    "MobileNetV3": {
        "model_name": "tf_mobilenetv3_small_100.in1k",
        "make_optim": make_mobilenetv3,
        "beta": 0.05,
        "batch_size": 32
    },
    "MobileVitV2": {
        "model_name": "mobilevitv2_200",
        "make_optim": make_mobilevitv2,
        "beta": 0.01,
        "batch_size": 32
     },
        "SwinTV2": {
           "model_name": "swinv2_cr_small_ns_224.sw_in1k",
           "make_optim": make_swin_transformerv2,
           "beta": 0.001,
          "batch_size": 16
     },
}

DB_REGISTRY = {
    "UTKFace" : {'root': Path(PROJECT_ROOT / 'skin/UTKFace/'), 'num_classes': 2, 'epochs': 50},
    "dermoscopic": {'root': Path(PROJECT_ROOT / 'skin/milk10k/dermoscopic/'), 'num_classes': 6, 'epochs': 200},
    "clinical": {'root': Path(PROJECT_ROOT / 'skin/milk10k/clinical/'), 'num_classes': 6, 'epochs': 200},
}

def main():

    for db_name, db in DB_REGISTRY.items():
        root = DB_REGISTRY[db_name]['root']
        num_classes = DB_REGISTRY[db_name]['num_classes']
        epochs = DB_REGISTRY[db_name]['epochs']

        for model_reg, model_conf in MODEL_REGISTRY.items():

            batch_size = MODEL_REGISTRY[model_reg]['batch_size']
            df_train = pd.read_csv(f"{root}/df_train.csv", dtype={"label": str, "skin tone": str})
            df_valid = pd.read_csv(f"{root}/df_valid.csv", dtype={"label": str, "skin tone": str})
            df_test = pd.read_csv(f"{root}/df_test.csv", dtype={"label": str, "skin tone": str})

            le = LabelEncoder()
            train_loader = LoadDataset(df_train, "label", le, batch_size=batch_size, isTrain=True).get_dataloader()
            val_loader = LoadDataset(df_valid, "label", le, batch_size=batch_size, isTrain=False).get_dataloader()
            test_loader = LoadDataset(df_test, "label", le, batch_size=batch_size, isTrain=False).get_dataloader()

            model_save_directory = f"{root}/models/{model_reg}/"
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
            trainer = ModelTrainer(train_loader, train_method="baseline", beta=model_conf["beta"])
            best_model_path = trainer.train(model, loaders, training_settings)

            # test
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            pred = trainer.test(model, test_loader)
            df_test['pred'] = pred
            df_test.to_csv(f"{root}/results/{model_reg}_test_results.csv", index=False)

            # valid by best model for fair train
            pred = trainer.test(model, val_loader)
            df_valid['pred'] = pred
            df_valid.to_csv(f"{root}/results/{model_reg}_valid_results.csv", index=False)

if __name__== "__main__":

    main()
