import os
import random
import numpy as np
import torch
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import timm
import argparse
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
PROJECT_ROOT = Path(__file__).parent # Path to folder opened files


def make_mobilenetv3(model, args=None, train_loader_len=None):
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, weight_decay=1e-5, momentum=0.9, nesterov=True)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=5)
    return opt, sch, False

def make_mobilevitv2(model, args, train_loader_len):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = train_loader_len * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    return opt, sch, True

def make_swin_transformerv2(model, args, train_loader_len):
    no_decay = ["bias", "ln_1.weight", "ln_2.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-5, amsgrad=True)
    num_training_steps = train_loader_len * args.epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return opt, sch, True


MODEL_REGISTRY = {
    "MobileVitV2": {
      "model_name": "mobilevitv2_200.cvnets_in1k",
      "make_optim": make_mobilevitv2,
  },
  "MobileNetV3": {
      "model_name": "mobilenetv3_small_100.lamb_in1k",
      "make_optim": make_mobilenetv3,
  },

  "SwinTV2": {
      "model_name": "swinv2_cr_small_ns_224.sw_in1k",
      "make_optim": make_swin_transformerv2,
  },
}

def main():
    root = PROJECT_ROOT / 'skin/UTKFace'
    df_train = pd.read_csv(f"{root}/df_train.csv", dtype={"label": str, "skin tone": str})
    df_valid = pd.read_csv(f"{root}/df_valid.csv", dtype={"label": str, "skin tone": str})
    df_test = pd.read_csv(f"{root}/df_test.csv", dtype={"label": str, "skin tone": str})

    le = LabelEncoder()
    train_loader = LoadDataset(df_train, "label", le, isTrain=True).get_dataloader()
    val_loader = LoadDataset(df_valid, "label", le, isTrain=False).get_dataloader()
    test_loader = LoadDataset(df_test, "label", le, isTrain=False).get_dataloader()

    for model_reg, model_conf in MODEL_REGISTRY.items():
        model_save_directory = f"{root}/models/{model_reg}/"
        utils.create_directory(model_save_directory)
        utils.delete_subfolders(model_save_directory)

        model = timm.create_model(model_conf['model_name'], pretrained=True, num_classes=2)
        make_optim = model_conf["make_optim"]
        optimizer, scheduler, batch_scheduler = make_optim(model, args, len(train_loader))

        loaders = {"train": train_loader, "valid": val_loader}
        training_settings = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            'batch_scheduler': batch_scheduler,
            "epochs": args.epochs,
            "model_save_directory": model_save_directory,
        }

        # train
        trainer = ModelTrainer()
        best_model_path = trainer.train(model, loaders, training_settings)

        # test
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        pred = trainer.test(model, test_loader)
        df_test['pred'] = pred
        df_test.to_csv(f"{root}/results/{model_conf['model_name']}_test_results.csv", index=False)

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    main()
