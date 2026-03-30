import torch
from transformers import get_cosine_schedule_with_warmup

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

def make_cnn_base(model, epochs, train_loader_len):
    opt = torch.optim.AdamW(param_groups_weight_decay(model, 1e-4), lr=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    return opt, sch, False

def make_hybrid(model, epochs, train_loader_len):
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

def make_transformers(model, epochs, train_loader_len):
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


MODEL_REGISTRY = {
    "ConvNext": {
        "model_name": "convnext_tiny.fb_in22k",
        "make_optim": make_cnn_base,
        "batch_size": 32
    },
    "FastVit": {
        "model_name": "fastvit_s12.apple_in1k",
        "make_optim": make_hybrid,
        "batch_size": 32
    },
    "MobileVitV2": {
        "model_name": "mobilevitv2_200",
        "make_optim": make_hybrid,
        "batch_size": 32
    },
    "VitTiny": {
        "model_name": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "make_optim": make_transformers,
        "batch_size": 32
    },
    "DeitTiny": {
        "model_name": "deit_tiny_patch16_224.fb_in1k",
        "make_optim": make_transformers,
        "batch_size": 32
    },
}