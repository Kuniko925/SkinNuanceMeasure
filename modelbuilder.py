import torch

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

