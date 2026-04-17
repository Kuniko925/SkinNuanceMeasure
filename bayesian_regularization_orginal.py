import torch


class BRFairLoss(torch.nn.Module):
    def __init__(self, br_models, reduction="mean"):
        super().__init__()
        self.br_models = br_models
        self.reduction = reduction

    def forward(self, y_pred, att, y_gt):

        att = att.reshape(-1)
        y_gt = y_gt.reshape(-1)

        class_losses = []
        classes = torch.unique(y_gt)

        for c in classes:
            mask = (y_gt == c)
            if mask.sum() == 0:
                continue
            att_c = att[mask]
            att_c = att_c.detach().cpu().numpy().reshape(-1, 1)
            class_tp = y_pred[mask][:, c]

            br_objects = self.br_models[int(c)]
            br = br_objects['brr_poly'] # regressor
            mu = br_objects['mu_value']

            br_value = br.predict(att_c)
            br_value = torch.tensor(br_value, device=y_pred.device, dtype=y_pred.dtype)
            mu = torch.tensor(mu, device=y_pred.device, dtype=y_pred.dtype)
            br_value = torch.maximum(br_value, mu)
            loss_c = torch.relu(br_value - class_tp) ** 2
            loss_c = loss_c.mean()
            class_losses.append(loss_c)

        if len(class_losses) == 0:
            return torch.tensor(0.0, device=y_pred.device)

        class_losses = torch.stack(class_losses)

        if self.reduction == "mean":
            return class_losses.mean()
        if self.reduction == "sum":
            return class_losses.sum()

        raise ValueError

