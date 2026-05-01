import torch

class BRFairLoss(torch.nn.Module):
    def __init__(self, br_models):
        super().__init__()
        self.br_models = br_models

    def forward(self, y_pred, att, y_gt):

        att = att.reshape(-1)
        y_gt = y_gt.reshape(-1)
        probs = torch.softmax(y_pred, dim=1)

        class_losses = []
        classes = torch.unique(y_gt)

        for c in classes:
            mask = (y_gt == c)
            if mask.sum() == 0:
                continue
            att_c = att[mask]
            att_c = att_c.detach().cpu().numpy().reshape(-1, 1)

            br_objects = self.br_models[int(c)]
            br = br_objects['brr_poly']  # regressor
            mu = br_objects['mu_value']
            br_value = br.predict(att_c)
            br_value = torch.tensor(br_value, device=y_pred.device, dtype=y_pred.dtype)
            mu = torch.tensor(mu, device=y_pred.device, dtype=y_pred.dtype)

            weight = (1.0 - br_value).detach()
            ce_c = torch.nn.functional.cross_entropy(y_pred[mask], y_gt[mask], reduction="none")
            loss_c = weight * ce_c
            class_losses.append(loss_c.mean())

        if len(class_losses) == 0:
            return torch.tensor(0.0, device=y_pred.device)
        return torch.stack(class_losses).mean()

