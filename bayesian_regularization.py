import torch

class BRFairLoss(torch.nn.Module):
    def __init__(self, br_models):
        super().__init__()
        self.br_models = br_models

    def forward(self, ce_loss, att, y_gt):
        att = att.reshape(-1)
        y_gt = y_gt.reshape(-1)

        losses = []

        for c in torch.unique(y_gt):
            mask = (y_gt == c)
            att_c = att[mask].detach().cpu().numpy().reshape(-1, 1)

            br = self.br_models[int(c)]['brr_poly']
            br_value = br.predict(att_c)

            br_value = torch.tensor(
                br_value,
                device=ce_loss.device,
                dtype=ce_loss.dtype
            )

            weight = 1.0 - br_value
            loss_c = weight * ce_loss[mask]
            losses.append(loss_c.mean())

        if len(losses) == 0:
            return torch.tensor(0.0, device=ce_loss.device)

        return torch.stack(losses).mean()

