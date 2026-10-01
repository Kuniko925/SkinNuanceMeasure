import torch

class BRFairLoss(torch.nn.Module):
    """Normalize WD weights within each class, then average present classes."""

    def __init__(self, br_models, eps=1e-8):
        super().__init__()
        self.br_models = br_models
        self.eps = eps

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
            weighted_loss = (
                (weight * ce_loss[mask]).sum()
                / (weight.sum() + self.eps)
            )
            losses.append(weighted_loss)

        if len(losses) == 0:
            return ce_loss.sum() * 0.0

        return torch.stack(losses).mean()

