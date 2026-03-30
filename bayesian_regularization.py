import torch
import torch.nn.functional as F

class DiffEOpp(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(DiffEOpp, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, s, y_gt):
        if y_pred.ndim != 2:
            raise ValueError("y_pred must have shape [N, C]")

        s = s.reshape(-1)
        y_gt = y_gt.reshape(-1)

        probs = F.softmax(y_pred, dim=1)

        classes = torch.unique(y_gt)
        groups = torch.unique(s)

        class_losses = []

        for c in classes:
            pos_mask = (y_gt == c)
            if pos_mask.sum() == 0:
                continue

            class_scores = probs[pos_mask, c]
            class_groups = s[pos_mask]

            group_means = []
            for g in groups:
                g_mask = (class_groups == g)
                if g_mask.sum() == 0:
                    continue
                group_means.append(class_scores[g_mask].mean())

            if len(group_means) <= 1:
                continue

            group_means = torch.stack(group_means)

            # max-min gap
            loss_c = group_means.max() - group_means.min()
            class_losses.append(loss_c)

        if len(class_losses) == 0:
            return torch.tensor(0.0, device=y_pred.device)

        class_losses = torch.stack(class_losses)

        if self.reduction == "mean":
            return class_losses.mean()
        elif self.reduction == "sum":
            return class_losses.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")
