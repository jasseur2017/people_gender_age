import torch
from torch import nn


class Loss(nn.Module):

    def __init__(self, image_size,):
        super().__init__()
        self.image_size = image_size
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, preds, centers, labels):
        bs, nf, out_h, out_w = preds.shape
        width, height = self.image_size
        ratio = torch.as_tensor([width / out_w, height / out_h], device=preds.device)
        losses = []
        for i, (center, label) in enumerate(zip(centers, labels)):
            if (len(center) == 0):
                continue
            center = center / ratio[None, :]
            ci = center[:, 0].long()
            cj = center[:, 1].long()
            pred = preds[i, :, cj, ci].permute(1, 0)
            assert pred.shape[-1] == nf
            mask = ~torch.isnan(label)
            losses.append(self.criterion(pred[mask], label[mask].to(torch.int64)))

        if len(losses) == 0:
            loss = torch.tensor(0.0, device=preds.device, requires_grad=True)
        else:
            loss = sum(losses)
        loss_stats = {"loss": loss}
        return loss, loss_stats
