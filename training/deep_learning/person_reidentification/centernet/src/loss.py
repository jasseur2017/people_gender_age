import math
import torch
from torch import nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, image_size, reid_dim, nID):
        super().__init__()
        self.image_size = image_size
        self.classifier = nn.Linear(reid_dim, nID)
        self.reid_criterion = nn.CrossEntropyLoss(reduction="sum")
        self.emb_scale = math.sqrt(2) * math.log(nID - 1)

    def forward(self, preds_id, centers, person_ids):
        bs, nf, out_h, out_w = preds_id.shape
        width, height = self.image_size
        ratio = torch.as_tensor([width / out_w, height / out_h], device=preds_id.device)
        reid_losses = []
        for i, (center, person_id) in enumerate(zip(centers, person_ids)):
            if (len(center) == 0):
                continue
            center = center / ratio[None, :]
            ci = center[:, 0].long()
            cj = center[:, 1].long()
            pred_id = preds_id[i, :, cj, ci].permute(1, 0)
            assert pred_id.shape[-1] == nf
            pred_id = self.emb_scale * F.normalize(pred_id, dim=-1)
            pred_id = self.classifier(pred_id)
            reid_losses.append(self.reid_criterion(pred_id, person_id.to(torch.int64)))

        if len(reid_losses) == 0:
            loss = torch.tensor(0.0, device=preds_id.device, requires_grad=True)
        else:
            loss = sum(reid_losses)
        loss_stats = {"loss": loss}
        return loss, loss_stats
