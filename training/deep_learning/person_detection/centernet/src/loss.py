import torch
from torch import nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        # self.obj_criterion = nn.BCEWithLogitsLoss(reduction="sum")
        self.obj_criterion = self.__focal_with_logits_loss
        self.reg_criterion = nn.L1Loss(reduction="sum")
        # self.reg_criterion = nn.MSELoss(reduction="mean")
        self.hm_weight = 1
        self.wh_weight = 0.1
        self.off_weight = 1

    @classmethod
    def __focal_with_logits_loss(cls, logits, obj):
        pos_inds = obj.eq(1).float()
        neg_inds = obj.lt(1).float()
        neg_weights = torch.pow(1 - obj, 4)
        p = logits.sigmoid()
        pos_loss = -F.logsigmoid(logits) * torch.pow(1 - p, 2) * pos_inds
        neg_loss = -F.logsigmoid(-logits) * torch.pow(p, 2) * neg_weights * neg_inds
        num_pos = pos_inds.sum()
        num_neg = neg_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # loss = pos_loss / max(num_pos, 1) + neg_loss / num_neg
        loss = (pos_loss + neg_loss) / max(num_pos, 1)
        return loss

    @classmethod
    def draw_gaussian(cls, thm, gi, gj, dimension):
        for x, y, (l, t, r, b) in zip(gi, gj, dimension):
            w = max(l, r)
            h = max(t, b)
            gaussian_x = torch.arange(-w, w + 1, device=thm.device)
            gaussian_y = torch.arange(-h, h + 1, device=thm.device)
            grid_y, grid_x = torch.meshgrid(gaussian_y, gaussian_x, indexing="ij")
            sigma_x = max(w, 1) / 2
            sigma_y = max(h, 1) / 2
            gaussian = torch.exp(
                - grid_x ** 2 / (2 * sigma_x ** 2) - grid_y ** 2 / (2 * sigma_y ** 2)
            )
            thm[y - t: y + b + 1, x - l: x + r + 1, :] = torch.maximum(
                thm[y - t: y + b + 1, x - l: x + r + 1, :],
                gaussian[h - t: h + b + 1, w - l: w + r + 1, None]
            )

    def encode(self, centers, dimensions, out_w, out_h):
        num_samples = len(centers)
        device = centers[0].device
        thm = torch.zeros((num_samples, out_h, out_w, 1), device=device)
        twh = torch.zeros((num_samples, out_h, out_w, 4), device=device)
        treg = torch.zeros((num_samples, out_h, out_w, 2), device=device)
        width, height = self.image_size
        ratio = torch.as_tensor([width / out_w, height / out_h], device=device)
        for i, (center, dimension) in enumerate(zip(centers, dimensions)):
            center = center / ratio[None, :]
            dimension = dimension / torch.tile(ratio, (2,))[None, :]
            gi = center[:, 0].long()
            gj = center[:, 1].long()
            self.draw_gaussian(thm[i, :, :, :], gi, gj, dimension.long())
            twh[i, gj, gi, :] = dimension
            treg[i, gj, gi, :] = center - center.long()
        return thm, twh, treg

    def forward(
        self, preds_hm, preds_wh, preds_reg, centers, dimensions
    ):
        _, _, out_h, out_w = preds_hm.shape
        thm, twh, treg = self.encode(centers, dimensions, out_w, out_h)
        preds_hm = preds_hm.permute(0, 2, 3, 1)
        preds_wh = preds_wh.permute(0, 2, 3, 1)
        preds_reg = preds_reg.permute(0, 2, 3, 1)
        mask = (thm == 1).squeeze(dim=-1)
        # TODO self.obj_criterion has reduction mean !
        hm_loss = self.obj_criterion(preds_hm , thm)
        wh_loss = self.reg_criterion(preds_wh[mask, :], twh[mask, :]) * mask.size(0) / mask.sum()
        off_loss = self.reg_criterion(preds_reg[mask, :], treg[mask, :]) * mask.size(0) / mask.sum()
        loss = (
            self.hm_weight * hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss
        )
        loss_stats = {
            "loss": loss, "hm_loss": hm_loss, "wh_loss": wh_loss, "off_loss": off_loss
        }
        return loss, loss_stats
