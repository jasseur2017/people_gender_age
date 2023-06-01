import torch
from torch import nn
# TODO from skimage.measure import block_reduce


class PreDetector(nn.Module):

    def __init__(self,):
        super().__init__()
        self.max_pool = nn.MaxPool2d((3, 3), stride=1, padding=1)

    @torch.no_grad()
    def forward(self, preds_hm):
        preds_hm = preds_hm.sigmoid()
        hmax = self.max_pool(preds_hm)
        keep = (hmax == preds_hm).float()
        preds_hm = preds_hm * keep
        return preds_hm
