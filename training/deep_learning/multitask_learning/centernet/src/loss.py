import torch
from torch import nn
import sys
sys.path.append("../../../")
from person_detection.centernet.src.loss import Loss as DetectionLoss
from person_reidentification.centernet.src.loss import Loss as ReidLoss
from person_classification.centernet.src.loss import Loss as ClassificationLoss


class Loss(nn.Module):

    def __init__(self, image_size, reid_dim):
        super().__init__()
        self.det_loss = DetectionLoss(image_size)
        self.reid_loss = ReidLoss(image_size, reid_dim, nID=1000)
        self.gender_loss = ClassificationLoss(image_size)
        self.age_loss = ClassificationLoss(image_size)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.s_gender = nn.Parameter(-1.05 * torch.ones(1))
        self.s_age = nn.Parameter(-1.05 * torch.ones(1))

    def forward(
        self, preds, centers, dimensions, i_dimensions, person_ids, i_person_ids, genders, i_genders, ages, i_ages
    ):
        preds_hm, preds_wh, preds_reg, preds_id, preds_gender, preds_age = (
            preds[0]["hm"], preds[0]["wh"], preds[0]["reg"], preds[0]["id"],
            preds[0]["gender"], preds[0]["age"]
        )
        det_loss, _ = self.det_loss(
            preds_hm[i_dimensions, ...], preds_wh[i_dimensions, ...], preds_reg[i_dimensions, ...],
            [centers[i] for i in i_dimensions], dimensions
        )
        reid_loss, _ = self.reid_loss(
            preds_id[i_person_ids, ...], [centers[i] for i in i_person_ids], person_ids
        )
        gender_loss, _ = self.gender_loss(
            preds_gender[i_genders, ...], [centers[i] for i in i_genders], genders
        )
        age_loss, _ = self.age_loss(
            preds_age[i_ages, ...], [centers[i] for i in i_ages], ages
        )
        loss = 0.5 * (
           torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * reid_loss +
           torch.exp(-self.s_gender) * gender_loss + torch.exp(-self.s_age) * age_loss
           + self.s_det + self.s_id + self.s_gender + self.s_age
        )
        loss_stats = {
            "loss": loss, "det_loss": det_loss, "reid_loss": reid_loss,
            "gender_loss": gender_loss, "age_loss": age_loss
        }
        return loss, loss_stats
