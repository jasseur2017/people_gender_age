import numpy as np
# from scipy.special import expit as sigmoid


class PostProcessor(object):

    def __init__(self, image_size,):
        super().__init__()
        self.image_size = image_size
        self.conf_threshold = 0.3
        # self.min_area = 10.0
        # self.min_aspect_ratio = 0.9

    def decode(self, preds_hm, preds_wh, preds_reg):
        width, height = self.image_size
        num_samples, _, out_h, out_w = preds_hm.shape
        ratio = np.array([width / out_w, height / out_h], dtype=np.float32)
        preds_center, preds_dimension, preds_score = [], [], []
        for i in range(num_samples):
            pred_hm = preds_hm[i, :, :, :]
            c, y, x = np.nonzero(pred_hm > self.conf_threshold)
            pred_score = pred_hm[c, y, x]
            pred_x = x.astype(np.float32) + preds_reg[i, 0, y, x]
            pred_y = y.astype(np.float32) + preds_reg[i, 1, y, x]
            pred_center = np.stack((pred_x, pred_y), axis=1)
            pred_center = pred_center * ratio[None, :]
            pred_dimension = preds_wh[i, :, y, x] * np.tile(ratio, 2)[None, :]
            preds_center.append(pred_center)
            preds_dimension.append(pred_dimension)
            preds_score.append(pred_score)
        return preds_center, preds_dimension, preds_score

    def filter(self, preds_center, preds_dimension, preds_score):
        processed_preds_center = []
        processed_preds_dimension = []
        processed_preds_score = []
        width, height = self.image_size
        for pred_center, pred_dimension, pred_score in zip(
            preds_center, preds_dimension, preds_score
        ):
            pred_dimension[:, 0] = np.clip(pred_dimension[:, 0], 0, pred_center[:, 0])
            pred_dimension[:, 1] = np.clip(pred_dimension[:, 1], 0, pred_center[:, 1])
            pred_dimension[:, 2] = np.clip(pred_dimension[:, 2], 0, width - 1 - pred_center[:, 0])
            pred_dimension[:, 3] = np.clip(pred_dimension[:, 3], 0, height - 1 - pred_center[:, 1])
            # # remove small areas and bad aspect ratio
            # pred_area = (
            #     (pred_dimension[:, 0] + pred_dimension[:, 2]) *
            #     (pred_dimension[:, 1] + pred_dimension[:, 3])
            # )
            # pred_aspect_ratio = (
            #     (pred_dimension[:, 1] + pred_dimension[:, 3]) /
            #     (pred_dimension[:, 0] + pred_dimension[:, 2])
            # )
            # keep = (pred_area > self.min_area) & (pred_aspect_ratio > self.min_aspect_ratio)
            # pred_center = pred_center[keep, :]
            # pred_dimension = pred_dimension[keep, :]
            # pred_score = pred_score[keep]
            processed_preds_center.append(pred_center)
            processed_preds_dimension.append(pred_dimension)
            processed_preds_score.append(pred_score)
        return processed_preds_center, processed_preds_dimension, processed_preds_score


    def __call__(self, preds_hm, preds_wh, preds_reg):
        preds_center, preds_dimension, preds_score = self.decode(
            preds_hm, preds_wh, preds_reg
        )
        preds_center, preds_dimension, preds_score = self.filter(
            preds_center, preds_dimension, preds_score
        )
        return preds_center, preds_dimension, preds_score
