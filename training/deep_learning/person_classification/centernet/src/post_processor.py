import numpy as np
from scipy.special import softmax


class PostProcessor(object):

    def __init__(self, image_size,):
        super().__init__()
        self.image_size = image_size

    def __call__(self, preds, centers):
        _, nf, out_h, out_w = preds.shape
        width, height = self.image_size
        ratio = np.array([width / out_w, height / out_h], dtype=np.float32)
        processed_preds = []
        processed_scores = []
        for i, center in enumerate(centers):
            if len(center) == 0:
                pred = np.zeros(0, dtype=np.int64)
                score = np.ones(0, dtype=np.int64)
                processed_preds.append(pred)
                processed_scores.append(score)
                continue
            center = center / ratio[None, :]
            ci = center[:, 0].astype(np.long)
            cj = center[:, 1].astype(np.long)
            pred = preds[i, :, cj, ci]
            assert pred.shape[-1] == nf, pred.shape
            pred = softmax(pred, axis=-1)
            pred_i = np.argmax(pred, axis=-1)
            processed_preds.append(pred_i)
            score = pred[np.arange(len(center)), pred_i]
            processed_scores.append(score)
        return processed_preds, processed_scores