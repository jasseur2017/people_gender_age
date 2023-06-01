import numpy as np


class PostProcessor(object):

    def __init__(self, image_size,):
        super().__init__()
        self.image_size = image_size

    def __call__(self, preds_id, centers):
        _, nf, out_h, out_w = preds_id.shape
        width, height = self.image_size
        ratio = np.array([width / out_w, height / out_h], dtype=np.float32)
        processed_preds_id = []
        for i, center in enumerate(centers):
            if len(center) == 0:
                pred_id = np.zeros((0, nf), dtype=np.float32)
                processed_preds_id.append(pred_id)
                continue
            center = center / ratio[None, :]
            ci = center[:, 0].astype(np.long)
            cj = center[:, 1].astype(np.long)
            pred_id = preds_id[i, :, cj, ci]
            assert pred_id.shape[-1] == nf, pred_id.shape
            pred_id = pred_id / (np.linalg.norm(pred_id, axis=-1, keepdims=True) + 1e-12)
            processed_preds_id.append(pred_id)
        return processed_preds_id