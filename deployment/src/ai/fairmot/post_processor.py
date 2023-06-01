from .det_post_processor import PostProcessor as DetectionPostProcessor
from .reid_post_processor import PostProcessor as ReidPostProcessor


class PostProcessor(object):

    def __init__(self, image_size,):
        super().__init__()
        self.detection_post_processor = DetectionPostProcessor(image_size)
        self.reid_post_processor = ReidPostProcessor(image_size)

    def __call__(self, preds_hm, preds_wh, preds_reg, preds_id):
        preds_center, preds_dimension, preds_score = self.detection_post_processor(
            preds_hm, preds_wh, preds_reg
        )
        preds_id = self.reid_post_processor(preds_id, preds_center)
        return preds_center, preds_dimension, preds_score, preds_id
