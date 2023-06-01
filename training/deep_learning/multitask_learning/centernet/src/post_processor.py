import sys
sys.path.append("../../../")
from person_detection.centernet.src.post_processor import PostProcessor as DetectionPostProcessor
from person_reidentification.centernet.src.post_processor import PostProcessor as ReidPostProcessor
from person_classification.centernet.src.post_processor import PostProcessor as ClassificationPostProcessor


class PostProcessor(object):

    def __init__(self, image_size,):
        super().__init__()
        self.det_post_processor = DetectionPostProcessor(image_size, conf_threshold=0.3)
        self.reid_post_processor = ReidPostProcessor(image_size)
        self.gender_post_processor = ClassificationPostProcessor(image_size)
        self.age_post_processor = ClassificationPostProcessor(image_size)

    def __call__(self, preds, centers):
        preds_hm, preds_wh, preds_reg, preds_id, preds_gender, preds_age = (
            preds[0]["hm"].detach().cpu().numpy(), preds[0]["wh"].detach().cpu().numpy(),
            preds[0]["reg"].detach().cpu().numpy(), preds[0]["id"].detach().cpu().numpy(),
            preds[0]["gender"].detach().cpu().numpy(), preds[0]["age"].detach().cpu().numpy()
        )
        centers = [center.cpu().numpy() for center in centers]
        preds_center, preds_dimension, preds_score = self.det_post_processor(
            preds_hm, preds_wh, preds_reg
        )
        preds_id = self.reid_post_processor(preds_id, centers)
        preds_gender, scores_gender = self.gender_post_processor(preds_gender, centers)
        preds_age, scores_age = self.age_post_processor(preds_age, centers)
        return preds_center, preds_dimension, preds_id, preds_gender, preds_age
