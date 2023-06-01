import sys
sys.path.append("../../../")
from person_detection.centernet.src.post_processor import PostProcessor as DetectionPostProcessor
from person_reidentification.centernet.src.post_processor import PostProcessor as ReidPostProcessor
from person_classification.centernet.src.post_processor import PostProcessor as ClassificationPostProcessor
from .kalman_filter import KalmanFilter
from .matcher import CompositeMatcher
from .embedding_matcher import EmbeddingMatcher
from .iou_matcher import IouMatcher
from .track import Track
from .tracker import Tracker


class PostProcessor(object):

    def __init__(self, image_size,):
        super().__init__()
        self.detection_post_processor = DetectionPostProcessor(image_size, conf_threshold=0.2)
        self.reid_post_processor = ReidPostProcessor(image_size)
        self.gender_post_processor = ClassificationPostProcessor(image_size)
        self.age_post_processor = ClassificationPostProcessor(image_size)
        kf = KalmanFilter()
        matcher = CompositeMatcher(
            EmbeddingMatcher(kf, cost_limit=0.4),
            IouMatcher(state=Track.TrackState.TRACKED, cost_limit=0.5),
            IouMatcher(state=Track.TrackState.LOST, cost_limit=0.5)
        )
        self.tracker = Tracker(kf, matcher, new_track_threshold=0.4, max_lost_time=4 / 30)

    def __call__(self, t, state_tracks, state_count, preds_hm, preds_wh, preds_reg, preds_id, preds_gender, preds_age):
        preds_center, preds_dimension, preds_score = self.detection_post_processor(
            preds_hm, preds_wh, preds_reg
        )
        preds_id = self.reid_post_processor(preds_id, preds_center)
        preds_gender, scores_gender = self.gender_post_processor(preds_gender, preds_center)
        preds_age, scores_age = self.age_post_processor(preds_age, preds_center)
        state_tracks, state_count = self.tracker(
            t, state_tracks, state_count,
            preds_center[0], preds_dimension[0], preds_score[0], preds_id[0], preds_gender[0], preds_age[0]
        )
        return state_tracks, state_count
