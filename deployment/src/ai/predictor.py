import numpy as np
from .trt_model import TrtModel
from .fairmot.post_processor import PostProcessor
from .fairmot.kalman_filter import KalmanFilter
from .fairmot.matcher import CompositeMatcher
from .fairmot.embedding_matcher import EmbeddingMatcher
from .fairmot.iou_matcher import IouMatcher
from .fairmot.tracker import Tracker
from .fairmot.track import Track
from .direction_classifier import DirectionClassifier


class Predictor(object):

    def __init__(self, onnx_file, image_size):
        super().__init__()
        self.trt_model = TrtModel(onnx_file)
        self.post_processor = PostProcessor(image_size)
        kf = KalmanFilter()
        matcher = CompositeMatcher(
            EmbeddingMatcher(kf, cost_limit=0.4),
            IouMatcher(state=Track.TrackState.TRACKED, cost_limit=0.5),
            IouMatcher(state=Track.TrackState.LOST, cost_limit=0.7)
        )
        direction_classifier = DirectionClassifier(
            (0, 0.8 * image_size[1]), (image_size[0], 0.8 * image_size[1])
        )
        self.tracker = Tracker(kf, matcher, direction_classifier)

    def predict(self, time, states_tracks, states_count, images):
        images = images.transpose(0, 3, 1, 2)
        images = np.ascontiguousarray(images)
        preds = self.trt_model(images)
        preds_center, preds_dimension, preds_score, preds_id = self.post_processor(
            preds["hm"], preds["wh"], preds["reg"], preds["id"]
        )
        for i, (pred_center, pred_dimension, pred_score, pred_id) in enumerate(
            zip(preds_center, preds_dimension, preds_score, preds_id)
            ):
            states_tracks[i], states_count[i] = self.tracker(
                time, states_tracks[i], states_count[i],
                pred_center, pred_dimension, pred_score, pred_id
            )
        return states_tracks, states_count
