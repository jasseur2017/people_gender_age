from enum import Enum
import numpy as np


class Track(object):

    class TrackState(Enum):
        NEW = 0
        TRACKED = 1
        LOST = 2
        REMOVED = 3

    def __init__(self, detection, id, time):
        super().__init__()
        x, y = detection["center"]
        l, t, r, b = detection["dimension"]
        w = l + r
        h = t + b
        self.mean = np.array([x - l, y - t, w / h, h, 0, 0, 0], dtype=np.float32)
        self.covariance = np.diag(
            np.square([h / 10, h / 10, 1e-1, h / 10, 100, 100, 100], dtype=np.float32)
        )
        self.direction = 0
        self.bbox = np.array([x - l, y - t, x + r, y + b], dtype=np.float32)
        self.id = id
        self.alpha = 0.9
        self.smooth_embedding = detection["embedding"]
        # TODO self.state = Track.TrackState.New
        self.state = Track.TrackState.TRACKED
        self.last_predict_time = time
        self.last_update_time = time

    def predict(self, kf, time):
        self.mean, self.covariance = kf.predict(
            time - self.last_predict_time, self.mean, self.covariance
        )
        self.last_predict_time = time

    def update(self, kf, direction_classifier, time, detection):
        x, y = detection["center"]
        l, t, r, b = detection["dimension"]
        w = l + r
        h = t + b
        measurement = np.array([x - l, y - t, w / h, h], dtype=np.float32)
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, measurement
        )
        self.direction = direction_classifier(
            old_center=(self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2),
            center=(x + (r - l) / 2, y + (b - t) / 2)
        )
        self.bbox = np.array([x - l, y - t, x + r, y + b], dtype=np.float32)
        self.smooth_embedding = (
            self.alpha * self.smooth_embedding + (1 - self.alpha) * detection["embedding"]
        )
        self.smooth_embedding /= (np.linalg.norm(self.smooth_embedding) + 1e-12)
        self.state = Track.TrackState.TRACKED
        self.last_update_time = time

    def gating_distance(self, kf, detection):
        x, y = detection["center"]
        l, t, r, b = detection["dimension"]
        w = l + r
        h = t + b
        measurement = np.array([x - l, y - t, w / h, h], dtype=np.float32)
        return kf.gating_distance(self.mean, self.covariance, measurement)

    def get_state_bbox(self,):
        x, y, r, h = self.mean[:4]
        w = r * h
        return np.array([x, y, x + w, y + h], dtype=np.float32)
