from enum import IntEnum
import numpy as np


class Track(object):

    class TrackState(IntEnum):
        # Enum causes equality issue between jupyter and python
        NEW = 0
        TRACKED = 1
        LOST = 2
        REMOVED = 3

    def __init__(self, detection, id, time):
        super().__init__()
        x, y = detection["center"]
        wm, hm, wp, hp = detection["dimension"]
        w = wm + wp
        h = hm + hp
        self.mean = np.array([x - wm, y - hm, w / h, h, 0, 0, 0], dtype=np.float32)
        self.covariance = np.diag(
            np.square([w / 10, h / 10, (w / h) / 100, h / 10, 1000, 1000, 1000], dtype=np.float32)
        )
        self.bbox = np.array([x - wm, y - hm, x + wp, y + hp], dtype=np.float32)
        self.id = id
        self.alpha = 0.9
        self.smooth_embedding = detection["embedding"]
        self.gender = [detection["gender"]]
        self.age = [detection["age"]]
        # TODO self.state = Track.TrackState.New
        self.state = Track.TrackState.TRACKED
        self.last_predict_time = time
        self.last_update_time = time

    def predict(self, kf, time):
        self.mean, self.covariance = kf.predict(
            time - self.last_predict_time, self.mean, self.covariance
        )
        self.last_predict_time = time

    def update(self, kf, time, detection):
        x, y = detection["center"]
        wm, hm, wp, hp = detection["dimension"]
        w = wm + wp
        h = hm + hp
        measurement = np.array([x - wm, y - hm, w / h, h], dtype=np.float32)
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, measurement
        )
        self.bbox = np.array([x - wm, y - hm, x + wp, y + hp], dtype=np.float32)
        self.smooth_embedding = (
            self.alpha * self.smooth_embedding + (1 - self.alpha) * detection["embedding"]
        )
        self.smooth_embedding /= (np.linalg.norm(self.smooth_embedding) + 1e-12)
        self.gender.append(detection["gender"])
        self.age.append(detection["age"])
        self.state = Track.TrackState.TRACKED
        self.last_update_time = time

    def gating_distance(self, kf, detection):
        x, y = detection["center"]
        wm, hm, wp, hp = detection["dimension"]
        w = wm + wp
        h = hm + hp
        measurement = np.array([x - wm, y - hm, w / h, h], dtype=np.float32)
        return kf.gating_distance(self.mean, self.covariance, measurement)

    def get_state_bbox(self,):
        x, y, r, h = self.mean[:4]
        w = r * h
        return np.array([x, y, x + w, y + h], dtype=np.float32)
    
    def getGender(self,):
        return max(set(self.gender), key=self.gender.count)
    
    def getAge(self,):
        return max(set(self.age), key=self.age.count)
