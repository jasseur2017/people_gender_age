import numpy as np
from .matcher import Matcher


class IouMatcher(Matcher):

    def __init__(self, state, cost_limit):
        super().__init__(cost_limit=cost_limit)
        self.state = state

    @classmethod
    def bboxes_iou_distance(cls, bboxes_1, bboxes_2):
        area1 = (bboxes_1[:, 2] - bboxes_1[:, 0]) * (bboxes_1[:, 3] - bboxes_1[:, 1])
        area2 = (bboxes_2[:, 2] - bboxes_2[:, 0]) * (bboxes_2[:, 3] - bboxes_2[:, 1])
        width = np.maximum(0.0, (
            np.minimum(bboxes_1[:, 2, None], bboxes_2[:, 2]) -
            np.maximum(bboxes_1[:, 0, None], bboxes_2[:, 0])
        ))
        height = np.maximum(0.0, (
            np.minimum(bboxes_1[:, 3, None], bboxes_2[:, 3]) -
            np.maximum(bboxes_1[:, 1, None], bboxes_2[:, 1])
        ))
        inter = width * height
        ovr = inter / (area1[:, None] + area2[None, :] - inter)
        return 1 - ovr

    @classmethod
    def iou_distance(cls, tracks, detections):
        if (len(tracks) == 0) or (len(detections) == 0):
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        tracks_bbox = np.asarray([track.get_state_bbox() for track in tracks])
        detections_bbox = np.asarray([[
            detection["center"][0] - detection["dimension"][0],
            detection["center"][1] - detection["dimension"][1],
            detection["center"][0] + detection["dimension"][2],
            detection["center"][1] + detection["dimension"][3],
            ] for detection in detections
        ])
        return cls.bboxes_iou_distance(tracks_bbox, detections_bbox)

    def associate(self, unassociated_tracks, unassociated_detections):
        state_unassociated_tracks = [
            track for track in unassociated_tracks if track.state == self.state
        ]
        not_state_unassociated_tracks = [
            track for track in unassociated_tracks if track.state != self.state
        ]
        dists = self.iou_distance(state_unassociated_tracks, unassociated_detections)
        (
            associated_tracks_id, associated_detections_id,
            unassociated_tracks_id, unassociated_detections_id
        ) = self.linear_assignment(dists)
        associated_tracks = [
            state_unassociated_tracks[i] for i in associated_tracks_id
        ]
        state_unassociated_tracks = [
            state_unassociated_tracks[i] for i in unassociated_tracks_id
        ]
        unassociated_tracks = state_unassociated_tracks + not_state_unassociated_tracks
        associated_detections = [
            unassociated_detections[i] for i in associated_detections_id
        ]
        unassociated_detections = [
            unassociated_detections[i] for i in unassociated_detections_id
        ]
        return associated_tracks, associated_detections, unassociated_tracks, unassociated_detections
