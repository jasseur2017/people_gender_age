from .matcher import Matcher
import sys
sys.path.append("../../..")
from person_detection.centernet.src.accumulator import AveragePrecisionAccumulator
import numpy as np


class IouMatcher(Matcher):

    def __init__(self, state, cost_limit):
        super().__init__(cost_limit=cost_limit)
        self.state = state

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
        return AveragePrecisionAccumulator.iou_distance(tracks_bbox, detections_bbox)

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
