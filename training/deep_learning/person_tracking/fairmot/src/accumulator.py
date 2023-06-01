from .track import Track
import numpy as np
import motmetrics as mm
mm.lap.default_solver = "lap"


class Accumulator(object):

    def __init__(self,):
        super().__init__()
        self.mot_metric = mm.metrics.create()
        self.mot_accumulator = mm.MOTAccumulator(auto_id=True)

    @classmethod
    def tlbr2tlwh(cls, tlbr):
        return np.concatenate((tlbr[:, :2], tlbr[:, 2:] - tlbr[:, :2]), axis=-1)

    def reset(self,):
        self.mot_accumulator = mm.MOTAccumulator(auto_id=True)

    def update(self, state_tracks, bbox, person_id):
        tracks_bbox = np.asarray([track.bbox for track in state_tracks
            if track.state == Track.TrackState.TRACKED
        ], dtype=np.float32)
        if len(tracks_bbox) == 0:
            tracks_bbox = np.zeros((0, 4), dtype=np.float32)
        tracks_person_id = np.asarray([track.id for track in state_tracks
            if track.state == Track.TrackState.TRACKED
        ])
        iou_distance = mm.distances.iou_matrix(
            self.tlbr2tlwh(bbox), self.tlbr2tlwh(tracks_bbox), max_iou=0.5
        )
        self.mot_accumulator.update(
            person_id, tracks_person_id, iou_distance
        )

    def summary(self,):
        summary = self.mot_metric.compute(
            self.mot_accumulator, metrics=["mota", "motp"], name="acc"
        )
        return summary
