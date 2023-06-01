import numpy as np
import lap


class Matcher(object):

    def __init__(self, cost_limit):
        super().__init__()
        self.cost_limit = cost_limit

    def linear_assignment(self, cost_matrix):
        if cost_matrix.size == 0:
            tracks_size, detections_size = cost_matrix.shape
            return (
                np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
                np.arange(tracks_size), np.arange(detections_size)
            )
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=self.cost_limit)
        associated_tracks_id, = np.nonzero(x != -1)
        associated_detections_id = x[associated_tracks_id]
        unassociated_tracks_id, = np.nonzero(x == -1)
        unassociated_detections_id, = np.nonzero(y == -1)
        return (
            associated_tracks_id, associated_detections_id,
            unassociated_tracks_id, unassociated_detections_id
        )

    def associate(self, unassociated_tracks, unassociated_detections):
        raise NotImplementedError


class CompositeMatcher(Matcher):

    def __init__(self, *matchers):
        super().__init__(cost_limit=0.)
        self.matchers = matchers

    def associate(self, unassociated_tracks, unassociated_detections):
        associated_tracks = []
        associated_detections = []
        for matcher in self.matchers:
            (
                sub_associated_tracks, sub_associated_detections,
                unassociated_tracks, unassociated_detections
                ) = matcher.associate(
                unassociated_tracks, unassociated_detections
            )
            associated_tracks.extend(sub_associated_tracks)
            associated_detections.extend(sub_associated_detections)
        return associated_tracks, associated_detections, unassociated_tracks, unassociated_detections
