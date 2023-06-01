import numpy as np
from scipy.spatial.distance import cdist
from .matcher import Matcher


class EmbeddingMatcher(Matcher):

    def __init__(self, kf, cost_limit=0.4):
        super().__init__(cost_limit=cost_limit)
        self.kf = kf
        self.lambda_ = 0.98

    @classmethod
    def embedding_distance(cls, tracks, detections):
        if (len(tracks) == 0) or (len(detections) == 0):
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        tracks_embedding = np.asarray([track.smooth_embedding for track in tracks])
        detections_embedding = np.asarray([detection["embedding"] for detection in detections])
        cost_matrix = cdist(tracks_embedding, detections_embedding, metric="cosine")
        return cost_matrix

    def gating_distance(self, tracks, detections):
        if (len(tracks) == 0) or (len(detections) == 0):
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        cost_matrix = np.array([
            [track.gating_distance(self.kf, detection) for detection in detections]
            for track in tracks
        ], dtype=np.float32)
        gating_cost_limit = self.kf.chi2inv95[4] # Normally it is 2 but here 4 works better
        cost_matrix[cost_matrix > gating_cost_limit] = np.inf
        return cost_matrix

    def associate(self, unassociated_tracks, unassociated_detections):
        embedding_dists = self.embedding_distance(unassociated_tracks, unassociated_detections)
        gating_dists = self.gating_distance(unassociated_tracks, unassociated_detections)
        dists = self.lambda_ * embedding_dists + (1 - self.lambda_) * gating_dists
        (
            associated_tracks_id, associated_detections_id,
            unassociated_tracks_id, unassociated_detections_id
        ) = self.linear_assignment(dists)
        associated_tracks = [
            unassociated_tracks[i] for i in associated_tracks_id
        ]
        unassociated_tracks = [
            unassociated_tracks[i] for i in unassociated_tracks_id
        ]
        associated_detections = [
            unassociated_detections[i] for i in associated_detections_id
        ]
        unassociated_detections = [
            unassociated_detections[i] for i in unassociated_detections_id
        ]
        return associated_tracks, associated_detections, unassociated_tracks, unassociated_detections
