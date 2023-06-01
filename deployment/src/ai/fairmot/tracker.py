from .track import Track


class Tracker(object):

    def __init__(self, kf, matcher, direction_classifier):
        super().__init__()
        self.kf = kf
        self.matcher = matcher
        self.direction_classifier = direction_classifier
        self.new_track_threshold = 0.5
        self.max_lost_time = 1.2

    def __call__(
        self, time, state_tracks, state_count,
        pred_center, pred_dimension, pred_score, pred_id
    ):
        nb_objects = pred_center.shape[0]
        detections = [{
            "center": pred_center[j, :], "dimension": pred_dimension[j, :], "score": pred_score[j],
            "embedding": pred_id[j, :]
            } for j in range(nb_objects)
        ]
        for track in state_tracks:
            track.predict(self.kf, time)
        (
            associated_tracks, associated_detections, unassociated_tracks, unassociated_detections
            ) = self.matcher.associate(
            state_tracks, detections
        )
        for track, detection in zip(associated_tracks, associated_detections):
            track.update(self.kf, self.direction_classifier, time, detection)

        for track in unassociated_tracks:
            track.state = Track.TrackState.LOST

        for detection in unassociated_detections:
            if detection["score"] > self.new_track_threshold:
                associated_tracks.append(Track(detection, state_count, time))
                state_count += 1

        left_tracks = [
            track for track in unassociated_tracks
            if (time - track.last_update_time <= self.max_lost_time)
            # TODO remove border persons within less time
        ]
        state_tracks = associated_tracks + left_tracks
        return state_tracks, state_count
