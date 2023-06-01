from .track import Track


class Tracker(object):

    def __init__(self, kf, matcher, new_track_threshold, max_lost_time):
        super().__init__()
        self.kf = kf
        self.matcher = matcher
        self.new_track_threshold = new_track_threshold
        self.max_lost_time = max_lost_time

    def __call__(
        self, time, state_tracks, state_count,
        pred_center, pred_dimension, pred_score, pred_id, pred_gender, pred_age
    ):
        nb_objects = pred_center.shape[0]
        detections = [{
            "center": pred_center[j, :], "dimension": pred_dimension[j, :], "score": pred_score[j],
            "embedding": pred_id[j, :], "gender": pred_gender[j], "age": pred_age[j]
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
            track.update(self.kf, time, detection)

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
