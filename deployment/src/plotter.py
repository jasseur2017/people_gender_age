import cv2


class Plotter(object):

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    @classmethod
    def get_color(cls, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

    def __call__(self, frame, state_tracks):
        frame = cv2.resize(frame, self.image_size)
        for track in state_tracks:
            (x1, y1, x2, y2) = track.bbox.astype(int).tolist()
            color = self.get_color(track.id)
            if track.state == track.TrackState.TRACKED:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA
                )
            else:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA
                )
            cv2.putText(
                frame, "%d" % track.id, (x1, y1 + 30),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2
            )
        return frame
