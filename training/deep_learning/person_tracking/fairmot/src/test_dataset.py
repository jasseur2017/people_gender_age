from tqdm.notebook import tqdm
import numpy as np
import cv2


class TestDataset(object):

    def __init__(self, data_dir, transform, desired_fps=None):
        super().__init__()
        self.data_dir = data_dir
        self.desired_fps = desired_fps
        self.data_groups = list(data_dir.iterdir())
        self.transform = transform

    def read_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        try:
            for i in tqdm(range(video_length)):
                t = i / fps
                if self.desired_fps and not(t % (1 / self.desired_fps) < 1 / fps):
                    has_frame = cap.grab()
                    continue
                has_frame, image = cap.read()
                assert has_frame
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                sample = self.transform(image=image, bboxes=[], person_ids=[])
                image = sample["image"]
                image = image.astype(np.float32) / 255.
                yield {"frame_id": i, "time": (i + 1) / fps, "image": image}
        finally:
            cap.release()

    def __getitem__(self, idx):
        video_path = self.data_groups[idx]
        video = self.read_video(video_path)
        return video_path.name, video

    def __len__(self,):
        return len(self.data_groups)
