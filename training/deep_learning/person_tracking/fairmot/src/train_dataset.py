from pathlib import Path
import numpy as np
import cv2


class TrainDataset(object):

    def __init__(self, data_dir, data_df, transform):
        super().__init__()
        self.data_dir = data_dir
        self.data_groups = list(data_df.groupby("video_name"))
        self.transform = transform

    def read_image(self, image_path):
        assert image_path.is_file()
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_frames(self, video_name, frames):
        for frame_id, labels in frames.groupby("frame_id"):
            image_path = Path(
                self.data_dir, video_name, "img1", "%06d.jpg" % frame_id
            )
            image = self.read_image(image_path)
            bboxes = labels[["x", "y", "w", "h"]].values.astype(np.float32)
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
            height, width, _ = image.shape
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height - 1)
            person_ids = labels["person_id"].values
            if (len(bboxes) == 0):
                bboxes = np.zeros((0, 4), dtype=np.float32)
            sample = self.transform(
                image=image, bboxes=bboxes, person_ids=person_ids
            )
            image = sample["image"]
            image = image.astype(np.float32) / 255.
            if (len(sample["bboxes"]) > 0):
                bboxes = np.asarray(sample["bboxes"], dtype=np.float32)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
            person_ids = np.asarray(sample["person_ids"], dtype=np.int64)
            t = labels["time"].iloc[0]
            sample = {
                "time": t, "image": image, "bbox": bboxes, "person_id": person_ids
            }
            yield sample

    def __getitem__(self, video_id):
        video_name, frames = self.data_groups[video_id]
        video_length = frames["video_length"].iloc[0]
        video = self.read_frames(video_name, frames)
        return video_name, video_length, video

    def __len__(self,):
        return len(self.data_groups)
