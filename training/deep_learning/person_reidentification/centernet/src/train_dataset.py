from pathlib import Path
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder


class TrainDataset(object):

    def __init__(self, data_dir, data_df, transform, nID=0):
        super().__init__()
        self.data_dir = data_dir
        data_df = data_df.copy()
        le = LabelEncoder()
        data_df["reid"] = nID + le.fit_transform(
            data_df.apply(lambda d: d["video_name"] + str(d["person_id"]), axis=1)
        )
        self.nID = data_df["reid"].max() + 1
        self.data_groups = list(data_df.groupby(["video_name", "frame_id"]))
        self.transform = transform

    def read_image(self, image_path):
        assert image_path.is_file()
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        group = self.data_groups[idx]
        video_name, frame_id = group[0]
        image_path = Path(self.data_dir, video_name, "img1", "%06d.jpg" % frame_id)
        image = self.read_image(image_path)
        labels = group[1]
        bboxes = labels[["x", "y", "w", "h"]].values.astype(np.float32)
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
        height, width, _ = image.shape
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height - 1)
        person_ids = labels["reid"].values
        if (len(bboxes) == 0):
            bboxes = np.zeros((0, 4), dtype=np.float32)
        sample = self.transform(
            image=image, bboxes=bboxes, person_ids=person_ids
        )
        image = sample["image"]
        image = image.astype(np.float32) / 255.
        if (len(sample["bboxes"]) > 0):
            bboxes = np.asarray(sample["bboxes"], dtype=np.float32)
            centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        else:
            centers = np.zeros((0, 2), dtype=np.float32)
        person_ids = np.asarray(sample["person_ids"], dtype=np.int64)
        return {"image": image, "center": centers, "person_id": person_ids}

    def __len__(self,):
        return len(self.data_groups)
