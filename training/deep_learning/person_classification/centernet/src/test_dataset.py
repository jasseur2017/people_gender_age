from pathlib import Path
import numpy as np
import cv2
import json


class TestDataset(object):

    def __init__(self, data_dir, data_df, transform):
        super().__init__()
        self.data_dir = data_dir
        self.data_groups = list(data_df.groupby("id"))
        self.transform = transform

    def read_image(self, image_path):
        assert image_path.is_file()
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        group = self.data_groups[idx]
        image_name = group[0]
        image_path = Path(self.data_dir, image_name + ".jpg")
        image = self.read_image(image_path)
        labels = group[1]
        bboxes_id = labels["extra.box_id"]
        bboxes = labels["hbox"]
        bboxes = np.asarray([json.loads(bbox) for bbox in bboxes])
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        height, width, c = image.shape
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, width - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, height - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, width - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, height - 1)
        mask = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) > 0
        bboxes_id = bboxes_id[mask]
        bboxes = bboxes[mask, :]
        sample = self.transform(image=image, bboxes=bboxes, bboxes_id=bboxes_id)
        image = sample["image"]
        image = image.astype(np.float32) / 255.
        bboxes_id = sample["bboxes_id"]
        bboxes = np.asarray(sample["bboxes"], dtype=np.float32)
        if len(bboxes) == 0:
            centers = np.zeros((0, 4), dtype=np.float32)
            wh = np.zeros((0, 2), dtype=np.float32)
        else:
            centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
            wh = bboxes[:, 2:] - bboxes[:, :2]
        return {
            "image_name": image_name, "bbox_id": bboxes_id, "image": image,
            "center": centers
        }

    def __len__(self,):
        return len(self.data_groups)
