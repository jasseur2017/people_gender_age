from pathlib import Path
import numpy as np
import cv2
import json


class TrainDataset(object):

    def __init__(self, data_dir, data_df, transform):
        super().__init__()
        self.data_dir = data_dir
        self.data_groups = list(data_df.groupby("id"))
        self.transform = transform

    def read_image(self, image_path):
        assert image_path.is_file(), image_path
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        group = self.data_groups[idx]
        image_name = group[0]
        image_path = Path(self.data_dir, image_name + ".jpg")
        image = self.read_image(image_path)
        labels = group[1]
        bboxes = labels["fbox"]
        bboxes = np.asarray([json.loads(bbox) for bbox in bboxes])
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        height, width, c = image.shape
        # bboxes start from 1 to handle albumentations flip augmentation issue
        bboxes[:, 0] = np.clip(bboxes[:, 0], 1, width - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 1, height - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 1, width - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 1, height - 1)
        # TODO built-in min_area
        mask = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) > 0
        bboxes = bboxes[mask, :]
        sample = self.transform(image=image, bboxes=bboxes)
        image = sample["image"]
        image = image.astype(np.float32) / 255.
        if len(sample["bboxes"]) == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            bboxes = np.asarray(sample["bboxes"], dtype=np.float32)
        centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        wh = bboxes[:, 2:] - bboxes[:, :2]
        dimensions = np.tile(wh / 2, (1, 2))
        return {"image": image, "center": centers, "dimension": dimensions}

    def __len__(self,):
        return len(self.data_groups)
