from pathlib import Path
import numpy as np
import cv2


class TestDataset(object):

    def __init__(self, data_dir, data_df, transform):
        super().__init__()
        self.data_dir = data_dir
        self.data_df = data_df
        self.transform = transform

    def read_image(self, image_path):
        assert image_path.is_file(), image_path
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        image_name = row["id"]
        image_path = Path(self.data_dir, image_name + ".jpg")
        image = self.read_image(image_path)
        image_size = (image.shape[1], image.shape[0])
        sample = self.transform(image=image)
        image = sample["image"]
        image = image.astype(np.float32) / 255.
        return {
            "image_name": image_name, "image": image, "image_size": image_size
        }

    def __len__(self,):
        return len(self.data_df)
