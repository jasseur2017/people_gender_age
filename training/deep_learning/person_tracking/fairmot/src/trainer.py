from tqdm.notebook import tqdm
# from collections import OrderedDict
import pandas as pd
import torch
import sys
sys.path.append("../../..")
from person_detection.centernet.src.pre_detector import PreDetector
from .post_processor import PostProcessor
from .accumulator import Accumulator


class Trainer(object):

    def __init__(self, net, image_size, device, model_path=None):
        super().__init__()
        self.device = device
        self.net = net
        self.net.to(device)
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
        self.pre_detector = PreDetector()
        self.pre_detector.to(device)
        self.pre_detector.eval()
        self.post_processor = PostProcessor(image_size)
        self.accumulator = Accumulator()

    def __validate(self, val_dataset):
        self.net.eval()
        for video_name, video_length, video in val_dataset:
            self.accumulator.reset()
            state_tracks, state_count = [], 0
            tqdm_video = tqdm(video, total=video_length)
            for sample in tqdm_video:
                image = torch.as_tensor(sample["image"], device=self.device)
                image = image.permute(2, 0, 1)
                bbox = torch.as_tensor(sample["bbox"], device=self.device)
                person_id = torch.as_tensor(sample["person_id"], device=self.device)
                with torch.no_grad():
                    preds = self.net(image.unsqueeze(0))
                preds_hm, preds_wh, preds_reg, preds_id, preds_gender, preds_age = (
                    preds[0]["hm"], preds[0]["wh"], preds[0]["reg"], preds[0]["id"], preds[0]["gender"], preds[0]["age"]
                )
                preds_hm = self.pre_detector(preds_hm)
                state_tracks, state_count = self.post_processor(
                    sample["time"], state_tracks, state_count,
                    preds_hm.cpu().numpy(), preds_wh.cpu().numpy(),
                    preds_reg.cpu().numpy(), preds_id.cpu().numpy(),
                    preds_gender.cpu().numpy(), preds_age.cpu().numpy()
                )
                self.accumulator.update(
                    state_tracks, bbox.cpu().numpy(), person_id.cpu().numpy()
                )
                summary = self.accumulator.summary()
                tqdm_video.set_description(
                    "video name %s mota %.2f, motp %.2f" % (
                        video_name, summary["mota"], summary["motp"]
                    )
                )

    def eval(self, val_dataset):
        self.__validate(val_dataset)

    def predict_video(self, video):
        self.net.eval()
        self.pre_detector.eval()
        state_tracks, state_count = [], 0
        for sample in video:
            image = torch.as_tensor(sample["image"], device=self.device)
            image = image.permute(2, 0, 1)
            with torch.no_grad():
                preds = self.net(image.unsqueeze(0))
            preds_hm, preds_wh, preds_reg, preds_id, preds_gender, preds_age = (
                preds[0]["hm"], preds[0]["wh"], preds[0]["reg"], preds[0]["id"], preds[0]["gender"], preds[0]["age"]
            )
            preds_hm = self.pre_detector(preds_hm)
            state_tracks, state_count = self.post_processor(
                sample["time"], state_tracks, state_count,
                preds_hm.cpu().numpy(), preds_wh.cpu().numpy(),
                preds_reg.cpu().numpy(), preds_id.cpu().numpy(),
                preds_gender.cpu().numpy(), preds_age.cpu().numpy()
            )
            yield sample["image"], sample["frame_id"], state_tracks
