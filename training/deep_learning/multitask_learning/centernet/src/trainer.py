from pathlib import Path
# from collections import OrderedDict
from tqdm.notebook import tqdm
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from .loss import Loss
import sys
sys.path.append("../../..")
from person_detection.centernet.src.pre_detector import PreDetector
from .post_processor import PostProcessor
from .accumulator import Accumulator


class Trainer(object):

    def __init__(self, net, image_size, device, checkpoint_dir, log_dir, model_path=None):
        super().__init__()
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.net = net
        self.net.to(device)
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
        self.pre_detector = PreDetector()
        self.pre_detector.to(device)
        self.pre_detector.eval()
        self.post_processor = PostProcessor(image_size)
        self.criterion = Loss(image_size, net.heads["id"])
        self.criterion.to(device)
        self.optimizer = optim.Adam([
            {"params": self.net.parameters(), "lr": 1e-4},
            {"params": self.criterion.parameters(), "lr": 1e-3}
            ], lr=1e-3, weight_decay=1e-5
        )
        self.accumulator = Accumulator()
        self.scaler = torch.cuda.amp.GradScaler()
        # self.writer = SummaryWriter(log_dir)

    def __train(self, train_loader):
        self.net.train()
        self.criterion.train()
        self.pre_detector.eval()
        self.accumulator.reset()
        cum_loss = 0.0
        cum_len = 0
        with tqdm(train_loader) as progress_bar:
            for images, centers, dimensions, i_dimensions, person_ids, i_person_ids, genders, i_genders, ages, i_ages in progress_bar:
                images = images.to(self.device)
                centers = [center.to(self.device) for center in centers]
                dimensions = [dimension.to(self.device) for dimension in dimensions]
                person_ids = [person_id.to(self.device) for person_id in person_ids]
                genders = [gender.to(self.device) for gender in genders]
                ages = [age.to(self.device) for age in ages]
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = self.net(images)
                    loss, loss_stats = self.criterion(
                        preds, centers, dimensions, i_dimensions, person_ids, i_person_ids, genders, i_genders, ages, i_ages
                    )
                cum_loss += loss.item()
                cum_len += images.size(0)
                self.scaler.scale(loss).backward()
                # self.optimizer.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                preds[0]["hm"] = self.pre_detector(preds[0]["hm"])
                (preds_center, preds_dimension, preds_id, preds_gender, preds_age
                ) = self.post_processor(preds, centers)
                self.accumulator.update(
                    preds_center, preds_dimension, preds_id, preds_gender, preds_age,
                    [center.cpu().numpy() for center in centers],
                    [dimension.cpu().numpy() for dimension in dimensions],
                    i_dimensions,
                    [person_id.cpu().numpy() for person_id in person_ids],
                    i_person_ids,
                    [gender.cpu().numpy() for gender in genders],
                    i_genders,
                    [age.cpu().numpy() for age in ages],
                    i_ages
                )
                # score = self.accumulator.summary()
                # progress_bar.set_description(
                #     "Train detection %.2f, gender %.2f, age %.2f, reid %.2f" % (
                #         score["detection"], score["gender"], score["age"], score["reid"]
                #     )
                # )
            score = self.accumulator.summary()
            print(score)

    def __validate(self, val_loader):
        self.net.eval()
        self.pre_detector.eval()
        self.accumulator.reset()
        with tqdm(val_loader) as progress_bar:
            for images, centers, dimensions, i_dimensions, person_ids, i_person_ids, genders, i_genders, ages, i_ages in progress_bar:
                images = images.to(self.device)
                centers = [center.to(self.device) for center in centers]
                dimensions = [dimension.to(self.device) for dimension in dimensions]
                person_ids = [person_id.to(self.device) for person_id in person_ids]
                genders = [gender.to(self.device) for gender in genders]
                ages = [age.to(self.device) for age in ages]
                with torch.cuda.amp.autocast(), torch.no_grad():
                    preds = self.net(images)
                preds[0]["hm"] = self.pre_detector(preds[0]["hm"])
                (preds_center, preds_dimension, preds_id, preds_gender, preds_age
                ) = self.post_processor(preds, centers)
                self.accumulator.update(
                    preds_center, preds_dimension, preds_id, preds_gender, preds_age,
                    [center.cpu().numpy() for center in centers],
                    [dimension.cpu().numpy() for dimension in dimensions],
                    i_dimensions,
                    [person_id.cpu().numpy() for person_id in person_ids],
                    i_person_ids,
                    [gender.cpu().numpy() for gender in genders],
                    i_genders,
                    [age.cpu().numpy() for age in ages],
                    i_ages
                )
                # score = self.accumulator.summary()
                # progress_bar.set_description(
                #     "Val detection %.2f, gender %.2f, age %.2f, reid %.2f" % (
                #         score["detection"], score["gender"], score["age"], score["reid"]
                #     )
                # )
            score = self.accumulator.summary()
            print(score)

    @classmethod
    def collate_fn(cls, batch):
        images = [torch.from_numpy(sample["image"]) for sample in batch]
        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2)
        centers = [torch.from_numpy(sample["center"]) for sample in batch]
        dimensions = [
            torch.from_numpy(sample["dimension"]) for sample in batch if "dimension" in sample
        ]
        i_dimensions = [
            i for i, sample in enumerate(batch) if "dimension" in sample
        ]
        person_ids = [
            torch.from_numpy(sample["person_id"].astype(np.float32)) for sample in batch if "person_id" in sample
        ]
        i_person_ids = [
            i for i, sample in enumerate(batch) if "person_id" in sample
        ]
        genders = [
            torch.from_numpy(sample["gender"].astype(np.float32)) for sample in batch if "gender" in sample
        ]
        i_genders = [
            i for i, sample in enumerate(batch) if "gender" in sample
        ]
        ages = [
            torch.from_numpy(sample["age"].astype(np.float32)) for sample in batch if "age" in sample
        ]
        i_ages = [
            i for i, sample in enumerate(batch) if "age" in sample
        ]
        return (
            images, centers, dimensions, i_dimensions, person_ids, i_person_ids, genders, i_genders, ages, i_ages
        )

    def fit(self, train_dataset, val_dataset, batch_size=2, epochs=2):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn
        )
        # best_score = 0
        # best_epoch = -1
        for epoch in range(epochs):
            print("Epoch:", epoch)
            self.__train(train_loader)
            self.__validate(val_loader)
            torch.save(self.net.state_dict(), Path(self.checkpoint_dir, f"{epoch}.pth"))
            # if score > best_score:
            #     best_score = score
            #     best_epoch = epoch
            #     torch.save(self.net.state_dict(), TORCH_FILE)
            # self.writer.add_scalar('Loss/train', ?, epoch)
        # self.net.load_state_dict(torch.load(TORCH_FILE))
        # self.writer.close() ?? called at SummaryWriter.__exit__

    def eval(self, val_dataset, batch_size=2):
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn
        )
        self.__validate(val_loader)
