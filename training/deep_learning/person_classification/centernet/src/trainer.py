from pathlib import Path
# from collections import OrderedDict
from tqdm.notebook import tqdm
# import random
# import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from .post_processor import PostProcessor
from .loss import Loss
from .accumulator import Accumulator


class Trainer(object):

    def __init__(self, net, image_size, device, checkpoint_dir, log_dir, model_path=None):
        super().__init__()
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.net = net
        self.net.to(device)
        if model_path:
            state_dict = torch.load(model_path)
            # state_dict = OrderedDict(
            #     (k, state_dict.get(k, v)) for k, v in self.net.state_dict().items()
            # )
            self.net.load_state_dict(state_dict)
        self.post_processor = PostProcessor(image_size)
        self.accumulator = Accumulator()
        self.criterion = Loss(image_size)
        self.criterion.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler()
        self.writer = SummaryWriter(log_dir)

    def __train(self, train_loader):
        self.net.train()
        self.criterion.train()
        cum_loss = 0.0
        cum_len = 0
        self.accumulator.reset()
        with tqdm(train_loader) as progress_bar:
            for images, centers, labels in progress_bar:
                images = images.to(self.device)
                centers = [center.to(self.device) for center in centers]
                labels = [label.to(self.device) for label in labels]
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = self.net(images)
                    preds = preds[0]["clss"]
                    assert preds.dtype is torch.float16
                    loss, _ = self.criterion(preds, centers, labels)
                    assert loss.dtype is torch.float32
                cum_loss += loss.item()
                cum_len += images.size(0)
                self.scaler.scale(loss).backward()
                # self.optimizer.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                preds, scores = self.post_processor(
                    preds.detach().cpu().numpy(), [center.cpu().numpy() for center in centers]
                )
                self.accumulator.update(preds, [label.cpu().numpy() for label in labels])
                accuracy = self.accumulator.summary()
                progress_bar.set_description("Train accuracy %.2f" % (accuracy))

    def __validate(self, val_loader):
        self.net.eval()
        self.accumulator.reset()
        with tqdm(val_loader) as progress_bar:
            for images, centers, labels in progress_bar:
                images = images.to(self.device)
                centers = [center.to(self.device) for center in centers]
                labels = [label.to(self.device) for label in labels]
                with torch.no_grad():
                    preds = self.net(images)
                preds = preds[0]["clss"]
                preds, scores = self.post_processor(
                    preds.cpu().numpy(), [center.cpu().numpy() for center in centers]
                )
                self.accumulator.update(preds, [label.cpu().numpy() for label in labels])
                accuracy = self.accumulator.summary()
                progress_bar.set_description("Val accuracy %.2f" % accuracy)
        score = self.accumulator.summary()
        print("Val score:", score)
        return score

    @classmethod
    def val_collate_fn(cls, batch):
        images = [torch.from_numpy(sample["image"]) for sample in batch]
        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2)
        centers = [torch.from_numpy(sample["center"]) for sample in batch]
        labels = [torch.from_numpy(sample["label"]) for sample in batch]
        return images, centers, labels

    def fit(self, train_dataset, val_dataset, batch_size=16, start_epoch=0, end_epoch=10):
        # train_sampler = DistributedSampler(dataset=train_dataset)
        # sampler=train_sampler,
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.val_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.val_collate_fn,
        )
        # dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        try:
    #        best_score = 0
    #        best_epoch = -1
            for epoch in range(start_epoch, end_epoch):
                print("Epoch:", epoch)
                self.__train(train_loader)
                train_score = self.__validate(train_loader)
                val_score = self.__validate(val_loader)
                self.writer.add_scalar('Accuracy/train', train_score, epoch)
                self.writer.add_scalar('Accuracy/test', val_score, epoch)
                torch.save(self.net.state_dict(), Path(self.checkpoint_dir, f"{epoch}.pth"))
    #            if score > best_score:
    #                best_score = score
    #                best_epoch = epoch
    #                torch.save(self.net.state_dict(), TORCH_FILE)
    #        self.net.load_state_dict(torch.load(TORCH_FILE))
        finally:
            # dist.destroy_process_group() ??
            pass

    def eval(self, val_dataset, batch_size=2):
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.val_collate_fn
        )
        score = self.__validate(val_loader)
        return score

    def __predict(self, test_loader):
        self.net.eval()
        test_df = []
        for images_name, bboxes_id, images, centers in tqdm(test_loader):
            images = images.to(self.device)
            centers = [center.to(self.device) for center in centers]
            with torch.no_grad():
                preds = self.net(images)
            preds = preds[0]["clss"]
            preds, scores = self.post_processor(
                preds.cpu().numpy(), [center.cpu().numpy() for center in centers]
            )
            test_df.extend([
                {"id": image_name, "extra.box_id": bi, "pred": p, "score": s}
                for image_name, bbox_id, pred, score in zip(
                    images_name, bboxes_id, preds, scores
                )
                for bi, p, s in zip(bbox_id, pred, score)
            ])
        return pd.DataFrame.from_records(test_df)

    @classmethod
    def test_collate_fn(cls, batch):
        images_name = [sample["image_name"] for sample in batch]
        bboxes_id = [sample["bbox_id"] for sample in batch]
        images = [torch.from_numpy(sample["image"]) for sample in batch]
        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2)
        centers = [torch.from_numpy(sample["center"]) for sample in batch]
        return images_name, bboxes_id, images, centers

    def predict(self, test_dataset, batch_size=2):
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.test_collate_fn
        )
        test_df = self.__predict(test_loader)
        return test_df
