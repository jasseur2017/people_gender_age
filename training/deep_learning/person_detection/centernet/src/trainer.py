from pathlib import Path
# from collections import OrderedDict
from tqdm.notebook import tqdm
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .pre_detector import PreDetector
from .post_processor import PostProcessor
from .accumulator import AveragePrecisionAccumulator
from .loss import Loss


class Trainer(object):

    def __init__(self, net, image_size, device, checkpoint_dir, log_dir, model_path=None):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.net = net
        self.net.to(device)
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
        self.pre_detector = PreDetector()
        self.pre_detector.to(device)
        self.post_processor = PostProcessor(image_size)
        self.accumulator = AveragePrecisionAccumulator(mode="macro")
        self.criterion = Loss(image_size)
        self.criterion.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=2e-4)
        self.scaler = torch.cuda.amp.GradScaler()
        self.writer = SummaryWriter(log_dir)

    def __train(self, train_loader):
        self.net.train()
        self.criterion.train()
        cum_loss = 0.0
        cum_loss_stats = {"hm_loss": 0.0, "wh_loss": 0.0, "off_loss": 0.0}
        cum_len = 0
        self.pre_detector.eval()
        self.accumulator.reset()
        tqdm_train_loader = tqdm(train_loader)
        for images, centers, dimensions in tqdm_train_loader:
            images = images.to(self.device)
            centers = [center.to(self.device) for center in centers]
            dimensions = [dimension.to(self.device) for dimension in dimensions]
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = self.net(images)
                preds_hm, preds_wh, preds_reg = (
                    preds[0]["hm"], preds[0]["wh"], preds[0]["reg"]
                )
                loss, loss_stats = self.criterion(
                    preds_hm, preds_wh, preds_reg, centers, dimensions
                )
            cum_loss += loss.item()
            cum_loss_stats = {k: cum_loss_stats[k] + loss_stats[k].item() for k in cum_loss_stats}
            cum_len += images.size(0)
            self.scaler.scale(loss).backward()
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            preds_hm = self.pre_detector(preds_hm)
            preds_center, preds_dimension, preds_score = self.post_processor(
                preds_hm.detach().cpu().numpy(),
                preds_wh.detach().cpu().numpy(),
                preds_reg.detach().cpu().numpy()
            )
            self.accumulator.update(
                preds_center,
                preds_dimension,
                [center.cpu().numpy() for center in centers],
                [dimension.cpu().numpy() for dimension in dimensions]
            )
            summary = self.accumulator.summary()
            # tqdm_train_loader.set_description(
            #     "Train loss %.2f, hm %.2f, wh %.2f, off %.2f" % (
            #         cum_loss / cum_len, cum_loss_stats["hm_loss"] / cum_len,
            #         cum_loss_stats["wh_loss"] / cum_len, cum_loss_stats["off_loss"] / cum_len
            #     )
            # )
            tqdm_train_loader.set_description(
                "Train loss %.2f, avg pr %.2f, rec %.2f, pr %.2f" % (
                    cum_loss / cum_len,
                    summary["avg_precision"], summary["recall"], summary["precision"]
                )
            )

    def __validate(self, val_loader):
        self.net.eval()
        self.pre_detector.eval()
        self.accumulator.reset()
        tqdm_val_loader = tqdm(val_loader)
        for images, centers, dimensions in tqdm_val_loader:
            images = images.to(self.device)
            centers = [center.to(self.device) for center in centers]
            dimensions = [dimension.to(self.device) for dimension in dimensions]
            with torch.no_grad():
                preds = self.net(images)
            preds_hm, preds_wh, preds_reg = (
                preds[0]["hm"], preds[0]["wh"], preds[0]["reg"]
            )
            preds_hm = self.pre_detector(preds_hm)
            preds_center, preds_dimension, preds_score = self.post_processor(
                preds_hm.cpu().numpy(), preds_wh.cpu().numpy(), preds_reg.cpu().numpy()
            )
            self.accumulator.update(
                preds_center,
                preds_dimension,
                [center.cpu().numpy() for center in centers],
                [dimension.cpu().numpy() for dimension in dimensions]
            )
            summary = self.accumulator.summary()
            tqdm_val_loader.set_description(
                "Val avg pr %.2f, rec %.2f, pr %.2f" % (
                    summary["avg_precision"], summary["recall"], summary["precision"]
                ))
        return self.accumulator.summary()["avg_precision"]

    @classmethod
    def val_collate_fn(cls, batch):
        images = [torch.as_tensor(sample["image"]) for sample in batch]
        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2)
        centers = [torch.as_tensor(sample["center"]) for sample in batch]
        dimensions = [torch.as_tensor(sample["dimension"]) for sample in batch]
        return images, centers, dimensions

    def fit(self, train_dataset, val_dataset, batch_size=16, start_epoch=0, end_epoch=10):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=self.val_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self.val_collate_fn
        )
#        best_score = 0
#        best_epoch = -1
        for epoch in range(start_epoch, end_epoch):
            print("Epoch:", epoch)
            self.__train(train_loader)
            train_score = self.__validate(train_loader)
            val_score = self.__validate(val_loader)
            self.writer.add_scalar('MAP/train', train_score, epoch)
            self.writer.add_scalar('MAP/test', val_score, epoch)
            torch.save(self.net.state_dict(), Path(self.checkpoint_dir, f"{epoch}.pth"))
#            if score > best_score:
#                best_score = score
#                best_epoch = epoch
#                torch.save(self.net.state_dict(), TORCH_FILE)
#        self.net.load_state_dict(torch.load(TORCH_FILE))

    def eval(self, val_dataset, batch_size=2):
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.val_collate_fn
        )
        score = self.__validate(val_loader)
        return score

    def __predict(self, test_loader):
        self.net.eval()
        self.pre_detector.eval()
        test_df = []
        w0, h0 = self.image_size
        for images_name, images, images_size in tqdm(test_loader):
            images = images.to(self.device)
            with torch.no_grad():
                preds = self.net(images)
            preds_hm, preds_wh, preds_reg = (
                preds[0]["hm"], preds[0]["wh"], preds[0]["reg"]
            )
            preds_hm = self.pre_detector(preds_hm)
            preds_center, preds_dimension, preds_score = self.post_processor(
                preds_hm.cpu().numpy(), preds_wh.cpu().numpy(), preds_reg.cpu().numpy()
            )
            for image_name, (w, h), pred_center, pred_dimension, pred_score in zip(
                images_name, images_size, preds_center, preds_dimension, preds_score
            ):
                for i, ((x, y), (l, t, r, b), s) in enumerate(
                    zip(pred_center, pred_dimension, pred_score)
                ):
                    ratio = min(w0 / w, h0 / h)
                    test_df.append(
                        {"id": image_name,
                        "box_id": i,
                        "center": [
                            (x - 0.5 * (w0 - ratio * w)) / ratio,
                            (y - 0.5 * (h0 - ratio * h)) / ratio
                        ],
                        "dimension": [l / ratio, t / ratio, r / ratio, b / ratio],
                        "score": s}
                        )
        return pd.DataFrame.from_records(test_df)

    @classmethod
    def test_collate_fn(cls, batch):
        images_name = [sample["image_name"] for sample in batch]
        images = [torch.from_numpy(sample["image"]) for sample in batch]
        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2)
        images_size = [sample["image_size"] for sample in batch]
        return images_name, images, images_size

    def predict(self, test_dataset, batch_size=2):
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.test_collate_fn
        )
        test_df = self.__predict(test_loader)
        return test_df
