from pathlib import Path
from tqdm.notebook import tqdm
from torch import optim
import torch
from torch.utils.data import DataLoader
from .post_processor import PostProcessor
from .loss import Loss
from .accumulator import Accumulator


class Trainer(object):

    def __init__(self, net, image_size, device, checkpoint_dir, nID, model_path=None):
        super().__init__()
        self.device = device
        self.checkpoint_dir = checkpoint_dir        
        self.net = net
        self.net.to(device)
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
        self.post_processor = PostProcessor(image_size)
        self.accumulator = Accumulator()
        self.criterion = Loss(image_size, net.heads["id"], nID)
        self.criterion.to(device)
        self.optimizer = optim.Adam(
            [{"params": self.net.parameters(), "lr": 1e-3},
             {"params": self.criterion.parameters(), "lr": 1e-3}
             ],
          lr=1e-3, weight_decay=1e-4
        )

    def __train(self, train_loader):
        self.net.train()
        self.criterion.train()
        cum_loss = 0.0
        cum_len = 0
        tqdm_train_loader = tqdm(train_loader)
        for images, centers, person_ids in tqdm_train_loader:
            images = images.to(self.device)
            centers = [center.to(self.device) for center in centers]
            person_ids = [person_id.to(self.device) for person_id in person_ids]
            self.optimizer.zero_grad()
            preds = self.net(images)
            preds_id = preds[0]["id"]
            loss, loss_stats = self.criterion(preds_id, centers, person_ids)
            cum_loss += loss.item()
            cum_len += images.size(0)
            loss.backward()
            self.optimizer.step()
            tqdm_train_loader.set_description("Train loss %.2f" % (cum_loss / cum_len))

    def __validate(self, val_loader):
        self.net.eval()
        self.accumulator.reset()
        tqdm_val_loader = tqdm(val_loader)
        for images, centers, person_ids in tqdm_val_loader:
            images = images.to(self.device)
            centers = [center.to(self.device) for center in centers]
            person_ids = [person_id.to(self.device) for person_id in person_ids]
            with torch.no_grad():
                preds = self.net(images)
            preds_id = preds[0]["id"]
            preds_id = self.post_processor(
                preds_id.cpu().numpy(), [center.cpu().numpy() for center in centers]
            )
            for pred_id, person_id in zip(preds_id, person_ids):
                self.accumulator.update(pred_id, person_id.cpu().numpy())
        score = self.accumulator.summary()
        print("Val score:", score)
        return score

    @classmethod
    def collate_fn(cls, batch):
        images = [torch.from_numpy(sample["image"]) for sample in batch]
        centers = [torch.from_numpy(sample["center"]) for sample in batch]
        person_ids = [torch.from_numpy(sample["person_id"]) for sample in batch]
        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2)
        return images, centers, person_ids

    def fit(self, train_dataset, val_dataset, batch_size=2,  epochs=2):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn,
        )
#        best_score = 0
#        best_epoch = -1
        for epoch in range(epochs):
            self.__train(train_loader)
            with torch.no_grad():
                score = self.__validate(val_loader)
            torch.save(self.net.state_dict(), Path(self.checkpoint_dir, f"{epoch}.pth"))
#            if score > best_score:
#                best_score = score
#                best_epoch = epoch
#                torch.save(self.net.state_dict(), TORCH_FILE)
#        self.net.load_state_dict(torch.load(TORCH_FILE))

    def eval(self, val_dataset, batch_size=2):
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn
        )
        score = self.__validate(val_loader)
        return score
