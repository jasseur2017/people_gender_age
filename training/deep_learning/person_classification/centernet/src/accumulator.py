import numpy as np


class Accumulator(object):

    def __init__(self,):
        super().__init__()
        self.samples = []

    def reset(self,):
        self.samples.clear()

    def update(self, preds, labels):
        for pred, label in zip(preds, labels):
            mask = ~np.isnan(label)
            self.samples.extend((pred[mask] == label[mask]).tolist())

    def summary(self,):
        if len(self.samples) == 0:
            return 1.0
        return np.mean(self.samples)
