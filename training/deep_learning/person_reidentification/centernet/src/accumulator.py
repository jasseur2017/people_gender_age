import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


class Accumulator(object):

    def __init__(self,):
        super().__init__()
        self.X = []
        self.y = []

    def reset(self,):
        self.X.clear()
        self.y.clear()

    def update(self, preds_id, person_ids):
        for pred_id, person_id in zip(preds_id, person_ids):
            self.X.append(pred_id)
            self.y.append(person_id)

    def summary(self,):
        if len(self.X) == 0:
            return 1.0
        X = np.concatenate(self.X, axis=0)
        Y = np.concatenate(self.y, axis=0)
        le = LabelEncoder()
        y = le.fit_transform(Y)
        classifier = KNeighborsClassifier(n_neighbors=y.max() + 1)
        classifier.fit(X, y)
        return classifier.score(X, y)
