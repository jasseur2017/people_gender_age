import numpy as np
import lap


class AveragePrecisionAccumulator(object):

    def __init__(self, mode="micro"):
        super().__init__()
        assert mode in ["micro", "macro"]
        self.mode = mode
#         self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        self.iou_thresholds = [0.5]
        self.cum_tp = []
        self.cum_fp = []
        self.cum_fn = []

    @classmethod
    def center1dimension2bbox(cls, center, dimension):
        bbox = np.stack((
            center[:, 0] - dimension[:, 0],
            center[:, 1] - dimension[:, 1],
            center[:, 0] + dimension[:, 2],
            center[:, 1] + dimension[:, 3],
        ), axis=1)
        return bbox

    @classmethod
    def iou_distance(cls, bboxes_1, bboxes_2):
        area1 = (bboxes_1[:, 2] - bboxes_1[:, 0]) * (bboxes_1[:, 3] - bboxes_1[:, 1])
        area2 = (bboxes_2[:, 2] - bboxes_2[:, 0]) * (bboxes_2[:, 3] - bboxes_2[:, 1])
        width = np.maximum(0.0, (
            np.minimum(bboxes_1[:, 2, None], bboxes_2[:, 2]) -
            np.maximum(bboxes_1[:, 0, None], bboxes_2[:, 0])
        ))
        height = np.maximum(0.0, (
            np.minimum(bboxes_1[:, 3, None], bboxes_2[:, 3]) -
            np.maximum(bboxes_1[:, 1, None], bboxes_2[:, 1])
        ))
        inter = width * height
        ovr = inter / (area1[:, None] + area2[None, :] - inter)
        return 1 - ovr

    def reset(self,):
        self.cum_tp = []
        self.cum_fp = []
        self.cum_fn = []

    def update(self, preds_center, preds_dimension, targets_center, targets_dimension):
        for pred_center, pred_dimension, target_center, target_dimension in zip(
            preds_center, preds_dimension, targets_center, targets_dimension
        ):
            pred = self.center1dimension2bbox(pred_center, pred_dimension)
            target = self.center1dimension2bbox(target_center, target_dimension)
            distance_matrix = self.iou_distance(pred, target)
            # distance_matrix = distance_matrix[np.argsort(scores)[::-1], :]
            nb_preds, nb_targets = distance_matrix.shape
            n = len(self.iou_thresholds)
            tp = np.zeros(n, dtype=np.float32)
            fp = np.zeros(n, dtype=np.float32)
            fn = np.zeros(n, dtype=np.float32)
            if (nb_targets == 0) or (nb_preds == 0):
                fp += nb_preds
                fn += nb_targets
            else:
                for iou_index, iou_threshold in enumerate(self.iou_thresholds):
                    cost, x, y = lap.lapjv(distance_matrix, extend_cost=True, cost_limit=1 - iou_threshold)
                    tp[iou_index] += (x != -1).sum()
                    fp[iou_index] += (x == -1).sum()
                    fn[iou_index] += (y == -1).sum()
            self.cum_tp.append(tp)
            self.cum_fp.append(fp)
            self.cum_fn.append(fn)
    
    def __summary_macro(self,):
        recalls = np.mean([
            np.divide(tp, tp + fn, out=np.ones_like(tp), where=~((tp == 0) & (tp + fn == 0)))
            for tp, fn in zip(self.cum_tp, self.cum_fn)
        ], axis=0)
        recall = np.mean(recalls)
        precisions = np.mean([
            np.divide(tp, tp + fp, out=np.ones_like(tp), where=~((tp == 0) & (tp + fp == 0)))
            for tp, fp in zip(self.cum_tp, self.cum_fp)
        ], axis=0)
        precision = np.mean(precisions)
        avg_precisions = np.mean([
            np.divide(tp, tp + fp + fn, out=np.ones_like(tp), where=~((tp == 0) & (tp + fp + fn == 0)))
            for tp, fp, fn in zip(self.cum_tp, self.cum_fp, self.cum_fn)
        ], axis=0)
        avg_precision = np.mean(avg_precisions)
        return dict(precision=precision, recall=recall, avg_precision=avg_precision)

    def __summary_micro(self,):
        a = np.sum(self.cum_tp, axis=0)
        b = np.sum(self.cum_tp, axis=0) + np.sum(self.cum_fn, axis=0)
        recalls = np.divide(
            a, b, out=np.ones_like(a), where=~((a == 0) & (b == 0))
        )
        recall = np.mean(recalls)
        a = np.sum(self.cum_tp, axis=0)
        b = np.sum(self.cum_tp, axis=0) + np.sum(self.cum_fp, axis=0)
        precisions = np.divide(
            a, b, out=np.ones_like(a), where=~((a == 0) & (b == 0))
        )
        precision = np.mean(precisions)
        a = np.sum(self.cum_tp, axis=0)
        b = np.sum(self.cum_tp, axis=0) + np.sum(self.cum_fp, axis=0) + np.sum(self.cum_fn, axis=0)
        avg_precisions = np.divide(
            a, b, out=np.ones_like(a), where=~((a == 0) & (b == 0))
        )
        avg_precision = np.mean(avg_precisions)
        return dict(precision=precision, recall=recall, avg_precision=avg_precision)

    def summary(self,):
        if self.mode == "macro":
            return self.__summary_macro()
        elif self.mode == "micro":
            return self.__summary_micro()
        else:
            raise ValueError
