import sys
sys.path.append("../../../")
from person_detection.centernet.src.accumulator import AveragePrecisionAccumulator as DetectionAccumulator
from person_reidentification.centernet.src.accumulator import Accumulator as ReidAccumulator
from person_classification.centernet.src.accumulator import Accumulator as ClassificationAccumulator


class Accumulator(object):

    def __init__(self,):
        super().__init__()
        self.det_accumulator = DetectionAccumulator()
        self.reid_accumulator = ReidAccumulator()
        self.gender_accumulator = ClassificationAccumulator()
        self.age_accumulator = ClassificationAccumulator()

    def reset(self,):
        self.det_accumulator.reset()
        self.reid_accumulator.reset()
        self.gender_accumulator.reset()
        self.age_accumulator.reset()

    def update(
        self, preds_center, preds_dimension, preds_id, preds_gender, preds_age,
        centers, dimensions, i_dimensions, person_ids, i_person_ids, genders, i_genders, ages, i_ages
        ):
        self.det_accumulator.update(
            [preds_center[i] for i in i_dimensions], [preds_dimension[i] for i in i_dimensions],
            [centers[i] for i in i_dimensions], dimensions
        )
        self.reid_accumulator.update([preds_id[i] for i in i_person_ids], person_ids)
        self.gender_accumulator.update([preds_gender[i] for i in i_genders], genders)
        self.age_accumulator.update([preds_age[i] for i in i_ages], ages)

    def summary(self,):
        return {
            "detection": self.det_accumulator.summary()["avg_precision"],
            "reid": self.reid_accumulator.summary(),
            "gender": self.gender_accumulator.summary(),
            "age": self.age_accumulator.summary()
        }