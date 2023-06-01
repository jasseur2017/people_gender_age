
class DirectionClassifier(object):

    def __init__(self, x1y1, x2y2):
        super().__init__()
        self.a = x1y1
        self.b = x2y2

    def __call__(self, old_center, center):
        xa, ya = self.a
        xb, yb = self.b
        xo, yo = old_center
        xc, yc = center
        det_ab_ao = (xb - xa) * (yo - ya) - (yb - ya) * (xo - xa)
        det_ab_ac = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)
        if det_ab_ao < 0 and det_ab_ac >= 0:
            return 1
        elif det_ab_ao >= 0 and det_ab_ac < 0:
            return -1
        else:
            return 0
