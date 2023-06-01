import numpy as np
import scipy as sp


class KalmanFilter(object):

    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919
    }

    def __init__(self,):
        super().__init__()
        self.F = lambda dt: np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.Q = np.diag([1, 1, 1e-2, 1, 1e-2, 1e-2, 1e-2]).astype(np.float32)
        self.H = np.eye(4, 7, dtype=np.float32)
        self.R = np.diag([1, 1, 1e-1, 1]).astype(np.float32)
        self.metric = "Mahalanobis"
        assert self.metric in ["Gaussian", "Mahalanobis"]

    def predict(self, dt, mean, covariance):
        F = self.F(dt)
        mean = np.dot(mean, F.T)
        covariance = np.linalg.multi_dot((F, covariance, F.T)) + self.Q
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean = np.dot(self.H, mean)
        projected_cov = np.linalg.multi_dot((self.H, covariance, self.H.T)) + self.R
        chol_factor, lower = sp.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = sp.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self.H.T).T,
            check_finite=False
        ).T
        innovation = measurement - projected_mean
        mean = mean + np.dot(innovation, kalman_gain.T)
        covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return mean, covariance

    def gating_distance(self, mean, covariance, measurement):
        projected_mean = np.dot(self.H, mean)
        projected_cov = np.linalg.multi_dot((self.H, covariance, self.H.T)) + self.R
        d = measurement - projected_mean
        if self.metric == "Gaussian":
            return np.sum(d * d, axis=0)
        elif self.metric == "Mahalanobis":
            cholesky_factor = np.linalg.cholesky(projected_cov)
            z = sp.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")
