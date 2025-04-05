import abc

import pandas as pd
import torch

from tools.preprocessing.transforms import Transform


class DataUnscaler(abc.ABC):
    @abc.abstractmethod
    def unscale(self, x):
        pass


class NormalScaler(Transform, DataUnscaler):
    def __init__(self, target_mu=0, target_sigma=1, device="cpu"):
        self.target_mu = target_mu
        self.target_sigma = target_sigma

        self.mu = None
        self.sigma = None

        self.t_mu = None
        self.t_sigma = None

        self.device = device

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders):
        df = pd.concat([x, y], axis=1)

        if self.mu is None:
            self.mu = df.mean()
            self.sigma = df.std()

            self.sigma /= self.target_sigma
            self.mu -= self.target_mu * self.sigma

            self.t_mu = torch.tensor(self.mu.to_numpy(dtype='float32'), device=self.device)
            self.t_sigma = torch.tensor(self.sigma.to_numpy(dtype='float32'), device=self.device)

        df = (df - self.mu) / self.sigma

        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]

    def unscale(self, x):
        return x * self.t_sigma + self.t_mu


class RobustScaler(Transform, DataUnscaler):
    def __init__(self, q_width=0.5, device="cpu"):
        self.q_width = q_width

        self.median = None
        self.iqr = None

        self.t_median = None
        self.t_iqr = None

        self.device = device

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame, borders):
        df = pd.concat([x, y], axis=1)

        if self.median is None:
            self.median = df.median()
            self.iqr = df.quantile(0.5 + self.q_width / 2) - df.quantile(0.5 - self.q_width / 2)

            self.t_median = torch.tensor(self.median.to_numpy(dtype='float32'), device=self.device)
            self.t_iqr = torch.tensor(self.iqr.to_numpy(dtype='float32'), device=self.device)

        df = (df - self.median) / self.iqr

        return df.iloc[:, :len(x.columns)], df.iloc[:, len(x.columns):]

    def unscale(self, x):
        return x * self.t_iqr + self.t_median
