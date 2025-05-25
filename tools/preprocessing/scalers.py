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

    def __call__(self, parts, borders):
        df, *merge_meta = self._merge_parts(parts)

        if self.mu is None:
            self.mu = df.mean()
            self.sigma = df.std()

            self.sigma /= self.target_sigma
            self.mu -= self.target_mu * self.sigma

            self.t_mu = torch.tensor(self.mu.to_numpy(dtype='float32'), device=self.device)
            self.t_sigma = torch.tensor(self.sigma.to_numpy(dtype='float32'), device=self.device)

        df = (df - self.mu) / self.sigma

        return self._split_parts(df, *merge_meta)

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

    def __call__(self, parts, borders):
        df, *merge_meta = self._merge_parts(parts)

        if self.median is None:
            self.median = df.median()
            self.iqr = df.quantile(0.5 + self.q_width / 2) - df.quantile(0.5 - self.q_width / 2)

            self.t_median = torch.tensor(self.median.to_numpy(dtype='float32'), device=self.device)
            self.t_iqr = torch.tensor(self.iqr.to_numpy(dtype='float32'), device=self.device)

        df = (df - self.median) / self.iqr

        return self._split_parts(df, *merge_meta)

    def unscale(self, x):
        return x * self.t_iqr + self.t_median


class FunctionScaler(Transform, DataUnscaler):
    def __init__(self, f, f_inv=None, device="cpu"):
        """
        f: функция для применения к данным (например, torch.log)
        f_inv: обратная функция (например, torch.exp)
        """
        self.f = f
        self.f_inv = f_inv
        self.device = device

    def __call__(self, parts, borders):
        df, *merge_meta = self._merge_parts(parts)
        scaled = self.f(torch.tensor(df.to_numpy(dtype='float32'), device=self.device))
        df = pd.DataFrame(scaled.cpu().numpy(), columns=df.columns.tolist(), index=df.index)

        return self._split_parts(df, *merge_meta)

    def unscale(self, x):
        if self.f_inv is None:
            raise NotImplementedError("Обратная функция не задана!")
        return self.f_inv(x)
