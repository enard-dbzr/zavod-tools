import abc
from copy import copy
from typing import Callable

import pandas as pd
import torch

from tools.consts import FEATURES, TARGETS
from tools.preprocessing.transforms import Transform


class PEMSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, window: int = 60, stride: int = 1,
                 filler: Transform = None, transform: Transform = None,
                 features=None, targets=None, dtype="float32"):

        self.features = features or FEATURES
        self.targets = targets or TARGETS

        self.window_size = window
        self.stride = stride

        self.dtype = dtype

        self.transform = transform

        df = pd.read_csv(dataset_path, index_col="Time", parse_dates=["Time"])

        self.x = df[self.features]
        self.y = df[self.targets]

        if filler is not None:
            self.apply_filler(filler)

    def apply_filler(self, filler: Transform):
        self.x, self.y = filler(self.x, self.y)

    def split(self, sizes: list[float]) -> list["PEMSDataset"]:
        borders = torch.tensor([0] + sizes).cumsum(0)
        original_size = len(self.x)

        splits = []

        for i in range(len(borders) - 1):
            split = copy(self)

            split.x = self.x.iloc[int(original_size * borders[i]): int(original_size * borders[i + 1])].copy()
            split.y = self.y.iloc[int(original_size * borders[i]): int(original_size * borders[i + 1])].copy()

            splits.append(split)

        return splits

    def __len__(self):
        return (len(self.x) - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        x_sample: pd.DataFrame = self.x.iloc[idx * self.stride: idx * self.stride + self.window_size]
        y_sample: pd.DataFrame = self.y.iloc[idx * self.stride: idx * self.stride + self.window_size]

        if self.transform is not None:
            x_sample, y_sample = self.transform(x_sample, y_sample)

        return x_sample.astype(self.dtype).to_numpy(), y_sample.astype(self.dtype).to_numpy()
