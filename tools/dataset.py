import abc
import itertools
from copy import copy
from functools import cache
from typing import Callable

import numpy as np
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

        self._cumulative_sizes = None

        self.independent_borders = [self.x.index.min(), self.x.index.max() + self.x.index[0].resolution]
        self.update_sizes()

    def apply_filler(self, filler: Transform):
        self.x, self.y = filler(self.x, self.y, self.independent_borders)

        self.update_sizes()

    def split(self, sizes: list[float]) -> list["PEMSDataset"]:
        borders = torch.tensor([0] + sizes).cumsum(0)
        original_size = len(self.x)

        splits = []

        for i in range(len(borders) - 1):
            split = copy(self)

            split.x = self.x.iloc[int(original_size * borders[i]): int(original_size * borders[i + 1])].copy()
            split.y = self.y.iloc[int(original_size * borders[i]): int(original_size * borders[i + 1])].copy()

            split.independent_borders = [split.x.index.min(), split.x.index.max() + split.x.index[0].resolution]
            split.update_sizes()

            splits.append(split)

        return splits

    def random_split(self, sizes: list[float], resample="14d", seed=None):
        borders = torch.tensor([0] + sizes).cumsum(0)

        resampled_indices = self.x.resample(resample).indices
        resampled_size = len(resampled_indices)

        shuffled_indices = np.array(list(resampled_indices.keys()))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(shuffled_indices)

        splits = []

        for i in range(len(borders) - 1):
            split = copy(self)

            i_time_split = sorted(shuffled_indices[int(resampled_size * borders[i]): int(resampled_size * borders[i + 1])])

            source_indices = [
                j
                for i in i_time_split
                for j in resampled_indices[i]
            ]

            split.x = self.x.iloc[source_indices].copy()
            split.y = self.y.iloc[source_indices].copy()

            split.independent_borders = i_time_split + [split.x.index.max() + split.x.index[0].resolution]
            split.update_sizes()

            splits.append(split)

        return splits

    def update_sizes(self):
        cur = 0
        self._cumulative_sizes = [0]
        for i in range(len(self.independent_borders) - 1):
            split = self.x.loc[self.independent_borders[i]:
                               self.independent_borders[i + 1] - self.x.index[0].resolution]

            cur += (len(split) - self.window_size) // self.stride + 1
            self._cumulative_sizes.append(cur)

    @cache
    def get_sample(self, idx):
        gr_size_index = next((i for i, v in enumerate(self._cumulative_sizes) if v > idx))
        accum_size = self._cumulative_sizes[gr_size_index - 1]

        idx -= accum_size

        borders = self.independent_borders + [self.x.index.max()]

        x = self.x.loc[borders[gr_size_index - 1]: borders[gr_size_index]]
        y = self.y.loc[borders[gr_size_index - 1]: borders[gr_size_index]]

        x_sample: pd.DataFrame = x.iloc[idx * self.stride: idx * self.stride + self.window_size]
        y_sample: pd.DataFrame = y.iloc[idx * self.stride: idx * self.stride + self.window_size]

        return x_sample, y_sample

    def __len__(self):
        return self._cumulative_sizes[-1]

    def __getitem__(self, idx):
        x_sample, y_sample = self.get_sample(idx)

        if self.transform is not None:
            x_sample, y_sample = self.transform(x_sample, y_sample, [])

        return x_sample.astype(self.dtype).to_numpy(), y_sample.astype(self.dtype).to_numpy()
