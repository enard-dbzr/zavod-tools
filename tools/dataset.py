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
    def __init__(self, dataframe: pd.DataFrame, window: int = 60, stride: int = 1,
                 filler: Transform = None, transform: Transform = None,
                 features=None, targets=None, dtype="float32"):

        self.features = features or FEATURES
        self.targets = targets or TARGETS

        self.window_size = window
        self.stride = stride

        self.dtype = dtype

        self.transform = transform

        self.x = dataframe[self.features]
        self.y = dataframe[self.targets]

        self.x_tensor = None
        self.y_tensor = None

        self.independent_borders = [self.x.index.min(), self.x.index.max() + self.x.index[0].resolution]

        self._start_indices = []

        if filler is not None:
            self.apply_filler(filler)
        else:
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

    def random_split(self, sizes: list[float], resample="14d", seed=None) -> list["PEMSDataset"]:
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
        self._start_indices = []

        for i in range(len(self.independent_borders) - 1):
            start_time = self.independent_borders[i]
            end_time = self.independent_borders[i + 1]

            start_idx = self.x.index.searchsorted(start_time)
            end_idx = self.x.index.searchsorted(end_time)

            length = end_idx - start_idx
            n_windows = (length - self.window_size) // self.stride + 1

            self._start_indices.extend([
                start_idx + j * self.stride for j in range(n_windows)
            ])

        self.x_tensor = torch.tensor(self.x.to_numpy(dtype=self.dtype))
        self.y_tensor = torch.tensor(self.y.to_numpy(dtype=self.dtype))

    def get_sample(self, idx):
        start = self._start_indices[idx]
        end = start + self.window_size

        x_sample = self.x.iloc[start:end]
        y_sample = self.y.iloc[start:end]

        return x_sample, y_sample, start

    def __len__(self):
        return len(self._start_indices)

    def __getitem__(self, idx):
        start = self._start_indices[idx]
        end = start + self.window_size

        # TODO: Заменить местный трансформ на другой интерфейс, который будет работать с тензорами
        # if self.transform is not None:
        #     x_sample, y_sample = self.transform(x_sample, y_sample, borders=[])

        x_sample = self.x_tensor[start:end]
        y_sample = self.y_tensor[start:end]
        return x_sample, y_sample, start
