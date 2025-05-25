import abc
import itertools
from copy import deepcopy
from functools import cache
from typing import Callable

import numpy as np
import pandas as pd
import torch

from tools.consts import FEATURES, TARGETS
from tools.preprocessing.transforms import Transform


class PEMSDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, window: int = 60, stride: int = 1,
                 parts: dict[str, list[str]] = None, prefixes: dict[str, str] = None,
                 preprocessor: Transform = None, transform: Transform = None, dtype="float32"):
        if prefixes is None:
            prefixes = {}

        self.part_columns = {
            "x": FEATURES,
            "y": TARGETS,
        }
        self.part_columns.update(parts or {})

        self.window_size = window
        self.stride = stride

        self.dtype = dtype

        self.transform = transform

        self.dfs = {part: dataframe[cols].add_prefix(prefixes.get(part, ""))
                    for part, cols in self.part_columns.items()}

        self.tensors = None

        self.independent_borders = [self.dfs["x"].index.min(),
                                    self.dfs["x"].index.max() + self.dfs["x"].index[0].resolution]

        self._start_indices = []

        if preprocessor is not None:
            self.apply_preprocessor(preprocessor)
        else:
            self.update_sizes()

    def apply_preprocessor(self, filler: Transform):
        self.dfs = filler(self.dfs, self.independent_borders)

        self.update_sizes()

    def split(self, sizes: list[float]) -> list["PEMSDataset"]:
        borders = torch.tensor([0] + sizes).cumsum(0)
        original_size = len(self.dfs["x"])

        splits = []

        for i in range(len(borders) - 1):
            split = deepcopy(self)

            for part in self.dfs:
                split.dfs[part] = self.dfs[part].iloc[int(original_size * borders[i]):
                                                      int(original_size * borders[i + 1])].copy()

            split.independent_borders = [split.dfs["x"].index.min(),
                                         split.dfs["x"].index.max() + split.dfs["x"].index[0].resolution]
            split.update_sizes()

            splits.append(split)

        return splits

    def random_split(self, sizes: list[float], resample="14d", seed=None) -> list["PEMSDataset"]:
        borders = torch.tensor([0] + sizes).cumsum(0)

        resampled_indices = self.dfs["x"].resample(resample).indices
        resampled_size = len(resampled_indices)

        shuffled_indices = np.array(list(resampled_indices.keys()))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(shuffled_indices)

        splits = []

        for i in range(len(borders) - 1):
            split = deepcopy(self)

            i_time_split = sorted(shuffled_indices[int(resampled_size * borders[i]): int(resampled_size * borders[i + 1])])

            source_indices = [
                j
                for i in i_time_split
                for j in resampled_indices[i]
            ]

            for part in self.dfs:
                split.dfs[part] = self.dfs[part].iloc[source_indices].copy()

            split.independent_borders = i_time_split + [split.dfs["x"].index.max() + split.dfs["x"].index[0].resolution]
            split.update_sizes()

            splits.append(split)

        return splits

    def update_sizes(self):
        self._start_indices = []

        for i in range(len(self.independent_borders) - 1):
            start_time = self.independent_borders[i]
            end_time = self.independent_borders[i + 1]

            start_idx = self.dfs["x"].index.searchsorted(start_time)
            end_idx = self.dfs["x"].index.searchsorted(end_time)

            length = end_idx - start_idx
            n_windows = (length - self.window_size) // self.stride + 1

            self._start_indices.extend([
                start_idx + j * self.stride for j in range(n_windows)
            ])

        self.tensors = {part: torch.tensor(df.to_numpy(dtype=self.dtype)) for part, df in self.dfs.items()}

    def get_sample(self, idx):
        start = self._start_indices[idx]
        end = start + self.window_size

        sample = {}

        for part, df in self.dfs.items():
            sample[part] = df.iloc[start:end]

        sample["iloc"] = start

        return sample

    def __len__(self):
        return len(self._start_indices)

    def __getitem__(self, idx):
        start = self._start_indices[idx]
        end = start + self.window_size

        # TODO: Заменить местный трансформ на другой интерфейс, который будет работать с тензорами
        # if self.transform is not None:
        #     x_sample, y_sample = self.transform(x_sample, y_sample, borders=[])

        sample = {}

        for part, tensor in self.tensors.items():
            sample[part] = tensor[start:end]

        sample["iloc"] = start

        return sample
