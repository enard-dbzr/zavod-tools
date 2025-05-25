import warnings

import pandas as pd
import torch

from tools.dataset import PEMSDataset
from tools.objectives.metrics import MetricContext


class CovarianceDatasetAdapter:
    """
    Адаптер, получающий матрицы ковариаций cov(XY, XY) датасета по индексам.
    """

    def __init__(self, datasets: dict[str, PEMSDataset], x_part_label="x", y_part_label="y",
                 resample="100YS", device="cpu"):
        """
        :param datasets: Датасеты с метками.
        :param x_part_label: Метка части признаков.
        :param y_part_label: Метка части целевых переменных
        :param resample: По сегментам какой длинны, будут вычисляться матрицы.
        :param device: Устройство.
        """
        self.datasets = datasets
        self.x_part_label = x_part_label
        self.y_part_label = y_part_label
        self.resample = resample
        self.device = device

        if self.resample.endswith("E"):
            warnings.warn("Be careful with end frequency in resample")

        self.dataset_stats = {k: self.process_dataset(v) for k, v in self.datasets.items()}

    def process_dataset(self, dataset: PEMSDataset):
        df = pd.concat([dataset.dfs[self.x_part_label], dataset.dfs[self.y_part_label]], axis=1)

        group = df.groupby(pd.Grouper(freq=self.resample, closed="left"))

        cov = group.cov().to_numpy(dtype='float32')
        cov = cov.reshape((-1, cov.shape[-1], cov.shape[-1]))

        stat_loc = group.ngroup().to_numpy()

        cov = torch.tensor(cov, device=self.device)
        stat_loc = torch.tensor(stat_loc, device=self.device)
        print(f"Grouped into {len(group)} group(s).")

        return cov, stat_loc

    def __call__(self, ctx: MetricContext) -> torch.Tensor:
        cov, stat_loc = self.dataset_stats[ctx.dataloader_tag]

        return cov[stat_loc[ctx.iloc]]


class GetDatasetItemAdapter:
    """
    .. deprecated::
        Для подмены данных используйте OverridePartMetric
    """
    def __init__(self, datasets: dict[str, PEMSDataset], device="cpu"):
        warnings.warn("Для подмены данных используйте OverridePartMetric",
                      category=DeprecationWarning, stacklevel=2)

        self.datasets = datasets
        self.device = device
        
    def __call__(self, ctx: MetricContext) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = self.datasets.get(ctx.dataloader_tag)
        idx = ctx.iloc

        if dataset is None:
            dataset = next(iter(self.datasets.values()))

        indices = idx.unsqueeze(1) + torch.arange(dataset.window_size)
        
        x = dataset.tensors["x"][indices].to(self.device)
        y = dataset.tensors["y"][indices].to(self.device)
        
        return x, y
