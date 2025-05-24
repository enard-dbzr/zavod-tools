import warnings

import pandas as pd
import torch

from tools.dataset import PEMSDataset
from tools.objectives.metrics import MetricContext


class CovarianceDatasetAdapter:
    def __init__(self, datasets: dict[str, PEMSDataset], resample="100YS", device="cpu"):
        self.datasets = datasets
        self.resample = resample
        self.device = device

        if self.resample.endswith("E"):
            warnings.warn("Be careful with end frequency in resample")

        self.dataset_stats = {k: self.process_dataset(v) for k, v in self.datasets.items()}

    def process_dataset(self, dataset: PEMSDataset):
        df = pd.concat([dataset.x, dataset.y], axis=1)

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
    def __init__(self, datasets: dict[str, PEMSDataset], device="cpu"):
        self.datasets = datasets
        self.device = device
        
    def __call__(self, ctx: MetricContext) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = self.datasets.get(ctx.dataloader_tag)
        idx = ctx.iloc

        if dataset is None:
            dataset = next(iter(self.datasets.values()))

        indices = idx.unsqueeze(1) + torch.arange(dataset.window_size)
        
        x = dataset.x_tensor[indices].to(self.device)
        y = dataset.y_tensor[indices].to(self.device)
        
        return x, y
