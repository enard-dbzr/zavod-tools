import pandas as pd
import torch

from tools.dataset import PEMSDataset


class CovarianceDatasetAdapter:
    def __init__(self, dataset: PEMSDataset, resample="2ME", device="cpu"):
        self.dataset = dataset

        df = pd.concat([self.dataset.x, self.dataset.y], axis=1)

        group = df.groupby(pd.Grouper(freq=resample))

        cov = group.cov().to_numpy(dtype='float32')
        cov = cov.reshape((-1, cov.shape[-1], cov.shape[-1]))

        stat_loc = group.ngroup().to_numpy()

        self.cov = torch.tensor(cov, device=device)
        self.stat_loc = torch.tensor(stat_loc, device=device)

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.cov[self.stat_loc[idx]]
