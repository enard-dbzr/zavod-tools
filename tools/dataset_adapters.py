import warnings

import pandas as pd
import torch

from tools.dataset import PEMSDataset


class CovarianceDatasetAdapter:
    def __init__(self, dataset: PEMSDataset, resample="100YS", device="cpu"):
        self.dataset = dataset

        if resample.endswith("E"):
            warnings.warn("Be careful with end frequency in resample")

        df = pd.concat([self.dataset.x, self.dataset.y], axis=1)

        group = df.groupby(pd.Grouper(freq=resample, closed="left"))

        cov = group.cov().to_numpy(dtype='float32')
        cov = cov.reshape((-1, cov.shape[-1], cov.shape[-1]))

        stat_loc = group.ngroup().to_numpy()

        self.cov = torch.tensor(cov, device=device)
        self.stat_loc = torch.tensor(stat_loc, device=device)
        print(f"Grouped into {len(group)} group(s).")

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        return self.cov[self.stat_loc[idx]]
    
    
class GetDatasetItemAdapter:
    def __init__(self, dataset: PEMSDataset, device="cpu"):
        self.dataset = dataset
        self.device = device
        
    def __call__(self, idx: torch.Tensor):
        indices = idx.unsqueeze(1) + torch.arange(self.dataset.window_size)
        
        x = self.dataset.x_tensor[indices].to(self.device)
        y = self.dataset.y_tensor[indices].to(self.device)
        
        return x, y
