import numpy as np
import pandas as pd
import torch

from tools.dataset import PEMSDataset
from tools.objectives.metrics import Metric


class NormalizedCovarianceLoss(Metric):
    def __init__(self, corr_matr: torch.Tensor, std: torch.Tensor):
        super().__init__(None)

        self.corr_matr = corr_matr.unsqueeze(0)

        self.outer_std = torch.outer(std, std).unsqueeze(0)

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xy_data = torch.concat([x, y_pred], dim=-1)

        means = xy_data.nanmean(dim=1, keepdim=True)

        # # Заменяем NaN на среднее по колонке (вдоль window).
        # # От этого не изменится ковариация, за исключением нормировки на количество
        # xy_data = torch.where(xy_data.isnan(), means, xy_data)

        centered = xy_data - means

        corr = torch.bmm(centered.transpose(1, 2), centered) / self.outer_std

        delta_corr = (corr - self.corr_matr)

        corr_loss = torch.linalg.matrix_norm(delta_corr, dim=(1, 2)).nanmean()

        return corr_loss
