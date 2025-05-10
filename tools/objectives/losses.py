from typing import Callable

import torch

from tools.objectives.metrics import Metric


class NormalizedCovarianceLoss(Metric):
    def __init__(self, corr_matr: torch.Tensor, std: torch.Tensor):
        super().__init__(None)

        self.corr_matr = corr_matr.unsqueeze(0)

        self.outer_std = torch.outer(std, std).unsqueeze(0)

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
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


class NormalizedCovarianceWindowLoss(Metric):
    def __init__(self, cov_adapter: Callable[[torch.Tensor], torch.Tensor], zero_x: bool = True):
        """
        stats_adapter должен по батчу индексов возвращать соответствующие им матрицы ковариаций (B, F, F)
        """
        super().__init__(None)

        self.stats_adapter = cov_adapter
        self.zero_x = zero_x

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        source_cov = self.stats_adapter(iloc)
        source_std = torch.diagonal(source_cov, dim1=1, dim2=2).sqrt()

        outer_std = source_std.unsqueeze(2) * source_std.unsqueeze(1)

        xy_data = torch.concat([x, y_pred], dim=-1)

        means = xy_data.nanmean(dim=1, keepdim=True)

        # # Заменяем NaN на среднее по колонке (вдоль window).
        # # От этого не изменится ковариация, за исключением нормировки на количество
        # xy_data = torch.where(xy_data.isnan(), means, xy_data)

        centered = xy_data - means
        cov = torch.bmm(centered.transpose(1, 2), centered)

        outer_std = outer_std.masked_fill(outer_std == 0, torch.nan)

        # d((cov - source_cov) / outer_std)/dcov = 1 / outer_std
        mask = (~outer_std.isnan()) & (~source_cov.isnan())
        diff = torch.where(mask, cov - source_cov, torch.zeros_like(cov))
        delta_corr = diff / outer_std.nan_to_num(1.0)

        if self.zero_x:
            num_features = x.shape[-1]
            delta_corr[:, :num_features, :num_features] = 0

        corr_loss = torch.linalg.matrix_norm(delta_corr, dim=(1, 2)).nanmean()

        return corr_loss


class MomentLoss(Metric):
    def __init__(self, moments=(1, 2), weights=None, central=True):
        """
        moments: tuple из порядков моментов (например, (1, 2, 3))
        weights: веса для каждого момента, по умолчанию равные
        central: использовать центральные моменты
        """
        super().__init__()
        self.moments = moments
        self.central = central
        self.weights = weights if weights is not None else [1.0] * len(moments)

    def compute(self, y_pred, y_true, x, iloc):
        loss = 0.0
        for k, w in zip(self.moments, self.weights):
            if self.central and k != 1:
                mean = y_true.mean(dim=1, keepdim=True)
                diff = y_pred - mean
            else:
                diff = y_pred
            moment_pred = (diff ** k).mean()
            moment_true = ((y_true - y_true.mean(dim=1, keepdim=True)) if self.central else y_true) ** k
            moment_true = moment_true.mean()
            loss += w * (moment_pred - moment_true).abs()
        return loss


class RobustHuberCovLoss(Metric):
    def __init__(self, corr_target: torch.Tensor, std: torch.Tensor, delta=1.0):
        """
        corr_target: матрица целевой корреляции
        std: вектор std для нормировки ковариации
        delta: параметр Huber loss
        """
        super().__init__()
        self.corr_target = corr_target.unsqueeze(0)
        self.outer_std = torch.outer(std, std).unsqueeze(0)
        self.delta = delta

    def compute(self, y_pred, y_true, x, iloc):
        xy_data = torch.cat([x, y_pred], dim=-1)
        means = xy_data.mean(dim=1, keepdim=True)
        centered = xy_data - means
        cov = torch.bmm(centered.transpose(1, 2), centered) / self.outer_std
        error = cov - self.corr_target

        # Huber Loss по матрице
        abs_err = error.abs()
        loss = torch.where(abs_err <= self.delta, 0.5 * error ** 2, self.delta * (abs_err - 0.5 * self.delta))
        return loss.mean()


class TemporalSpectralDivergenceLoss(Metric):
    def __init__(self, window_size=32, stride=16, method='kl', eps=1e-8):
        """
        window_size: размер окна БПФ
        stride: шаг скользящего окна
        method: метод сравнения спектров ('kl' | 'bhatt')
        eps: защита от логов нуля
        """
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.method = method
        self.eps = eps

    def _power_spectrum(self, signal):
        windows = signal.unfold(dimension=1, size=self.window_size, step=self.stride)
        fft = torch.fft.rfft(windows, dim=2)
        power = torch.abs(fft) ** 2
        power = power / (power.sum(dim=2, keepdim=True) + self.eps)  # нормировка на вероятности
        return power

    def compute(self, y_pred, y_true, x, iloc):
        spec_pred = self._power_spectrum(y_pred)
        spec_true = self._power_spectrum(y_true)

        if self.method == 'kl':
            # KL-divergence между спектрами :3
            kl = spec_true * (torch.log(spec_true + self.eps) - torch.log(spec_pred + self.eps))
            loss = kl.sum(dim=2).mean()
        elif self.method == 'bhatt':
            # Bhattacharyya distance
            bc = torch.sqrt(spec_pred * spec_true).sum(dim=2)
            loss = -torch.log(bc + self.eps).mean()
        else:
            raise ValueError("method must be 'kl' or 'bhatt'")
        
        return loss
