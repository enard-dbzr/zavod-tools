from typing import Callable, Union

import torch

from tools.objectives.metrics import PredictionBasedMetric, Metric


class NormalizedCovarianceWindowLoss(PredictionBasedMetric):
    def __init__(self, cov_adapter: Callable[[torch.Tensor], torch.Tensor],
                 vanish_xx: bool = True,
                 vanish_xy = False,
                 vanish_yy = False,
                 merge_batch_window: bool = False,
                 diag_multiplier: float = 1.0,
                 aggregation_fn=lambda x: torch.linalg.matrix_norm(x, dim=(1, 2)).nanmean(),
                 center_data: bool = True):
        """
        Функция потерь, вычисляющая разность предсказанной и действительной матриц ковариации с нормировкой.

        :arg cov_adapter: Должен по батчу индексов возвращать соответствующие им матрицы ковариаций (B, F, F).
        :arg vanish_xx: Определяет, будет ли обнуляться срез матрицы, соответсвующий ковариациям X на X.
        :arg merge_batch_window: Объединять размерности батча и окна, для подсчета статистики.
        :arg diag_multiplier: Множитель диагональных элементов матриц.
        :arg aggregation_fn: Функция агграгации ошибки. По умолчанию среднее F-норм матриц.
        :arg center_data: Параметр, определяющий надо ли центрировать данные.
        """
        super().__init__(aggregation_fn, None)

        self.stats_adapter = cov_adapter
        self.vanish_xx = vanish_xx
        self.vanish_yy = vanish_yy
        self.vanish_xy = vanish_xy
        self.merge_batch_window = merge_batch_window
        self.diag_multiplier = diag_multiplier
        self.center_data = center_data

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        xy_data = torch.concat([x, y_pred], dim=-1)

        source_cov = self.stats_adapter(iloc)

        if self.merge_batch_window:
            xy_data = xy_data.view((1, -1, xy_data.shape[-1]))

            # FIXME: все хорошо, но есть нюанс
            source_cov = source_cov.nanmean(dim=0, keepdim=True)

        source_std = torch.diagonal(source_cov, dim1=1, dim2=2).sqrt()
        outer_std = source_std.unsqueeze(2) * source_std.unsqueeze(1)

        # # Заменяем NaN на среднее по колонке (вдоль window).
        # # От этого не изменится ковариация, за исключением нормировки на количество
        # xy_data = torch.where(xy_data.isnan(), means, xy_data)

        means = xy_data.nanmean(dim=1, keepdim=True)
        centered = xy_data - means if self.center_data else xy_data

        # TODO: optimize calculations and помнимм про возможные наны при нормировке
        cov = torch.bmm(centered.transpose(1, 2), centered)
        cov /= centered.shape[1]

        outer_std = outer_std.masked_fill(outer_std == 0, torch.nan)

        # d((cov - source_cov) / outer_std)/dcov = 1 / outer_std
        mask = (~outer_std.isnan()) & (~source_cov.isnan())
        diff = torch.where(mask, cov - source_cov, torch.zeros_like(cov))
        delta_corr = diff / outer_std.nan_to_num(1.0)
        
        diag_idx = torch.arange(delta_corr.shape[-1])
        delta_corr[:, diag_idx, diag_idx] *= self.diag_multiplier

        if self.vanish_xx:
            num_features = x.shape[-1]
            delta_corr[:, :num_features, :num_features] = 0
        
        if self.vanish_yy:
            num_features = x.shape[-1]
            delta_corr[:, num_features:, num_features:] = 0

        if self.vanish_xy:
            num_features = x.shape[-1]
            delta_corr[:, num_features:, :num_features] = 0
            delta_corr[:, :num_features, num_features:] = 0
        
        return delta_corr


class MomentLoss(PredictionBasedMetric):
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


class RobustHuberCovLoss(PredictionBasedMetric):
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


class TemporalSpectralDivergenceLoss(PredictionBasedMetric):
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


class KlDivergenceToStandard(Metric):
    def __init__(self, mean_index: int = 1, logvar_index: int = 2):
        super().__init__()
        self.mean_index = mean_index
        self.logvar_index = logvar_index

    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor, x: torch.Tensor,
                 iloc: torch.Tensor) -> torch.Tensor:
        mean, logvar = y_pred[self.mean_index], y_pred[self.logvar_index]

        return torch.mean(- 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
