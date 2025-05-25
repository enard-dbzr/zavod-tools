import abc
import dataclasses
import warnings
from typing import Callable, Optional, Union, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tools.preprocessing.scalers import DataUnscaler


# По-хорошему это косвенно вносит знание о датасете в метрики, но я **** адаптеры выносить из метрик
@dataclasses.dataclass
class MetricContext:
    iloc: torch.Tensor
    dataset: Any
    dataloader_tag: str
    extra: dict[str, torch.Tensor]


class Metric(abc.ABC):
    """Абстрактный базовый класс для всех метрик"""

    def __init__(self,
                 aggregation_fn: Optional[Callable] = torch.nanmean,
                 axis: Union[int, tuple, None] = None):
        self.aggregation_fn = aggregation_fn
        self.axis = axis
        if axis is not None and not isinstance(axis, tuple):
            self.axis = (axis,)

    def aggregate(self, errors: torch.Tensor) -> torch.Tensor:
        """Агрегирует ошибки с использованием заданной функции"""
        if self.aggregation_fn is None:
            return errors
        if self.axis is not None:
            dim = [d if d >= 0 else errors.ndim + d for d in self.axis]  # Обработка отрицательных индексов
            keep_dims = [d for d in range(errors.ndim) if d not in dim]  # Измерения, которые остаются
            flattened = errors.permute(*keep_dims, *dim).flatten(
                start_dim=len(keep_dims))  # Группируем dim в конец и объединяем
            aggregated = self.aggregation_fn(flattened, dim=-1)
            # FIXME: Very very very fucked
            return aggregated[0] if isinstance(aggregated, tuple) else aggregated
        else:
            return self.aggregation_fn(errors)

    @abc.abstractmethod
    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor,
                 x: torch.Tensor, ctx: MetricContext) -> torch.Tensor:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PredictionBasedMetric(Metric):
    @abc.abstractmethod
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx: MetricContext) -> torch.Tensor:
        """Вычисляет значение ошибки для каждого элемента"""
        pass

    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor, x: torch.Tensor,
                 ctx) -> torch.Tensor:
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        errors = self.compute(y_pred, y_true, x, ctx)
        return self.aggregate(errors)


class UnscaledMetric(PredictionBasedMetric):
    """Метрика с преобразованием шкалы"""

    def __init__(self, unscaler: DataUnscaler, metric: PredictionBasedMetric):
        super().__init__(None)
        self.unscaler = unscaler
        self.base_metric = metric

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        y_pred_unscaled = self.unscaler.unscale(y_pred)
        y_true_unscaled = self.unscaler.unscale(y_true)
        return self.base_metric(y_pred_unscaled, y_true_unscaled, x, ctx)

    def __repr__(self):
        return f"UnscaledMetric({repr(self.base_metric)})"


class NanMaskedMetric(PredictionBasedMetric):
    """Метрика с маскированием NaN значений"""

    def __init__(self, metric: PredictionBasedMetric):
        super().__init__(None)
        self.base_metric = metric

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        mask = ~torch.isnan(y_true)
        return self.base_metric(y_pred[mask], y_true[mask], x, ctx)

    def __repr__(self):
        return f"NanMaskedMetric({repr(self.base_metric)})"


class WeightedMetricsCombination(Metric):
    def __init__(self, metrics: list[Metric], weights: torch.Tensor = None, aggregation_fn: Callable = torch.sum):
        super().__init__(aggregation_fn, 0)

        self.metrics = metrics
        self.weights = weights

    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor, x: torch.Tensor,
                 ctx) -> torch.Tensor:
        res = torch.stack([m(y_pred, y_true, x, ctx) for m in self.metrics])
        if self.weights is not None:
            res = self.weights.broadcast_to(res.shape) * res
        return self.aggregate(res)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.metrics)}, {self.weights})"


class PermuteOutputsMetric(Metric):
    def __init__(self, metric: Metric, positions: tuple[int]):
        super().__init__(None)
        self.metric = metric
        self.positions = positions

    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor, x: torch.Tensor,
                 ctx) -> torch.Tensor:
        y_pred = tuple(y_pred[pos] for pos in self.positions)
        return self.metric(y_pred, y_true, x, ctx)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metric}, {self.positions})"


class OverrideDataMetric(Metric):
    """
    .. deprecated::
        Для подмены данных используйте OverridePartMetric
    """

    def __init__(self, metric: Metric, adapter: Callable[[MetricContext], tuple[torch.Tensor, torch.Tensor]],
                 override_x: bool = True, override_y: bool = True):
        warnings.warn("Для подмены данных используйте OverridePartMetric",
                      category=DeprecationWarning, stacklevel=2)
        super().__init__()
        self.metric = metric
        self.adapter = adapter
        
        self.override_x = override_x
        self.override_y = override_y

    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor, x: torch.Tensor,
                 ctx) -> torch.Tensor:
        o_x, o_y = self.adapter(ctx)
        
        if self.override_x:
            x = o_x
        if self.override_y:
            y_true = o_y
        
        return self.metric(y_pred, y_true, x, ctx)


class OverridePartMetric(Metric):
    """
    Заменят указанным образом части данных.

    При словаре замен {"x": "slice", "a": "b"} часть **x** заменится на **slice**, а часть **a** на **b**.
    В том числе происходят замены аргументных частей y_true и x, как частей с метками x и y.
    Неуказанные части остаются на своих местах.
    """

    def __init__(self, metric: Metric, rules: dict[str, str]):
        """
        :param metric: Базовая метрика.
        :param rules: Словарь замен.
        """
        super().__init__(None)

        self.metric = metric
        self.map = rules

    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor, x: torch.Tensor,
                 ctx: MetricContext) -> torch.Tensor:
        ctx.extra["x"] = x
        ctx.extra["y"] = y_true

        ctx.extra = {k: ctx.extra[self.map.get(k, k)] for k in self.map.keys() | ctx.extra.keys()}

        return self.metric(y_pred, ctx.extra["y"], ctx.extra["x"], ctx)


class NanWeightedMetric(PredictionBasedMetric):
    """Метрика с взвешиванием по наличию NaN"""

    def __init__(self,
                 metric: PredictionBasedMetric,
                 weights: torch.Tensor,
                 aggregation_fn: Optional[Callable] = None):
        super().__init__(aggregation_fn or metric.aggregation_fn, metric.axis)
        self.base_metric = metric
        self.weights = weights

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        errors = self.base_metric.compute(y_pred, y_true, x, ctx)
        weights = self.weights.to(errors.device)
        return errors * weights


class ZeroMetric(PredictionBasedMetric):
    """Метрика, независящая от предсказаний"""

    def __init__(self):
        super().__init__()

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        return y_pred * 0


class MAPE(PredictionBasedMetric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        safe_y_true = torch.where(y_true == 0, self.epsilon, y_true)
        return torch.abs((y_true - y_pred) / safe_y_true)


class MAE(PredictionBasedMetric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        return torch.abs(y_true - y_pred)


class MSE(PredictionBasedMetric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        return (y_true - y_pred) ** 2


class MASE(PredictionBasedMetric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 naive_axis: int = -1,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon
        self.naive_axis = naive_axis

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        naive = torch.nanmean(y_true, dim=self.naive_axis, keepdim=True)
        mae_model = torch.abs(y_true - y_pred)
        mae_naive = torch.abs(y_true - naive)
        return mae_model / (mae_naive + self.epsilon)


class WAPE(PredictionBasedMetric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        absolute_error = self.aggregate(torch.abs(y_true - y_pred))
        total_actual = self.aggregate(torch.abs(y_true))
        return absolute_error / (total_actual + 1e-8)


class RMSSE(PredictionBasedMetric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 naive_axis: int = -1,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon
        self.naive_axis = naive_axis

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        naive = torch.nanmean(y_true, dim=self.naive_axis, keepdim=True)
        squared_error = (y_true - y_pred) ** 2
        squared_error_naive = (y_true - naive) ** 2
        return squared_error / (squared_error_naive + self.epsilon)

    def aggregate(self, errors: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(super().aggregate(errors))


class PeakDeviation(PredictionBasedMetric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.max):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        true = torch.abs(y_true)
        error = torch.abs(y_true - y_pred)
        return (error / (true + self.epsilon)).nan_to_num(0.0)  # Типо избавляемся от nan таким образом


class RTCS(PredictionBasedMetric):
    def __init__(self,
                 axis: Optional[int] = None,
                 diff_axis: Optional[int] = 1,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.diff_axis = diff_axis

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        diff_pred = torch.diff(y_pred, dim=self.diff_axis)
        diff_true = torch.diff(y_true, dim=self.diff_axis)
        return (diff_pred ** 2) / (diff_true ** 2 + 1e-8)


class RED(PredictionBasedMetric):
    def __init__(self,
                 p: int = 2,
                 epsilon: float = 1e-8,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.p = p
        self.epsilon = epsilon

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        # Такое вот интересное решение вроде работает с данной метрикой и не искажает ее
        mask = ~torch.isnan(y_true) & ~torch.isnan(y_pred)

        y_true_filled = torch.where(mask, y_true, torch.zeros_like(y_true))
        y_pred_filled = torch.where(mask, y_pred, torch.zeros_like(y_pred))

        error_norm = torch.norm(y_true_filled - y_pred_filled, p=self.p, dim=list(self.axis))
        true_norm = torch.norm(y_true_filled, p=self.p, dim=list(self.axis))

        result = error_norm / (true_norm + self.epsilon)

        return result


class GradientNorm(PredictionBasedMetric):
    def __init__(self, net: nn.Module):
        super().__init__(None)

        self.net = net

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        total_norm = torch.tensor(0, dtype=y_pred.dtype, device=y_pred.device)
        for p in self.net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2) ** 2
        return total_norm ** 0.5


class PredMetric(PredictionBasedMetric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, ctx) -> torch.Tensor:
        self.test = y_true, x, ctx
        return y_pred
