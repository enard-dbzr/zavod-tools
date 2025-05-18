import abc
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tools.preprocessing.scalers import DataUnscaler


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
                 x: torch.Tensor, iloc: torch.Tensor) -> torch.Tensor:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PredictionBasedMetric(Metric):
    @abc.abstractmethod
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc: torch.Tensor) -> torch.Tensor:
        """Вычисляет значение ошибки для каждого элемента"""
        pass

    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor, x: torch.Tensor,
                 iloc: torch.Tensor) -> torch.Tensor:
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        errors = self.compute(y_pred, y_true, x, iloc)
        return self.aggregate(errors)


class UnscaledMetric(PredictionBasedMetric):
    """Метрика с преобразованием шкалы"""

    def __init__(self, unscaler: DataUnscaler, metric: PredictionBasedMetric):
        super().__init__(None)
        self.unscaler = unscaler
        self.base_metric = metric

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        y_pred_unscaled = self.unscaler.unscale(y_pred)
        y_true_unscaled = self.unscaler.unscale(y_true)
        return self.base_metric(y_pred_unscaled, y_true_unscaled, x, iloc)

    def __repr__(self):
        return f"UnscaledMetric({repr(self.base_metric)})"


class NanMaskedMetric(PredictionBasedMetric):
    """Метрика с маскированием NaN значений"""

    def __init__(self, metric: PredictionBasedMetric):
        super().__init__(None)
        self.base_metric = metric

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        mask = ~torch.isnan(y_true)
        return self.base_metric(y_pred[mask], y_true[mask], x, iloc)

    def __repr__(self):
        return f"NanMaskedMetric({repr(self.base_metric)})"


class WeightedMetricsCombination(Metric):
    def __init__(self, metrics: list[Metric], weights: torch.Tensor = None, aggregation_fn: Callable = torch.sum):
        super().__init__(aggregation_fn, 0)

        self.metrics = metrics
        self.weights = weights

    def __call__(self, y_pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], y_true: torch.Tensor, x: torch.Tensor,
                 iloc: torch.Tensor) -> torch.Tensor:
        res = torch.stack([m(y_pred, y_true, x, iloc) for m in self.metrics])
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
                 iloc: torch.Tensor) -> torch.Tensor:
        y_pred = tuple(y_pred[pos] for pos in self.positions)
        return self.metric(y_pred, y_true, x, iloc)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metric}, {self.positions})"


class NanWeightedMetric(PredictionBasedMetric):
    """Метрика с взвешиванием по наличию NaN"""

    def __init__(self,
                 metric: PredictionBasedMetric,
                 weights: torch.Tensor,
                 aggregation_fn: Optional[Callable] = None):
        super().__init__(aggregation_fn or metric.aggregation_fn, metric.axis)
        self.base_metric = metric
        self.weights = weights

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        errors = self.base_metric.compute(y_pred, y_true, x, iloc)
        weights = self.weights.to(errors.device)
        return errors * weights


class ZeroMetric(PredictionBasedMetric):
    """Метрика, независящая от предсказаний"""

    def __init__(self):
        super().__init__()

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        return y_pred * 0


class MAPE(PredictionBasedMetric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        safe_y_true = torch.where(y_true == 0, self.epsilon, y_true)
        return torch.abs((y_true - y_pred) / safe_y_true)


class MAE(PredictionBasedMetric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        return torch.abs(y_true - y_pred)


class MSE(PredictionBasedMetric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
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

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        naive = torch.nanmean(y_true, dim=self.naive_axis, keepdim=True)
        mae_model = torch.abs(y_true - y_pred)
        mae_naive = torch.abs(y_true - naive)
        return mae_model / (mae_naive + self.epsilon)


class WAPE(PredictionBasedMetric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
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

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
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

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
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

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
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

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc) -> torch.Tensor:
        # Такое вот интересное решение вроде работает с данной метрикой и не искажает ее
        mask = ~torch.isnan(y_true) & ~torch.isnan(y_pred)

        y_true_filled = torch.where(mask, y_true, torch.zeros_like(y_true))
        y_pred_filled = torch.where(mask, y_pred, torch.zeros_like(y_pred))

        error_norm = torch.norm(y_true_filled - y_pred_filled, p=self.p, dim=self.axis)
        true_norm = torch.norm(y_true_filled, p=self.p, dim=self.axis)

        result = error_norm / (true_norm + self.epsilon)

        return result


class GradientNorm(PredictionBasedMetric):
    def __init__(self, net: nn.Module):
        super().__init__(None)

        self.net = net

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc: torch.Tensor) -> torch.Tensor:
        total_norm = torch.tensor(0, dtype=y_pred.dtype)
        for p in self.net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2) ** 2
        return total_norm ** 0.5


class PredMetric(PredictionBasedMetric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor, iloc: torch.Tensor) -> torch.Tensor:
        return y_pred


def calculate_metrics(
        data: Union[Dataset, DataLoader],
        metrics: dict[str, Metric],
        model: torch.nn.Module,
        device: str = "cpu",
        batch_processing: Optional[Callable] = None,
        batch_size: int = 512,
        show_progress: bool = True,
        num_workers: int = 0
) -> dict[str, float]:
    model.to(device)
    model.eval()

    if isinstance(data, Dataset):
        dataloader = DataLoader(data, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
    else:
        dataloader = data

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not show_progress):
            batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]

            if batch_processing:
                x, y_true = batch_processing(batch)
            else:
                x, y_true = batch

            y_pred = model(x)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_true.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    results = {}
    for name, metric in metrics.items():
        results[name] = metric(all_preds, all_targets)

    return results
