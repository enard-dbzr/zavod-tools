import torch
import abc
from typing import Callable, Optional, Union
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tools.preprocessing.scalers import DataUnscaler


class Metric(abc.ABC):
    """Абстрактный базовый класс для всех метрик"""

    def __init__(self,
                 aggregation_fn: Optional[Callable] = torch.nanmean,
                 axis: Optional[int] = None):
        self.aggregation_fn = aggregation_fn
        self.axis = axis and (axis if isinstance(axis, tuple) else (axis,))

    @abc.abstractmethod
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Вычисляет значение ошибки для каждого элемента"""
        pass

    def aggregate(self, errors: torch.Tensor) -> torch.Tensor:
        """Агрегирует ошибки с использованием заданной функции"""
        if self.aggregation_fn is None:
            return errors
        if self.axis is not None:
            dim = [d if d >= 0 else errors.ndim + d for d in self.axis]  # Обработка отрицательных индексов
            keep_dims = [d for d in range(errors.ndim) if d not in dim]  # Измерения, которые остаются
            flattened = errors.permute(*keep_dims, *dim).flatten(start_dim=len(keep_dims))  # Группируем dim в конец и объединяем
            aggregated = self.aggregation_fn(flattened, dim=-1)
            # FIXME: Very very very fucked
            return aggregated[0] if isinstance(aggregated, tuple) else aggregated
        else:
            return self.aggregation_fn(errors)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        errors = self.compute(y_pred, y_true)
        return self.aggregate(errors)

    def __repr__(self):
        return f"{self.__class__.__name__}(axis={self.axis}, aggregation={self.aggregation_fn.__name__})"


class UnscaledMetric(Metric):
    """Метрика с преобразованием шкалы"""

    def __init__(self,
                 unscaler: DataUnscaler,
                 metric: Metric,
                 aggregation_fn: Optional[Callable] = None):
        super().__init__(aggregation_fn or metric.aggregation_fn, metric.axis)
        self.unscaler = unscaler
        self.base_metric = metric

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_unscaled = self.unscaler.unscale(y_pred)
        y_true_unscaled = self.unscaler.unscale(y_true)
        return self.base_metric.compute(y_pred_unscaled, y_true_unscaled)

    def __repr__(self):
        return f"UnscaledMetric({repr(self.base_metric)})"


class NanMaskedMetric(Metric):
    """Метрика с маскированием NaN значений"""

    def __init__(self,
                 metric: Metric,
                 aggregation_fn: Optional[Callable] = None):
        super().__init__(aggregation_fn or metric.aggregation_fn, metric.axis)
        self.base_metric = metric

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(y_true)
        return self.base_metric.compute(y_pred[mask], y_true[mask])

    def __repr__(self):
        return f"NanMaskedMetric({repr(self.base_metric)})"


class NanWeightedMetric(Metric):
    """Метрика с взвешиванием по наличию NaN"""

    def __init__(self,
                 metric: Metric,
                 weights: torch.Tensor,
                 aggregation_fn: Optional[Callable] = None):
        super().__init__(aggregation_fn or metric.aggregation_fn, metric.axis)
        self.base_metric = metric
        self.weights = weights

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        errors = self.base_metric.compute(y_pred, y_true)
        weights = self.weights.to(errors.device)
        return errors * weights


class MAPE(Metric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        safe_y_true = torch.where(y_true == 0, self.epsilon, y_true)
        return torch.abs((y_true - y_pred) / safe_y_true)


class MAE(Metric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.abs(y_true - y_pred)


class MSE(Metric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return (y_true - y_pred) ** 2


class MASE(Metric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 naive_axis: int = -1,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon
        self.naive_axis = naive_axis

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        naive = torch.nanmean(y_true, dim=self.naive_axis, keepdim=True)
        mae_model = torch.abs(y_true - y_pred)
        mae_naive = torch.abs(y_true - naive)
        return mae_model / (mae_naive + self.epsilon)


class WAPE(Metric):
    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        absolute_error = self.aggregate(torch.abs(y_true - y_pred))
        total_actual = self.aggregate(torch.abs(y_true))
        return absolute_error / (total_actual + 1e-8)


class RMSSE(Metric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 naive_axis: int = -1,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon
        self.naive_axis = naive_axis

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        naive = torch.nanmean(y_true, dim=self.naive_axis, keepdim=True)
        squared_error = (y_true - y_pred) ** 2
        squared_error_naive = (y_true - naive) ** 2
        return squared_error / (squared_error_naive + self.epsilon)

    def aggregate(self, errors: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(super().aggregate(errors))


class PeakDeviation(Metric):
    def __init__(self,
                 epsilon: float = 1e-8,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.max):
        super().__init__(aggregation_fn, axis)
        self.epsilon = epsilon

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        true = torch.abs(y_true)
        error = torch.abs(y_true - y_pred)
        return (error / (true + self.epsilon)).nan_to_num(0.0)  # Типо избавляемся от nan таким образом


class RTCS(Metric):
    def __init__(self,
                 axis: Optional[int] = None,
                 diff_axis: Optional[int] = 1,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.diff_axis = diff_axis

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff_pred = torch.diff(y_pred, dim=self.diff_axis)
        diff_true = torch.diff(y_true, dim=self.diff_axis)
        return (diff_pred ** 2) / (diff_true ** 2 + 1e-8)


class RED(Metric):
    def __init__(self,
                 p: int = 2,
                 epsilon: float = 1e-8,
                 axis: Optional[int] = None,
                 aggregation_fn: Callable = torch.nanmean):
        super().__init__(aggregation_fn, axis)
        self.p = p
        self.epsilon = epsilon

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Такое вот интересное решение вроде работает с данной метрикой и не искажает ее
        mask = ~torch.isnan(y_true) & ~torch.isnan(y_pred)

        y_true_filled = torch.where(mask, y_true, torch.zeros_like(y_true))
        y_pred_filled = torch.where(mask, y_pred, torch.zeros_like(y_pred))

        error_norm = torch.norm(y_true_filled - y_pred_filled, p=self.p, dim=self.axis)
        true_norm = torch.norm(y_true_filled, p=self.p, dim=self.axis)

        result = error_norm / (true_norm + self.epsilon)

        return result


def exponential_aggregation(values: torch.Tensor,
                            alpha: float = 0.9,
                            dim: int = 0) -> torch.Tensor:
    weights = torch.tensor([alpha ** i for i in range(values.shape[dim])],
                           device=values.device)
    weights /= weights.sum()
    return (values * weights.view(-1, 1)).sum(dim=dim)


def linear_aggregation(values: torch.Tensor,
                       dim: int = 0) -> torch.Tensor:
    weights = torch.linspace(0, 1, values.shape[dim], device=values.device)
    weights /= weights.sum()
    return (values * weights.view(-1, 1)).sum(dim=dim)


class QuantileAggregation:
    def __init__(self, q: float):
        self.q = q

    def __call__(self, values: torch.Tensor, dim=None):
        if isinstance(dim, tuple):
            # Объединяем указанные измерения в одно
            dim = [d if d >= 0 else values.ndim + d for d in dim]  # Обработка отрицательных индексов
            keep_dims = [d for d in range(values.ndim) if d not in dim]  # Измерения, которые остаются
            flattened = values.permute(*keep_dims, *dim).flatten(start_dim=len(keep_dims))  # Группируем dim в конец и объединяем
            return torch.nanquantile(flattened, q=self.q, dim=-1)
        return torch.nanquantile(values, q=self.q, dim=dim)


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
