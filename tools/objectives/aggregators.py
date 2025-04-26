import torch


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
