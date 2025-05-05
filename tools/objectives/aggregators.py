import torch


class ExponentialAggregation:
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha

    def __call__(self, values: torch.Tensor, dim: int = 0) -> torch.Tensor:
        size = values.shape[dim]
        weights = torch.tensor([self.alpha ** i for i in range(size)], device=values.device)
        weights /= weights.sum()
        shape = [1] * values.ndim
        shape[dim] = size
        weights = weights.view(*shape)
        return (values * weights).sum(dim=dim)


class LinearAggregation:
    def __init__(self):
        pass

    def __call__(self, values: torch.Tensor, dim: int = 0) -> torch.Tensor:
        size = values.shape[dim]
        weights = torch.linspace(0, 1, steps=size, device=values.device)
        weights /= weights.sum()
        shape = [1] * values.ndim
        shape[dim] = size
        weights = weights.view(*shape)
        return (values * weights).sum(dim=dim)



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
