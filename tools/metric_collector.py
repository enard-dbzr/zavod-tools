import torch
from torch import Tensor


class MetricCollector:
    def __init__(self, metrics, aggregation_fn=None):
        self.metrics = metrics
        self.aggregation_fn = aggregation_fn or (lambda x: torch.nanmean(x, dim=0))

        self.collected = {}
        self.aggregate_and_release()

    def calculate_metrics(self, y_pred, y_batch) -> dict[str, Tensor]:
        current = {}
        for k, v in self.metrics.items():
            m = v(y_pred, y_batch).detach()
            self.collected[k].append(m)
            current[k] = m

        return current

    def aggregate_and_release(self) -> dict[str, Tensor]:
        result = {}
        for k in self.collected:
            result[k] = self.aggregation_fn(torch.stack(self.collected[k]))

        self.collected = {k: [] for k in self.metrics}

        return result
