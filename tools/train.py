from typing import Callable, Optional

import torch
from torch import nn, Tensor
from tqdm.notebook import trange, tqdm

from tools.objectives.metrics import Metric
from tools.model_logger.model_logger import ModelLogger


class MetricCollector:
    def __init__(self, metrics: dict[str, Metric], aggregation_fn=None):
        self.metrics = metrics
        self.aggregation_fn = aggregation_fn or (lambda x: torch.nanmean(x, dim=0))

        self.collected = {}
        self.aggregate_and_release()

    def calculate_metrics(self, y_pred, y_batch, x_batch, iloc) -> dict[str, Tensor]:
        current = {}
        for k, v in self.metrics.items():
            m = v(y_pred, y_batch, x_batch, iloc).detach()
            self.collected[k].append(m)
            current[k] = m

        return current

    def aggregate_and_release(self) -> dict[str, Tensor]:
        result = {}
        for k in self.collected:
            result[k] = self.aggregation_fn(torch.stack(self.collected[k]))

        self.collected = {k: [] for k in self.metrics}

        return result


def train_eval(net: nn.Module,
               optimizer: torch.optim.Optimizer,
               criterion: Metric,
               num_epochs: int,
               train_dataloader: torch.utils.data.DataLoader,
               val_dataloader: torch.utils.data.DataLoader,
               metric_collector: MetricCollector,
               logger: ModelLogger,
               log_params: bool = False,
               device="cpu"):

    net.to(device)

    with logger(net, optimizer, criterion, train_dataloader):
        logger.log_model()
        logger.log_hparams()

        for epoch in trange(num_epochs, desc="Epoch"):
            net.train()
            for i, (x_batch, y_batch, iloc) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = net(x_batch)

                loss = criterion(y_pred, y_batch, x=x_batch, iloc=iloc)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

                step = epoch * len(train_dataloader) + i

                metrics = metric_collector.calculate_metrics(y_pred, y_batch, x_batch, iloc)
                logger.log_batch_metrics(metrics, step, "train")

            metrics = metric_collector.aggregate_and_release()
            logger.log_epoch_metrics(metrics, epoch, "train")

            if log_params:
                logger.log_params(epoch)

            net.eval()
            with torch.no_grad():
                for i, (x_batch, y_batch, iloc) in enumerate(tqdm(val_dataloader)):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    y_pred = net(x_batch)

                    step = epoch * len(val_dataloader) + i

                    logger.log_predictions(step, y_pred, y_batch)

                    metrics = metric_collector.calculate_metrics(y_pred, y_batch, x_batch, iloc)
                    logger.log_batch_metrics(metrics, step, "val")

            metrics = metric_collector.aggregate_and_release()
            logger.log_epoch_metrics(metrics, epoch, "val")

            logger.save_model(epoch)



# def evaluate(net: nn.Module,
#              dataloader: torch.utils.data.DataLoader,
#              metrics: Optional[dict[str, Callable]] = None,
#              device="cpu",
#              comment="",
#              log_tensor_mode="subplots"):
#     if metrics is None:
#         metrics = {}
#
#     net.to(device)
#
#     writer = SummaryWriter(comment=(f"_{comment}_EVAL" if comment else ""))
#
#     metric_values = {k: [] for k in metrics}
#
#     net.eval()
#     with torch.no_grad():
#         for i, (x_batch, y_batch) in enumerate(tqdm(dataloader)):
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             y_pred = net(x_batch)
#
#             for k, v in metrics.items():
#                 m = v(y_pred, y_batch).detach()
#                 metric_values[k].append(m)
#
#                 _log_metric(writer, f"{k.capitalize()}/val", m, step=i, log_tensor_mode=log_tensor_mode)
#
#     print(writer.log_dir)
#     for k in metric_values:
#         u_metrics = torch.stack(metric_values[k])
#         m_val = u_metrics.nanmean()
#         print(f"Mean {k}: {m_val}")
#
#     writer.close()
#
#     return metric_values
