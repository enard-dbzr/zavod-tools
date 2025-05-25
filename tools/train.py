from typing import Union

import torch
from torch import nn, Tensor
from tqdm.notebook import trange, tqdm

from tools.model_logger.model_logger import ModelLogger
from tools.objectives.metrics import Metric, MetricContext


class MetricCollector:
    def __init__(self, metrics: dict[str, Metric], aggregation_fn=None):
        self.metrics = metrics
        self.aggregation_fn = aggregation_fn or (lambda x: torch.nanmean(x, dim=0))

        self.collected = {}
        self.aggregate_and_release()

    def calculate_metrics(self, y_pred: torch.Tensor, batch: dict[str, torch.Tensor],
                          dataset, dataloader_tag: str) -> dict[str, Tensor]:
        current = {}
        for k, v in self.metrics.items():
            m = v(y_pred, batch["y"], x=batch["x"],
                  ctx=MetricContext(
                      iloc=batch["iloc"],
                      dataset=dataset,
                      dataloader_tag=dataloader_tag,
                      extra=batch
                  )).detach()
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
               val_dataloaders: Union[torch.utils.data.DataLoader, dict[str, torch.utils.data.DataLoader]],
               metric_collector: MetricCollector,
               logger: ModelLogger,
               log_params: bool = False,
               leave_progress_bars=False,
               device="cpu"):
    if isinstance(val_dataloaders, torch.utils.data.DataLoader):
        val_dataloaders = {"val": val_dataloaders}

    net.to(device)

    with logger(net, optimizer, criterion, train_dataloader, num_epochs):
        logger.log_model()
        logger.log_hparams()

        for epoch in trange(num_epochs, desc="Epoch"):
            net.train()
            for i, batch in enumerate(tqdm(train_dataloader, leave=leave_progress_bars)):
                batch: dict[str, torch.Tensor]
                optimizer.zero_grad()

                batch = {k: v.to(device) for k, v in batch.items()}

                y_pred = net(**batch)

                loss = criterion(y_pred, batch["y"], x=batch["x"],
                                 ctx=MetricContext(
                                     iloc=batch["iloc"],
                                     dataset=train_dataloader.dataset,
                                     dataloader_tag="train",
                                     extra=batch
                                 ))
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    step = epoch * len(train_dataloader) + i

                    metrics = metric_collector.calculate_metrics(y_pred, batch,
                                                                 train_dataloader.dataset, "train")
                    logger.log_batch_metrics(metrics, step, "train")

            metrics = metric_collector.aggregate_and_release()
            logger.log_epoch_metrics(metrics, epoch, "train")

            if log_params:
                logger.log_params(epoch)

            net.eval()

            for val_name, val_dataloader in val_dataloaders.items():
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(val_dataloader, leave=leave_progress_bars)):
                        batch: dict[str, torch.Tensor]

                        batch = {k: v.to(device) for k, v in batch.items()}

                        y_pred = net(**batch)

                        step = epoch * len(val_dataloader) + i

                        logger.log_predictions(step, y_pred, batch["y"])

                        metrics = metric_collector.calculate_metrics(y_pred, batch,
                                                                     val_dataloader.dataset, val_name)
                        logger.log_batch_metrics(metrics, step, val_name)

                metrics = metric_collector.aggregate_and_release()
                logger.log_epoch_metrics(metrics, epoch, val_name)

            logger.save_model(epoch)
