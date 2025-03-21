import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange, tqdm
import seaborn as sns

from tools.metric_collector import MetricCollector
from tools.model_logger import ModelLogger


def _log_metric(writer: SummaryWriter, title: str, m: torch.Tensor, step, log_tensor_mode: str = ""):
    if m.dim() == 0:
        writer.add_scalar(title, m, step)
        return

    if log_tensor_mode == "multiple":
        for im, vm in enumerate(m):
            writer.add_scalar(f"{title}/target_{im}", vm, step)

        return

    if log_tensor_mode == "subplots":
        writer.add_scalars(title, {f"target_{im}": vm for im, vm in enumerate(m)}, step)
        return

    if log_tensor_mode == "barplot":
        ax = sns.barplot(m)
        ax.bar_label(ax.containers[0])
        writer.add_figure(title, ax.figure, step)
        return


def train_eval(net: nn.Module,
               optimizer: torch.optim.Optimizer,
               criterion: Callable,
               num_epochs: int,
               train_dataloader: torch.utils.data.DataLoader,
               val_dataloader: torch.utils.data.DataLoader,
               metric_collector: MetricCollector,
               logger: ModelLogger,
               device="cpu"):

    net.to(device)

    with logger(net, optimizer, criterion, train_dataloader):
        logger.log_model()
        logger.log_hparams()

        for epoch in trange(num_epochs, desc="Epoch"):
            net.train()
            for i, (x_batch, y_batch) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = net(x_batch)

                loss = criterion(y_pred, y_batch)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

                step = epoch * len(train_dataloader) + i

                metrics = metric_collector.calculate_metrics(y_pred, y_batch)
                logger.log_batch_metrics(metrics, step, "train")

            metrics = metric_collector.aggregate_and_release()
            logger.log_epoch_metrics(metrics, epoch, "train")

            logger.log_params(epoch)

            net.eval()
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(tqdm(val_dataloader)):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    y_pred = net(x_batch)

                    step = epoch * len(val_dataloader) + i

                    metrics = metric_collector.calculate_metrics(y_pred, y_batch)
                    logger.log_batch_metrics(metrics, step, "val")

            metrics = metric_collector.aggregate_and_release()
            logger.log_epoch_metrics(metrics, epoch, "val")

            logger.save_model(epoch)



def evaluate(net: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             metrics: Optional[dict[str, Callable]] = None,
             device="cpu",
             comment="",
             log_tensor_mode="subplots"):
    if metrics is None:
        metrics = {}

    net.to(device)

    writer = SummaryWriter(comment=(f"_{comment}_EVAL" if comment else ""))

    metric_values = {k: [] for k in metrics}

    net.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(tqdm(dataloader)):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = net(x_batch)

            for k, v in metrics.items():
                m = v(y_pred, y_batch).detach()
                metric_values[k].append(m)

                _log_metric(writer, f"{k.capitalize()}/val", m, step=i, log_tensor_mode=log_tensor_mode)

    print(writer.log_dir)
    for k in metric_values:
        u_metrics = torch.stack(metric_values[k])
        m_val = u_metrics.nanmean()
        print(f"Mean {k}: {m_val}")

    writer.close()

    return metric_values
