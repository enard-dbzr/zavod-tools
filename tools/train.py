import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange, tqdm


def _compose_hparams(net: nn.Module, optimizer: torch.optim.Optimizer,
                     criterion: Callable, dataloader: torch.utils.data.DataLoader):
    res = {
        "net": repr(net),
        "input_shape": repr(dataloader.dataset[0][0].shape),
        "criterion": repr(criterion),
        "batch_size": dataloader.batch_size,
        "optimizer": repr(optimizer),
        "optimizer_lr": optimizer.defaults["lr"]
    }

    return res


def train_eval(net: nn.Module,
               optimizer: torch.optim.Optimizer,
               criterion: Callable,
               num_epochs: int,
               train_dataloader: torch.utils.data.DataLoader,
               val_dataloader: torch.utils.data.DataLoader,
               metrics: Optional[dict[str, Callable]] = None,
               device="cpu",
               comment="",
               checkpoint_every=1,
               log_in_epoch=False,
               log_params_every=5,
               log_tensor_metric_mode="scalars"):
    if metrics is None:
        metrics = {}

    metrics = {"Loss": criterion, **metrics}

    net.to(device)

    writer = SummaryWriter(comment=(f"_{comment}" if comment else ""))
    checkpoints_folder = Path(f"checkpoints/{Path(writer.log_dir).name}")
    os.makedirs(checkpoints_folder, exist_ok=True)

    # логирование архитектуры модели
    dummy_input = torch.randn(1, *train_dataloader.dataset[0][0].shape).to(device)
    writer.add_graph(net, dummy_input)
    writer.add_hparams(_compose_hparams(net, optimizer, criterion, train_dataloader), {})

    for epoch in trange(num_epochs, desc="Epoch", position=0):
        epochwise_metrics_train = {k: [] for k in metrics}
        epochwise_metrics_val = {k: [] for k in metrics}

        net.train()
        for i, (x_batch, y_batch) in enumerate(tqdm(train_dataloader, position=1)):
            optimizer.zero_grad()

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = net(x_batch)

            loss = criterion(y_pred, y_batch)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()

            step = epoch * len(train_dataloader) + i

            for k, v in metrics.items():
                m = v(y_pred, y_batch).detach()
                epochwise_metrics_train[k].append(m)
                if log_in_epoch:
                    writer.add_scalar(f"{k.capitalize()}/train", m, step)

        # логирование параметров модели
        if epoch % log_params_every == 0:
            for name, param in net.named_parameters():
                writer.add_histogram(name, param, step)

        net.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(tqdm(val_dataloader)):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = net(x_batch)

                step = epoch * len(val_dataloader) + i

                for k, v in metrics.items():
                    m = v(y_pred, y_batch).detach()
                    epochwise_metrics_val[k].append(m)
                    if log_in_epoch:
                        writer.add_scalar(f"{k.capitalize()}/val", m, step)

        for k in epochwise_metrics_train:
            m_train = torch.stack(epochwise_metrics_train[k]).mean(dim=-1)
            m_val = torch.stack(epochwise_metrics_val[k]).mean(dim=-1)
            writer.add_scalar(f"{k.capitalize()}/train/epochwise", m_train, epoch + 1)
            writer.add_scalar(f"{k.capitalize()}/val/epochwise", m_val, epoch + 1)

        if (epoch + 1) % checkpoint_every == 0:
            torch.save(net.state_dict(), checkpoints_folder / f'epoch_{epoch + 1}.pth')

    writer.close()


def evaluate(net: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             metrics: Optional[dict[str, Callable]] = None,
             device="cpu",
             comment=""):
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

                if m.dim() == 0:
                    writer.add_scalar(f"{k.capitalize()}/val", m, i)
                else:
                    for im, vm in enumerate(m):
                        writer.add_scalar(f"{k.capitalize()}/val/target_{im}", vm, i)
                    # writer.add_scalars(f"{k.capitalize()}/val",
                    #                    {str(im): vm for im, vm in enumerate(m)}, i)

    print(writer.log_dir)
    for k in metric_values:
        u_metrics = torch.stack(metric_values[k])
        m_val = u_metrics.nanmean()
        print(f"Mean {k}: {m_val}")

    writer.close()

    return metric_values
