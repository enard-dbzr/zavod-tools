import abc
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class ModelLogger(abc.ABC):
    @abc.abstractmethod
    def __call__(self, net: nn.Module, optimizer: torch.optim.Optimizer, criterion, train_dataloader):
        pass

    @abc.abstractmethod
    def log_model(self):
        pass

    @abc.abstractmethod
    def log_hparams(self):
        pass

    @abc.abstractmethod
    def log_params(self, step=0):
        pass

    @abc.abstractmethod
    def log_batch_metrics(self, metrics: dict[str, Tensor], step=0, tag=""):
        pass

    @abc.abstractmethod
    def log_epoch_metrics(self, metrics: dict[str, Tensor], step=0, tag=""):
        pass

    @abc.abstractmethod
    def save_model(self, step):
        pass

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TensorBoardLogger(ModelLogger):
    def __init__(self, comment, log_params_every=5, log_in_epoch=False, checkpoint_every=5):
        self.comment = comment
        self.log_params_every = log_params_every
        self.log_in_epoch = log_in_epoch
        self.checkpoint_every = checkpoint_every

        self.writer: Optional[SummaryWriter] = None

        self.net: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion = None
        self.train_dataloader: Optional[DataLoader] = None

    def __call__(self, net: nn.Module, optimizer: torch.optim.Optimizer, criterion, train_dataloader):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader

        return self

    def __enter__(self):
        self.writer = SummaryWriter(comment=(f"_{self.comment}" if self.comment else ""))
        self.checkpoints_folder = Path(f"checkpoints/{Path(self.writer.log_dir).name}")
        os.makedirs(self.checkpoints_folder, exist_ok=True)

        return self

    def log_model(self):
        dummy_input = torch.randn(1, *self.train_dataloader.dataset[0][0].shape).to(next(self.net.parameters()).device)
        self.writer.add_graph(self.net, dummy_input)

    def log_hparams(self):
        hparams = {
            "net": repr(self.net),
            "input_shape": repr(self.train_dataloader.dataset[0][0].shape),
            "criterion": repr(self.criterion),
            "batch_size": self.train_dataloader.batch_size,
            "optimizer": repr(self.optimizer),
            "optimizer_lr": self.optimizer.defaults["lr"]
        }

        self.writer.add_hparams(hparams, {})

    def log_params(self, step=0):
        if step % self.log_params_every == 0:
            for name, param in self.net.named_parameters():
                self.writer.add_histogram(name, param, step)

    def log_batch_metrics(self, metrics: dict[str, Tensor], step=0, tag=""):
        if self.log_in_epoch:
            for k, m in metrics.items():
                self.writer.add_scalar(f"{k.capitalize()}/{tag}", m, step)

    def log_epoch_metrics(self, metrics: dict[str, Tensor], step=0, tag=""):
        for k, m in metrics.items():
            self.writer.add_scalar(f"{k.capitalize()}/{tag}/epochwise", m, step + 1)

    def save_model(self, step):
        if (step + 1) % self.checkpoint_every == 0:
            torch.save(self.net.state_dict(), self.checkpoints_folder / f'epoch_{step + 1}.pth')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
