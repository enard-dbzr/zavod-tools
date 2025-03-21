import abc

import torch
from torch import nn, Tensor


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
