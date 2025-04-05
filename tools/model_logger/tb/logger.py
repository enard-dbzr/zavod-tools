import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tools.dataset import PEMSDataset
from tools.model_logger.model_logger import ModelLogger
from tools.model_logger.tb.plotters import TensorPlotter, BarPlotter, PredictionPlotter
from tools.preprocessing.scalers import DataUnscaler


class TensorBoardLogger(ModelLogger):
    def __init__(
        self,
        comment: str,
        log_params_every: int = 5,
        log_in_epoch: bool = False,
        checkpoint_every: int = 5,
        default_tensor_plotter: TensorPlotter = None,
        tensor_plotters: Dict[str, TensorPlotter] = None,
        val_dataset: Optional[PEMSDataset] = None,
        unscaler: Optional[DataUnscaler] = None,
        target_columns: List[str] = None
    ):
        if default_tensor_plotter is None:
            default_tensor_plotter = BarPlotter()
        if tensor_plotters is None:
            tensor_plotters = {}

        self.comment = comment
        self.log_params_every = log_params_every
        self.log_in_epoch = log_in_epoch
        self.checkpoint_every = checkpoint_every
        self.default_tensor_plotter = default_tensor_plotter
        self.specific_tensor_plotter = tensor_plotters
        self.val_dataset = val_dataset
        self.unscaler = unscaler
        self.target_columns = target_columns

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
        if self.train_dataloader and self.net:
            dummy_input = torch.randn(1, *self.train_dataloader.dataset[0][0].shape).to(next(self.net.parameters()).device)
            self.writer.add_graph(self.net, dummy_input)

    def log_hparams(self):
        if self.net and self.train_dataloader and self.optimizer:
            hparams = {
                "net": repr(self.net),
                "input_shape": repr(self.train_dataloader.dataset[0][0].shape),
                "criterion": repr(self.criterion),
                "batch_size": self.train_dataloader.batch_size,
                "optimizer": repr(self.optimizer),
                "optimizer_lr": self.optimizer.defaults.get("lr", 0)
            }
            self.writer.add_hparams(hparams, {})

    def log_params(self, step=0):
        if self.net and step % self.log_params_every == 0:
            for name, param in self.net.named_parameters():
                self.writer.add_histogram(name, param, step)

    def log_predictions(self, step=0):
        plotter = self.specific_tensor_plotter.get("predictions")
        if plotter and isinstance(plotter, PredictionPlotter):
            plotter.plot(self.writer, "Predictions", step)

    def log_batch_metrics(self, metrics: Dict[str, Tensor], step=0, tag=""):
        if self.log_in_epoch:
            for k, m in metrics.items():
                self._plot_metric(f"{k.capitalize()}/{tag}", k, m, step)

    def log_epoch_metrics(self, metrics: Dict[str, Tensor], step=0, tag=""):
        for k, m in metrics.items():
            self._plot_metric(f"{k.capitalize()}/{tag}/epochwise", k, m, step + 1)
        self.log_predictions(step + 1)

    def save_model(self, step):
        if (step + 1) % self.checkpoint_every == 0 and self.net:
            torch.save(self.net.state_dict(), self.checkpoints_folder / f'epoch_{step + 1}.pth')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()

    def _plot_metric(self, title: str, m_key: str, m: torch.Tensor, step):
        if m.dim() == 0:
            self.writer.add_scalar(title, m, step)
            return

        plotter = self.specific_tensor_plotter.get(m_key, self.default_tensor_plotter)
        if plotter:
            plotter.plot(self.writer, title, m.detach().cpu(), step)
