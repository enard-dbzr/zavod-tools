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
        log_intervals: Dict[str, int] = None,
        log_in_epoch: bool = False,
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
        if log_intervals is None:
            log_intervals = {}

        self.comment = comment
        self.log_in_epoch = log_in_epoch
        self.default_tensor_plotter = default_tensor_plotter
        self.specific_tensor_plotter = tensor_plotters
        self.val_dataset = val_dataset
        self.unscaler = unscaler
        self.target_columns = target_columns

        default_intervals = {
            'params': 5,
            'checkpoint': 5,
            'predictions': 5,
            'batch_metrics': 25,
            'epoch_metrics': 1,
        }
        self.log_intervals = default_intervals.copy()
        self.log_intervals.update(log_intervals)

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
        if self._should_log('params', step) and self.net:
            for name, param in self.net.named_parameters():
                self.writer.add_histogram(name, param, step)

    def log_predictions(self, step=0, y_pred=None, y_true=None, idx=None):
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        if self._should_log('predictions', step):
            plotter = self.specific_tensor_plotter.get("predictions")
            if plotter:
                plotter.set_y(y_pred, y_true)
                plotter.set_idx(idx)
                plotter.plot(self.writer, "Predictions", step)

    def log_batch_metrics(self, metrics: Dict[str, Tensor], step=0, tag=""):
        if self._should_log('batch_metrics', step) and self.log_in_epoch:
            for k, m in metrics.items():
                self._plot_metric(f"{k.capitalize()}/{tag}", k, m, step)

    def log_epoch_metrics(self, metrics: Dict[str, Tensor], step=0, tag=""):
        if self._should_log('epoch_metrics', step):
            for k, m in metrics.items():
                self._plot_metric(f"{k.capitalize()}/{tag}/epochwise", k, m, step + 1)

    def save_model(self, step):
        if self._should_log('checkpoint', step) and self.net:
            torch.save(self.net.state_dict(), self.checkpoints_folder / f'epoch_{step}.pth')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()

    def _should_log(self, log_type: str, step: int) -> bool:
        interval = self.log_intervals.get(log_type, 0)
        return interval > 0 and step % interval == 0

    def _plot_metric(self, title: str, m_key: str, m: torch.Tensor, step):
        if m.dim() == 0:
            self.writer.add_scalar(title, m, step)
            return

        plotter = self.specific_tensor_plotter.get(m_key, self.default_tensor_plotter)
        if plotter:
            plotter.plot(self.writer, title, m.detach().cpu(), step)
