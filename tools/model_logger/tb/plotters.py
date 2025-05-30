import abc
from typing import Dict, List

import seaborn as sns
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from torch.utils.tensorboard import SummaryWriter
from matplotlib.colors import LogNorm, Normalize

from tools.preprocessing.scalers import DataUnscaler

matplotlib.use('Agg')


class TensorPlotter(abc.ABC):
    @abc.abstractmethod
    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        pass


class MultiplePlotter(TensorPlotter):
    def __init__(self, labels: list[str] = None):
        self.labels = labels

    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        for im, vm in enumerate(m):
            tag = f"{title}/target_{im}" if self.labels is None else f"{title}/target_{self.labels[im]}"
            writer.add_scalar(tag, vm, step)


class ScalarsPlotter(TensorPlotter):
    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        writer.add_scalars(title, {f"target_{im}": vm for im, vm in enumerate(m)}, step)


class BarPlotter(TensorPlotter):
    def __init__(self, figsize=(6.4, 4.8), show_values=False, log_scale=False, y_lims=(None, None), n_ticks=None, labels=None):
        self.figsize = figsize
        self.show_values = show_values
        self.log_scale = log_scale
        self.y_lims = y_lims
        self.n_ticks = n_ticks
        self.labels = labels

    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = sns.barplot(m, ax=ax)

        if self.log_scale and 0 in self.y_lims:
            raise ValueError('Один из пределов 0, при log_scale=True!')
        
        if self.n_ticks is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(self.n_ticks))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        ax.set_ylim(*self.y_lims)

        if self.show_values:
            ax.bar_label(
                bars.containers[0], 
                label_type='center',
                color="white",
                fmt="%.1E",
                fontweight="bold",
                fontsize=8
            )

        if self.log_scale:
            ax.set_yscale("log")
            
        if self.labels:
            ax.set_xticks(ax.get_xticks(), self.labels, rotation=20, ha='right')
            # ax.set_xticklabels()

        writer.add_figure(title, fig, step)
        plt.close(fig)


class HeatmapPlotter(TensorPlotter):
    def __init__(self, figsize=(6.4, 4.8), permute=None, show_values=False, cbar_lims=(None, None), log_scale=False):
        self.figsize = figsize
        self.permute = permute
        self.show_values = show_values
        self.cbar_lims = cbar_lims
        self.log_scale = log_scale

    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        fig, ax = plt.subplots(figsize=self.figsize)

        m = m.unsqueeze(0) if m.dim() == 1 else m
        if self.log_scale:
            norm = LogNorm(*self.cbar_lims)
        else:
            norm = None if self.cbar_lims == (None, None) else Normalize(*self.cbar_lims)
        
        if self.permute is not None:
            m = m.permute(self.permute)

        sns.heatmap(m, annot=self.show_values, ax=ax, norm=norm)

        writer.add_figure(title, fig, step)
        plt.close(fig)


class PredictionPlotter(TensorPlotter):
    def __init__(
        self,
        unscaler: DataUnscaler = None,
        model_output_indices: Dict[str, int] = None,
        max_samples: int = 1000,
        figsize: tuple = (12, 5),
        style_dict: Dict = None,
        batchwise = True
    ):
        self.unscaler = unscaler
        self.model_output_indices = model_output_indices or {}
        self.max_samples = max_samples
        self.figsize = figsize
        self.style_dict = style_dict or {
            'true': {'color': 'blue', 'linestyle': '-', 'label': 'True'},
            'pred': {'color': 'orange', 'linestyle': '--', 'label': 'Predicted'}
        }
        self.batchwise = batchwise
        self.y_pred = None
        self.y_true = None
        self.val_idx = None

    def plot(self, writer: SummaryWriter, title: str, step: int):
        if self.y_pred is None or self.y_true is None:
            raise ValueError("Predictions were not set! Use set_y for that purpose!")
    
        if self.unscaler:
            y_true = self.unscaler.unscale(self.y_true)
            y_pred = self.unscaler.unscale(self.y_pred)
        else:
            y_true = self.y_true
            y_pred = self.y_pred
        
        fig = self._create_figure(y_true.cpu() , y_pred.cpu())
        
        writer.add_figure(title, fig, step)
        plt.close(fig)
        
    def set_y(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
    
    def set_idx(self, idx):
        self.val_idx = idx

    def _create_figure(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        n_targets = len(self.model_output_indices)
        fig, axes = plt.subplots(n_targets, 1, figsize=(self.figsize[0], self.figsize[1]*n_targets))
        
        if n_targets == 1:
            axes = [axes]

        for ax, (target_name, output_idx) in zip(axes, self.model_output_indices.items()):
            y_true_line = y_true[:self.max_samples, 0, output_idx] if self.batchwise else y_true[0, :self.max_samples, output_idx]
            y_pred_line = y_pred[:self.max_samples, 0, output_idx] if self.batchwise else y_pred[0, :self.max_samples, output_idx]
            
            sns.lineplot(
                x=torch.arange(0, len(y_true_line)),
                y=y_true_line,
                ax=ax,
                **self.style_dict['true']
            )
            
            sns.lineplot(
                x=torch.arange(0, len(y_pred_line)),
                y=y_pred_line,
                ax=ax,
                **self.style_dict['pred']
            )
            
            ax.set_title(f'Predictions for {target_name}', fontsize=12)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        return fig
