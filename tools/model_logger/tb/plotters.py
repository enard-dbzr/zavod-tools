import abc
from typing import Dict, List

import seaborn as sns
import torch
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker 
from torch.utils.tensorboard import SummaryWriter
from matplotlib.colors import LogNorm, NoNorm

from tools.dataset import PEMSDataset
from tools.preprocessing.scalers import DataUnscaler

class TensorPlotter(abc.ABC):
    @abc.abstractmethod
    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        pass


class MultiplePlotter(TensorPlotter):
    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        for im, vm in enumerate(m):
            writer.add_scalar(f"{title}/target_{im}", vm, step)


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
        norm = LogNorm(*self.cbar_lims) if self.log_scale else NoNorm(*self.cbar_lims)
        
        if self.permute is not None:
            m = m.permute(self.permute)

        sns.heatmap(m, annot=self.show_values, ax=ax, norm=norm)

        # fig.tight_layout()
        writer.add_figure(title, fig, step)

class PredictionPlotter(TensorPlotter):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: PEMSDataset,
        unscaler: DataUnscaler,
        target_columns: List[str],
        model_output_indices: Dict[str, int] = None,
        max_samples: int = 1000,
        figsize: tuple = (16, 8),
        style_dict: Dict = None
    ):
        self.dataset = dataset
        self.unscaler = unscaler
        self.target_columns = target_columns
        self.model_output_indices = model_output_indices or {}
        self.max_samples = max_samples
        self.figsize = figsize
        self.style_dict = style_dict or {
            'true': {'color': 'blue', 'linestyle': '-', 'label': 'True'},
            'pred': {'color': 'orange', 'linestyle': '--', 'label': 'Predicted'}
        }
        self.model = model

    def plot(self, writer: SummaryWriter, title: str, step: int):
        self.model.eval()
        device = next(self.model.parameters()).device
        
        x, y_true = self.dataset[0]
        
        with torch.no_grad():
            y_pred = self.model(x.to(device)).cpu()

        y_true_unscaled = self.unscaler.unscale(y_true)
        y_pred_unscaled = self.unscaler.unscale(y_pred)

        fig = self._create_figure(y_true_unscaled, y_pred_unscaled)
        writer.add_figure(title, fig, step)

    def _create_figure(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        n_targets = len(self.target_columns)
        fig, axes = plt.subplots(n_targets, 1, figsize=(self.figsize[0], self.figsize[1]*n_targets))
        
        if n_targets == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            target_name = self.target_columns[i]
            output_idx = self.model_output_indices.get(target_name, i)
            
            idx = torch.randperm(len(y_true))[:self.max_samples]
            sorted_idx = idx.sort().values

            sns.lineplot(
                x=sorted_idx.numpy(),
                y=y_true[sorted_idx, output_idx].numpy(),
                ax=ax,
                **self.style_dict['true']
            )
            
            sns.lineplot(
                x=sorted_idx.numpy(),
                y=y_pred[sorted_idx, output_idx].numpy(),
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