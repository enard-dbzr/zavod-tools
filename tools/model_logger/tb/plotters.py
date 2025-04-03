import abc

import seaborn as sns
import torch
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker 
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, figsize=(6.4, 4.8), show_values=False, log_scale=False, y_lims=(None, None), n_ticks=None):
        self.figsize = figsize
        self.show_values = show_values
        self.log_scale = log_scale
        self.y_lims = y_lims
        self.n_ticks = n_ticks

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
                fontweight="bold"
            )

        if self.log_scale:
            ax.set_yscale("log")

        writer.add_figure(title, fig, step)


class HeatmapPlotter(TensorPlotter):
    def __init__(self, figsize=(6.4, 4.8), permute=None, show_values=False, cbar_lims=None):
        self.figsize = figsize
        self.permute = permute
        self.show_values = show_values
        self.cbar_lims = cbar_lims

    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        fig, ax = plt.subplots(figsize=self.figsize)

        m = m.unsqueeze(0) if m.dim() == 1 else m
        if self.permute is not None:
            m = m.permute(self.permute)

        sns.heatmap(m, annot=self.show_values, ax=ax, vmin=self.cbar_lims[0], vmax=self.cbar_lims[1])

        # fig.tight_layout()
        writer.add_figure(title, fig, step)
