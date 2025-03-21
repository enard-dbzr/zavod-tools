import abc

import seaborn as sns
import torch
from matplotlib import pyplot as plt
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
    def __init__(self, figsize=(6.4, 4.8), show_values=False, log_scale=False):
        self.figsize = figsize
        self.show_values = show_values
        self.log_scale = log_scale

    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.barplot(m, ax=ax)

        if self.show_values:
            ax.bar_label(ax.containers[0])

        if self.log_scale:
            ax.set_yscale("log")

        # fig.tight_layout()

        writer.add_figure(title, fig, step)


class HeatmapPlotter(TensorPlotter):
    def __init__(self, figsize=(6.4, 4.8), permute=None, show_values=False):
        self.figsize = figsize
        self.permute = permute
        self.show_values = show_values

    def plot(self, writer: SummaryWriter, title: str, m: torch.Tensor, step):
        fig, ax = plt.subplots(figsize=self.figsize)

        m = m.unsqueeze(0) if m.dim() == 1 else m
        if self.permute is not None:
            m = m.permute(self.permute)

        sns.heatmap(m, annot=self.show_values, ax=ax)

        # fig.tight_layout()
        writer.add_figure(title, fig, step)
