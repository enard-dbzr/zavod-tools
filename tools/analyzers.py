import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from tools.objectives.metrics import MSE

class TimeSeriesAnalyzer:
    def __init__(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, loss_fn=MSE(), mc_iterations: int = 100):
        self.model = model
        self.x = inputs
        self.y = targets
        self.loss_fn = loss_fn
        self.mc_iterations = mc_iterations
        self.device = next(model.parameters()).device
        self.x, self.y = self.x.to(self.device), self.y.to(self.device)
            
    def _monte_carlo_noise(self, noise_std=1):
        self.model.eval()
        preds = []
        for _ in tqdm(range(self.mc_iterations), desc="MC Noise"):
            noisy_input = self.x + torch.randn_like(self.x) * noise_std
            with torch.no_grad():
                preds.append(self.model(noisy_input).cpu().numpy())
        preds = np.stack(preds)
        return preds.mean(axis=0), preds.std(axis=0)


    def _sensitivity_analysis(self):
        self.model.eval()
        x = self.x.clone().detach().requires_grad_(True)
        output = self.model(x)
        
        loss = output.pow(2).sum()
        loss.backward()

        grads = x.grad.detach().cpu().abs().numpy()
        return grads

    def plot_uncertainty(self, batch_idx=0, feature_idx=0, input_std=1):
        self.mean_pred, self.std_pred = self._monte_carlo_noise(noise_std=input_std)
        
        true = self.y[batch_idx, :, feature_idx].cpu().numpy()
        mean = self.mean_pred[batch_idx, :, feature_idx]
        std = self.std_pred[batch_idx, :, feature_idx]
        t = np.arange(len(mean))

        plt.figure(figsize=(10, 4))
        plt.plot(t, true, label="True")
        plt.plot(t, mean, label="Mean Prediction")
        plt.fill_between(t, mean - std, mean + std, alpha=0.3, label=r"$\pm1$ std")
        plt.legend(); plt.grid(True)
        plt.title(fr"Uncertainty | Input noise $\sigma={input_std}$ | Feature {feature_idx}")
        plt.tight_layout(); plt.show()

    def plot_sensitivity(self, batch_idx=0):
        self.sensitivity = self._sensitivity_analysis()
    
        sns.heatmap(self.sensitivity[batch_idx].T, cmap="magma", cbar_kws={"label": "Sensitivity"})
        plt.xlabel("Time"); plt.ylabel("Feature")
        plt.title(f"Sensitivity | Batch {batch_idx}")
        plt.tight_layout(); plt.show()
        
    def plot_input_output_sensitivity(self, p=1):
        sens_matrix = self.input_output_sensitivity_matrix(percent_delta=p / 100)
        plt.figure(figsize=(10, 6))
        sns.heatmap(sens_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=[f"Out {i}" for i in range(sens_matrix.shape[1])],
                    yticklabels=[f"In {i}" for i in range(sens_matrix.shape[0])])
        plt.xlabel("Output Feature")
        plt.ylabel("Input Feature")
        plt.title(f"Sensitivity: {p}% Input Change -> % Output Change")
        plt.tight_layout()
        plt.show()


    def input_output_sensitivity_matrix(self, percent_delta=0.01):
        self.model.eval()
        B, T, F = self.x.shape
        Y = self.y.shape[-1]
        base = self.model(self.x).detach().cpu().numpy()  # (B, T, Y)
        
        sensitivities = np.zeros((F, Y))  # Input feature -> output feature

        for f in range(F):
            x_mod = self.x.clone()
            delta = self.x[:, :, f] * percent_delta
            x_mod[:, :, f] += delta

            with torch.no_grad():
                perturbed = self.model(x_mod).detach().cpu().numpy()  # (B, T, Y)

            diff = np.abs(perturbed - base) / (np.abs(base) + 1e-8)  # relative change
            sensitivities[f] = diff.mean(axis=(0, 1)) * 100  # percentage

        return sensitivities
