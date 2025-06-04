import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Tuple
import warnings
from scipy import stats

class TimeSeriesAnalyzer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 dataset: torch.utils.data.Dataset,
                 loss_fn: Optional[torch.nn.Module] = None,
                 mc_iterations: int = 100):
        """
        Анализатор временных рядов для моделей PyTorch.
        
        Args:
            model: Модель PyTorch для анализа
            dataset: Датасет временных рядов (возвращает тензоры формы [T, F])
            loss_fn: Функция потерь (по умолчанию MSELoss)
            mc_iterations: Количество итераций для Монте-Карло анализа
        """
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
        self.mc_iterations = mc_iterations
        self.device = next(model.parameters()).device
        
        self._init_data()
        
    def _init_data(self):
        """Загрузить все данные из датасета в память"""
        self.x = self.dataset.tensors['x'].unsqueeze(0)
        self.y = self.dataset.tensors['y'].unsqueeze(0)
        
        self.x_subset = self.x
        self.y_subset = self.y
        self.current_range = (0, len(self.x))
        
    def set_data_range(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None):
        """
        Установить диапазон данных для анализа.
        :params start_idx: Начальный индекс
        :params end_idx: Конечный индекс
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.x)
            
        self.x_subset = self.x[:, start_idx:end_idx]
        self.y_subset = self.y[:, start_idx:end_idx]
        self.current_range = (start_idx, end_idx)
        return self.x_subset, self.y_subset
        
    def reset_data_range(self):
        """Сбросить диапазон до полного датасета"""
        self.x_subset = self.x
        self.y_subset = self.y
        self.current_range = (0, self.x.shape[1])
        return self.x_subset, self.y_subset
        
    def _monte_carlo_noise(self, noise_std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Монте-Карло анализ с шумом во входных данных"""
        self.model.eval()
        preds = []
        
        for _ in tqdm(range(self.mc_iterations), desc="MC Noise"):
            noisy_input = self.x_subset + torch.randn_like(self.x_subset) * noise_std
            with torch.no_grad():
                preds.append(self.model(noisy_input).cpu().numpy())
                
        preds = np.stack(preds)
        return preds.mean(axis=0), preds.std(axis=0)
        
    def _sensitivity_analysis(self) -> np.ndarray:
        """Анализ чувствительности через градиенты"""
        self.model.eval()
        x = self.x_subset.clone().detach().requires_grad_(True)
        output = self.model(x)
        
        loss = output.pow(2).sum()
        loss.backward()

        grads = x.grad.detach().cpu().abs().numpy()
        return grads

    def get_jacobian(self):
        self.model.eval()

        x = self.x_subset.requires_grad_(True)  # [1, T, F_in]
        output = self.model(x)
        
        # Собираем якобиан для всех выходов по всем входам
        jacobians = []
        for j in tqdm(range(output.shape[-1]), desc="Computing Jacobian condition numbers"):  # По каждому выходному признаку
            grad_output = torch.zeros_like(output)
            grad_output[..., j] = 1
            jacobian = torch.autograd.grad(output, x, grad_outputs=grad_output,
                                        retain_graph=True, create_graph=False)[0]
            jacobians.append(jacobian.flatten())  # [T*F_in]
        
        return torch.stack(jacobians).detach().cpu().numpy()
    
    def compute_jacobian_condition_number(self) -> float:
        """
        Вычисляет статистику чисел обусловленности якобиана для выборки из датасета.
        :returns cond: - число обусловленности якобиана
        """
        J = self.get_jacobian()
        if torch.Size(J.shape).numel() > 100_000:
            warnings.warn("The size of Jacobian is too big! It may take some time to calculate the Singular Value Decomposition!")
        
        u, s, vh = np.linalg.svd(J)
        cond = s.max() / (s.min() + 1e-8)
        
        return cond
   
    def plot_uncertainty(self, feature_idx: int = 0, input_std: float = 1.0):
        """Визуализация неопределенности предсказаний"""
        mean_pred, std_pred = self._monte_carlo_noise(noise_std=input_std)
        
        true = self.y_subset[0, :, feature_idx].cpu().numpy()
        mean = mean_pred[0, :, feature_idx]
        std = std_pred[0, :, feature_idx]
        t = np.arange(len(mean))

        plt.figure(figsize=(10, 4))
        plt.plot(t, true, label="True")
        plt.plot(t, mean, label="Mean Prediction")
        plt.fill_between(t, mean - std, mean + std, alpha=0.3, label=r"$\pm1$ std")
        plt.legend()
        plt.grid(True)
        plt.title(fr"Uncertainty | Input noise $\sigma={input_std}$ | Sample {self.current_range} | Feature {feature_idx}")
        plt.tight_layout()
        plt.show()
        
    def plot_sensitivity(self):
        """Визуализация чувствительности модели"""
        sensitivity = self._sensitivity_analysis()
    
        plt.figure(figsize=(10, 4))
        sns.heatmap(sensitivity[0].T, cmap="magma", cbar_kws={"label": "Sensitivity"})
        plt.xlabel("Time")
        plt.ylabel("Feature")
        plt.title(f"Sensitivity | Sample {self.current_range}")
        plt.tight_layout()
        plt.show()
        
    def plot_input_output_sensitivity(self, p: float = 1.0):
        """Визуализация матрицы чувствительности вход-выход"""
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
        
    def input_output_sensitivity_matrix(self, percent_delta: float = 0.01) -> np.ndarray:
        """
        Матрица чувствительности входных признаков к выходным.
        """
        self.model.eval()
        N, T, F_in = self.x_subset.shape
        F_out = self.y_subset.shape[-1]
        
        with torch.no_grad():
            base = self.model(self.x_subset).cpu().numpy()  # [1, T, F_out]
        
        sensitivities = np.zeros((F_in, F_out))

        for f in range(F_in):
            x_mod = self.x_subset.clone()
            delta = self.x_subset[..., f] * percent_delta
            x_mod[..., f] += delta

            with torch.no_grad():
                perturbed = self.model(x_mod).detach().cpu().numpy()  # [1, T, F_out]

            diff = np.abs(perturbed - base) / (np.abs(base) + 1e-8)  # relative change
            sensitivities[f] = diff.mean(axis=(0, 1)) * 100  # percentage

        return sensitivities
        
    def plot_error_distribution(self, feature_idx: int = 0):
        """Визуализация распределения ошибок предсказания"""
        with torch.no_grad():
            preds = self.model(self.x_subset).cpu().numpy()
        
        errors = preds[..., feature_idx] - self.y_subset[..., feature_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 4))
        sns.histplot(errors.flatten(), kde=True, bins=50)
        plt.xlabel("Prediction Error")
        plt.title(f"Error Distribution for Feature {feature_idx}")
        plt.grid(True)
        plt.show()
        
    def plot_error_correlation(self):
        """Матрица корреляции между ошибками разных признаков"""
        with torch.no_grad():
            preds = self.model(self.x_subset).cpu().numpy()
        
        errors = preds - self.y_subset.cpu().numpy()
        N, T, F = errors.shape
        errors_reshaped = errors.reshape(-1, F)
        
        corr_matrix = np.corrcoef(errors_reshaped.T)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=[f"Feature {i}" for i in range(F)],
                    yticklabels=[f"Feature {i}" for i in range(F)])
        plt.title("Error Correlation Between Features")
        plt.tight_layout()
        plt.show()
        
    def plot_autocorrelation(self, feature_idx: int = 0, max_lag: int = 50):
        """Анализ автокорреляции ошибок во времени"""
        with torch.no_grad():
            preds = self.model(self.x_subset).cpu().numpy()
        
        errors = preds[..., feature_idx] - self.y_subset[..., feature_idx].cpu().numpy()
        errors_flat = errors.flatten()
        
        plt.figure(figsize=(10, 4))
        plt.acorr(errors_flat - np.mean(errors_flat), maxlags=max_lag)
        plt.xlim(0, max_lag)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title(f"Error Autocorrelation for Feature {feature_idx}")
        plt.grid(True)
        plt.show()
        
    def feature_importance_analysis(self, n_permutations: int = 100) -> torch.Tensor:
        """
        Анализ важности признаков через пермутационный тест.
        """
        with torch.no_grad():
            baseline_loss = self.loss_fn(self.model(self.x_subset), self.y_subset).item()
            
        importance = torch.zeros(self.x_subset.shape[-1], device=self.device)
        
        for feature in tqdm(range(self.x_subset.shape[-1]), desc="Permuting..."):
            total_increase = 0.0
            for _ in range(n_permutations):
                x_permuted = self.x_subset.clone()
                perm_idx = torch.randperm(self.x_subset.shape[1])
                x_permuted[..., feature] = x_permuted[:, perm_idx, feature]

                with torch.no_grad():
                    perm_loss = self.loss_fn(self.model(x_permuted), self.y_subset)
                total_increase += (perm_loss.item() - baseline_loss)

            importance[feature] = total_increase / n_permutations

        plt.figure(figsize=(10, 4))
        plt.bar(range(len(importance)), importance.cpu().numpy())
        plt.xlabel("Feature Index")
        plt.ylabel("Importance Score")
        plt.title("Feature Importance via Permutation Test")
        plt.grid(True)
        plt.show()
        
        return importance
        
    def plot_multiple_features(self, n_features: int = 5):
        """Визуализация предсказаний для нескольких признаков"""
        with torch.no_grad():
            preds = self.model(self.x_subset).squeeze(0).cpu().numpy()
        
        true = self.y_subset.cpu().numpy()
        n_features = min(n_features, true.shape[-1])
        
        plt.figure(figsize=(12, 2*n_features))
        for i in range(n_features):
            plt.subplot(n_features, 1, i+1)
            plt.plot(true[:, i], label="True")
            plt.plot(preds[:, i], label="Predicted")
            plt.legend()
            plt.grid(True)
            plt.title(f"Feature {i}")
        plt.tight_layout()
        plt.show()
    
    def plot_qq_distribution(self, feature_idx: int = 0):
        """Визуализация распределения квантилей остатков (ошибок) для конкретного признака"""
        with torch.no_grad():
            preds = self.model(self.x_subset).cpu().numpy()
        residuals = self.y_subset.cpu().numpy()[..., feature_idx] - preds[..., feature_idx]

        stats.probplot(residuals.flatten(), plot=plt)
        plt.title("Q-Q Plot")
        plt.tight_layout()
        plt.show()