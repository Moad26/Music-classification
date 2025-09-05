from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """Visualization utilities for emotion prediction model analysis."""

    def __init__(self, log_dir: Path) -> None:
        """Initialize the visualizer with TensorBoard logging directory.

        Args:
            log_dir: Directory path for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def concordance_correlation_coefficient(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Calculate Lin's Concordance Correlation Coefficient.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Concordance correlation coefficient
        """
        # Pearson correlation coefficient
        cor = np.corrcoef(y_true, y_pred)[0][1]

        # Mean values
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)

        # Variance values
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)

        # Standard deviations
        sd_true = np.sqrt(var_true)
        sd_pred = np.sqrt(var_pred)

        # Concordance correlation coefficient
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        return numerator / denominator

    def adjusted_r2_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1
    ) -> float:
        """Calculate Adjusted R² score.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            n_features: Number of features used in the model

        Returns:
            Adjusted R² score
        """
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)

        # Avoid division by zero
        if n - n_features - 1 <= 0:
            return r2

        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adjusted_r2

    def calculate_all_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive metrics for both valence and arousal.

        Args:
            predictions: Model predictions tensor of shape (N, 2) [arousal, valence]
            targets: Ground truth tensor of shape (N, 2) [arousal, valence]

        Returns:
            Dictionary containing all metrics for valence and arousal
        """
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        metrics = {}
        dimensions = ["arousal", "valence"]

        for i, dim in enumerate(dimensions):
            y_true = targets_np[:, i]
            y_pred = preds_np[:, i]

            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)

            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)

            # R² score
            r2 = r2_score(y_true, y_pred)

            # Adjusted R² (assuming single feature for simplicity)
            adj_r2 = self.adjusted_r2_score(y_true, y_pred, n_features=1)

            # Concordance Correlation Coefficient
            ccc = self.concordance_correlation_coefficient(y_true, y_pred)

            # Mean Absolute Error
            mae = np.mean(np.abs(y_true - y_pred))

            # Root Mean Square Error
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

            metrics[dim] = {
                "pearson_correlation": pearson_corr,
                "pearson_p_value": pearson_p,
                "spearman_correlation": spearman_corr,
                "spearman_p_value": spearman_p,
                "r2_score": r2,
                "adjusted_r2": adj_r2,
                "concordance_correlation_coefficient": ccc,
                "mae": mae,
                "rmse": rmse,
            }

        return metrics

    def log_scatter_plots(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        tag_prefix: str = "Predictions",
    ) -> None:
        """Create scatter plots comparing predictions vs ground truth for valence and arousal.

        Args:
            predictions: Model predictions tensor of shape (N, 2) [arousal, valence]
            targets: Ground truth tensor of shape (N, 2) [arousal, valence]
            epoch: Current training epoch
            tag_prefix: Prefix for TensorBoard tag naming
        """
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Valence scatter plot
        ax1.scatter(targets_np[:, 1], preds_np[:, 1], alpha=0.6, s=20)
        ax1.plot([-1, 1], [-1, 1], "r--", linewidth=2, label="Perfect prediction")
        ax1.set_xlabel("True Valence")
        ax1.set_ylabel("Predicted Valence")
        ax1.set_title("Valence Predictions vs True Values")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Arousal scatter plot
        ax2.scatter(targets_np[:, 0], preds_np[:, 0], alpha=0.6, s=20)
        ax2.plot([-1, 1], [-1, 1], "r--", linewidth=2, label="Perfect prediction")
        ax2.set_xlabel("True Arousal")
        ax2.set_ylabel("Predicted Arousal")
        ax2.set_title("Arousal Predictions vs True Values")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        self.writer.add_figure(f"{tag_prefix}/Scatter_Plots", fig, epoch)
        plt.close(fig)

    def log_error_distributions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        tag_prefix: str = "Errors",
    ) -> None:
        """Create histograms showing prediction error distributions.

        Args:
            predictions: Model predictions tensor of shape (N, 2) [arousal, valence]
            targets: Ground truth tensor of shape (N, 2) [arousal, valence]
            epoch: Current training epoch
            tag_prefix: Prefix for TensorBoard tag naming
        """
        errors = predictions - targets
        errors_np = errors.cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Valence error distribution
        ax1.hist(
            errors_np[:, 1], bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        ax1.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
        ax1.set_xlabel("Valence Error")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Valence Error Distribution")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Arousal error distribution
        ax2.hist(
            errors_np[:, 0], bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
        ax2.set_xlabel("Arousal Error")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Arousal Error Distribution")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        self.writer.add_figure(f"{tag_prefix}/Distributions", fig, epoch)
        plt.close(fig)

    def log_comprehensive_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        tag_prefix: str = "Metrics",
    ) -> Dict[str, Dict[str, float]]:
        """Calculate and log comprehensive metrics including all requested correlations and scores.

        Args:
            predictions: Model predictions tensor of shape (N, 2) [arousal, valence]
            targets: Ground truth tensor of shape (N, 2) [arousal, valence]
            epoch: Current training epoch
            tag_prefix: Prefix for TensorBoard tag naming

        Returns:
            Dictionary containing all metrics for valence and arousal
        """
        metrics = self.calculate_all_metrics(predictions, targets)

        # Log all metrics to TensorBoard
        for dimension, dim_metrics in metrics.items():
            for metric_name, value in dim_metrics.items():
                if not np.isnan(value) and not np.isinf(value):
                    self.writer.add_scalar(
                        f"{tag_prefix}/{dimension.title()}_{metric_name.title()}",
                        value,
                        epoch,
                    )

        return metrics

    def log_correlation_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        tag_prefix: str = "Metrics",
    ) -> Tuple[float, float]:
        """Calculate and log Pearson correlation coefficients (maintained for backward compatibility).

        Args:
            predictions: Model predictions tensor of shape (N, 2) [arousal, valence]
            targets: Ground truth tensor of shape (N, 2) [arousal, valence]
            epoch: Current training epoch
            tag_prefix: Prefix for TensorBoard tag naming

        Returns:
            Tuple containing (valence_correlation, arousal_correlation)
        """
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        valence_corr, _ = pearsonr(targets_np[:, 1], preds_np[:, 1])
        arousal_corr, _ = pearsonr(targets_np[:, 0], preds_np[:, 0])

        self.writer.add_scalar(f"{tag_prefix}/Valence_Correlation", valence_corr, epoch)
        self.writer.add_scalar(f"{tag_prefix}/Arousal_Correlation", arousal_corr, epoch)

        return valence_corr, arousal_corr

    def log_all_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor, epoch: int
    ) -> Tuple[float, float]:
        """Log all visualization metrics and plots in one call.

        Args:
            predictions: Model predictions tensor of shape (N, 2) [arousal, valence]
            targets: Ground truth tensor of shape (N, 2) [arousal, valence]
            epoch: Current training epoch

        Returns:
            Tuple containing (valence_correlation, arousal_correlation) for backward compatibility
        """
        # Log comprehensive metrics
        all_metrics = self.log_comprehensive_metrics(predictions, targets, epoch)

        # Log visualizations every 5 epochs to avoid cluttering
        if epoch % 5 == 0 or epoch == 0:
            self.log_scatter_plots(predictions, targets, epoch)
            self.log_error_distributions(predictions, targets, epoch)

        # Print compact metrics summary for every epoch
        arousal_metrics = all_metrics["arousal"]
        valence_metrics = all_metrics["valence"]

        print(
            f"Metrics - Arousal: Pearson={arousal_metrics['pearson_correlation']:.3f}, "
            f"Spearman={arousal_metrics['spearman_correlation']:.3f}, "
            f"R²={arousal_metrics['r2_score']:.3f}, "
            f"Adj-R²={arousal_metrics['adjusted_r2']:.3f}, "
            f"CCC={arousal_metrics['concordance_correlation_coefficient']:.3f}"
        )

        print(
            f"Metrics - Valence: Pearson={valence_metrics['pearson_correlation']:.3f}, "
            f"Spearman={valence_metrics['spearman_correlation']:.3f}, "
            f"R²={valence_metrics['r2_score']:.3f}, "
            f"Adj-R²={valence_metrics['adjusted_r2']:.3f}, "
            f"CCC={valence_metrics['concordance_correlation_coefficient']:.3f}"
        )

        # Return Pearson correlations for backward compatibility
        valence_corr = all_metrics["valence"]["pearson_correlation"]
        arousal_corr = all_metrics["arousal"]["pearson_correlation"]

        return valence_corr, arousal_corr

    def close(self) -> None:
        """Close the TensorBoard writer and release resources."""
        self.writer.close()
