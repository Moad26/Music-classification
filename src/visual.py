from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """Visualization utilities for emotion prediction model analysis."""

    def __init__(self, log_dir: Path) -> None:
        """Initialize the visualizer with TensorBoard logging directory.

        Args:
            log_dir: Directory path for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir=log_dir)

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

    def log_correlation_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        tag_prefix: str = "Metrics",
    ) -> Tuple[float, float]:
        """Calculate and log Pearson correlation coefficients.

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
            Tuple containing (valence_correlation, arousal_correlation)
        """
        self.log_scatter_plots(predictions, targets, epoch)
        self.log_error_distributions(predictions, targets, epoch)
        valence_corr, arousal_corr = self.log_correlation_metrics(
            predictions, targets, epoch
        )

        return valence_corr, arousal_corr

    def close(self) -> None:
        """Close the TensorBoard writer and release resources."""
        self.writer.close()
