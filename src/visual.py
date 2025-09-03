from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scatter_plots(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        tag_prefix: str = "Predictions",
    ):
        """
        Create scatter plots for valence and arousal predictions vs ground truth
        """
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(targets_np[:, 1], preds_np[:, 1], alpha=0.5)
        ax1.plot([-1, 1], [-1, 1], "r--")
        ax1.set_xlabel("True Valence")
        ax1.set_ylabel("Predicted Valence")
        ax1.set_title("Valence Predictions vs True Values")
        ax1.grid(True)

        ax2.scatter(targets_np[:, 0], preds_np[:, 0], alpha=0.5)
        ax2.plot([-1, 1], [-1, 1], "r--")
        ax2.set_xlabel("True Arousal")
        ax2.set_ylabel("Predicted Arousal")
        ax2.set_title("Arousal Predictions vs True Values")
        ax2.grid(True)

        plt.tight_layout()

        self.writer.add_figure(f"{tag_prefix}/Scatter_Plots", fig, epoch)
        plt.close(fig)

    def log_error_distributions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        tag_prefix: str = "Errors",
    ):
        """
        Create histograms of prediction errors for valence and arousal
        """
        errors = predictions - targets
        errors_np = errors.cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(errors_np[:, 1], bins=50, alpha=0.7)
        ax1.set_xlabel("Valence Error")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Valence Error Distribution")
        ax1.grid(True)

        ax2.hist(errors_np[:, 0], bins=50, alpha=0.7)
        ax2.set_xlabel("Arousal Error")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Arousal Error Distribution")
        ax2.grid(True)

        plt.tight_layout()

        self.writer.add_figure(f"{tag_prefix}/Distributions", fig, epoch)
        plt.close(fig)

    def log_correlation_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        tag_prefix: str = "Metrics",
    ):
        """
        Calculate and log correlation coefficients for valence and arousal
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
    ):
        """
        Log all visualization metrics at once
        """
        self.log_scatter_plots(predictions, targets, epoch)
        self.log_error_distributions(predictions, targets, epoch)
        valence_corr, arousal_corr = self.log_correlation_metrics(
            predictions, targets, epoch
        )

        return valence_corr, arousal_corr

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
