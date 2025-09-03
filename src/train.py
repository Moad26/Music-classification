import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from paths import MODEL_DIR

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        verbose: bool = False,
        path: Path = MODEL_DIR / Path("checkpoint.pt"),
    ) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.verbose: bool = verbose
        self.path: Path = path
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min = np.inf

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: Adam,
    ):
        try:
            if self.verbose:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model to {self.path}..."
                )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                self.path,
            )
            self.val_loss_min = val_loss
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def __call__(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: Adam,
    ) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, train_loss, val_loss, model, optimizer)

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, train_loss, val_loss, model, optimizer)
            self.counter = 0

        return self.early_stop


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    es_patience: int = 7,
    lr_patience: int = 7,
    save_path: Path = MODEL_DIR / Path("model_checkpoint.pt"),
    resume: bool = True,
    start_epoch: int = 0,
):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience
    )
    early_stopping = EarlyStopping(patience=es_patience, path=save_path, verbose=True)

    if resume and save_path.exists():
        print(f"Resuming from checkpoint: {save_path}")
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        early_stopping.val_loss_min = checkpoint["val_loss"]
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            spectrograms = batch.spectrogram.to(device)
            targets = batch.targets.to(device)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validating"
            ):
                spectrograms = batch.spectrogram.to(device)
                targets = batch.targets.to(device)

                outputs = model(spectrograms)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        scheduler.step(avg_val_loss)

        if early_stopping(epoch, avg_train_loss, avg_val_loss, model, optimizer):
            print("Early stopping triggered")
            break
