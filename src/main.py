import argparse
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import music_dataset
from model import CNN_Music_classifier
from paths import ANNOTATION_DIR, AUDIO_DIR, MODEL_DIR
from train import train_model


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train CNN Music Emotion Recognition Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--annotation-dir",
        type=Path,
        default=ANNOTATION_DIR,
        help="Path to annotation CSV files directory",
    )
    data_group.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help="Path to audio files directory",
    )
    data_group.add_argument(
        "--max-ms", type=int, default=30000, help="Maximum audio length in milliseconds"
    )
    data_group.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate for audio resampling",
    )
    data_group.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of audio channels (1=mono, 2=stereo)",
    )

    # Spectrogram arguments
    spec_group = parser.add_argument_group("Spectrogram Configuration")
    spec_group.add_argument(
        "--n-mels", type=int, default=128, help="Number of mel frequency bins"
    )
    spec_group.add_argument("--n-fft", type=int, default=2048, help="FFT window size")
    spec_group.add_argument(
        "--hop-length", type=int, default=512, help="Hop length for STFT"
    )
    spec_group.add_argument(
        "--top-db", type=int, default=80, help="Top dB for amplitude to dB conversion"
    )

    # Data augmentation arguments
    aug_group = parser.add_argument_group("Data Augmentation")
    aug_group.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation for training",
    )
    aug_group.add_argument(
        "--aug-prob",
        type=float,
        default=0.5,
        help="Probability of applying augmentation",
    )
    aug_group.add_argument(
        "--shift-limit",
        type=float,
        default=0.2,
        help="Maximum time shift as fraction of audio length",
    )
    aug_group.add_argument(
        "--max-mask-pct",
        type=float,
        default=0.1,
        help="Maximum percentage of spectrogram to mask",
    )
    aug_group.add_argument(
        "--n-freq-masks", type=int, default=2, help="Number of frequency masks to apply"
    )
    aug_group.add_argument(
        "--n-time-masks", type=int, default=2, help="Number of time masks to apply"
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for model regularization",
    )
    model_group.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes (valence, arousal)",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    train_group.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    train_group.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    train_group.add_argument(
        "--val-batch-size", type=int, default=64, help="Validation batch size"
    )
    train_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    train_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )

    # Early stopping and scheduling
    train_group.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    train_group.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="Learning rate scheduler patience (epochs)",
    )

    # Paths and saving
    save_group = parser.add_argument_group("Saving and Paths")
    save_group.add_argument(
        "--model-save-path",
        type=Path,
        default=MODEL_DIR / "model_checkpoint.pt",
        help="Path to save the best model checkpoint",
    )
    save_group.add_argument(
        "--no-normalize", action="store_true", help="Disable target normalization"
    )
    save_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )

    # Reproducibility
    repro_group = parser.add_argument_group("Reproducibility")
    repro_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


def setup_training(args):
    # Determine device
    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.verbose:
        print(f"Using device: {device}")
        print(f"Configuration: {vars(args)}")

    scaler = StandardScaler()

    train_dataset = music_dataset(
        df_path=args.annotation_dir,
        audios_path=args.audio_dir,
        scaler=scaler,
        fit_scaler=True,
        new_ch=args.channels,
        newsr=args.sample_rate,
        max_ms=args.max_ms,
        n_mel=args.n_mels,
        n_fft=args.n_fft,
        hop_len=args.hop_length,
        top_db=args.top_db,
        apply_augmentation=not args.no_augmentation,
        augmentation_prob=args.aug_prob,
        shift_limit=args.shift_limit,
        max_mask_pct=args.max_mask_pct,
        n_freq_masks=args.n_freq_masks,
        n_time_masks=args.n_time_masks,
        normalize=not args.no_normalize,
        seed=args.seed,
    )

    val_dataset = music_dataset(
        df_path=args.annotation_dir,
        audios_path=args.audio_dir,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        new_ch=args.channels,
        newsr=args.sample_rate,
        max_ms=args.max_ms,
        n_mel=args.n_mels,
        n_fft=args.n_fft,
        hop_len=args.hop_length,
        top_db=args.top_db,
        apply_augmentation=False,
        normalize=not args.no_normalize,
        seed=args.seed,
    )

    if args.verbose:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = CNN_Music_classifier(
        num_channels=args.channels, num_classes=args.num_classes, dropout=args.dropout
    )

    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    args.model_save_path.parent.mkdir(parents=True, exist_ok=True)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        es_patience=args.early_stopping_patience,
        lr_patience=args.lr_patience,
        save_path=args.model_save_path,
        resume=args.resume,
    )


def main():
    parser = create_parser()
    args = parser.parse_args()
    setup_training(args)


if __name__ == "__main__":
    main()
