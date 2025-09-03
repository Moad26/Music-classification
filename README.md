# Music Emotion Recognition with Deep Learning

## Project Overview

This project implements a deep learning system for music emotion recognition using the DEAM (Dataset for Emotion Analysis in Music). The system processes audio files to predict two emotional dimensions: valence (pleasantness) and arousal (intensity) using a convolutional neural network architecture.

## Key Features

- **Audio Processing Pipeline**: Comprehensive audio preprocessing including resampling, channel conversion, and spectrogram generation
- **Data Augmentation**: Time shifting and spectrogram masking techniques to improve model generalization
- **Residual CNN Architecture**: Deep neural network with residual connections for effective feature extraction
- **Comprehensive Training**: Support for checkpointing, early stopping, and learning rate scheduling
- **Visualization Tools**: TensorBoard integration for tracking training progress and model performance

## Dataset

The project uses the DEAM (Dataset for Emotion Analysis in Music) which contains:

- 1,802 song excerpts (30-second clips)
- Annotations for valence and arousal dimensions
- Continuous annotations averaged per song

Dataset structure:

```
input/
├── MEMD_audio/          # Audio files
├── annotations/         # Emotion annotations
└── features/           # Precomputed features
```

## Project Structure

```
music-sentiment/
├── src/
│   ├── data.py         # Dataset class and data loading logic
│   ├── datautil.py     # Audio processing utilities
│   ├── model.py        # CNN model architecture
│   ├── train.py        # Training loop and early stopping
│   ├── visual.py       # Visualization utilities
│   ├── main.py         # Main training script
│   └── paths.py        # Path configuration
├── input/              # Dataset storage
├── model/              # Trained model checkpoints
├── visualisation/      # Training visualizations
├── pyproject.toml      # Project dependencies
├── setup.bat          # Windows setup script
└── setup.sh           # Linux/Mac setup script
```

## Installation

### Prerequisites

- Python 3.10 or higher
- UV package manager (recommended)

### Using UV

1. Install UV:

```bash
# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# On Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Set up the project:

```bash
# Windows
setup.bat
uv sync

# Linux/Mac
chmod +x setup.sh
./setup.sh
uv sync
```

## Usage

### Training the Model

The main training script provides numerous configuration options:

```bash
python src/main.py --help  # View all available options
```

Basic training command:

```bash
python src/main.py \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --n-mels 128 \
    --n-fft 2048 \
    --aug-prob 0.5
```

### Configuration Options

Key training parameters:

- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Initial learning rate (default: 1e-3)
- `--n-mels`: Number of mel frequency bins (default: 128)
- `--n-fft`: FFT window size (default: 2048)
- `--aug-prob`: Probability of applying augmentation (default: 0.5)

Audio processing parameters:

- `--max-ms`: Maximum audio length in milliseconds (default: 30000)
- `--sample-rate`: Target sample rate (default: 22050)
- `--channels`: Number of audio channels (1=mono, 2=stereo)

## Model Architecture

The system uses a residual CNN architecture with:

- 7×7 convolutional stem layer
- 4 residual blocks with increasing channels (64→128→256→512)
- Batch normalization and ReLU activations
- Adaptive average pooling before final classification
- Dropout for regularization

## Training Process

The training pipeline includes:

1. Data loading with on-the-fly augmentation
2. Spectrogram generation using Mel-frequency spectrograms
3. Model training with Adam optimizer
4. Validation with early stopping
5. Learning rate scheduling based on validation performance
6. Checkpointing of best model

## Visualization

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir visualisation/runs
```

The visualization includes:

- Training and validation loss curves
- Scatter plots of predictions vs ground truth
- Error distributions for valence and arousal
- Correlation metrics for both dimensions

## Results

The model outputs continuous values for both valence and arousal dimensions. Performance is measured using:

- Mean Squared Error (MSE) loss
- Pearson correlation coefficients for valence and arousal

## Acknowledgments

- DEAM dataset providers: <https://cvml.unige.ch/databases/DEAM/>
