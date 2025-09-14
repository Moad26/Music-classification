# Music Emotion Recognition with Deep Learning

## Project Overview

This project implements a comprehensive deep learning system for music emotion recognition using the DEAM (Dataset for Emotion Analysis in Music). The system processes audio files to predict two emotional dimensions: valence (pleasantness) and arousal (intensity) using a convolutional neural network architecture, complete with a web-based interface for real-time emotion analysis.

## Key Features

- **Audio Processing Pipeline**: Comprehensive audio preprocessing including resampling, channel conversion, and spectrogram generation
- **Data Augmentation**: Time shifting and spectrogram masking techniques to improve model generalization
- **Residual CNN Architecture**: Deep neural network with residual connections for effective feature extraction
- **RESTful API Backend**: FastAPI-powered backend for serving model predictions
- **Interactive Web Interface**: Modern React-based frontend for audio file upload and emotion visualization
- **Real-time Processing**: End-to-end system supporting multiple audio formats with instant emotion analysis
- **Comprehensive Training**: Support for checkpointing, early stopping, and learning rate scheduling
- **Visualization Tools**: TensorBoard integration for tracking training progress and model performance

## System Architecture

The project consists of three main components:

1. **Core ML Pipeline** (`src/`): Training, model definition, and data processing
2. **API Backend** (`src/backend.py`): FastAPI server for model inference
3. **Web Frontend** (`web/`): React application for user interaction

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
├── features/           # Precomputed features
├── DEAM_Annotations.zip
├── DEAM_audio.zip
└── features.zip
```

## Project Structure

```
music-sentiment/
├── src/
│   ├── __init__.py
│   ├── backend.py       # FastAPI backend server
│   ├── data.py         # Dataset class and data loading logic
│   ├── datautil.py     # Audio processing utilities
│   ├── model.py        # CNN model architecture
│   ├── train.py        # Training loop and early stopping
│   ├── visual.py       # Visualization utilities
│   ├── main.py         # Main training script
│   └── paths.py        # Path configuration
├── web/                # React frontend application
│   ├── src/
│   │   ├── App.jsx     # Main application component
│   │   ├── components/
│   │   │   ├── EmotionDisplay.jsx    # Emotion visualization
│   │   │   ├── FileUpload.jsx        # File upload interface
│   │   │   └── LoadingSpinner.jsx    # Loading indicator
│   │   └── [CSS files]
│   ├── public/         # Static assets
│   └── package.json    # Frontend dependencies
├── input/              # Dataset storage
├── model/              # Trained model checkpoints
│   └── model_checkpoint.pt
├── visualisation/      # Training visualizations
│   └── runs/          # TensorBoard logs
├── pyproject.toml      # Python project dependencies
├── setup.bat          # Windows setup script
└── setup.sh           # Linux/Mac setup script
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 16+ (for web interface)
- UV package manager (recommended)

### Backend Setup

1. Install UV:

```bash
# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# On Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Set up the Python environment:

```bash
# Windows
setup.bat
uv sync

# Linux/Mac
chmod +x setup.sh
./setup.sh
uv sync
```

### Frontend Setup

1. Navigate to the web directory:

```bash
cd web
```

2. Install Node.js dependencies:

```bash
npm install
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

### Running the Complete System

#### 1. Start the Backend API

```bash
# From the root directory
python src/backend.py
```

The API will be available at `http://localhost:8000`

#### 2. Start the Frontend

```bash
# From the web directory
cd web
npm run dev
```

The web interface will be available at `http://localhost:3000`

### API Endpoints

- `GET /`: API status and information
- `GET /health`: Health check and model status
- `POST /predict`: Upload audio file for emotion prediction
- `GET /supported-formats`: List of supported audio formats

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

## Web Interface Features

### File Upload Component

- Drag-and-drop interface for audio files
- Support for multiple audio formats (MP3, WAV, M4A, FLAC, OGG)
- Visual feedback for file selection

### Emotion Visualization

- Interactive 2D emotion quadrant display
- Real-time positioning based on arousal/valence values
- Detailed emotion labels and numerical values
- Four-quadrant emotion mapping:
  - **Happy/Excited**: High arousal, positive valence
  - **Angry/Tense**: High arousal, negative valence
  - **Peaceful/Content**: Low arousal, positive valence
  - **Sad/Depressed**: Low arousal, negative valence

### User Experience

- Loading indicators during processing
- Error handling and user feedback
- Responsive design for various screen sizes
- Clean, modern interface

## Training Process

The training pipeline includes:

1. Data loading with on-the-fly augmentation
2. Spectrogram generation using Mel-frequency spectrograms
3. Model training with Adam optimizer
4. Validation with early stopping
5. Learning rate scheduling based on validation performance
6. Checkpointing of best model

## Monitoring and Visualization

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir visualisation/runs
```

The visualization includes:

- Training and validation loss curves
- Scatter plots of predictions vs ground truth
- Error distributions for valence and arousal
- Correlation metrics for both dimensions

## Supported Audio Formats

The system supports the following audio formats:

- MP3
- WAV
- M4A
- FLAC
- OGG

**Limitations:**

- Maximum file size: 10MB
- Maximum duration: 30 seconds (longer files will be truncated)
- Files are automatically converted to mono, 22.05kHz sample rate

## API Response Format

Successful prediction response:

```json
{
  "success": true,
  "prediction": {
    "arousal": 0.25,
    "valence": 0.67,
    "arousal_label": "Low",
    "valence_label": "Positive",
    "quadrant": "Peaceful/Content"
  }
}
```

Error response:

```json
{
  "success": false,
  "error": "Error message describing the issue"
}
```

## Results

The model outputs continuous values for both valence and arousal dimensions. Performance is measured using:

- Mean Squared Error (MSE) loss
- Pearson correlation coefficients for valence and arousal

## Development and Deployment

### Development Mode

- Backend: `python src/backend.py` (auto-reload enabled)
- Frontend: `npm run dev` (hot module replacement)

### Production Deployment

- Backend: Use a production ASGI server like Gunicorn with Uvicorn workers
- Frontend: `npm run build` to create optimized production build
- Configure CORS settings for your production domains

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure `model_checkpoint.pt` exists in the `model/` directory
2. **CORS errors**: Check that the frontend URL is included in the backend CORS settings
3. **Audio processing errors**: Verify that uploaded files are valid audio formats
4. **Memory issues**: Reduce batch size during training or use CPU mode if GPU memory is limited

### Dependencies

The project uses modern dependency management:

- **Backend**: UV for Python package management
- **Frontend**: NPM for Node.js package management
- **Key libraries**: PyTorch, FastAPI, React, Lucide React icons

## Acknowledgments

- DEAM dataset providers: <https://cvml.unige.ch/databases/DEAM/>
