import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.datautil import AudioUtils
from src.model import CNN_Music_classifier
from src.paths import MODEL_DIR


class EmotionPrediction(BaseModel):
    arousal: float
    valence: float
    arousal_label: str
    valence_label: str
    quadrant: str


class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[EmotionPrediction] = None
    error: Optional[str] = None


model = None
audio_tool = AudioUtils()


def process_audio(
    audio_path: Path,
    new_ch: int = 1,
    newsr: int = 22050,
    max_ms: int = 30000,
    n_mel: int = 128,
    n_fft: int = 2048,
    hop_len: Optional[int] = 512,
    top_db: int = 80,
) -> torch.Tensor:
    try:
        aud = audio_tool.open(audio_path)

        aud = audio_tool.rechannel(aud, new_ch)
        aud = audio_tool.resample(aud, newsr)
        aud = audio_tool.pad_trunc(aud, max_ms)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading audio: {str(e)}")
    try:
        spec = audio_tool.spectro_gram(
            aud,
            n_mel=n_mel,
            n_fft=n_fft,
            hop_len=hop_len,
            top_db=top_db,
        )
        spec = spec.unsqueeze(0)
        return spec
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating spectrogram: {str(e)}"
        )


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global model
    try:
        save_path: Path = MODEL_DIR / Path("model_checkpoint.pt")
        model = CNN_Music_classifier()
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"error loading the model {e}")
        model = CNN_Music_classifier()
        model.to(device)


def interpret_emotions(arousal: float, valence: float) -> Dict[str, str]:
    """Convert numerical values to emotion labels"""
    arousal_label = "High" if arousal > 0 else "Low"
    valence_label = "Positive" if valence > 0 else "Negative"

    if arousal > 0 and valence > 0:
        quadrant = "Happy/Excited"
    elif arousal > 0 and valence <= 0:
        quadrant = "Angry/Tense"
    elif arousal <= 0 and valence > 0:
        quadrant = "Peaceful/Content"
    else:
        quadrant = "Sad/Depressed"

    return {
        "arousal_label": arousal_label,
        "valence_label": valence_label,
        "quadrant": quadrant,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    print("Model loaded!")
    yield
    print("App shutting down...")


app = FastAPI(title="Music Emotion Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Music Emotion Recognition API", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)

        try:
            spectrogram = process_audio(tmp_file_path)
            spectrogram = spectrogram.to(device)

            model.eval()
            with torch.no_grad():
                prediction = model(spectrogram)
                arousal, valence = prediction[0].tolist()

            emotion_labels = interpret_emotions(arousal, valence)

            emotion_prediction = EmotionPrediction(
                arousal=float(arousal),
                valence=float(valence),
                arousal_label=emotion_labels["arousal_label"],
                valence_label=emotion_labels["valence_label"],
                quadrant=emotion_labels["quadrant"],
            )

            return PredictionResponse(success=True, prediction=emotion_prediction)

        finally:
            os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        return PredictionResponse(success=False, error=f"Prediction failed: {str(e)}")


@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported audio formats"""
    return {
        "formats": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
        "max_file_size": "10MB",
        "max_duration": "30 seconds (will be truncated if longer)",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
