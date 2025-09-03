from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "input"
AUDIO_DIR = DATA_DIR / "MEMD_audio"
ANNOTATION_DIR = (
    DATA_DIR / "annotations" / "annotations averaged per song" / "song_level"
)
DF1_DIR = ANNOTATION_DIR / "static_annotations_averaged_songs_1_2000.csv"
DF2_DIR = ANNOTATION_DIR / "static_annotations_averaged_songs_2000_2058.csv"

MODEL_DIR = ROOT_DIR / "model"
