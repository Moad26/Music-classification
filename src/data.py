import random
from collections import namedtuple
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from datautil import AudioUtils

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "input"
AUDIO_DIR = DATA_DIR / "MEMD_audio"
ANNOTATION_DIR = (
    DATA_DIR / "annotations" / "annotations averaged per song" / "song_level"
)
DF1_DIR = ANNOTATION_DIR / "static_annotations_averaged_songs_1_2000.csv"
DF2_DIR = ANNOTATION_DIR / "static_annotations_averaged_songs_2000_2058.csv"

BatchItem = namedtuple("BatchItem", ["spectrogram", "targets", "song_id"])


class music_dataset(Dataset):
    def __init__(
        self,
        df_path: Path = ANNOTATION_DIR,
        audios_path: Path = AUDIO_DIR,
        new_ch: int = 1,
        newsr: int = 22050,
        max_ms: int = 30000,
        n_mel: int = 128,
        n_fft: int = 2048,
        hop_len: Optional[int] = 512,
        top_db: int = 80,
        apply_augmentation: bool = True,
        augmentation_prob: float = 0.5,
        shift_limit: float = 0.2,
        max_mask_pct: float = 0.1,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        normalize: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.audios_path = audios_path
        self.audio_tool = AudioUtils()
        self.annotation_df = self.load_all_df(df_path)
        self.new_ch = new_ch
        self.newsr = newsr
        self.max_ms = max_ms
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.top_db = top_db
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob
        self.shift_limit = shift_limit
        self.max_mask_pct = max_mask_pct
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.normalize = normalize

        self.set_seed(seed)

        if normalize:
            self.normalize_df()

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def normalize_df(self):
        scaler = StandardScaler()
        self.annotation_df[["valence_mean", "arousal_mean"]] = scaler.fit_transform(
            self.annotation_df[["valence_mean", "arousal_mean"]]
        )

    @staticmethod
    def load_all_df(df_dir: Path) -> pd.DataFrame:
        df_list: List[pd.DataFrame] = []
        for file in df_dir.glob("*.csv"):
            df: pd.DataFrame = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            df = df[["song_id", "valence_mean", "arousal_mean"]]
            df_list.append(df)
        if not df_list:
            return pd.DataFrame(columns=["song_id", "valence_mean", "arousal_mean"])
        return pd.concat(df_list, ignore_index=True)

    def __len__(self) -> int:
        return len(self.annotation_df)

    def __getitem__(self, index: int):
        song_id = int(self.annotation_df.iloc[index]["song_id"])
        arousal = self.annotation_df.iloc[index]["arousal_mean"]
        valence = self.annotation_df.iloc[index]["valence_mean"]

        audio_path = self.audios_path / f"{song_id}.mp3"

        aud = self.audio_tool.open(audio_path)

        aud = self.audio_tool.rechannel(aud, self.new_ch)
        aud = self.audio_tool.resample(aud, self.newsr)
        aud = self.audio_tool.pad_trunc(aud, self.max_ms)

        if self.apply_augmentation and random.random() < self.augmentation_prob:
            aud = self.audio_tool.time_shift(aud, self.shift_limit)
        # Convert to spectrogram
        spec = self.audio_tool.spectro_gram(
            aud,
            n_mel=self.n_mel,
            n_fft=self.n_fft,
            hop_len=self.hop_len,
            top_db=self.top_db,
        )

        # Apply spectrogram augmentation if enabled
        if self.apply_augmentation and random.random() < self.augmentation_prob:
            spec = self.audio_tool.spectro_augment(
                spec,
                max_mask_pct=self.max_mask_pct,
                n_freq_masks=self.n_freq_masks,
                n_time_masks=self.n_time_masks,
            )

        return BatchItem(
            spectrogram=spec,
            targets=torch.tensor([arousal, valence], dtype=torch.float32),
            song_id=song_id,
        )
