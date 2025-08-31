import random
from typing import Optional, Tuple

import torch
import torchaudio
from torchaudio import transforms


class AudioUtils:
    @staticmethod
    def open(audio_file_path: str) -> Tuple[torch.Tensor, int]:
        sig, sr = torchaudio.load(audio_file_path)
        return (sig, sr)

    @staticmethod
    def rechannel(
        aud: Tuple[torch.Tensor, int], new_ch: int
    ) -> Tuple[torch.Tensor, int]:
        sig, sr = aud

        if sig.shape[0] == new_ch:
            return aud
        if new_ch == 1:
            # Convert to mono
            resig = torch.mean(sig, dim=0, keepdim=True)
        else:
            # Convert to stereo
            resig = torch.cat([sig, sig])
        return (resig, sr)

    @staticmethod
    def resample(aud: Tuple[torch.Tensor, int], newsr) -> Tuple[torch.Tensor, int]:
        sig, sr = aud

        if sr == newsr:
            return aud
        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return (resig, newsr)

    @staticmethod
    def pad_trunc(aud: Tuple[torch.Tensor, int], max_ms) -> Tuple[torch.Tensor, int]:
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(
                0, max_len - sig_len
            )  # Consider it a way of data augmentation
            """linter gives error for this idk why
            pad_begin_len_tensor = torch.randint(0, max_len - sig_len + 1, (1,))
            pad_begin_len = pad_begin_len_tensor.item()
            """
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return sig, sr

    @staticmethod
    def time_shift(
        aud: Tuple[torch.Tensor, int], shift_limit: float
    ) -> Tuple[torch.Tensor, int]:
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(
        aud: Tuple[torch.Tensor, int],
        n_mel: int = 65,
        n_fft: int = 1024,
        top_db: int = 80,
        hop_len: Optional[int] = None,
    ) -> torch.Tensor:
        sig, sr = aud

        spec = transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mel
        )(sig)

        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spectro_augment(
        spec: torch.Tensor,
        max_mask_pct: float = 0.1,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ) -> torch.Tensor:
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(
                aug_spec, mask_value
            )

        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
