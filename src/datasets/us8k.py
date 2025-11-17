# src/datasets/us8k.py
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset


# -----------------------------
# Robust audio loading helpers
# -----------------------------
def safe_load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Robust audio loader:
    - Try soundfile (fast); on failure, fallback to librosa(audioread).
    - Stereo -> mono
    - Resample to target_sr
    - Clean NaN/Inf, handle empty clips, soft peak normalize
    """
    try:
        y, sr = sf.read(path, always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        if y is None or (isinstance(y, np.ndarray) and y.size == 0):
            raise RuntimeError("Empty audio from soundfile")
    except Exception:
        # Fallback through audioread backend
        y, sr = librosa.load(path, sr=None, mono=True)

    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Clean NaN/Inf
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Guarantee minimum length
    if y.size == 0:
        y = np.zeros(int(target_sr * 0.5), dtype=np.float32)
        sr = target_sr

    # Resample
    if sr != target_sr and sr is not None:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Soft peak normalize
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / (peak + 1e-6)

    return y.astype(np.float32), sr


def load_logmel(
    path: str,
    sr: int = 16000,
    n_mels: int = 64,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    power: float = 2.0,
) -> np.ndarray:
    """
    Audio -> log-mel (per-clip standardized).
    Returns: (n_mels, T) float32
    """
    y, _ = safe_load_audio(path, target_sr=sr)

    n_fft = max(256, int(sr * win_ms / 1000.0))
    hop_length = max(1, int(sr * hop_ms / 1000.0))

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    logmel = librosa.power_to_db(S, ref=np.max).astype(np.float32)

    # Per-clip mean/var norm
    mu = float(np.mean(logmel))
    sigma = float(np.std(logmel)) + 1e-6
    logmel = (logmel - mu) / sigma
    return logmel.astype(np.float32)  # (M, T)


# -----------------------------
# Dataset
# -----------------------------
class UrbanSoundSubset(Dataset):
    """
    Works with either:
      1) Original layout: <audio_root>/foldX/<file>.wav    (with metadata CSV)
      2) Subset layout :  <audio_root>/<class>/<file>.wav  (pre-filtered)

    It will check both path styles automatically.
    """
    def __init__(
        self,
        meta_csv: str,
        audio_root: str,
        class_list: List[str],
        folds: List[int],
        sr: int,
        n_mels: int,
        win_ms: float,
        hop_ms: float,
        augment: Optional[torch.nn.Module] = None,
    ):
        df = pd.read_csv(meta_csv)
        df["fold"] = df["fold"].astype(int)
        df = df[df["class"].isin(class_list)]
        df = df[df["fold"].isin(folds)]
        self.df = df.reset_index(drop=True)

        self.audio_root = audio_root
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        self.sr, self.n_mels, self.win_ms, self.hop_ms = sr, n_mels, win_ms, hop_ms
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, row: pd.Series) -> str:
        fold = f"fold{int(row['fold'])}"
        # 1) Original layout
        cand1 = os.path.join(self.audio_root, fold, row["slice_file_name"])
        # 2) Subset layout
        cand2 = os.path.join(self.audio_root, row["class"], row["slice_file_name"])
        if os.path.exists(cand1):
            return cand1
        if os.path.exists(cand2):
            return cand2
        raise FileNotFoundError(f"Audio not found: {cand1} or {cand2}")

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        wav = self._resolve_path(row)

        x = load_logmel(
            wav,
            sr=self.sr,
            n_mels=self.n_mels,
            win_ms=self.win_ms,
            hop_ms=self.hop_ms,
        )  # (M, T)
        x = torch.from_numpy(x).unsqueeze(0)  # (1, M, T)

        y = torch.tensor(self.class_to_idx[row["class"]], dtype=torch.long)

        if self.augment is not None:
            x = self.augment(x)
        return x, y


# -----------------------------
# Optional: simple collate_fn
# -----------------------------
def pad_collate(batch, pad_value: float = 0.0):
    """
    Pads variable-length time axes to the max T within the batch.
    Input: list of (x=(1,M,T), y)
    Returns:
      xs: (B,1,M,T_max), ys: (B,)
    """
    xs, ys = zip(*batch)
    M = xs[0].shape[1]
    T_max = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        T = x.shape[-1]
        if T < T_max:
            pad = (0, T_max - T)
            x = torch.nn.functional.pad(x, pad, value=pad_value)
        padded.append(x)
    xs = torch.stack(padded, dim=0)
    ys = torch.stack(list(ys), dim=0)
    return xs, ys
