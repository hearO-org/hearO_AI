import os, numpy as np, pandas as pd, soundfile as sf, librosa, torch
from torch.utils.data import Dataset

def load_logmel(path, sr=16000, n_mels=64, win_ms=25, hop_ms=10):
    y, orig_sr = sf.read(path, always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    n_fft = int(sr*win_ms/1000)
    hop_length = int(sr*hop_ms/1000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels, power=2.0)
    logmel = librosa.power_to_db(S).astype(np.float32)
    # per-clip mean/var norm
    mu, sigma = logmel.mean(), logmel.std() + 1e-6
    logmel = (logmel - mu) / sigma
    return logmel  # (n_mels, T)

class UrbanSoundSubset(Dataset):
    def __init__(self, meta_csv, audio_root, class_list, folds, sr, n_mels, win_ms, hop_ms, augment=None):
        df = pd.read_csv(meta_csv)
        df["fold"] = df["fold"].astype(int)
        df = df[df["class"].isin(class_list)]
        df = df[df["fold"].isin(folds)]
        self.df = df.reset_index(drop=True)
        self.audio_root = audio_root
        self.class_to_idx = {c:i for i,c in enumerate(class_list)}
        self.sr, self.n_mels, self.win_ms, self.hop_ms = sr, n_mels, win_ms, hop_ms
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        fold = f"fold{int(row['fold'])}"
        wav = os.path.join(self.audio_root, fold, row["slice_file_name"])
        x = load_logmel(wav, self.sr, self.n_mels, self.win_ms, self.hop_ms)  # (M, T)
        x = torch.from_numpy(x).unsqueeze(0)  # (1, M, T)
        y = torch.tensor(self.class_to_idx[row["class"]], dtype=torch.long)
        if self.augment is not None:
            x = self.augment(x)
        return x, y
