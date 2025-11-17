import torch
import torchaudio

class SpecAugment(torch.nn.Module):
    def __init__(self, time_mask_param=24, freq_mask_param=8):
        super().__init__()
        self.tmask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    def forward(self, x):  # x: (B, 1, n_mels, T)
        x = self.tmask(x)
        x = self.fmask(x)
        return x
