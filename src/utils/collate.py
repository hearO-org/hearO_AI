# src/utils/collate.py
import torch
import torch.nn.functional as F

def pad_collate(batch):
    xs, ys = zip(*batch)  # xs: (B, 1, M, T_i)
    max_T = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        pad_T = max_T - x.shape[-1]
        if pad_T > 0:
            # pad last dimension (T): (pad_left, pad_right)
            x = F.pad(x, (0, pad_T))
        padded.append(x)
    X = torch.stack(padded, dim=0)          # (B, 1, M, max_T)
    y = torch.stack(ys, dim=0)              # (B,)
    return X, y
