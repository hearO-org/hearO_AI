# src/infer_api.py
import os, tempfile
from typing import List
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.models.cnn_small import CNN_Small
from src.datasets.us8k import load_logmel  # 우리가 방금 강화한 함수
import yaml

# ---------- Config ----------
def get_cfg(path="./configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = get_cfg()
classes: List[str] = list(cfg["class_list"])
num_cls = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device","cpu")=="cuda" else "cpu")

# ---------- Model (load-once) ----------
mcfg = cfg["model"]
_model = CNN_Small(
    in_ch=mcfg["in_channels"],
    num_classes=num_cls,
    num_filters=tuple(mcfg["num_filters"]),
    dropout=mcfg["dropout"]
).to(device).eval()

# ckpt 경로 결정 (없으면 에러)
ckpt_path = os.path.join(cfg["out_dir"], "best_fold1.pt")
if not os.path.exists(ckpt_path):
    # 가장 최신 pt 한번 찾아보기
    outs = os.path.abspath(cfg["out_dir"])
    cand = []
    if os.path.isdir(outs):
        for root,_,files in os.walk(outs):
            for f in files:
                if f.lower().endswith(".pt"):
                    cand.append(os.path.join(root,f))
    ckpt_path = max(cand, key=os.path.getmtime) if cand else None

if not ckpt_path or not os.path.exists(ckpt_path):
    raise RuntimeError(f"Checkpoint(.pt) not found in {cfg['out_dir']}")

_state = torch.load(ckpt_path, map_location=device)
_model.load_state_dict(_state)
print(f"[API] Loaded checkpoint: {ckpt_path} on {device}")

# ---------- FastAPI ----------
app = FastAPI(title="hearO Sound Alert API", version="1.0.0")

class InferResponse(BaseModel):
    filename: str
    predicted: str
    probs: dict

@app.get("/ping")
def ping():
    return {"status": "ok", "device": str(device), "classes": classes}

@app.get("/classes")
def get_classes():
    return {"classes": classes}

def _infer_wavfile(wav_path: str):
    x = load_logmel(
        wav_path,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        win_ms=cfg["win_ms"],
        hop_ms=cfg["hop_ms"]
    )
    if isinstance(x, np.ndarray):
        import torch as _t
        x = _t.tensor(x, dtype=_t.float32)
    if x.ndim == 2:             # (M,T) -> (1,1,M,T)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:           # (1,M,T) -> (1,1,M,T)
        x = x.unsqueeze(0)

    with torch.no_grad():
        logits = _model(x.to(device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    return pred_label, {c: float(probs[i]) for i, c in enumerate(classes)}

@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Please upload a .wav file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        pred, probs = _infer_wavfile(tmp_path)
        return InferResponse(filename=file.filename, predicted=pred, probs=probs)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
