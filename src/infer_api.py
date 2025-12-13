# src/infer_api.py
import os, tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import yaml

from src.models.cnn_small import CNN_Small
from src.datasets.us8k import load_logmel  # 우리가 강화한 함수

# ---------- Config ----------
def get_cfg(path: str = None):
    # /app/src/infer_api.py 기준으로 /app 를 루트로 잡음
    ROOT = Path(__file__).resolve().parents[1]  # /app/src -> /app
    cfg_path = Path(path) if path else (ROOT / "configs" / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = get_cfg()

classes: List[str] = list(cfg["class_list"])
num_cls = len(classes)

device = torch.device(
    "cuda" if torch.cuda.is_available() and cfg.get("device", "cpu") == "cuda" else "cpu"
)

# ---------- Inference Settings ----------
# config.yaml에 아래가 있으면 적용됨:
# inference:
#   threshold: 0.88
#   margin: 0.10
#   unknown_label: null   # 또는 "unknown" / "" 등
icfg = cfg.get("inference", {}) if isinstance(cfg, dict) else {}
THRESHOLD = float(icfg.get("threshold", 0.0))       # 0이면 기존처럼 항상 반환
MARGIN = float(icfg.get("margin", 0.0))             # 0이면 margin 조건 미사용
UNKNOWN_LABEL = icfg.get("unknown_label", None)     # None이면 predicted=None

# ---------- Model (load-once) ----------
mcfg = cfg["model"]
_model = CNN_Small(
    in_ch=mcfg["in_channels"],
    num_classes=num_cls,
    num_filters=tuple(mcfg["num_filters"]),
    dropout=mcfg["dropout"],
).to(device).eval()

# ckpt 경로 결정 (없으면 에러)
ckpt_path = "../outputs/models/best_fold1.pt"
if not os.path.exists(ckpt_path):
    # 가장 최신 pt 한번 찾아보기
    outs = os.path.abspath(cfg["out_dir"])
    cand = []
    if os.path.isdir(outs):
        for root, _, files in os.walk(outs):
            for f in files:
                if f.lower().endswith(".pt"):
                    cand.append(os.path.join(root, f))
    ckpt_path = max(cand, key=os.path.getmtime) if cand else None

if not ckpt_path or not os.path.exists(ckpt_path):
    raise RuntimeError(f"Checkpoint(.pt) not found in {cfg['out_dir']}")

_state = torch.load(ckpt_path, map_location=device)
_model.load_state_dict(_state)
print(f"[API] Loaded checkpoint: {ckpt_path} on {device}")
print(f"[API] Inference threshold={THRESHOLD}, margin={MARGIN}, unknown_label={UNKNOWN_LABEL}")

# ---------- FastAPI ----------
app = FastAPI(title="hearO Sound Alert API", version="1.1.0")

class InferResponse(BaseModel):
    filename: str
    detected: bool                 # 확신해서 감지했는지
    predicted: Optional[str]       # detected=False면 None 또는 unknown_label
    confidence: float              # top1 확률
    probs: Dict[str, float]        # 전체 클래스 확률

@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "device": str(device),
        "classes": classes,
        "threshold": THRESHOLD,
        "margin": MARGIN,
        "unknown_label": UNKNOWN_LABEL,
    }

@app.get("/classes")
def get_classes():
    return {"classes": classes}

def _infer_wavfile(wav_path: str):
    # 1) feature
    x = load_logmel(
        wav_path,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        win_ms=cfg["win_ms"],
        hop_ms=cfg["hop_ms"],
    )

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    # x shape: (M,T) or (1,M,T) -> (1,1,M,T)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)

    # 2) infer
    with torch.no_grad():
        logits = _model(x.to(device))             # [1, C]
        probs_t = F.softmax(logits, dim=1)[0].detach().cpu()  # [C]

    # numpy probs dict
    probs_np = probs_t.numpy()
    probs_dict = {c: float(probs_np[i]) for i, c in enumerate(classes)}

    # 3) top1/top2 + confidence
    top2_vals, top2_idx = torch.topk(probs_t, k=min(2, probs_t.numel()))
    conf1 = float(top2_vals[0].item())
    pred_idx = int(top2_idx[0].item())
    pred_label = classes[pred_idx]

    conf2 = float(top2_vals[1].item()) if top2_vals.numel() > 1 else 0.0
    margin = conf1 - conf2

    # 4) gating (확신 조건)
    detected = True
    if THRESHOLD > 0.0 and conf1 < THRESHOLD:
        detected = False
    if MARGIN > 0.0 and margin < MARGIN:
        detected = False

    if not detected:
        # 결과 "안 보냄" 처리: predicted=None 또는 UNKNOWN_LABEL
        return None, conf1, probs_dict

    return pred_label, conf1, probs_dict

@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Please upload a .wav file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pred, conf, probs = _infer_wavfile(tmp_path)

        if pred is None:
            return InferResponse(
                filename=file.filename,
                detected=False,
                predicted=UNKNOWN_LABEL,   # None / "" / "unknown" 등 config대로
                confidence=float(conf),
                probs=probs,
            )

        return InferResponse(
            filename=file.filename,
            detected=True,
            predicted=pred,
            confidence=float(conf),
            probs=probs,
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
