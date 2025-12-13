# src/infer_api.py
import os
import tempfile
import subprocess
from collections import deque
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
import yaml

from src.models.cnn_small import CNN_Small
from src.datasets.us8k import load_logmel

# ---------- Config ----------
def get_cfg(path="./configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = get_cfg()
device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cpu") == "cuda" else "cpu")

# ---------- Classes ----------
classes: List[str] = list(cfg["class_list"])
num_cls = len(classes)

# ---------- Model (load-once) ----------
mcfg = cfg["model"]
_model = CNN_Small(
    in_ch=mcfg["in_channels"],
    num_classes=num_cls,
    num_filters=tuple(mcfg["num_filters"]),
    dropout=mcfg["dropout"],
).to(device).eval()

# ---------- checkpoint resolve ----------
def _find_latest_pt(out_dir: str) -> Optional[str]:
    outs = os.path.abspath(out_dir)
    cand = []
    if os.path.isdir(outs):
        for root, _, files in os.walk(outs):
            for f in files:
                if f.lower().endswith(".pt"):
                    cand.append(os.path.join(root, f))
    return max(cand, key=os.path.getmtime) if cand else None

ckpt_path = "./outputs/models/best_fold1.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = _find_latest_pt(cfg["out_dir"])

if not ckpt_path or not os.path.exists(ckpt_path):
    raise RuntimeError(f"Checkpoint(.pt) not found in {cfg['out_dir']}")

ckpt = torch.load(ckpt_path, map_location=device)
# 호환: state_dict만 저장된 경우 vs dict 저장된 경우
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    _model.load_state_dict(ckpt["state_dict"])
    # ckpt에 classes가 있으면 그걸 우선 사용(매핑 꼬임 방지)
    if "classes" in ckpt and isinstance(ckpt["classes"], list):
        classes = list(ckpt["classes"])
        num_cls = len(classes)
else:
    _model.load_state_dict(ckpt)

print(f"[API] Loaded checkpoint: {ckpt_path} on {device} | classes={classes}")

# ---------- FastAPI ----------
app = FastAPI(title="hearO Sound Alert API", version="1.1.0")

class InferResponse(BaseModel):
    filename: str
    predicted: str
    conf: float
    probs: Dict[str, float]
    duration_sec: float

# ---------- Real-time smoothing buffer ----------
# session_id 별로 최근 N개 확률을 평균
_SMOOTH_N = 5
_session_probs: Dict[str, deque] = {}

def _ffmpeg_to_wav16k_mono(in_path: str, out_path: str):
    """
    어떤 오디오든 ffmpeg로 16kHz mono wav로 변환
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", str(int(cfg["sample_rate"])),
        "-f", "wav",
        out_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise HTTPException(400, "Audio decode/convert failed (ffmpeg). Check input format or install ffmpeg.")

def _estimate_duration_sec(wav_path: str) -> float:
    # load_logmel을 이미 쓰니까, 여기서는 간단히 mel frame 길이로 duration 근사
    # hop_ms 기준: T frames -> T * hop_ms / 1000
    x = load_logmel(
        wav_path,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        win_ms=cfg["win_ms"],
        hop_ms=cfg["hop_ms"],
    )
    if isinstance(x, np.ndarray):
        T = x.shape[-1]
    else:
        T = int(x.shape[-1])
    return float(T) * float(cfg["hop_ms"]) / 1000.0

def _infer_wavfile(wav_path: str) -> Tuple[str, float, Dict[str, float], float]:
    # duration check (너무 짧으면 한 클래스로 쏠림)
    duration_sec = _estimate_duration_sec(wav_path)

    x = load_logmel(
        wav_path,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        win_ms=cfg["win_ms"],
        hop_ms=cfg["hop_ms"],
    )
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    # (M,T) -> (1,1,M,T)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)

    with torch.no_grad():
        logits = _model(x.to(device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    conf = float(probs[pred_idx])
    probs_dict = {c: float(probs[i]) for i, c in enumerate(classes)}
    return pred_label, conf, probs_dict, duration_sec

def _smooth_probs(session_id: str, probs_dict: Dict[str, float]) -> Dict[str, float]:
    dq = _session_probs.get(session_id)
    if dq is None:
        dq = deque(maxlen=_SMOOTH_N)
        _session_probs[session_id] = dq

    dq.append(np.array([probs_dict[c] for c in classes], dtype=np.float32))
    avg = np.mean(np.stack(list(dq), axis=0), axis=0)

    return {c: float(avg[i]) for i, c in enumerate(classes)}

@app.get("/ping")
def ping():
    return {"status": "ok", "device": str(device), "classes": classes, "ckpt": os.path.basename(ckpt_path)}

@app.get("/classes")
def get_classes():
    return {"classes": classes}

@app.post("/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(...),
    session_id: str = Query("default", description="실시간 smoothing을 위한 세션 키"),
    smooth: bool = Query(True, description="최근 확률 평균(smoothing) 적용"),
    min_sec: float = Query(0.8, description="이보다 짧으면 unknown 처리"),
    threshold: float = Query(0.55, description="confidence threshold (미만이면 unknown)"),
):
    # 확장자 강제하지 않음 (webm/ogg 등 실시간 chunk 대응)
    raw = await file.read()
    if not raw or len(raw) < 32:
        raise HTTPException(400, "Empty/too small audio payload")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_in:
        tmp_in.write(raw)
        tmp_in_path = tmp_in.name

    tmp_wav_path = None
    try:
        # 변환된 wav 경로
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()

        # 어떤 포맷이든 wav로 변환
        _ffmpeg_to_wav16k_mono(tmp_in_path, tmp_wav_path)

        pred, conf, probs, duration_sec = _infer_wavfile(tmp_wav_path)

        # 너무 짧으면 unknown (실시간 chunk 흔한 문제)
        if duration_sec < float(min_sec):
            return InferResponse(
                filename=file.filename,
                predicted="unknown",
                conf=float(conf),
                probs=probs,
                duration_sec=float(duration_sec),
            )

        # smoothing 적용
        if smooth:
            probs_s = _smooth_probs(session_id, probs)
            # smoothing 결과로 최종 결정
            best_label = max(probs_s.keys(), key=lambda k: probs_s[k])
            best_conf = float(probs_s[best_label])
            pred, conf, probs = best_label, best_conf, probs_s

        # threshold
        if float(conf) < float(threshold):
            pred = "unknown"

        return InferResponse(
            filename=file.filename,
            predicted=pred,
            conf=float(conf),
            probs=probs,
            duration_sec=float(duration_sec),
        )

    finally:
        for p in [tmp_in_path, tmp_wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
