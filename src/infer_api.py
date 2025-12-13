# src/infer_api.py
import os
import tempfile
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
def get_cfg(path: str = "./configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = get_cfg()
device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cpu") == "cuda" else "cpu")


# ---------- Helpers ----------
def _find_latest_pt(out_dir: str) -> Optional[str]:
    outs = os.path.abspath(out_dir)
    cand = []
    if os.path.isdir(outs):
        for root, _, files in os.walk(outs):
            for f in files:
                if f.lower().endswith(".pt"):
                    cand.append(os.path.join(root, f))
    return max(cand, key=os.path.getmtime) if cand else None


def _load_ckpt_any(ckpt_path: str, map_location) -> Tuple[dict, Optional[List[str]], Optional[dict], Optional[dict]]:
    """
    return: (state_dict, classes, audio_cfg, model_cfg)
    - old format: torch.save(state_dict)
    - new format: torch.save({"state_dict":..., "classes":..., "audio_cfg":..., "model_cfg":...})
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("classes"), ckpt.get("audio_cfg"), ckpt.get("model_cfg")
    if isinstance(ckpt, dict):
        return ckpt, None, None, None
    raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _is_wav_bytes(b: bytes) -> bool:
    # 최소 헤더 검사: "RIFF" .... "WAVE"
    return len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WAVE"


def _estimate_duration_sec_from_logmel(wav_path: str) -> float:
    # mel frame 길이로 duration 근사: T * hop_ms / 1000
    x = load_logmel(
        wav_path,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        win_ms=cfg["win_ms"],
        hop_ms=cfg["hop_ms"],
    )
    T = x.shape[-1] if isinstance(x, np.ndarray) else int(x.shape[-1])
    return float(T) * float(cfg["hop_ms"]) / 1000.0


# ---------- Load classes (default from cfg) ----------
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
ckpt_path = "./outputs/models/best_fold1.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = _find_latest_pt(cfg["out_dir"])

if not ckpt_path or not os.path.exists(ckpt_path):
    raise RuntimeError(f"Checkpoint(.pt) not found in {cfg['out_dir']}")

state_dict, ckpt_classes, ckpt_audio_cfg, ckpt_model_cfg = _load_ckpt_any(ckpt_path, map_location=device)

# ckpt에 classes가 있으면 우선 사용 (라벨 매핑 꼬임 방지)
if ckpt_classes is not None and isinstance(ckpt_classes, list) and len(ckpt_classes) > 0:
    classes = list(ckpt_classes)
    num_cls = len(classes)

# ckpt에 model_cfg가 있으면 우선 사용 (구조 mismatch 방지)
if ckpt_model_cfg is not None:
    mcfg = ckpt_model_cfg
    _model = CNN_Small(
        in_ch=mcfg["in_channels"],
        num_classes=num_cls,
        num_filters=tuple(mcfg["num_filters"]),
        dropout=mcfg["dropout"],
    ).to(device).eval()

_model.load_state_dict(state_dict)
print(f"[API] Loaded checkpoint: {ckpt_path} on {device} | classes={classes}")


# ---------- FastAPI ----------
app = FastAPI(title="hearO Sound Alert API", version="1.2.0")


class InferResponse(BaseModel):
    filename: str
    predicted: str
    conf: float
    probs: Dict[str, float]
    duration_sec: float


@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "device": str(device),
        "classes": classes,
        "ckpt": os.path.basename(ckpt_path),
    }


@app.get("/classes")
def get_classes():
    return {"classes": classes}


# ---------- Real-time smoothing ----------
_SMOOTH_N = 5
_session_probs: Dict[str, deque] = {}


def _smooth_probs(session_id: str, probs_dict: Dict[str, float]) -> Dict[str, float]:
    dq = _session_probs.get(session_id)
    if dq is None:
        dq = deque(maxlen=_SMOOTH_N)
        _session_probs[session_id] = dq

    dq.append(np.array([probs_dict[c] for c in classes], dtype=np.float32))
    avg = np.mean(np.stack(list(dq), axis=0), axis=0)
    return {c: float(avg[i]) for i, c in enumerate(classes)}


@torch.no_grad()
def _infer_wavfile(wav_path: str) -> Tuple[str, float, Dict[str, float], float]:
    duration_sec = _estimate_duration_sec_from_logmel(wav_path)

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

    logits = _model(x.to(device))
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    conf = float(probs[pred_idx])
    probs_dict = {c: float(probs[i]) for i, c in enumerate(classes)}
    return pred_label, conf, probs_dict, duration_sec


@app.post("/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(...),
    session_id: str = Query("default", description="실시간 smoothing용 세션 키"),
    smooth: bool = Query(True, description="최근 확률 평균(smoothing) 적용"),
    min_sec: float = Query(0.8, description="이보다 짧으면 unknown"),
    threshold: float = Query(0.55, description="confidence threshold"),
):
    raw = await file.read()
    if not raw or len(raw) < 32:
        raise HTTPException(400, "Empty/too small audio payload")

    # ✅ WAV 전용: 헤더 검사 (프론트가 wav라고 했으니 여기서 확실히 잡아줌)
    if not _is_wav_bytes(raw):
        raise HTTPException(
            400,
            "Input is not a valid WAV (missing RIFF/WAVE header). "
            "Please upload PCM 16-bit mono WAV."
        )

    # 그대로 wav로 저장 (ffmpeg 없음)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(raw)
        tmp_wav_path = tmp.name

    try:
        pred, conf, probs, duration_sec = _infer_wavfile(tmp_wav_path)

        # 너무 짧으면 unknown
        if float(duration_sec) < float(min_sec):
            return InferResponse(
                filename=file.filename,
                predicted="unknown",
                conf=float(conf),
                probs=probs,
                duration_sec=float(duration_sec),
            )

        # smoothing
        if smooth:
            probs_s = _smooth_probs(session_id, probs)
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
        try:
            os.remove(tmp_wav_path)
        except Exception:
            pass
