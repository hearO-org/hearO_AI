import argparse
import os
import yaml
import numpy as np
import torch

from src.models.cnn_small import CNN_Small
from src.datasets.us8k import load_logmel  # 학습과 동일 로직 사용


# -----------------------------
# Config loader
# -----------------------------
def get_cfg(path: str = "./configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------
# Checkpoint loader (호환)
# - old ckpt: state_dict만 저장된 경우(torch.save(state_dict))
# - new ckpt: {"state_dict": ..., "classes": [...], "audio_cfg": {...}, ...}
# -----------------------------
def load_checkpoint(ckpt_path: str, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        classes = ckpt.get("classes", None)
        audio_cfg = ckpt.get("audio_cfg", None)
        model_cfg = ckpt.get("model_cfg", None)
        return state_dict, classes, audio_cfg, model_cfg

    # 예전 포맷: state_dict만 저장
    if isinstance(ckpt, dict):
        return ckpt, None, None, None

    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


# -----------------------------
# Preprocess wav -> logmel tensor
# (1, 1, M, T) 형태로 맞춤
# -----------------------------
def wav_to_input_tensor(wav_path: str, sr: int, n_mels: int, win_ms: int, hop_ms: int):
    x = load_logmel(
        wav_path,
        sr=sr,
        n_mels=n_mels,
        win_ms=win_ms,
        hop_ms=hop_ms,
    )

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    # 예상 형태:
    # - (M, T)  -> (1,1,M,T)
    # - (1,M,T) -> (1,1,M,T)
    # - (B,1,M,T) already ok (but for single file we expect not)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        # (C, M, T) 형태면 C=1 가정
        x = x.unsqueeze(0)
        if x.shape[1] != 1:
            # (1, C, M, T)에서 C가 1이 아니면 강제로 1채널로 축소(평균)
            x = x.mean(dim=1, keepdim=True)
    elif x.ndim == 4:
        # already (B, C, M, T) — 그대로 둠
        pass
    else:
        raise ValueError(f"Unexpected logmel ndim={x.ndim}, shape={tuple(x.shape)}")

    # channel=1 보장
    if x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)

    return x


# -----------------------------
# Infer function
# -----------------------------
@torch.no_grad()
def infer(
    wav_path: str,
    cfg_path: str = "./configs/config.yaml",
    ckpt_path: str | None = None,
    device_str: str | None = None,
    topk: int = 5,
    debug: bool = False,
):
    cfg = get_cfg(cfg_path)

    # device
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # ckpt path default
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg["out_dir"], "best_fold1.pt")

    # load ckpt (classes/audio_cfg가 있으면 그걸 우선)
    state_dict, ckpt_classes, ckpt_audio_cfg, ckpt_model_cfg = load_checkpoint(ckpt_path, map_location=device)

    # classes: ckpt 우선, 없으면 cfg
    classes = ckpt_classes if ckpt_classes is not None else cfg["class_list"]
    num_cls = len(classes)

    # model cfg: ckpt 우선, 없으면 cfg
    mcfg = ckpt_model_cfg if ckpt_model_cfg is not None else cfg["model"]

    model = CNN_Small(
        in_ch=mcfg["in_channels"],
        num_classes=num_cls,
        num_filters=tuple(mcfg["num_filters"]),
        dropout=mcfg["dropout"],
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # audio cfg: ckpt 우선, 없으면 cfg
    a = ckpt_audio_cfg if ckpt_audio_cfg is not None else {
        "sample_rate": cfg["sample_rate"],
        "n_mels": cfg["n_mels"],
        "win_ms": cfg["win_ms"],
        "hop_ms": cfg["hop_ms"],
    }

    x = wav_to_input_tensor(
        wav_path,
        sr=int(a["sample_rate"]),
        n_mels=int(a["n_mels"]),
        win_ms=int(a["win_ms"]),
        hop_ms=int(a["hop_ms"]),
    ).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    pred_conf = float(probs[pred_idx])

    # output
    print(f"\n🎧 File: {os.path.basename(wav_path)}")
    print(f"🧠 Model: {os.path.basename(ckpt_path)} | Device: {device}")
    if debug:
        print(f"🔎 Input shape: {tuple(x.shape)}")
        print(f"🔎 Classes(from {'ckpt' if ckpt_classes is not None else 'cfg'}): {classes}")

    # topk
    k = min(topk, len(classes))
    top_idx = probs.argsort()[-k:][::-1]

    print("\n📊 Top predictions")
    for i in top_idx:
        print(f"  {classes[i]:15s}: {probs[i]*100:.2f}%")

    print(f"\n✅ Predicted: {pred_label.upper()} ({pred_conf*100:.2f}%)\n")

    return pred_label, pred_conf, probs


# -----------------------------
# CLI Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", help="추론할 wav 파일 경로")
    parser.add_argument("--cfg", default="./configs/config.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:0 등")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--debug", action="store_true")

    # 서버로 실행할지 여부
    parser.add_argument("--serve", action="store_true", help="FastAPI 서버 실행")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        # infer_api.py 안에 FastAPI app이 있어야 함 (권장)
        uvicorn.run("src.infer_api:app", host=args.host, port=args.port)
        return

    if not args.wav:
        raise SystemExit("❌ --wav 경로를 입력해줘. (서버면 --serve 사용)")

    infer(
        wav_path=args.wav,
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        device_str=args.device,
        topk=args.topk,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
