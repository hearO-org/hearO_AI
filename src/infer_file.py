import argparse, os, torch, torch.nn as nn, yaml, numpy as np
from src.models.cnn_small import CNN_Small
from src.datasets.us8k import load_logmel  # 학습과 동일한 로직 사용

# === config 읽기 ===
def get_cfg(path="./configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def infer(wav_path, cfg_path="./configs/config.yaml", ckpt_path=None):
    # Config 및 환경 설정
    cfg = get_cfg(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = cfg["class_list"]
    num_cls = len(classes)

    # === 모델 초기화 ===
    mcfg = cfg["model"]
    model = CNN_Small(
        in_ch=mcfg["in_channels"],
        num_classes=num_cls,
        num_filters=tuple(mcfg["num_filters"]),
        dropout=mcfg["dropout"]
    ).to(device)

    # === checkpoint 불러오기 ===
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg["out_dir"], "best_fold1.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # === 오디오 → logmel (학습과 동일한 방식) ===
    x = load_logmel(
        wav_path,
        sr=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        win_ms=cfg["win_ms"],
        hop_ms=cfg["hop_ms"]
    )

    # numpy → tensor 변환 (자동 감지)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    # (1, M, T) → (1, 1, M, T)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    # === 짧은 오디오 padding (학습과 frame 길이 맞추기용) ===
    min_frames = 100  # 대략 1초 정도
    if x.shape[-1] < min_frames:
        pad_len = min_frames - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, pad_len))

    x = x.to(device)

    # === 예측 ===
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]

    # === 결과 출력 ===
    print(f"\n🎧 File: {os.path.basename(wav_path)}")
    for i, c in enumerate(classes):
        print(f"  {c:15s}: {probs[i]*100:.2f}%")
    print(f"\n✅ Predicted: {pred_label.upper()} (model: {os.path.basename(ckpt_path)})\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="추론할 wav 파일 경로")
    parser.add_argument("--cfg", default="./configs/config.yaml")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    infer(args.wav, args.cfg, args.ckpt)
