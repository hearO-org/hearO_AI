# scripts/test_infer.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.infer_api import _infer_wavfile

# 여기만 바꿔도 됨: 폴더 또는 파일 경로 둘 다 OK
TEST_PATH = Path(r"C:\Users\catholic\PycharmProjects\hear\data\test_samples\car_horn.wav")
# TEST_PATH = ROOT / "data" / "test_samples"  # 폴더로 돌리고 싶으면 이걸로

def print_one(wav: Path):
    pred, conf, probs = _infer_wavfile(str(wav))

    print("=" * 70)
    print(f"File       : {wav.name}")
    print(f"Detected   : {pred is not None}")
    print(f"Prediction : {pred}")
    print(f"Confidence : {conf:.4f}")
    print("- probs (top) -")
    for k, v in sorted(probs.items(), key=lambda x: -x[1])[:10]:
        print(f"{k:15s}: {v:.4f}")
    print("=" * 70)

def main():
    if not TEST_PATH.exists():
        print(f"[ERROR] Path not found: {TEST_PATH}")
        return

    # ✅ 파일 1개면 그대로 테스트
    if TEST_PATH.is_file():
        if TEST_PATH.suffix.lower() != ".wav":
            print(f"[ERROR] Not a wav file: {TEST_PATH}")
            return
        print_one(TEST_PATH)
        return

    # ✅ 폴더면 wav 전부 테스트
    wav_files = sorted(TEST_PATH.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No wav files found in folder: {TEST_PATH}")
        return

    print(f"[INFO] Found {len(wav_files)} wav files in {TEST_PATH}\n")

    detected_cnt = 0
    for wav in wav_files:
        pred, conf, _ = _infer_wavfile(str(wav))
        status = "DETECTED" if pred is not None else "-----"
        detected_cnt += int(pred is not None)
        print(f"{status:8s} | {wav.name:30s} | pred={str(pred):10s} | conf={conf:.3f}")

    print("\n" + "=" * 60)
    print(f"Detected {detected_cnt}/{len(wav_files)} files")
    print("=" * 60)

if __name__ == "__main__":
    main()
