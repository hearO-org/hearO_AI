# download_us8k.py
import argparse, sys
import soundata

def main(data_home=None):
    ds = soundata.initialize("urbansound8k", data_home=data_home)
    print(f"[INFO] data_home = {ds.data_home}")
    try:
        ds.download()  # 원본 다운로드 (약 6GB)
        print("[OK] Download complete")
    except Exception as e:
        print("[ERR] Download failed:", e)
        sys.exit(1)
    # 간단 검증(파일 존재/인덱스 확인)
    try:
        ds.validate()
        print("[OK] Validate complete")
    except Exception as e:
        print("[WARN] Validate warnings:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_home", default=None, help="원하는 저장 경로 (예: C:\\Users\\me\\sound_datasets\\UrbanSound8K)")
    args = ap.parse_args()
    main(args.data_home)
