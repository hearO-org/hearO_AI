import os, re, time, json, shutil, argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

"""
UrbanSound8K에서 상업적 사용 가능한(Commercial OK) 클립만 걸러내어
- licenses.csv (fsID별 라이선스 요약)
- filtered_metadata.csv (허용된 라이선스만)
을 저장하고,
오디오 파일을 새 디렉터리로 복사(fold 구조 유지)합니다.

허용 라이선스: CC0, CC BY (기본값)
제외 라이선스: CC BY-NC 계열, CC BY-ND 계열, 기타 비허용
"""

FREESOUND_HTML_URL = "https://freesound.org/s/{fsid}/"

# 라이선스 URL/문구를 타입으로 정규화
def normalize_license(license_text_or_url: str) -> str:
    s = (license_text_or_url or "").lower().strip()
    # 흔한 형태들 처리
    if "cc0" in s or "creative commons 0" in s or "public domain" in s:
        return "CC0"
    if "creativecommons.org/licenses/by/" in s or "cc by" in s:
        return "CC BY"
    if "by-nc" in s:
        return "CC BY-NC"
    if "by-nd" in s:
        return "CC BY-ND"
    if "sampling+" in s:
        return "Sampling+"
    # 기타
    if "creative commons" in s:
        return "CC (unspecified)"
    return "Unknown"

def get_license_via_api(fsid: int, token: str) -> str | None:
    """
    Freesound API로 정확한 라이선스 URL을 받아옴.
    https://freesound.org/apiv2/sounds/{id}/?token=YOUR_TOKEN
    """
    url = f"https://freesound.org/apiv2/sounds/{fsid}/?token={token}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            lic = data.get("license", "")
            return normalize_license(lic or "")
    except Exception:
        pass
    return None

def get_license_via_html(fsid: int) -> str | None:
    """
    Freesound 공개 웹페이지를 파싱해서 라이선스 문구를 추출.
    과도한 요청을 피하기 위해 rate-limit 필요.
    """
    url = FREESOUND_HTML_URL.format(fsid=fsid)
    try:
        r = requests.get(url, timeout=12, allow_redirects=True)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        # 라이선스 텍스트가 있는 섹션 탐색 (페이지 구조가 바뀌면 여기 수정)
        # 보통 "License" 라벨 또는 Creative Commons 링크가 있음
        text = soup.get_text(" ", strip=True).lower()
        # 자주 보이는 패턴부터 시도
        m = re.search(r"creative commons[^\.]*", text)
        if m:
            return normalize_license(m.group(0))

        # a[href]에 라이선스 링크가 걸리는 경우
        for a in soup.select("a[href]"):
            href = a["href"].lower()
            if "creativecommons.org" in href or "cc0" in href:
                return normalize_license(href)
        return None
    except Exception:
        return None

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def copy_audio(row, audio_root, out_root):
    fold = f"fold{int(row['fold'])}"
    fname = row["slice_file_name"]
    src = os.path.join(audio_root, fold, fname)
    dst_dir = os.path.join(out_root, "audio", fold)
    safe_makedirs(dst_dir)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)

def main(args):
    # 경로들
    meta_csv = os.path.join(args.urbansound_root, "metadata", "UrbanSound8K.csv")
    audio_root = os.path.join(args.urbansound_root, "audio")
    out_root  = args.out_dir
    cache_json = os.path.join(out_root, "license_cache.json")
    safe_makedirs(out_root)

    # 메타데이터 읽기
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Metadata not found: {meta_csv}")
    df = pd.read_csv(meta_csv)

    # 필요한 컬럼 존재 확인
    required = {"slice_file_name", "fsID", "class", "classID", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in UrbanSound8K.csv: {missing}")

    # (선택) 특정 클래스만 대상으로 줄이기 (요청시)
    if args.include_classes:
        wanted = set([c.strip() for c in args.include_classes.split(",") if c.strip()])
        df = df[df["class"].isin(wanted)].copy()

    # 중복 fsID 줄이기 (같은 음원이 여러 슬라이스로 존재 가능)
    fsids = sorted(df["fsID"].dropna().astype(int).unique())

    # 캐시 로드
    cache = {}
    if os.path.exists(cache_json):
        try:
            cache = json.load(open(cache_json, "r", encoding="utf-8"))
        except Exception:
            cache = {}

    # 라이선스 조회
    results = {}
    use_api = bool(args.freesound_token)

    print(f"[INFO] Unique fsIDs to check: {len(fsids)} | API token: {use_api}")
    for fsid in tqdm(fsids, desc="Fetching licenses"):
        sfsid = str(fsid)
        if sfsid in cache:
            results[sfsid] = cache[sfsid]
            continue

        lic = None
        if use_api:
            lic = get_license_via_api(fsid, args.freesound_token)
        if lic is None:
            lic = get_license_via_html(fsid)
            # 예의상 rate-limit (과한 트래픽 방지)
            time.sleep(args.delay)

        results[sfsid] = lic or "Unknown"
        cache[sfsid] = results[sfsid]

        # 주기적으로 캐시 저장
        if len(cache) % 100 == 0:
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    # 최종 캐시 저장
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # fsID → license 매핑 join
    lic_df = pd.DataFrame(
        {"fsID": [int(k) for k in results.keys()],
         "license": list(results.values())}
    )
    lic_df.to_csv(os.path.join(out_root, "licenses.csv"), index=False, encoding="utf-8")

    # 허용 라이선스 규칙
    allowed = set([s.strip().upper() for s in args.allowed.split(",") if s.strip()])
    # 기본은 {"CC0", "CC BY"}
    def is_allowed(lic: str) -> bool:
        if lic is None:
            return False
        L = lic.upper()
        if L in allowed:
            return True
        return False

    # 메타데이터에 license 추가 & 필터
    df["license"] = df["fsID"].astype(int).astype(str).map(results).fillna("Unknown")
    filtered = df[df["license"].apply(is_allowed)].copy()

    # 저장
    filtered_csv = os.path.join(out_root, "filtered_metadata.csv")
    filtered.to_csv(filtered_csv, index=False, encoding="utf-8")

    # 요약
    print("\n=== SUMMARY ===")
    print("Total clips:", len(df))
    print("Allowed clips:", len(filtered))
    print("License breakdown (in dataset slice rows):")
    print(df["license"].value_counts(dropna=False))

    # 파일 복사 (fold 구조 유지)
    if args.copy_audio:
        print("\n[INFO] Copying audio files ...")
        for _, row in tqdm(filtered.iterrows(), total=len(filtered)):
            copy_audio(row, audio_root, out_root)
        print("[INFO] Done copying.")

    # 간단한 리포트 저장
    summary_txt = os.path.join(out_root, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== UrbanSound8K Commercial Subset Summary ===\n")
        f.write(f"Allowed licenses: {', '.join(sorted(allowed))}\n")
        f.write(f"Include classes: {args.include_classes or 'ALL'}\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Allowed rows: {len(filtered)}\n\n")
        f.write("License counts (all rows):\n")
        f.write(df['license'].value_counts(dropna=False).to_string())
        f.write("\n")

    print(f"\n[OK] Wrote:\n- {os.path.join(out_root, 'licenses.csv')}\n- {filtered_csv}\n- {summary_txt}")
    if args.copy_audio:
        print(f"- Copied audio under: {os.path.join(out_root, 'audio')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urbansound_root", type=str, required=True,
                        help="UrbanSound8K 루트 경로 (예: ~/sound_datasets/UrbanSound8K/UrbanSound8K)")
    parser.add_argument("--out_dir", type=str, default="./US8K_commercial_subset",
                        help="결과물(licenses.csv, filtered_metadata.csv, 복사본 audio/) 저장 폴더")
    parser.add_argument("--allowed", type=str, default="CC0,CC BY",
                        help="허용 라이선스 콤마구분 (기본: CC0,CC BY)")
    parser.add_argument("--include_classes", type=str, default="",
                        help="필요시 클래스 필터(예: 'car_horn,siren') 없으면 전체")
    parser.add_argument("--copy_audio", action="store_true",
                        help="필터된 오디오 파일을 out_dir/audio/ 로 복사")
    parser.add_argument("--freesound_token", type=str, default="",
                        help="(선택) Freesound API 토큰. 주면 빠르고 정확.")
    parser.add_argument("--delay", type=float, default=0.6,
                        help="HTML 스크랩 모드에서 요청 간 대기(초). 과한 트래픽 방지용.")
    args = parser.parse_args()
    main(args)
