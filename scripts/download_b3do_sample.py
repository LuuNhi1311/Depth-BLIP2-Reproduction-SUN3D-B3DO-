"""
Download B3DO dataset và chỉ extract N cặp RGB+depth cần thiết.

Workflow:
1. Tải VOCB3DO.zip vào /tmp (streaming, hiển thị progress)
2. Đọc bảng mục lục zip (không extract toàn bộ)
3. Extract chỉ N cặp rgb+depth vào datasets/B3DO/
4. Xoá file zip khỏi /tmp
5. Sinh b3do_train/val/test.csv

Dùng: python scripts/download_b3do_sample.py --n_pairs 60
"""

import os, sys, zipfile, random, argparse, shutil, time
from pathlib import Path
from urllib.request import urlopen, Request
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

URL = "https://www.dropbox.com/s/yzrjtc87tfcr2c1/VOCB3DO.zip?dl=1"
ZIP_TMP   = Path("/tmp/VOCB3DO.zip")
OUT_ROOT  = ROOT / "datasets" / "B3DO"
CSV_DIR   = OUT_ROOT


def download_with_progress(url: str, dest: Path):
    """Download file với progress bar đơn giản."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 1 << 20  # 1 MB
        downloaded = 0
        start = time.time()
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                elapsed = time.time() - start
                speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded/1e6:.1f}/{total/1e6:.1f} MB  "
                          f"({pct:.1f}%)  {speed:.1f} MB/s", end="", flush=True)
                else:
                    print(f"\r  {downloaded/1e6:.1f} MB  {speed:.1f} MB/s", end="", flush=True)
    print()


def find_pairs_in_zip(zf: zipfile.ZipFile):
    """
    Tìm cặp (rgb_path, depth_path) trong zip VOCB3DO.
    Cấu trúc thực tế:
      VOCB3DO/KinectColor/img_XXXX.png
      VOCB3DO/RegisteredDepthData/img_XXXX_abs.png
    """
    names = zf.namelist()
    rgb_files   = {}  # stem (img_XXXX) → zip_path
    depth_files = {}  # stem (img_XXXX) → zip_path  (dùng _abs, bỏ _abs_smooth)

    for n in names:
        p = Path(n)
        parent = p.parent.name
        if parent == "KinectColor" and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            rgb_files[p.stem] = n
        elif parent == "RegisteredDepthData" and p.name.endswith("_abs.png"):
            # stem: img_XXXX_abs → key img_XXXX
            stem = p.stem[:-4]  # remove "_abs"
            depth_files[stem] = n

    pairs = []
    for stem, rgb_path in sorted(rgb_files.items()):
        if stem in depth_files:
            pairs.append((rgb_path, depth_files[stem]))

    return pairs


def extract_sample(zf: zipfile.ZipFile, pairs, out_root: Path, n: int, seed: int = 42):
    """Extract N cặp ngẫu nhiên, giữ đường dẫn tương đối trong zip."""
    random.seed(seed)
    selected = random.sample(pairs, min(n, len(pairs)))
    extracted = []
    for i, (rgb_zip, dep_zip) in enumerate(selected):
        for zpath in (rgb_zip, dep_zip):
            dest = out_root / zpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zpath) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        extracted.append((rgb_zip, dep_zip))
        print(f"\r  Extracted {i+1}/{len(selected)}", end="", flush=True)
    print()
    return extracted


def write_csvs(pairs, out_dir: Path, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ps in splits.items():
        if not ps:
            continue
        csv_path = out_dir / f"b3do_{split}.csv"
        pd.DataFrame(ps, columns=["rgb", "depth"]).to_csv(
            csv_path, header=False, index=False
        )
        print(f"  {split:5s}: {len(ps):3d} pairs → {csv_path}")
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=60,
                        help="Số cặp RGB+depth cần lấy (default 60: ~42 train, 9 val, 9 test)")
    parser.add_argument("--keep_zip", action="store_true",
                        help="Giữ lại file zip sau khi extract")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = args.n_pairs
    print(f"=== B3DO Sample Downloader (n_pairs={n}) ===")
    print(f"Output: {OUT_ROOT}")

    # ── 1. Download ───────────────────────────────────────────────────────────
    if ZIP_TMP.exists():
        print(f"[SKIP] Zip đã tồn tại: {ZIP_TMP} ({ZIP_TMP.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"\n[1/4] Tải VOCB3DO.zip → {ZIP_TMP} ...")
        download_with_progress(URL, ZIP_TMP)
        print(f"  Kích thước: {ZIP_TMP.stat().st_size/1e6:.1f} MB")

    # ── 2. Scan zip TOC ───────────────────────────────────────────────────────
    print(f"\n[2/4] Đọc bảng mục lục zip ...")
    with zipfile.ZipFile(ZIP_TMP) as zf:
        all_pairs = find_pairs_in_zip(zf)
        print(f"  Tìm thấy {len(all_pairs)} cặp RGB+depth trong zip")

        if not all_pairs:
            print("[ERROR] Không tìm thấy cặp nào! Kiểm tra cấu trúc zip.")
            sys.exit(1)

        # ── 3. Extract sample ─────────────────────────────────────────────────
        print(f"\n[3/4] Extract {min(n, len(all_pairs))} cặp → {OUT_ROOT} ...")
        extracted = extract_sample(zf, all_pairs, OUT_ROOT, n=n, seed=args.seed)

    # ── 4. Sinh CSV ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Sinh CSV train/val/test ...")
    splits = write_csvs(extracted, CSV_DIR, seed=args.seed)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.keep_zip:
        ZIP_TMP.unlink()
        print(f"\n[OK] Đã xoá {ZIP_TMP}")

    print(f"\n✓ Hoàn tất! {len(extracted)} cặp tại {OUT_ROOT}")
    print(f"  Dùng ngay: scripts/train_b3do.sh")


if __name__ == "__main__":
    main()
