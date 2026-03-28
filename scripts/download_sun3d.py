#!/usr/bin/env python3
"""
Download thêm SUNRGBDv2 scenes (image/ + depth/ only).
Target: 60 pairs tổng (28 hiện có + ~32 mới) → split 42/9/9 giống B3DO.
Output: /media/ssd220/datasets/SUN3D/<batch>/<scene>/image/ + depth/
"""

import csv
import random
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE_URL = "https://sun3d.cs.princeton.edu/data/SUNRGBDv2/"
OUT_ROOT = Path("/media/ssd220/datasets/SUN3D")
CSV_DIR  = Path("/media/ssd1tb/Long/depth-blip2/datasets/SUN3D")

TARGET_TOTAL = 60   # tổng pairs sau khi download
RANDOM_SEED  = 99   # khác 42 để chọn scenes khác với lần trước


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v not in ("../", "/") and not v.startswith("?") and not v.startswith("http"):
                    self.links.append(v)


def list_dir(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
        p = LinkParser()
        p.feed(html)
        return p.links
    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return []


def wget_dir(url, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    items = list_dir(url)
    for item in items:
        if item.endswith("/"):
            continue
        file_path = local_dir / item
        if file_path.exists():
            continue
        try:
            urllib.request.urlretrieve(url + item, file_path)
        except Exception as e:
            print(f"    [WARN] skip {url + item}: {e}")


def download_scene(batch, scene):
    scene_url   = f"{BASE_URL}{batch}{scene}"
    scene_local = OUT_ROOT / batch.rstrip("/") / scene.rstrip("/")
    for subdir in ("image/", "depth/"):
        local_sub = scene_local / subdir.rstrip("/")
        if local_sub.exists() and any(local_sub.iterdir()):
            continue
        print(f"    → {subdir}")
        wget_dir(scene_url + subdir, local_sub)


def load_existing_csvs():
    """Load all existing pairs from train/val/test CSVs."""
    existing = []
    existing_scenes = set()
    for split in ("train", "val", "test"):
        csv_path = CSV_DIR / f"sun3d_{split}.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            existing.extend(rows)
            for row in rows:
                if row:
                    # Extract batch/scene from path like "11082015/scene/image/..."
                    parts = Path(row[0]).parts
                    if len(parts) >= 2:
                        existing_scenes.add(f"{parts[0]}/{parts[1]}")
    return existing, existing_scenes


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Load existing data
    existing_pairs, existing_scenes = load_existing_csvs()
    print(f"Existing pairs: {len(existing_pairs)}")
    need = TARGET_TOTAL - len(existing_pairs)
    if need <= 0:
        print(f"Already have {len(existing_pairs)} pairs >= target {TARGET_TOTAL}. Nothing to do.")
        return
    print(f"Need {need} more pairs to reach {TARGET_TOTAL} total.\n")

    # 2. Fetch available scenes from server
    print("Fetching batch list from server...")
    batches = [b for b in list_dir(BASE_URL) if b.endswith("/") and b not in ("../",)]
    print(f"  Batches: {[b.rstrip('/') for b in batches]}\n")

    random.seed(RANDOM_SEED)
    candidate_scenes = []
    for batch in batches:
        batch_url = BASE_URL + batch
        scenes = [s for s in list_dir(batch_url) if s.endswith("/") and not s.startswith("?")]
        for s in scenes:
            key = f"{batch.rstrip('/')}/{s.rstrip('/')}"
            if key not in existing_scenes:
                candidate_scenes.append((batch, s))

    print(f"Available new scenes: {len(candidate_scenes)}")
    random.shuffle(candidate_scenes)
    selected = candidate_scenes[:need]
    print(f"Selected {len(selected)} scenes to download.\n")

    # 3. Copy existing SUN3D data to ssd220 if not already there
    src = Path("/media/ssd1tb/Long/depth-blip2/datasets/SUN3D")
    if src.is_dir() and not src.is_symlink():
        print("Copying existing SUN3D data to ssd220...")
        import shutil
        for item in src.iterdir():
            if item.is_dir() and item.name not in ("sun3d_train.csv", "sun3d_val.csv", "sun3d_test.csv"):
                dest = OUT_ROOT / item.name
                if not dest.exists():
                    shutil.copytree(str(item), str(dest))
                    print(f"  Copied {item.name}")
        print()

    # 4. Download new scenes
    new_pairs = []
    for i, (batch, scene) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {batch}{scene}")
        download_scene(batch, scene)

        scene_local = OUT_ROOT / batch.rstrip("/") / scene.rstrip("/")
        img_dir   = scene_local / "image"
        depth_dir = scene_local / "depth"
        if not img_dir.exists() or not depth_dir.exists():
            continue

        imgs   = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        depths = sorted(depth_dir.glob("*.png"))
        n = min(len(imgs), len(depths))
        for j in range(n):
            rgb_rel   = str(imgs[j].relative_to(OUT_ROOT))
            depth_rel = str(depths[j].relative_to(OUT_ROOT))
            new_pairs.append((rgb_rel, depth_rel))

    print(f"\nNew pairs downloaded: {len(new_pairs)}")

    # 5. Re-split everything 70/15/15
    all_pairs = existing_pairs + new_pairs
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)
    n      = len(all_pairs)
    n_test = max(1, round(n * 0.15))
    n_val  = max(1, round(n * 0.15))
    test_p  = all_pairs[:n_test]
    val_p   = all_pairs[n_test:n_test + n_val]
    train_p = all_pairs[n_test + n_val:]

    # CSVs đi vào OUT_ROOT (ssd220) để symlink hoạt động đúng
    splits = {"train": train_p, "val": val_p, "test": test_p}
    for split, data in splits.items():
        csv_path = OUT_ROOT / f"sun3d_{split}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f"  {split}: {len(data)} pairs → {csv_path}")

    # 6. Replace datasets/SUN3D dir with symlink to ssd220
    link = Path("/media/ssd1tb/Long/depth-blip2/datasets/SUN3D")
    if link.is_dir() and not link.is_symlink():
        import shutil
        shutil.rmtree(str(link))
        link.symlink_to(OUT_ROOT)
        print(f"\nReplaced datasets/SUN3D → symlink to {OUT_ROOT}")

    print("\nDone!")


if __name__ == "__main__":
    main()
