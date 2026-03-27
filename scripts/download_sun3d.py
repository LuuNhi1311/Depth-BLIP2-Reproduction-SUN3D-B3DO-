#!/usr/bin/env python3
"""
Download a sample of SUNRGBDv2 scenes (image/ + depth/ only).
Target: ~20 scenes spread across date batches → ~2-4 GB total.
Output: /media/ssd220/datasets/SUN3D/<date>/<scene>/image/ + depth/
"""

import os
import csv
import random
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path

BASE_URL  = "https://sun3d.cs.princeton.edu/data/SUNRGBDv2/"
OUT_ROOT  = Path("/media/ssd220/datasets/SUN3D")
CSV_DIR   = Path("/media/ssd1tb/Long/depth-blip2/datasets/SUN3D")

# Số scenes muốn tải mỗi batch date
SCENES_PER_BATCH = 4
RANDOM_SEED      = 42

# ─────────────────────────────────────────────────
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
    """Download all files under url/ into local_dir/"""
    local_dir.mkdir(parents=True, exist_ok=True)
    items = list_dir(url)
    for item in items:
        if item.endswith("/"):
            continue  # skip sub-dirs (no sub-dirs expected in image/ or depth/)
        file_url  = url + item
        file_path = local_dir / item
        if file_path.exists():
            continue
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except Exception as e:
            print(f"    [WARN] skip {file_url}: {e}")

def download_scene(batch, scene):
    """Download only image/ and depth/ for one scene."""
    scene_url   = f"{BASE_URL}{batch}{scene}"
    scene_local = OUT_ROOT / batch / scene

    for subdir in ("image/", "depth/"):
        local_sub = scene_local / subdir.rstrip("/")
        if local_sub.exists() and any(local_sub.iterdir()):
            print(f"  [SKIP] {subdir} already exists")
            continue
        print(f"  → {subdir}")
        wget_dir(scene_url + subdir, local_sub)

# ─────────────────────────────────────────────────
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching batch list …")
    batches = [b for b in list_dir(BASE_URL) if b.endswith("/") and b not in ("../",)]
    print(f"  Batches: {[b.rstrip('/') for b in batches]}")

    random.seed(RANDOM_SEED)
    selected = []  # list of (batch, scene)

    for batch in batches:
        batch_url = BASE_URL + batch
        scenes = [s for s in list_dir(batch_url) if s.endswith("/") and not s.startswith("?")]
        if not scenes:
            continue
        chosen = random.sample(scenes, min(SCENES_PER_BATCH, len(scenes)))
        for s in chosen:
            selected.append((batch, s))

    print(f"\nSelected {len(selected)} scenes total. Downloading …")
    pairs = []  # (rgb_rel, depth_rel)

    for i, (batch, scene) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {batch}{scene}")
        download_scene(batch, scene)

        # Collect pairs
        scene_local = OUT_ROOT / batch / scene
        img_dir   = scene_local / "image"
        depth_dir = scene_local / "depth"
        if not img_dir.exists() or not depth_dir.exists():
            continue

        imgs   = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        depths = sorted(depth_dir.glob("*.png"))

        # Match by index (SUN3D frames are aligned by sequence)
        n = min(len(imgs), len(depths))
        for j in range(n):
            rgb_rel   = str(imgs[j].relative_to(OUT_ROOT))
            depth_rel = str(depths[j].relative_to(OUT_ROOT))
            pairs.append((rgb_rel, depth_rel))

    print(f"\nTotal pairs: {len(pairs)}")
    if not pairs:
        print("[ERROR] No pairs found. Check download.")
        return

    # ── Split train/val/test 70/15/15 ──────────────────────
    random.shuffle(pairs)
    n      = len(pairs)
    n_val  = max(1, int(n * 0.15))
    n_test = max(1, int(n * 0.15))
    test_p  = pairs[:n_test]
    val_p   = pairs[n_test:n_test + n_val]
    train_p = pairs[n_test + n_val:]

    splits = {"train": train_p, "val": val_p, "test": test_p}
    for split, data in splits.items():
        csv_path = CSV_DIR / f"sun3d_{split}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f"  {split}: {len(data)} pairs → {csv_path}")

    # Create symlink in project datasets/
    link = Path("/media/ssd1tb/Long/depth-blip2/datasets/SUN3D")
    if not link.exists():
        link.symlink_to(OUT_ROOT)
        print(f"\nSymlink: {link} → {OUT_ROOT}")

if __name__ == "__main__":
    main()
