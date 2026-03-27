#!/bin/bash
# =============================================================
# Download KITTI Eigen split + SUN3D vào /media/ssd220/datasets/
# Sau đó tạo symlink vào project datasets/
# Yêu cầu: kaggle CLI đã cấu hình (~/.kaggle/kaggle.json)
# =============================================================
set -e

DATASET_BASE="/media/ssd220/datasets"
PROJECT_DATASETS="/media/ssd1tb/Long/depth-blip2/datasets"

echo "=== Storage check ==="
df -h /media/ssd220/

# ------------------------------------------------------------------
# 1. Symlink datasets/ vào ssd220
# ------------------------------------------------------------------
echo ""
echo "=== Tạo symlink ==="
ln -sfn "$DATASET_BASE/KITTI"  "$PROJECT_DATASETS/KITTI"
ln -sfn "$DATASET_BASE/SUN3D"  "$PROJECT_DATASETS/SUN3D"
ln -sfn "$DATASET_BASE/B3DO"   "$PROJECT_DATASETS/B3DO"
echo "Symlinks OK:"
ls -la "$PROJECT_DATASETS/"

# ------------------------------------------------------------------
# 2. KITTI Eigen split (Kaggle, ~23GB)
# ------------------------------------------------------------------
echo ""
echo "=== Tải KITTI Eigen Split ==="
cd "$DATASET_BASE/KITTI"

if [ ! -f "kitti-eigen-split-dataset.zip" ]; then
    kaggle datasets download -d awsaf49/kitti-eigen-split-dataset
fi

if [ ! -d "test" ] && [ ! -d "train" ]; then
    echo "Giải nén KITTI..."
    unzip -q kitti-eigen-split-dataset.zip
    rm -f kitti-eigen-split-dataset.zip
fi

echo "Cấu trúc KITTI:"
ls "$DATASET_BASE/KITTI/"

# ------------------------------------------------------------------
# 3. Sinh file danh sách KITTI test
# ------------------------------------------------------------------
echo ""
echo "=== Sinh eigen_test_files_with_gt_dense.txt ==="
cd /media/ssd1tb/Long/depth-blip2
python - <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
from datasets.extract_kitti_summary import extract_summary

kitti_dir  = "/media/ssd220/datasets/KITTI/"
save_path  = "/media/ssd220/datasets/KITTI/"
extract_summary(kitti_dir, save_path)
print("[OK] eigen_test_files_with_gt_dense.txt")
PYEOF

# ------------------------------------------------------------------
# 4. SUN3D – tải một số scenes đại diện (~2-4 GB tổng)
# ------------------------------------------------------------------
echo ""
echo "=== Tải SUN3D scenes ==="
cd "$DATASET_BASE/SUN3D"

# Danh sách scenes nhỏ từ Princeton SUN3D server
# Mỗi scene ~200-600MB; tải 5 scenes đại diện (~2GB)
SCENES=(
    "MIT_76_studyroom"
    "MIT_lab_hj"
    "harvard_c5"
    "harvard_robotics_lab"
    "hotel_umd"
)

BASE_URL="http://sun3d.cs.princeton.edu/data"

for scene in "${SCENES[@]}"; do
    if [ ! -d "$scene" ]; then
        echo "Tải $scene..."
        wget -q --show-progress -O "${scene}.tar" "${BASE_URL}/${scene}.tar" 2>/dev/null \
            || wget -q --show-progress "${BASE_URL}/${scene}/" -r -np -nH --cut-dirs=2 -R "index.html*" \
            || echo "[WARN] Không tải được $scene từ server Princeton"
        if [ -f "${scene}.tar" ]; then
            tar -xf "${scene}.tar" && rm -f "${scene}.tar"
        fi
    else
        echo "  [SKIP] $scene đã có"
    fi
done

echo "Cấu trúc SUN3D:"
ls "$DATASET_BASE/SUN3D/"

# ------------------------------------------------------------------
# 5. Sinh CSV cho SUN3D
# ------------------------------------------------------------------
echo ""
echo "=== Sinh sun3d_*.csv ==="
python - <<'PYEOF'
import os, random
sun3d_root = "/media/ssd220/datasets/SUN3D/"
pairs = []
for scene in sorted(os.listdir(sun3d_root)):
    img_dir   = os.path.join(sun3d_root, scene, "image")
    depth_dir = os.path.join(sun3d_root, scene, "depth")
    if not (os.path.isdir(img_dir) and os.path.isdir(depth_dir)):
        continue
    for fname in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(fname)[0]
        rgb_rel   = f"{scene}/image/{fname}"
        depth_rel = f"{scene}/depth/{stem}.png"
        if os.path.isfile(os.path.join(sun3d_root, depth_rel)):
            pairs.append(f"{rgb_rel},{depth_rel}\n")

random.seed(42); random.shuffle(pairs)
n_test = max(1, int(len(pairs) * 0.2))
n_val  = max(1, int(len(pairs) * 0.1))
splits = {
    "sun3d_test.csv" : pairs[:n_test],
    "sun3d_val.csv"  : pairs[:n_val],
    "sun3d_train.csv": pairs[n_test:],
}
for name, data in splits.items():
    path = os.path.join(sun3d_root, name)
    with open(path, "w") as f: f.writelines(data)
    print(f"  {path}: {len(data)} samples")
PYEOF

echo ""
echo "=== HOÀN THÀNH ==="
df -h /media/ssd220/
echo ""
echo "Symlinks trong project:"
ls -la "$PROJECT_DATASETS/"
