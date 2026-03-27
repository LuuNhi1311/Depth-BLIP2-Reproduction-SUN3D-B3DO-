# %% [markdown]
"""
# DepthBLIP-2 — Thí nghiệm trên SUN3D

**Paper**: DepthBLIP-2: Leveraging Language to Guide BLIP-2 in Understanding Depth Information (ACCV 2024)

> **Luu y ve B3DO**: VOCB3DO.zip khong chua anh RGB (chi co depth maps + annotations VOC)
> nen khong dung duoc cho depth estimation. Thuc nghiem chi chay tren **SUN3D**.

## Thong tin dataset

| Dataset | Loai   | HxW     | depth_scale      | max_depth | Metrics   |
|---------|--------|---------|------------------|-----------|-----------|
| SUN3D   | Indoor | 480x640 | 10000 (0.1mm->m) | 10 m      | NYU-style |

## Ket qua paper (NYU — tham khao indoor)

| abs_rel | log10 | rmse  | a1    | a2    | a3    |
|---------|-------|-------|-------|-------|-------|
| 0.105   | 0.044 | 0.395 | 0.884 | 0.970 | 0.990 |
"""

# %%
# ============================================================
# Cell 2: Import va thiet lap duong dan
# ============================================================
import os
import sys
import re
import subprocess
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")

ROOT   = Path("/media/ssd1tb/Long/depth-blip2")
PYTHON = "/home/hoangtv/anaconda3/envs/dat310/bin/python"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# SUN3D config
SUN3D_ROOT        = "./datasets/SUN3D/"
SUN3D_TRAIN_CSV   = "./datasets/SUN3D/sun3d_train.csv"
SUN3D_VAL_CSV     = "./datasets/SUN3D/sun3d_val.csv"
SUN3D_TEST_CSV    = "./datasets/SUN3D/sun3d_test.csv"
SUN3D_DEPTH_SCALE = 10000.0   # SUNRGBDv2: 0.1mm -> m
SUN3D_HEIGHT      = 480
SUN3D_WIDTH       = 640
SUN3D_MAX_DEPTH   = 10.0

# Paper reference
PAPER_NYU = {
    "Dataset": "NYU (Paper)",
    "abs_rel": 0.105, "log10": 0.044, "rmse": 0.395,
    "a1": 0.884, "a2": 0.970, "a3": 0.990,
}

print(f"ROOT   = {ROOT}")
print(f"PYTHON = {PYTHON}")
print(f"CWD    = {os.getcwd()}")

# %% [markdown]
"""
## Cell 3: Kiem tra du lieu SUN3D

SUN3D da duoc tai san (28 frames tu 7 batches cua SUNRGBDv2).
"""

# %%
# ============================================================
# Cell 4: Kiem tra trang thai SUN3D
# ============================================================
def check_sun3d():
    ok = True
    for split, path in [("train", SUN3D_TRAIN_CSV),
                        ("val",   SUN3D_VAL_CSV),
                        ("test",  SUN3D_TEST_CSV)]:
        if not os.path.exists(path):
            print(f"[MISS] {split}: {path}")
            ok = False
            continue
        df = pd.read_csv(path, header=None, names=["rgb", "depth"])
        first_rgb = os.path.join(SUN3D_ROOT, df.iloc[0]["rgb"])
        status = "OK" if os.path.isfile(first_rgb) else "FILE NOT FOUND"
        print(f"  [{status}] {split:5s}: {len(df):3d} pairs — {path}")
    return ok

sun3d_ok = check_sun3d()
if sun3d_ok:
    df_train = pd.read_csv(SUN3D_TRAIN_CSV, header=None, names=["rgb", "depth"])
    print("\nMau train CSV:")
    print(df_train.head(3).to_string(index=False))

# %% [markdown]
"""
## Cell 5: Training SUN3D

Fine-tune MLP head cua DepthBLIP-2 tren 16 anh SUN3D (auto_bins=True).
BLIP-2 backbone dong bang — chi train MLP nhe.

**Uoc tinh thoi gian:** ~15-30 phut (50 epochs, batch=2, 1 GPU).
"""

# %%
# ============================================================
# Cell 6: Ham run_train va find_latest_checkpoint
# ============================================================
def run_train(dataset, data_root, train_file, val_file, height, width,
              depth_scale, max_depth, epochs=50, batch_size=2,
              train_limit=16, val_limit=8, gpu="0"):
    cmd = [
        PYTHON, "main.py",
        "--train",
        "--dataset",         dataset,
        "--method",          "second",
        "--auto_bins",       "True",
        "--height",          str(height),
        "--width",           str(width),
        "--max_depth",       str(max_depth),
        "--depth_scale",     str(depth_scale),
        "--data_root_path",  data_root,
        "--train_file",      train_file,
        "--val_file",        val_file,
        "--test_file",       val_file,
        "--train_limit",     str(train_limit),
        "--val_limit",       str(val_limit),
        "--batch_size",      str(batch_size),
        "--workers",         "2",
        "--epochs",          str(epochs),
        "--lr",              "0.001",
        "--weight_decay",    "0.01",
        "--gpu_num",         gpu,
        "--train_log_save",
        "--log_result_dir",  "./log_results/",
        "--model_save_path", "./checkpoints/",
    ]
    print("=" * 65)
    print(f"  Training {dataset}  (epochs={epochs}, train_limit={train_limit})")
    print("=" * 65)
    return subprocess.run(cmd, cwd=str(ROOT))


def find_latest_checkpoint(dataset, method="second"):
    base = ROOT / "checkpoints" / f"{method}_{dataset}_bins_True" / "train"
    if not base.exists():
        return None
    runs = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not runs:
        return None
    pths = sorted(
        runs[-1].glob("model_epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return str(pths[-1]) if pths else None

# %%
# ============================================================
# Cell 7: Train SUN3D
# ============================================================
_sun3d_ckpt = find_latest_checkpoint("SUN3D")
if _sun3d_ckpt:
    print(f"[SKIP] Checkpoint da co: {_sun3d_ckpt}")
    print("       Xoa checkpoints/second_SUN3D_bins_True/ de train lai.")
else:
    if not sun3d_ok:
        print("[ERROR] Du lieu SUN3D chua san sang.")
    else:
        run_train(
            dataset     = "SUN3D",
            data_root   = SUN3D_ROOT,
            train_file  = SUN3D_TRAIN_CSV,
            val_file    = SUN3D_VAL_CSV,
            height      = SUN3D_HEIGHT,
            width       = SUN3D_WIDTH,
            depth_scale = SUN3D_DEPTH_SCALE,
            max_depth   = SUN3D_MAX_DEPTH,
        )
        _sun3d_ckpt = find_latest_checkpoint("SUN3D")
        print(f"\n[OK] Checkpoint: {_sun3d_ckpt}")

# %% [markdown]
"""
## Cell 8: Testing & Danh gia
"""

# %%
# ============================================================
# Cell 9: Ham run_test va parse_metrics
# ============================================================
INDOOR_METRICS = ["abs_diff", "a1", "a2", "a3", "abs_rel", "log10", "rmse"]


def run_test(dataset, data_root, test_file, height, width,
             depth_scale, max_depth, model_path, gpu="0"):
    cmd = [
        PYTHON, "main.py",
        "--dataset",         dataset,
        "--method",          "second",
        "--auto_bins",       "True",
        "--height",          str(height),
        "--width",           str(width),
        "--max_depth",       str(max_depth),
        "--depth_scale",     str(depth_scale),
        "--data_root_path",  data_root,
        "--test_file",       test_file,
        "--class_name",      "all",
        "--gpu_num",         gpu,
        "--model_load_path", model_path,
    ]
    print("=" * 65)
    print(f"  Testing {dataset}")
    print("=" * 65)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    for line in result.stdout.strip().splitlines()[-50:]:
        print(line)
    if result.returncode != 0:
        print("[STDERR]", result.stderr[-500:])
    return result


def parse_metrics(stdout, metric_names):
    line = next((l for l in stdout.splitlines() if "* * Avg" in l), None)
    if not line:
        return {}
    metrics = {}
    for name in metric_names:
        m = re.search(rf"{re.escape(name)}\s*:\s*([\d.]+)", line)
        if m:
            metrics[name] = float(m.group(1))
    return metrics

# %%
# ============================================================
# Cell 10: Chay test SUN3D
# ============================================================
_sun3d_ckpt = find_latest_checkpoint("SUN3D")
metrics_sun3d = {}

if not _sun3d_ckpt:
    print("[WARN] Chua co checkpoint SUN3D. Hay chay cell Train truoc.")
else:
    result_test = run_test(
        dataset     = "SUN3D",
        data_root   = SUN3D_ROOT,
        test_file   = SUN3D_TEST_CSV,
        height      = SUN3D_HEIGHT,
        width       = SUN3D_WIDTH,
        depth_scale = SUN3D_DEPTH_SCALE,
        max_depth   = SUN3D_MAX_DEPTH,
        model_path  = _sun3d_ckpt,
    )
    metrics_sun3d = parse_metrics(result_test.stdout, INDOOR_METRICS)
    if metrics_sun3d:
        print("\n[Metrics SUN3D]")
        for k, v in metrics_sun3d.items():
            print(f"  {k:10s}: {v:.4f}")
    else:
        print("\n[WARN] Khong parse duoc metrics. Kiem tra output ben tren.")

# %% [markdown]
"""
## Cell 11: Bang so sanh & Bieu do
"""

# %%
# ============================================================
# Cell 12: Bang so sanh va bar chart
# ============================================================
compare_cols = ["abs_rel", "log10", "rmse", "a1", "a2", "a3"]
rows = [PAPER_NYU]
if metrics_sun3d:
    rows.append({
        "Dataset": "SUN3D (Ours)",
        **{k: metrics_sun3d.get(k, float("nan")) for k in compare_cols},
    })

df_results = pd.DataFrame(rows).set_index("Dataset")[compare_cols]

print("\n" + "=" * 58)
print("    Ket qua so sanh — Indoor Depth Estimation")
print("=" * 58)
print(df_results.to_string(float_format="%.3f"))
print("=" * 58)
print("  thap hon = tot hon : abs_rel, log10, rmse")
print("  cao hon  = tot hon : a1, a2, a3")

if metrics_sun3d:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    w = 0.35

    err_cols = ["abs_rel", "log10", "rmse"]
    x = np.arange(len(err_cols))
    axes[0].bar(x - w/2, [PAPER_NYU[c] for c in err_cols], w, label="NYU (Paper)", color="#4C72B0")
    axes[0].bar(x + w/2, [metrics_sun3d.get(c, 0) for c in err_cols], w, label="SUN3D (Ours)", color="#DD8452")
    axes[0].set_xticks(x); axes[0].set_xticklabels(err_cols)
    axes[0].set_title("Error metrics (thap hon = tot hon)"); axes[0].legend()

    acc_cols = ["a1", "a2", "a3"]
    x = np.arange(len(acc_cols))
    axes[1].bar(x - w/2, [PAPER_NYU[c] for c in acc_cols], w, label="NYU (Paper)", color="#4C72B0")
    axes[1].bar(x + w/2, [metrics_sun3d.get(c, 0) for c in acc_cols], w, label="SUN3D (Ours)", color="#DD8452")
    axes[1].set_xticks(x); axes[1].set_xticklabels(acc_cols)
    axes[1].set_title("Accuracy metrics (cao hon = tot hon)"); axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    plt.suptitle("DepthBLIP-2: SUN3D vs NYU Paper", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs("log_results", exist_ok=True)
    plt.savefig("log_results/comparison_chart.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[OK] Saved log_results/comparison_chart.png")

# %% [markdown]
"""
## Cell 13: Visualization — RGB | GT Depth | Predicted Depth
"""

# %%
# ============================================================
# Cell 14: Load model de visualize
# ============================================================
import argparse as _argparse
from torchvision import transforms as T

_PREPROC = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_depth_blip2_model(ckpt_path, gpu="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from lavis.models import load_model_and_preprocess
    blip2_base, _, _ = load_model_and_preprocess(
        name="depth_blip2_vicuna_instruct",
        model_type="vicuna7b",
        is_eval=True,
        device=device,
    )
    from models.model import DepthBLIP2
    args = _argparse.Namespace(
        dataset="SUN3D", method="second", auto_bins=True,
        height=SUN3D_HEIGHT, width=SUN3D_WIDTH,
        max_depth=SUN3D_MAX_DEPTH, temperature=0.5,
        depth_templates=["There is {} in the scene."],
        obj_classes=["object"], auto_prompt=False, flag=2,
    )
    model = DepthBLIP2(blip2_base, args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device

# %%
# ============================================================
# Cell 15: Visualization
# ============================================================
def visualize_sun3d(ckpt_path, n_samples=4):
    print("[INFO] Loading model ...")
    try:
        model, device = load_depth_blip2_model(ckpt_path)
    except Exception as e:
        print(f"[ERROR] Khong load duoc model: {e}")
        return

    df   = pd.read_csv(SUN3D_TEST_CSV, header=None, names=["rgb", "depth"])
    n    = min(n_samples, len(df))
    samp = df.sample(n, random_state=42).reset_index(drop=True)

    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, row in samp.iterrows():
        rgb_img = Image.open(os.path.join(SUN3D_ROOT, row["rgb"])).convert("RGB")
        dep_raw = np.array(Image.open(os.path.join(SUN3D_ROOT, row["depth"])), dtype=np.float32)
        dep_gt  = np.clip(dep_raw / SUN3D_DEPTH_SCALE, 0, SUN3D_MAX_DEPTH)

        inp = _PREPROC(rgb_img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp, 0)
            pred = F.interpolate(pred, size=(SUN3D_HEIGHT, SUN3D_WIDTH),
                                 mode="bilinear", align_corners=True)
        pred_np = np.clip(pred.squeeze().cpu().numpy(), 0, SUN3D_MAX_DEPTH)

        valid = (dep_gt > 1e-3) & (dep_gt < SUN3D_MAX_DEPTH)
        if valid.sum() > 0:
            ar = float(np.mean(np.abs(dep_gt[valid] - pred_np[valid]) / dep_gt[valid]))
            metric_str = f"abs_rel={ar:.3f}"
        else:
            metric_str = "no valid depth"

        axes[i, 0].imshow(rgb_img);                                          axes[i, 0].set_title("RGB")
        axes[i, 1].imshow(dep_gt,   cmap="plasma", vmin=0, vmax=SUN3D_MAX_DEPTH); axes[i, 1].set_title("GT Depth")
        im = axes[i, 2].imshow(pred_np, cmap="plasma", vmin=0, vmax=SUN3D_MAX_DEPTH)
        axes[i, 2].set_title(f"Predicted ({metric_str})")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04).set_label("m")
        for ax in axes[i]: ax.axis("off")

    plt.suptitle("DepthBLIP-2 — SUN3D Test Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs("log_results", exist_ok=True)
    out = "log_results/visualization_sun3d.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[OK] Saved {out}")


_sun3d_ckpt = find_latest_checkpoint("SUN3D")
if not _sun3d_ckpt:
    print("[WARN] Chua co checkpoint. Hay chay cell Train truoc.")
else:
    visualize_sun3d(_sun3d_ckpt, n_samples=4)

# %% [markdown]
"""
## Cell 16: Ket luan

### Thu tu chay
| Cell | Noi dung |
|------|----------|
| 2    | Import & setup |
| 4    | Kiem tra data SUN3D |
| 7    | Train SUN3D (bo qua neu co checkpoint) |
| 10   | Test SUN3D -> metrics |
| 12   | Bang so sanh + bieu do |
| 15   | Visualization |

### Ve B3DO
`VOCB3DO.zip` chi co depth maps + VOC annotations, **khong co anh RGB**.
Code B3DO da san sang tai `scripts/train_b3do.sh` — neu tim duoc anh RGB thi chay truc tiep.

### Dien giai ket qua SUN3D vs NYU
- Cung indoor Kinect setting -> so sanh hop le
- Ky vong kem hon NYU paper (chi fine-tune 16 anh)
- `a1 > 0.7` va `abs_rel < 0.25` -> model hoc duoc cau truc depth indoor
"""
