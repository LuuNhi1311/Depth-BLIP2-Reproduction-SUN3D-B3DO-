
---

## Experiments on B3DO & SUN3D (ACCV 2024 Extension)

This repository extends the original DepthBLIP-2 paper by evaluating on two new indoor datasets: **B3DO** and **SUN3D**. The full experiment notebook is at [`experiments/run_b3do_sun3d.ipynb`](experiments/run_b3do_sun3d.ipynb).

### Datasets

| Dataset | Samples | Split | Resolution | Depth scale |
|---|---|---|---|---|
| [B3DO](https://www.dropbox.com/s/yzrjtc87tfcr2c1/VOCB3DO.zip) | 60 pairs | 42 / 9 / 9 | 640×480 | 10000 (0.1mm→m) |
| [SUN3D](https://sun3d.cs.princeton.edu/data/SUNRGBDv2/) | 60 pairs | 42 / 9 / 9 | 640×480 | 10000 (0.1mm→m) |

Download scripts: `scripts/download_b3do_sample.py`, `scripts/download_sun3d.py`

### Main Results (validation, epoch 49)

| Dataset | Backbone | abs_rel↓ | log10↓ | rmse↓ | a1↑ | a2↑ | a3↑ |
|---|---|---|---|---|---|---|---|
| NYU (paper) | Vicuna-7B | 0.363 | 0.153 | 1.132 | 0.401 | 0.707 | 0.898 |
| B3DO | OPT-2.7B | 0.478 | 0.186 | 0.600 | 0.364 | 0.591 | 0.686 |
| B3DO | Vicuna-7B | 0.469 | 0.189 | 0.606 | 0.350 | 0.582 | 0.677 |
| SUN3D | OPT-2.7B | **0.256** | **0.121** | 0.941 | 0.472 | 0.799 | 0.957 |
| SUN3D | Vicuna-7B | 0.319 | 0.136 | 0.968 | 0.452 | 0.711 | 0.909 |

### Ablation Study — Bin Range (AB-2)

Effect of `bin_list` range on two datasets:

| Dataset | Backbone | Bin range | abs_rel↓ | rmse↓ | a1↑ |
|---|---|---|---|---|---|
| B3DO | Vicuna-7B | Narrow [1.0–3.5m] | **0.469** | **0.606** | **0.350** |
| B3DO | Vicuna-7B | Wide [0.5–9.5m] | 0.471 | 0.608 | 0.341 |
| SUN3D | Vicuna-7B | Narrow [1.0–3.5m] | 0.319 | 0.968 | 0.452 |
| SUN3D | Vicuna-7B | Wide [0.5–9.5m] | **0.276** | **0.878** | **0.518** |

**Finding:** Narrow bins suit B3DO (near-field objects); Wide bins improve SUN3D (diverse scene depth). Bin range should match the depth distribution of each domain.

### Quick Start

```bash
# Download datasets
python scripts/download_b3do_sample.py --n_pairs 60
python scripts/download_sun3d.py

# Train
bash scripts/train_b3do.sh    # B3DO, Vicuna-7B
bash scripts/train_sun3d.sh   # SUN3D, Vicuna-7B

# Or run full experiment notebook
jupyter notebook experiments/run_b3do_sun3d.ipynb
```

---

## Directory Structure
