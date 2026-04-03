
---

## Experiments on B3DO & SUN3D (ACCV 2024 Extension)

This repository extends the original DepthBLIP-2 paper by evaluating on two new indoor datasets: **B3DO** and **SUN3D**. The full experiment notebook is at [`experiments/run_b3do_sun3d.ipynb`](experiments/run_b3do_sun3d.ipynb).

### Datasets

| Dataset | Samples | Split (train/val/test) | Resolution | Depth scale |
|---|---|---|---|---|
| [B3DO](https://www.dropbox.com/s/yzrjtc87tfcr2c1/VOCB3DO.zip) | 60 pairs | 16 / 8 / 36 | 640×480 | 10000 (0.1mm→m) |
| [SUN3D](https://sun3d.cs.princeton.edu/data/SUNRGBDv2/) | 60 pairs | 16 / 8 / 36 | 640×480 | 10000 (0.1mm→m) |

> Same few-shot setting as the original paper (16 train / 8 val). 36 images reserved for test evaluation.

Download scripts: `scripts/download_b3do_sample.py`, `scripts/download_sun3d.py`

### Main Results (validation, epoch 49, 16 train images)

| Dataset | Backbone | abs_rel↓ | log10↓ | rmse↓ | a1↑ | a2↑ | a3↑ |
|---|---|---|---|---|---|---|---|
| NYU (paper) | OPT-2.7B | 0.380 | 0.155 | 1.148 | 0.397 | 0.701 | 0.890 |
| NYU (paper) | Vicuna-7B | **0.363** | **0.153** | **1.132** | **0.401** | **0.707** | **0.898** |
| B3DO | OPT-2.7B | 0.585 | 0.203 | 0.544 | 0.237 | 0.523 | 0.737 |
| B3DO | Vicuna-7B | 0.585 | 0.203 | 0.544 | 0.236 | 0.523 | 0.737 |
| SUN3D | OPT-2.7B | 0.300 | 0.122 | **0.866** | **0.494** | **0.807** | **0.923** |
| SUN3D | Vicuna-7B | **0.299** | **0.122** | 0.871 | 0.492 | 0.802 | 0.923 |

### Ablation Study — Bin Range (AB-2)

Effect of `bin_list` range on two datasets (Narrow: default paper setting, Wide: extended range):

| Dataset | Backbone | Bin range | abs_rel↓ | log10↓ | rmse↓ | a1↑ |
|---|---|---|---|---|---|---|
| B3DO | OPT-2.7B | Narrow [1.0–3.5m] | 0.585 | 0.203 | 0.544 | 0.237 |
| B3DO | OPT-2.7B | **Wide [0.5–9.5m]** | **0.544** | **0.198** | **0.542** | **0.246** |
| B3DO | Vicuna-7B | Narrow [1.0–3.5m] | 0.585 | 0.203 | 0.544 | 0.236 |
| B3DO | Vicuna-7B | **Wide [0.5–9.5m]** | **0.539** | **0.197** | **0.539** | **0.244** |
| SUN3D | OPT-2.7B | Narrow [1.0–3.5m] | **0.300** | **0.122** | **0.866** | **0.494** |
| SUN3D | OPT-2.7B | Wide [0.5–9.5m] | 0.299 | 0.122 | 0.868 | 0.490 |
| SUN3D | Vicuna-7B | Narrow [1.0–3.5m] | **0.299** | **0.122** | 0.871 | **0.492** |
| SUN3D | Vicuna-7B | Wide [0.5–9.5m] | 0.303 | 0.123 | **0.870** | 0.486 |

**Finding:** Wide bins [0.5–9.5m] improve B3DO by ~7% abs_rel, suggesting B3DO depth distribution extends beyond the default 3.5m ceiling. On SUN3D, both bin ranges perform equivalently. Bin range should be tuned to match the actual depth distribution of each dataset.

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

```
output
|–– configs/    # Configuration files for model loading.
|   |–– bert-base-uncased/
|   |–– blip2_pretrained_vitL/
|   |–– blip2-opt-2.7b/
|   |–– blip2-vicuna-instruct-7b/
|-- blip2_extractor/ # py file of pre-training model.
|–– datasets/   # Scripts for dataset storage and loading.
|–– models/     # DepthBLIP-2 model.
|–– log_results/     # train/test's logging file.
|–– checkpoints/     # pth file after training model.
|–– scripts/    # Shell scripts for quick project execution.
|–– utils/      # Commonly used scripts.
|–– main.py    # Project execution interface.
```

## Running

### All Scenarios

You have the option to configure parameters for `train.py` and run it, or you can use our provided `test.sh` script.

If you choose to run `test.sh`, you need to be in the root directory of the current project, then
execute `bash scripts/test.sh`. Also, please make sure to set the `HF_HOME` and `TORCH_HOME` environment variables,
which correspond to the model loading environment variables in Hugging Face and Lavis, respectively.

Upon completion of the experiment, you can find the evaluation results
in `/path/to/project/root/log_results/method_name/test/N/test_result.txt`. The results are formatted as follows:

```
---->class_name: all
---->depth_templates: ['This {} is {}']
---->obj_classes: ['object']
---->depth_classes: ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
---->bin_list: [1.0, 1.75, 2.25, 2.5, 2.75, 3.0, 3.5]
* * Avg abs_diff : 0.920, a1 : 0.393, a2 : 0.694, a3 : 0.861, abs_rel : 0.363, log10 : 0.153, rmse : 1.152
```

### Specific Scenarios

If you need to test the performance of a specific scenario, be sure to add the parameter `--class_name scenes_name` to
specify the scenario.

## Notes

- If you want to change configuration parameters such as the dataset path, batch size, training model etc., please refer to
  the `parser` in `main.py` for the corresponding modifications.
- Our code is modified based on [DepthCLIP](https://github.com/Adonis-galaxy/DepthCLIP), so there will be some
  similarities with that project.

## Acknowledgement

Our code borrows a lot from:

- [LAVIS](https://github.com/salesforce/LAVIS)

