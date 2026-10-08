# DATASET_README — 3DGrocery100 Common Data Pipeline

**Owner:** Dataset Preparation, Preprocessing and Common Data Pipeline  
**Project:** 3DGrocery100 Grocery Store Management  
**Last Updated:** October 2026

---

## Overview

This document is the **single source of truth** for every team member working with the 3DGrocery100 dataset. All four model tracks (PointNet, PointNet++, Point Transformer, and any additional models) must use this pipeline without modification. Any change to common preprocessing **must be discussed and agreed upon** before final comparison runs.

---

## Required Outputs — Status

| Required Output | File(s) | Status |
|---|---|---|
| Preprocessing module | `common/preprocessing.py` | ✅ Complete |
| Common Dataset & DataLoader module | `common/dataset.py` | ✅ Complete |
| 100-class label mapping file | `data/processed/labels.json` | ✅ Complete |
| Fixed train/test index files | `data/processed/splits/train.json` `data/processed/splits/test.json` | ✅ Complete |
| Dataset statistics report | `reports/dataset_statistics.json` `reports/dataset_statistics.csv` | ✅ Complete |
| Data-validation report (rejected/problematic samples) | `reports/data_validation_report_train.csv` `reports/data_validation_report_test.csv` | ✅ Complete |
| Representative point-cloud visualisations | `results/visualizations/` (14 files) | ✅ Complete |
| DATASET_README | `DATASET_README.md` | ✅ This file |

---

## Project Directory Structure

```
3DGrocery100/
├── DATASET_README.md          ← You are here
├── requirements.txt
├── configs/
│   └── data_config.yaml       ← Central config (paths, seed, normalization)
├── common/
│   ├── __init__.py
│   ├── preprocessing.py       ← validate / clean / normalize / sample
│   ├── dataset.py             ← Grocery3DDataset + make_dataloader
│   ├── metrics.py             ← Classification metrics helpers
│   └── visualization.py      ← Point-cloud plotting utilities
├── scripts/
│   ├── prepare_dataset.py     ← Run ONCE to generate splits + reports
│   ├── validate_dataset.py    ← Integrity checker (leakage / counts)
│   ├── inspect_dataset.py     ← Raw data sanity-check report
│   └── visualize_dataset.py  ← Generates all visualisations
├── tests/
│   ├── test_preprocessing.py  ← 36 unit tests (no processed data needed)
│   ├── test_dataset.py        ← 21 integration tests
│   └── test_split.py          ← 18 split-integrity tests
├── data/
│   ├── processed/
│   │   ├── labels.json        ← {class_id: {name, category}}  (100 classes)
│   │   └── splits/
│   │       ├── train.json     ← 66 032 entries
│   │       └── test.json      ← 21 866 entries
│   └── raw/                   ← (symlinks / copies if needed)
├── reports/
│   ├── dataset_statistics.json / .csv
│   ├── data_validation_report_train.csv
│   ├── data_validation_report_test.csv
│   ├── data_validation_summary_train.json
│   ├── data_validation_summary_test.json
│   ├── point_count_statistics_train.json / .csv
│   ├── point_count_statistics_test.json / .csv
│   └── dataset_inspection.json / .txt
└── results/
    └── visualizations/
        ├── class_distribution_train.png
        ├── point_count_distribution.png
        ├── contact_sheet_xyz.png
        ├── contact_sheet_rgb.png
        └── cloud_<class>.png  (×10 individual class examples)
```

---

## Setup Instructions

### 1. Prerequisites

```bash
pip install -r requirements.txt
```

Required packages: `torch>=2.0`, `numpy`, `matplotlib`, `tqdm`, `pyyaml`, `pytest`, `pandas`, `scipy`.

### 2. Raw Dataset Location

The raw dataset must be present at the path specified in `configs/data_config.yaml`:

```yaml
raw_root: "C:/Users/anees/Desktop/Clubs/MISC/misc projects/Groceries Project Datasets/3DGrocery100"
```

**Do NOT modify, move, or delete any file inside the raw dataset directory.**

The pipeline reads these files:
- `modelnet100_train_1024pts_fps.dat` (≈1.6 GB — 66 032 samples)
- `modelnet100_test_1024pts_fps.dat`  (≈540 MB — 21 866 samples)
- `modelnet100_train.txt` / `modelnet100_test.txt` (sample ID index)
- `modelnet100_shape_names.txt` (100 class names)

### 3. Generate Processed Splits (Run Once)

Run from the project root:

```bash
cd "C:\Users\anees\Desktop\Clubs\MISC\misc projects\3DGrocery100"
python scripts/prepare_dataset.py
```

This will:
1. Parse the 100 class names → `data/processed/labels.json`
2. Validate all 87 898 samples for NaN / Inf / shape errors
3. Write `data/processed/splits/train.json` (66 032 entries)
4. Write `data/processed/splits/test.json` (21 866 entries)
5. Confirm zero overlap between train and test
6. Save all reports to `reports/`

Expected output (last lines):
```
Classes           : 100
Train samples     : 66032
Test  samples     : 21866
Train/test overlap: 0 (expected 0)
```

### 4. Verify Integrity

```bash
python scripts/validate_dataset.py
```

Expected output:
```
ALL CHECKS PASSED (OK)
```

### 5. Run All Tests

```bash
python -m pytest tests/ -v
```

Expected: **75 passed** (36 unit + 21 dataset integration + 18 split integrity).

---

## Using the Dataset in Your Model

### Minimal Example

```python
from common.dataset import Grocery3DDataset, make_dataloader

# --- Training ---
train_ds = Grocery3DDataset(split='train', mode='xyz')   # XYZ only  → (1024, 3)
# OR
train_ds = Grocery3DDataset(split='train', mode='xyzrgb') # XYZ+RGB  → (1024, 6)

train_loader = make_dataloader(train_ds, batch_size=32, shuffle=True, num_workers=4)

# --- Testing ---
test_ds     = Grocery3DDataset(split='test', mode='xyz')
test_loader = make_dataloader(test_ds, batch_size=32, shuffle=False, num_workers=4)

# --- Iterating ---
for points, labels in train_loader:
    # points: torch.Tensor  shape (B, 1024, 3) or (B, 1024, 6),  dtype float32
    # labels: torch.Tensor  shape (B,),                           dtype int64
    pass
```

### Class Name Lookup

```python
ds = Grocery3DDataset(split='train', mode='xyz')
print(ds.get_class_name(4))      # e.g. "apple_honeycrisp"
print(ds.get_class_id('banana')) # e.g. 12
print(ds.num_classes)            # 100
```

### Label Map (for loss / metrics)

```python
from common.dataset import build_id_name_maps

id_to_name, name_to_id = build_id_name_maps('data/processed/labels.json')
# id_to_name: {0: 'apple_evercrisp', 1: 'apple_fuji', ..., 99: 'yoghurt_vanilla'}
```

---

## Preprocessing Details

All preprocessing is applied deterministically inside `Grocery3DDataset.__getitem__()` using `common/preprocessing.py`. The pipeline for each sample is:

| Step | Operation | Details |
|---|---|---|
| 1 | **Validate** | Reject if shape wrong, label out of range, or all-NaN |
| 2 | **Clean** | Drop rows with NaN/Inf XYZ; clamp RGB to [0, 1] |
| 3 | **Resample** | Ensure exactly 1024 points (random down/up-sample if needed) |
| 4 | **Normalize XYZ** | Unit-sphere: subtract centroid, divide by max L2 norm |
| 5 | **Select channels** | Return XYZ (cols 0-2) or XYZRGB (cols 0-5) |

**Normalization formula:**
```
centroid = mean(P_xyz, axis=0)
P_centered = P_xyz - centroid
scale = max(||P_centered||_2)
P_normalized = P_centered / scale
```

After normalization: centroid ≈ [0,0,0], all points within unit sphere (max dist = 1.0).

---

## Dataset Statistics

| Metric | Value |
|---|---|
| Total classes | 100 |
| Train samples | 66 032 |
| Test samples | 21 866 |
| Total samples | 87 898 |
| Points per sample | 1 024 (fixed, FPS pre-sampled) |
| Channels per point | 6 (X, Y, Z, R, G, B) |
| Valid samples (train) | 66 032 (100%) |
| Rejected samples (train) | 0 |
| Valid samples (test) | 21 866 (100%) |
| Rejected samples (test) | 0 |
| Train/test overlap | 0 |

Full statistics: `reports/dataset_statistics.json`  
Per-class counts: `reports/dataset_statistics.csv`  
Validation log: `reports/data_validation_report_train.csv`

---

## Visualisations

All visualisations are in `results/visualizations/`:

| File | Description |
|---|---|
| `class_distribution_train.png` | Bar chart of per-class sample counts |
| `point_count_distribution.png` | Histogram of raw point counts before resampling |
| `contact_sheet_xyz.png` | 4×5 grid of 20 point clouds (XYZ coloured by Z) |
| `contact_sheet_rgb.png` | 4×5 grid of 20 point clouds (true RGB colour) |
| `cloud_apple_*.png` | Individual class 3D scatter plots (10 apple varieties) |

---

## Important Rules for Team Members

> **Do NOT** re-implement your own data loader. Use `Grocery3DDataset` and `make_dataloader` from `common/dataset.py`.

> **Do NOT** change `data/processed/labels.json` or the split JSONs. These are frozen for the duration of the project.

> **Do NOT** apply additional normalization inside your model code — the pipeline already delivers unit-sphere normalized points.

> **If** you need a change to any common preprocessing (e.g., add normals, change point count), raise it with the dataset owner first. All four models must be re-validated after any such change.

---

## Quick-Reference Commands

```bash
# One-time setup
python scripts/prepare_dataset.py

# Integrity check
python scripts/validate_dataset.py

# Smoke test (copy-paste into terminal)
python -c "from common.dataset import Grocery3DDataset; ds = Grocery3DDataset(split='train', mode='xyz'); print(ds[0][0].shape)"
# Expected: torch.Size([1024, 3])

python -c "from common.dataset import Grocery3DDataset; ds = Grocery3DDataset(split='train', mode='xyzrgb'); print(ds[0][0].shape)"
# Expected: torch.Size([1024, 6])

# Full test suite
python -m pytest tests/ -v

# Regenerate visualisations
python scripts/visualize_dataset.py
```
