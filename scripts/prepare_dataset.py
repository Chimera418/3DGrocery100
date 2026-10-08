"""
scripts/prepare_dataset.py
==========================
Main dataset preparation pipeline for 3DGrocery100.

This script:
  1. Reads the raw dataset from the official .dat and .txt files
  2. Generates data/processed/labels.json  (100-class mapping)
  3. Validates every sample
  4. Generates data/processed/splits/train.json and test.json
  5. Generates reports/data_validation_report.csv
  6. Generates reports/data_validation_summary.json
  7. Generates reports/point_count_statistics.json / .csv
  8. Generates reports/dataset_statistics.json / .csv

Usage
-----
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --raw_root <path> --processed_root <path>
  python scripts/prepare_dataset.py --help

NOTE: This script does NOT copy or modify any raw files.
All outputs go to data/processed/ and reports/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow running from project root or scripts/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.preprocessing import (
    RANDOM_SEED,
    N_POINTS,
    set_global_seed,
    validate_points,
    load_shape_names,
    load_split_ids,
)

logger = logging.getLogger("prepare_dataset")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_RAW_ROOT = (
    "C:/Users/anees/Desktop/Clubs/MISC/misc projects/"
    "Groceries Project Datasets/3DGrocery100"
)
DEFAULT_PROCESSED_ROOT = "data/processed"
DEFAULT_REPORTS_DIR = "reports"

# Broad category assignment based on class position in modelnet100_shape_names.txt
# (verified against the actual class list in the dataset)
# Classes 0-9:   Apple varieties
# Classes 10-33: Other fruits
# Classes 34-61: Vegetables
# Classes 62-99: Packaged grocery items
_CATEGORY_MAP = {
    **{i: "fruit_apple" for i in range(0, 10)},
    **{i: "fruit_other" for i in range(10, 34)},
    **{i: "vegetable" for i in range(34, 62)},
    **{i: "packaged_grocery" for i in range(62, 100)},
}


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def build_label_map(shape_names: List[str]) -> Dict[str, dict]:
    """
    Build the 100-class label map from the ordered shape_names list.

    Returns a dict suitable for saving as labels.json.
    """
    label_map: Dict[str, dict] = {}
    for idx, name in enumerate(shape_names):
        category = _CATEGORY_MAP.get(idx, "other")
        label_map[str(idx)] = {
            "name": name,
            "category": category,
        }
    return label_map


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

def run_validation(
    point_clouds: list,
    labels_raw: list,
    split_ids: List[str],
    shape_names: List[str],
    split_name: str,
) -> tuple[List[Dict], List[Dict], Dict]:
    """
    Validate all samples in a split.

    Returns
    -------
    valid_entries   : list of dicts (samples that passed or warned)
    rejected_entries: list of dicts (samples that were rejected)
    summary         : high-level statistics dict
    """
    n_valid = 0
    n_warning = 0
    n_rejected = 0
    valid_entries: List[Dict] = []
    rejected_entries: List[Dict] = []

    n_classes = len(shape_names)
    name_to_id = {name: i for i, name in enumerate(shape_names)}

    for idx, (pc, lbl_arr, sample_id) in enumerate(
        zip(point_clouds, labels_raw, split_ids)
    ):
        arr = np.asarray(pc, dtype=np.float32)
        lbl = int(np.asarray(lbl_arr).flatten()[0])

        # Extract class name from sample ID
        class_name = sample_id.split("=")[0]

        result = validate_points(
            arr,
            lbl,
            n_classes=n_classes,
            require_rgb=False,
            sample_id=sample_id,
        )

        entry = {
            "index": idx,
            "sample_id": sample_id,
            "class_name": class_name,
            "class_id": lbl,
            "n_points": arr.shape[0] if arr.ndim == 2 else -1,
            "n_channels": arr.shape[1] if arr.ndim == 2 else -1,
            "status": result["status"],
            "issues": "; ".join(result["issues"]) if result["issues"] else "",
        }

        if result["status"] == "REJECTED":
            n_rejected += 1
            rejected_entries.append(entry)
        elif result["status"] == "WARNING":
            n_warning += 1
            valid_entries.append(entry)
        else:
            n_valid += 1
            valid_entries.append(entry)

    summary = {
        "split": split_name,
        "total": n_valid + n_warning + n_rejected,
        "valid": n_valid,
        "warning": n_warning,
        "rejected": n_rejected,
        "acceptance_rate_pct": round(
            100.0 * (n_valid + n_warning) / max(1, n_valid + n_warning + n_rejected), 2
        ),
    }
    return valid_entries, rejected_entries, summary


# ---------------------------------------------------------------------------
# Point-count statistics
# ---------------------------------------------------------------------------

def compute_point_stats(
    point_clouds: list,
    labels_raw: list,
    shape_names: List[str],
) -> Dict:
    """Compute per-sample and per-class point-count statistics."""
    counts = [np.asarray(pc).shape[0] for pc in point_clouds]
    labels = [int(np.asarray(l).flatten()[0]) for l in labels_raw]
    counts = np.array(counts)

    per_class: Dict[str, dict] = {}
    for cls_id, cls_name in enumerate(shape_names):
        mask = np.array(labels) == cls_id
        cls_counts = counts[mask]
        if len(cls_counts) == 0:
            continue
        per_class[cls_name] = {
            "class_id": cls_id,
            "n_samples": int(len(cls_counts)),
            "min": int(cls_counts.min()),
            "max": int(cls_counts.max()),
            "mean": float(cls_counts.mean()),
        }

    return {
        "global": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "std": float(counts.std()),
            "p5": float(np.percentile(counts, 5)),
            "p25": float(np.percentile(counts, 25)),
            "p75": float(np.percentile(counts, 75)),
            "p95": float(np.percentile(counts, 95)),
        },
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Split JSON generation
# ---------------------------------------------------------------------------

def build_split_json(
    valid_entries: List[Dict],
) -> List[Dict]:
    """
    Convert validated entries into the format used by Grocery3DDataset.

    Each item:
        {
          "index": int,           # position in .dat file
          "sample_id": str,
          "class_id": int,
          "class_name": str,
          "status": "VALID" | "WARNING"
        }
    """
    return [
        {
            "index": e["index"],
            "sample_id": e["sample_id"],
            "class_id": e["class_id"],
            "class_name": e["class_name"],
            "status": e["status"],
        }
        for e in valid_entries
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 3DGrocery100 dataset: validate, build label map, "
            "generate fixed splits, and save statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_root",
        default=DEFAULT_RAW_ROOT,
        help="Path to the 3DGrocery100 dataset directory (read-only source)",
    )
    parser.add_argument(
        "--processed_root",
        default=DEFAULT_PROCESSED_ROOT,
        help="Output directory for processed metadata and splits",
    )
    parser.add_argument(
        "--reports_dir",
        default=DEFAULT_REPORTS_DIR,
        help="Output directory for CSV/JSON reports",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Global random seed",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    reports_dir = Path(args.reports_dir)

    processed_root.mkdir(parents=True, exist_ok=True)
    (processed_root / "splits").mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load class names
    # -----------------------------------------------------------------------
    shape_names_path = raw_root / "modelnet100_shape_names.txt"
    logger.info("Loading class names from %s …", shape_names_path)
    shape_names = load_shape_names(shape_names_path)
    if len(shape_names) != 100:
        logger.warning(
            "Expected 100 classes, found %d in shape_names file!",
            len(shape_names),
        )
    else:
        logger.info("Found %d classes ✓", len(shape_names))

    # -----------------------------------------------------------------------
    # 2. Build and save labels.json
    # -----------------------------------------------------------------------
    label_map = build_label_map(shape_names)
    labels_json_path = processed_root / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2)
    logger.info("Saved: %s", labels_json_path)

    # -----------------------------------------------------------------------
    # 3. Load train and test .dat files + split .txt files
    # -----------------------------------------------------------------------
    for split_name in ("train", "test"):
        dat_file = raw_root / f"modelnet100_{split_name}_1024pts_fps.dat"
        txt_file = raw_root / f"modelnet100_{split_name}.txt"

        logger.info("\n%s\n=== Processing %s split ===\n%s",
                    "=" * 50, split_name.upper(), "=" * 50)
        logger.info("Loading %s (%.1f MB) …",
                    dat_file.name, dat_file.stat().st_size / 1e6)

        with open(dat_file, "rb") as fh:
            raw_dat = pickle.load(fh)

        point_clouds = raw_dat[0]
        labels_raw = raw_dat[1]

        split_ids = load_split_ids(txt_file)

        logger.info(
            "  DAT samples: %d | TXT entries: %d",
            len(point_clouds), len(split_ids),
        )
        assert len(point_clouds) == len(split_ids), (
            f"Mismatch: DAT has {len(point_clouds)} samples but "
            f"TXT has {len(split_ids)} entries"
        )

        # -------------------------------------------------------------------
        # 4. Validate
        # -------------------------------------------------------------------
        logger.info("Running validation on %d samples …", len(point_clouds))
        valid_entries, rejected_entries, val_summary = run_validation(
            point_clouds, labels_raw, split_ids, shape_names, split_name
        )
        logger.info(
            "  VALID=%d  WARNING=%d  REJECTED=%d",
            val_summary["valid"],
            val_summary["warning"],
            val_summary["rejected"],
        )

        # Save validation report CSV
        all_entries = valid_entries + rejected_entries
        all_entries.sort(key=lambda x: x["index"])
        val_csv_path = reports_dir / f"data_validation_report_{split_name}.csv"
        fieldnames = ["index", "sample_id", "class_name", "class_id",
                      "n_points", "n_channels", "status", "issues"]
        with open(val_csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_entries)
        logger.info("Saved: %s", val_csv_path)

        # Save validation summary JSON
        val_sum_path = reports_dir / f"data_validation_summary_{split_name}.json"
        with open(val_sum_path, "w", encoding="utf-8") as fh:
            json.dump(val_summary, fh, indent=2)
        logger.info("Saved: %s", val_sum_path)

        # -------------------------------------------------------------------
        # 5. Point-count statistics
        # -------------------------------------------------------------------
        logger.info("Computing point-count statistics …")
        pc_stats = compute_point_stats(point_clouds, labels_raw, shape_names)

        pc_stats_json_path = reports_dir / f"point_count_statistics_{split_name}.json"
        with open(pc_stats_json_path, "w", encoding="utf-8") as fh:
            json.dump(pc_stats, fh, indent=2)
        logger.info("Saved: %s", pc_stats_json_path)

        pc_stats_csv_path = reports_dir / f"point_count_statistics_{split_name}.csv"
        with open(pc_stats_csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["class_name", "class_id", "n_samples",
                             "min_pts", "max_pts", "mean_pts"])
            for cls_name, stats in pc_stats["per_class"].items():
                writer.writerow([
                    cls_name, stats["class_id"], stats["n_samples"],
                    stats["min"], stats["max"], round(stats["mean"], 1),
                ])
        logger.info("Saved: %s", pc_stats_csv_path)

        # -------------------------------------------------------------------
        # 6. Generate fixed split JSON
        # -------------------------------------------------------------------
        split_json = build_split_json(valid_entries)
        split_json_path = processed_root / "splits" / f"{split_name}.json"
        with open(split_json_path, "w", encoding="utf-8") as fh:
            json.dump(split_json, fh, indent=2)
        logger.info("Saved: %s  (%d entries)", split_json_path, len(split_json))

    # -----------------------------------------------------------------------
    # 7. Cross-split leakage check
    # -----------------------------------------------------------------------
    logger.info("\nChecking for train/test overlap …")
    with open(processed_root / "splits" / "train.json") as fh:
        train_entries = json.load(fh)
    with open(processed_root / "splits" / "test.json") as fh:
        test_entries = json.load(fh)

    train_ids = {e["sample_id"] for e in train_entries}
    test_ids = {e["sample_id"] for e in test_entries}
    overlap = train_ids & test_ids
    if overlap:
        logger.error(
            "TRAIN/TEST OVERLAP DETECTED: %d sample(s) in both splits!", len(overlap)
        )
    else:
        logger.info("No train/test overlap ✓  (train=%d, test=%d)",
                    len(train_ids), len(test_ids))

    # -----------------------------------------------------------------------
    # 8. Combined dataset statistics
    # -----------------------------------------------------------------------
    all_classes_in_train = Counter(e["class_id"] for e in train_entries)
    all_classes_in_test = Counter(e["class_id"] for e in test_entries)

    ds_stats = {
        "dataset_name": "3DGrocery100",
        "n_classes": len(shape_names),
        "class_names": shape_names,
        "total_train": len(train_entries),
        "total_test": len(test_entries),
        "total_samples": len(train_entries) + len(test_entries),
        "train_class_counts": {
            shape_names[k]: v
            for k, v in sorted(all_classes_in_train.items())
        },
        "test_class_counts": {
            shape_names[k]: v
            for k, v in sorted(all_classes_in_test.items())
        },
        "data_format": {
            "file_type": "pickle (.dat)",
            "structure": "[list_of_arrays(1024,6), list_of_int_labels]",
            "point_shape": [1024, 6],
            "channels": "XYZRGB",
            "xyz_range_approx": "[-1, 1] (unit-sphere normalized per object)",
            "rgb_range": "[0.0, 1.0]",
            "sampling_method": "Farthest Point Sampling (FPS) — pre-computed in .dat files",
            "normalization": "Unit-sphere per object — centroid centered, scaled by max L2",
        },
        "split_source": "Official split from modelnet100_train.txt / modelnet100_test.txt",
        "n_classes_in_train": len(all_classes_in_train),
        "n_classes_in_test": len(all_classes_in_test),
    }

    ds_stats_json_path = reports_dir / "dataset_statistics.json"
    with open(ds_stats_json_path, "w", encoding="utf-8") as fh:
        json.dump(ds_stats, fh, indent=2)
    logger.info("Saved: %s", ds_stats_json_path)

    ds_stats_csv_path = reports_dir / "dataset_statistics.csv"
    with open(ds_stats_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_id", "class_name", "train_count", "test_count"])
        for cls_id, cls_name in enumerate(shape_names):
            writer.writerow([
                cls_id, cls_name,
                all_classes_in_train.get(cls_id, 0),
                all_classes_in_test.get(cls_id, 0),
            ])
    logger.info("Saved: %s", ds_stats_csv_path)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Classes           : {len(shape_names)}")
    print(f"  Train samples     : {len(train_entries)}")
    print(f"  Test  samples     : {len(test_entries)}")
    print(f"  Train/test overlap: {len(overlap)} (expected 0)")
    print(f"  labels.json       : {labels_json_path}")
    print(f"  train.json        : {processed_root / 'splits' / 'train.json'}")
    print(f"  test.json         : {processed_root / 'splits' / 'test.json'}")
    print(f"  Reports           : {reports_dir.resolve()}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
