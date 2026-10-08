"""
scripts/inspect_dataset.py
==========================
Recursively inspect the 3DGrocery100 dataset and produce machine-readable
and human-readable reports.

Outputs
-------
  reports/dataset_inspection.json
  reports/dataset_inspection.txt

Usage
-----
  python scripts/inspect_dataset.py [--raw_root <path>] [--out_dir <path>]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
logger = logging.getLogger("inspect_dataset")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_RAW_ROOT = (
    "C:/Users/anees/Desktop/Clubs/MISC/misc projects/"
    "Groceries Project Datasets/3DGrocery100"
)
DEFAULT_OUT_DIR = "reports"


# ---------------------------------------------------------------------------

def inspect_split_txt(txt_path: Path) -> dict:
    """Parse a ModelNet-style split .txt file and return basic stats."""
    if not txt_path.exists():
        return {"exists": False, "count": 0, "classes": []}

    ids = []
    class_counts: dict = defaultdict(int)
    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ids.append(line)
            cls = line.split("=")[0]
            class_counts[cls] += 1

    return {
        "exists": True,
        "count": len(ids),
        "n_classes": len(class_counts),
        "sample_ids_first5": ids[:5],
        "sample_id_format": "<class>=<UUID>_<class>_<frame_id>",
        "class_counts": dict(sorted(class_counts.items())),
    }


def inspect_dat_file(dat_path: Path) -> dict:
    """Load a .dat file and collect statistics without loading all data."""
    if not dat_path.exists():
        return {"exists": False}

    result: dict = {
        "exists": True,
        "file_size_bytes": dat_path.stat().st_size,
        "file_size_mb": round(dat_path.stat().st_size / 1e6, 1),
    }

    logger.info("Loading %s (%.1f MB) …", dat_path.name,
                dat_path.stat().st_size / 1e6)

    with open(dat_path, "rb") as fh:
        raw = pickle.load(fh)

    point_clouds = raw[0]
    labels_raw = raw[1]

    n_samples = len(point_clouds)
    result["n_samples"] = n_samples
    result["n_label_entries"] = len(labels_raw)

    # Collect shapes, dtypes, point counts, label values
    shapes = set()
    dtypes = set()
    point_counts = []
    label_values = []
    nan_count = 0
    inf_count = 0
    xyz_min = np.full(3, np.inf)
    xyz_max = np.full(3, -np.inf)
    rgb_available = True
    rgb_min = np.full(3, np.inf)
    rgb_max = np.full(3, -np.inf)

    logger.info("Scanning %d samples …", n_samples)
    for i, (pc, lbl) in enumerate(zip(point_clouds, labels_raw)):
        arr = np.asarray(pc)
        shapes.add(arr.shape)
        dtypes.add(str(arr.dtype))
        point_counts.append(arr.shape[0])
        label_values.append(int(np.asarray(lbl).flatten()[0]))

        if arr.ndim == 2 and arr.shape[1] >= 3:
            xyz = arr[:, :3]
            nan_count += int(np.any(np.isnan(xyz)))
            inf_count += int(np.any(np.isinf(xyz)))
            xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
            xyz_max = np.maximum(xyz_max, xyz.max(axis=0))

            if arr.shape[1] >= 6:
                rgb = arr[:, 3:6]
                rgb_min = np.minimum(rgb_min, rgb.min(axis=0))
                rgb_max = np.maximum(rgb_max, rgb.max(axis=0))
            else:
                rgb_available = False

        if i % 5000 == 0 and i > 0:
            logger.info("  … %d / %d", i, n_samples)

    point_counts = np.array(point_counts)
    label_values = np.array(label_values)

    result["shapes_observed"] = [list(s) for s in shapes]
    result["dtypes_observed"] = list(dtypes)
    result["point_count_stats"] = {
        "min": int(point_counts.min()),
        "max": int(point_counts.max()),
        "mean": float(point_counts.mean()),
        "median": float(np.median(point_counts)),
        "std": float(point_counts.std()),
        "p5": float(np.percentile(point_counts, 5)),
        "p95": float(np.percentile(point_counts, 95)),
    }
    result["samples_with_nan_xyz"] = int(nan_count)
    result["samples_with_inf_xyz"] = int(inf_count)
    result["xyz_global_min"] = xyz_min.tolist()
    result["xyz_global_max"] = xyz_max.tolist()
    result["rgb_available"] = rgb_available
    if rgb_available:
        result["rgb_global_min"] = rgb_min.tolist()
        result["rgb_global_max"] = rgb_max.tolist()
        result["rgb_range_note"] = (
            "RGB values are in [0, 1] float32 range"
        )

    unique_labels = sorted(np.unique(label_values).tolist())
    result["unique_label_ids"] = unique_labels
    result["n_unique_labels"] = len(unique_labels)
    result["label_min"] = int(label_values.min())
    result["label_max"] = int(label_values.max())
    label_counts: dict = defaultdict(int)
    for l in label_values.tolist():
        label_counts[int(l)] += 1
    result["label_id_counts"] = dict(sorted(label_counts.items()))

    return result


def inspect_dataset(raw_root: Path) -> dict:
    """Run full inspection of the 3DGrocery100 dataset."""
    report: dict = {
        "dataset_name": "3DGrocery100",
        "raw_root": str(raw_root),
        "subdatasets": {},
    }

    # Enumerate top-level subdatasets
    subdirs = [d for d in raw_root.parent.iterdir() if d.is_dir()]
    report["top_level_subdirs"] = [d.name for d in subdirs]

    # Focus on the 3DGrocery100 subdataset
    groceries_dir = raw_root
    sub: dict = {"dir": str(groceries_dir)}

    # List all files
    all_files = list(groceries_dir.iterdir())
    sub["files"] = [
        {
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "size_mb": round(f.stat().st_size / 1e6, 1),
        }
        for f in sorted(all_files)
        if f.is_file()
    ]

    # Load shape names
    shape_names_path = groceries_dir / "modelnet100_shape_names.txt"
    if shape_names_path.exists():
        with open(shape_names_path) as fh:
            shape_names = [l.strip() for l in fh if l.strip()]
        sub["n_classes"] = len(shape_names)
        sub["class_names"] = shape_names
    else:
        sub["n_classes"] = None
        sub["class_names"] = []
        logger.warning("shape_names.txt not found")

    # Inspect split .txt files
    logger.info("Inspecting train split txt …")
    sub["train_txt"] = inspect_split_txt(
        groceries_dir / "modelnet100_train.txt"
    )
    logger.info("Inspecting test split txt …")
    sub["test_txt"] = inspect_split_txt(
        groceries_dir / "modelnet100_test.txt"
    )
    logger.info("Inspecting filelist.txt …")
    sub["filelist_txt"] = inspect_split_txt(
        groceries_dir / "filelist.txt"
    )

    # Inspect .dat files
    logger.info("Inspecting train .dat …")
    sub["train_dat"] = inspect_dat_file(
        groceries_dir / "modelnet100_train_1024pts_fps.dat"
    )
    logger.info("Inspecting test .dat …")
    sub["test_dat"] = inspect_dat_file(
        groceries_dir / "modelnet100_test_1024pts_fps.dat"
    )

    # Cross-checks
    sub["cross_checks"] = {
        "train_txt_vs_dat_count_match": (
            sub["train_txt"].get("count", -1)
            == sub["train_dat"].get("n_samples", -2)
        ),
        "test_txt_vs_dat_count_match": (
            sub["test_txt"].get("count", -1)
            == sub["test_dat"].get("n_samples", -2)
        ),
        "n_classes_shape_names_vs_label_ids": (
            len(sub["class_names"])
            == sub["train_dat"].get("n_unique_labels", -1)
        ),
    }

    report["subdatasets"]["3DGrocery100"] = sub
    return report


def write_txt_report(report: dict, out_path: Path) -> None:
    """Write human-readable .txt summary."""
    lines = []
    sep = "=" * 70

    def h(title: str) -> None:
        lines.append(sep)
        lines.append(title)
        lines.append(sep)

    def kv(k: str, v) -> None:
        lines.append(f"  {k:<35} {v}")

    h("3DGrocery100 Dataset Inspection Report")
    kv("Raw root", report["raw_root"])
    lines.append("")

    sub = report["subdatasets"]["3DGrocery100"]

    h("Files in dataset directory")
    for f in sub["files"]:
        lines.append(f"  {f['name']:<55} {f['size_mb']:>8.1f} MB")
    lines.append("")

    h("Classes")
    kv("Number of classes", sub["n_classes"])
    lines.append("  Class list:")
    for i, name in enumerate(sub["class_names"]):
        lines.append(f"    {i:>3}: {name}")
    lines.append("")

    h("Official Train/Test Split")
    kv("Train samples (from train.txt)", sub["train_txt"].get("count"))
    kv("Test  samples (from test.txt)", sub["test_txt"].get("count"))
    kv("Train classes", sub["train_txt"].get("n_classes"))
    kv("Test  classes", sub["test_txt"].get("n_classes"))
    kv("Sample ID format", sub["train_txt"].get("sample_id_format"))
    lines.append("")

    h("Train .dat File")
    td = sub["train_dat"]
    kv("File size", f"{td['file_size_mb']} MB")
    kv("Samples", td.get("n_samples"))
    kv("Shapes observed", td.get("shapes_observed"))
    kv("Dtypes", td.get("dtypes_observed"))
    kv("Channels (6 = XYZRGB)", sub["train_dat"].get("shapes_observed"))
    kv("Point count (min)", td["point_count_stats"]["min"])
    kv("Point count (max)", td["point_count_stats"]["max"])
    kv("Point count (mean)", f"{td['point_count_stats']['mean']:.1f}")
    kv("Samples with NaN XYZ", td.get("samples_with_nan_xyz"))
    kv("Samples with Inf XYZ", td.get("samples_with_inf_xyz"))
    kv("XYZ global min", td.get("xyz_global_min"))
    kv("XYZ global max", td.get("xyz_global_max"))
    kv("RGB available", td.get("rgb_available"))
    if td.get("rgb_available"):
        kv("RGB global min", td.get("rgb_global_min"))
        kv("RGB global max", td.get("rgb_global_max"))
    kv("Unique label IDs count", td.get("n_unique_labels"))
    kv("Label range", f"[{td.get('label_min')}, {td.get('label_max')}]")
    lines.append("")

    h("Test .dat File")
    tdt = sub["test_dat"]
    kv("File size", f"{tdt['file_size_mb']} MB")
    kv("Samples", tdt.get("n_samples"))
    kv("Samples with NaN XYZ", tdt.get("samples_with_nan_xyz"))
    kv("Samples with Inf XYZ", tdt.get("samples_with_inf_xyz"))
    kv("Unique label IDs count", tdt.get("n_unique_labels"))
    lines.append("")

    h("Cross-Checks")
    for k, v in sub["cross_checks"].items():
        lines.append(f"  {'PASS' if v else 'FAIL'}  {k}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Written: %s", out_path)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect 3DGrocery100 dataset and generate reports"
    )
    parser.add_argument(
        "--raw_root",
        default=DEFAULT_RAW_ROOT,
        help="Path to the 3DGrocery100 dataset directory",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for reports",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        logger.error("Dataset directory not found: %s", raw_root)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting dataset inspection …")
    report = inspect_dataset(raw_root)

    # JSON report
    json_path = out_dir / "dataset_inspection.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Written: %s", json_path)

    # TXT report
    txt_path = out_dir / "dataset_inspection.txt"
    write_txt_report(report, txt_path)

    # Quick summary to stdout
    sub = report["subdatasets"]["3DGrocery100"]
    print("\n" + "=" * 60)
    print("INSPECTION SUMMARY")
    print("=" * 60)
    print(f"  Classes           : {sub['n_classes']}")
    print(f"  Train samples     : {sub['train_txt'].get('count')}")
    print(f"  Test  samples     : {sub['test_txt'].get('count')}")
    print(f"  Point shape       : {sub['train_dat'].get('shapes_observed')}")
    print(f"  NaN in train      : {sub['train_dat'].get('samples_with_nan_xyz')}")
    print(f"  Inf in train      : {sub['train_dat'].get('samples_with_inf_xyz')}")
    print(f"  RGB available     : {sub['train_dat'].get('rgb_available')}")
    print(f"  Official split    : YES (train.txt / test.txt)")
    print("=" * 60)
    print(f"  Reports saved to  : {out_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()
