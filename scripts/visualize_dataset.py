"""
scripts/visualize_dataset.py
============================
Generate representative visualizations of the 3DGrocery100 point clouds.

Outputs (in results/visualizations/):
  - contact_sheet.png          : grid of one sample per category
  - contact_sheet_rgb.png      : same, with RGB colouring
  - point_count_distribution.png
  - class_distribution_train.png
  - cloud_<class_name>.png     : individual class examples

Usage
-----
  python scripts/visualize_dataset.py
  python scripts/visualize_dataset.py --n_classes 20 --out_dir results/visualizations
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.preprocessing import normalize_xyz, set_global_seed, load_shape_names
from common.visualization import (
    plot_contact_sheet,
    plot_point_count_distribution,
    plot_class_distribution,
    plot_point_cloud,
)

logger = logging.getLogger("visualize_dataset")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_RAW_ROOT = (
    "C:/Users/anees/Desktop/Clubs/MISC/misc projects/"
    "Groceries Project Datasets/3DGrocery100"
)
DEFAULT_OUT_DIR = "results/visualizations"
DEFAULT_PROCESSED_ROOT = "data/processed"


def pick_first_per_class(
    point_clouds: list,
    labels_raw: list,
    n_classes: int,
) -> List[Tuple[np.ndarray, int]]:
    """Pick the first sample from each class (deterministic)."""
    seen = {}
    for pc, lbl in zip(point_clouds, labels_raw):
        cls = int(np.asarray(lbl).flatten()[0])
        if cls not in seen:
            seen[cls] = np.asarray(pc, dtype=np.float32)
        if len(seen) == n_classes:
            break
    return [(seen[k], k) for k in sorted(seen.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3DGrocery100 dataset visualizations"
    )
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--processed_root", default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument(
        "--n_classes",
        type=int,
        default=20,
        help="How many classes to include in the contact sheet",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_root = Path(args.processed_root)

    # Load class names
    shape_names_path = raw_root / "modelnet100_shape_names.txt"
    shape_names = load_shape_names(shape_names_path)

    # Load train .dat
    dat_path = raw_root / "modelnet100_train_1024pts_fps.dat"
    logger.info("Loading %s (%.1f MB) …", dat_path.name,
                dat_path.stat().st_size / 1e6)
    with open(dat_path, "rb") as fh:
        raw = pickle.load(fh)
    point_clouds = raw[0]
    labels_raw = raw[1]

    # -------------------------------------------------------------------
    # 1. Point-count distribution
    # -------------------------------------------------------------------
    logger.info("Plotting point-count distribution …")
    counts = [np.asarray(pc).shape[0] for pc in point_clouds]
    plot_point_count_distribution(
        counts,
        out_path=out_dir / "point_count_distribution.png",
        title="3DGrocery100 Train — Point Count Distribution",
    )

    # -------------------------------------------------------------------
    # 2. Class distribution bar chart
    # -------------------------------------------------------------------
    logger.info("Plotting class distribution …")
    # Load from processed split if available
    train_json = processed_root / "splits" / "train.json"
    if train_json.exists():
        with open(train_json) as fh:
            train_entries = json.load(fh)
        from collections import Counter
        cls_counts = Counter(e["class_id"] for e in train_entries)
        cls_names = [shape_names[i] for i in range(len(shape_names))]
        cls_count_list = [cls_counts.get(i, 0) for i in range(len(shape_names))]
    else:
        from collections import Counter
        lbl_list = [int(np.asarray(l).flatten()[0]) for l in labels_raw]
        cls_counts = Counter(lbl_list)
        cls_names = [shape_names[i] for i in range(len(shape_names))]
        cls_count_list = [cls_counts.get(i, 0) for i in range(len(shape_names))]

    plot_class_distribution(
        cls_names,
        cls_count_list,
        out_path=out_dir / "class_distribution_train.png",
        title="3DGrocery100 Train — Samples per Class",
    )

    # -------------------------------------------------------------------
    # 3. Contact sheets
    # -------------------------------------------------------------------
    n = min(args.n_classes, len(shape_names))
    logger.info("Collecting %d representative samples …", n)
    per_class = pick_first_per_class(point_clouds, labels_raw, len(shape_names))
    selected = per_class[:n]

    # XYZ contact sheet (height-coloured)
    logger.info("Plotting XYZ contact sheet …")
    xyz_samples = [
        (normalize_xyz(arr)[:, :3], shape_names[cls_id])
        for arr, cls_id in selected
    ]
    plot_contact_sheet(
        xyz_samples,
        out_path=out_dir / "contact_sheet_xyz.png",
        cols=5,
        title="3DGrocery100 — Sample Point Clouds (XYZ, height-coloured)",
    )

    # XYZRGB contact sheet
    logger.info("Plotting XYZRGB contact sheet …")
    rgb_samples = [
        (normalize_xyz(arr)[:, :6], shape_names[cls_id])
        for arr, cls_id in selected
    ]
    plot_contact_sheet(
        rgb_samples,
        out_path=out_dir / "contact_sheet_rgb.png",
        cols=5,
        title="3DGrocery100 — Sample Point Clouds (RGB)",
    )

    # -------------------------------------------------------------------
    # 4. Individual class examples (first 10)
    # -------------------------------------------------------------------
    logger.info("Plotting individual class examples …")
    for arr, cls_id in per_class[:10]:
        cls_name = shape_names[cls_id]
        pts_norm = normalize_xyz(arr)
        plot_point_cloud(
            pts_norm,
            class_name=cls_name,
            mode="xyzrgb",
            out_path=out_dir / f"cloud_{cls_name.replace('-', '_')}.png",
        )

    logger.info("\nAll visualizations saved to: %s", out_dir.resolve())
    print(f"\nVisualizations saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
